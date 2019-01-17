from math import log
from gym.core import Wrapper
from pickle import dumps, loads
from collections import namedtuple

# a container for get_result function below. Works just like tuple, but prettier
ActionResult = namedtuple("action_result", ("snapshot", "observation", "reward", "is_done", "info"))


class WithSnapshots(Wrapper):
    """
    Creates a wrapper that supports saving and loading environemnt states.
    Required for planning algorithms.

    This class will have access to the core environment as self.env, e.g.:
    - self.env.reset()           #reset original env
    - self.env.ale.cloneState()  #make snapshot for atari. load with .restoreState()
    - ...

    You can also use reset, step and render directly for convenience.
    - s, r, done, _ = self.step(action)   #step, same as self.env.step(action)
    - self.render(close=True)             #close window, same as self.env.render(close=True)
    """

    def get_snapshot(self):
        """
        :returns: environment state that can be loaded with load_snapshot
        Snapshots guarantee same env behaviour each time they are loaded.

        Warning! Snapshots can be arbitrary things (strings, integers, json, tuples)
        Don't count on them being pickle strings when implementing MCTS.

        Developer Note: Make sure the object you return will not be affected by
        anything that happens to the environment after it's saved.
        You shouldn't, for example, return self.env.
        In case of doubt, use pickle.dumps or deepcopy.

        """
        self.render()  # close popup windows since we can't pickle them
        if self.unwrapped.viewer is not None:
            self.unwrapped.viewer.close()
            self.unwrapped.viewer = None
        return dumps(self.env)

    def load_snapshot(self, snapshot):
        """
        Loads snapshot as current env state.
        Should not change snapshot inplace (in case of doubt, deepcopy).
        """
        assert not hasattr(self, "_monitor") or hasattr(self.env, "_monitor"), "can't backtrack while recording"

        self.render()
        self.env.close()
        self.env = loads(snapshot)

    def get_result(self, snapshot, action):
        """
        A convenience function that
        - loads snapshot,
        - commits action via self.step,
        - and takes snapshot again :)

        :returns: next snapshot, next_observation, reward, is_done, info

        Basically it returns next snapshot and everything that env.step would have returned.
        """
        self.load_snapshot(snapshot)
        next_observation, reward, is_done, info = self.step(action)
        next_snapshot = self.get_snapshot()
        return ActionResult(next_snapshot, next_observation, reward, is_done, info)


class Node:
    """ a tree node for MCTS """

    # metadata:
    parent = None  # parent Node
    value_sum = 0  # sum of state values from all visits (numerator)
    times_visited = 0  # counter of visits (denominator)

    def __init__(self, env, parent=None, action=None):
        """
        Creates and empty node with no children.
        Does so by commiting an action and recording outcome.

        :param parent: parent Node
        :param action: action to commit from parent Node

        """
        self.env = env
        self.parent = parent
        self.action = action
        self.children = set()  # set of child nodes

        # get action outcome and save it
        if parent:
            res = env.get_result(parent.snapshot, action)
            self.snapshot, self.observation, self.immediate_reward, self.is_done, _ = res

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def get_mean_value(self):
        return self.value_sum / self.times_visited if self.times_visited != 0 else 0

    def ucb_score(self, scale=10, max_value=1e100):
        """
        Computes ucb1 upper bound using current value and visit counts for node and it's parent.

        :param scale: Multiplies upper bound by that. From hoeffding inequality, assumes reward range to be [0,scale].
        :param max_value: a value that represents infinity (for unvisited nodes)

        """
        if self.times_visited == 0:
            return max_value

        N, n = self.parent.times_visited, self.times_visited
        U = (2 * log(N) / n) ** 0.5
        return self.get_mean_value() + scale * U

    def select_best_leaf(self):
        """
        Picks the leaf with highest priority to expand
        Does so by recursively picking nodes with best UCB-1 score until it reaches the leaf.

        """
        if self.is_leaf():
            return self

        best_child = sorted(self.children, key=lambda c: c.ucb_score())[-1]
        return best_child.select_best_leaf()

    def expand(self):
        """
        Expands the current node by creating all possible child nodes.
        Then returns one of those children.
        """
        assert not self.is_done, "can't expand from terminal state"

        n_actions = 2  # КОСТЫЛЬ!
        for action in range(n_actions):
            self.children.add(Node(self, action=action))
        return self.select_best_leaf()

    def rollout(self, t_max=10 ** 4):
        """
        Play the game from this state to the end (done) or for t_max steps.

        On each step, pick action at random (hint: env.action_space.sample()).

        Compute sum of rewards from current state till
        Note 1: use env.action_space.sample() for random action
        Note 2: if node is terminal (self.is_done is True), just return 0

        """
        # set env into the appropriate state
        self.env.load_snapshot(self.snapshot)
        rollout_reward = 0
        if self.is_done:
            return rollout_reward

        for _ in range(t_max):
            action = self.env.action_space.sample()
            _, r, is_done, _ = self.env.step(action)
            rollout_reward += r
            if is_done:
                break
        return rollout_reward

    def propagate(self, child_value):
        """
        Uses child value (sum of rewards) to update parents recursively.
        """
        # compute node value
        my_value = self.immediate_reward + child_value

        # update value_sum and times_visited
        self.value_sum += my_value
        self.times_visited += 1

        # propagate upwards
        if not self.is_root():
            self.parent.propagate(my_value)

    def safe_delete(self):
        """safe delete to prevent memory leak in some python versions"""
        del self.parent
        for child in self.children:
            child.safe_delete()
            del child


class Root(Node):
    def __init__(self, env, snapshot, observation):
        """
        creates special node that acts like tree root
        :snapshot: snapshot (from env.get_snapshot) to start planning from
        :observation: last environment observation
        """
        super().__init__(env=env)
        self.children = set()  # set of child nodes

        # root: load snapshot and observation
        self.snapshot = snapshot
        self.observation = observation
        self.immediate_reward = 0
        self.is_done = False

    def from_node(self, node):
        """initializes node as root"""
        root = Root(self.env, node.snapshot, node.observation)
        copied_fields = ["value_sum", "times_visited", "children", "is_done"]
        for field in copied_fields:
            setattr(root, field, getattr(node, field))
        return root


def plan_mcts(root, n_iters=10):
    """
    builds tree with monte-carlo tree search for n_iters iterations
    :param root: tree node to plan from
    :param n_iters: how many select-expand-simulate-propagete loops to make
    """
    for _ in range(n_iters):
        node = root.select_best_leaf()
        if node.is_done:
            node.propagate(0)
        else:
            child = node.expand()
            print(child.env)
            rollout_reward = child.rollout()
            child.propagate(rollout_reward)
