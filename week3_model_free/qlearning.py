from collections import defaultdict, deque
import random
import math
import numpy as np


class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Q-Learning Agent
        based on http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
        Instance variables you have access to
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)

        Functions you should use
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value

        !!!Important!!!
        Note: please avoid using self._qValues directly. 
            There's a special self.get_qvalue/set_qvalue for that.
        """
        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        if len(possible_actions) == 0:
            return 0

        return max(self._qvalues[state][a] for a in possible_actions)

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """
        qsa = self.get_qvalue(state, action)
        qs_next = self.get_value(next_state)
        q_value = (1 - self.alpha) * qsa + self.alpha * (reward + self.discount * qs_next)
        self.set_qvalue(state, action, q_value)

    def get_action(self, state, greedy=False):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.getPolicy).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """
        possible_actions = self.get_legal_actions(state)

        if len(possible_actions) == 0:
            return None

        if not greedy and random.random() < self.epsilon:
            return random.choice(possible_actions)

        qs = self._qvalues[state]
        if len(qs) == 0:
            return random.choice(possible_actions)
        return max(qs, key=qs.get)


class EVSarsaAgent(QLearningAgent):
    """
    An agent that changes some of q-learning functions to implement Expected Value SARSA.
    Note: this demo assumes that your implementation of QLearningAgent.update uses get_value(next_state).
    If it doesn't, please add
        def update(self, state, action, reward, next_state):
            and implement it for Expected Value SARSA's V(s')
    """

    def get_value(self, state):
        """
        Returns Vpi for current state under epsilon-greedy policy:
          V_{pi}(s) = sum _{over a_i} {pi(a_i | s) * Q(s, a_i)}

        Hint: all other methods from QLearningAgent are still accessible.
        """
        possible_actions = self.get_legal_actions(state)
        n_actions = len(possible_actions)

        if n_actions == 0:
            return 0

        qs = self._qvalues[state]
        q_max = max(qs[a] for a in possible_actions)
        n_best = sum(v == q_max for k, v in qs.items() if k in possible_actions)
        eps = self.epsilon / max(1, n_actions - n_best)  # вероятность случайного действия
        value = 0
        for k, v in qs.items():
            value += v * (1 - eps) if v == q_max else v * eps
        return value


class ReplayBuffer:
    def __init__(self, size):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.

        Note: for this assignment you can pick any data structure you want.
              If you want to keep it simple, you can store a list of tuples of (s, a, r, s') in self._storage
              However you may find out there are faster and/or more memory-efficient ways to do so.
        """
        self._storage = deque(maxlen=size)

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        Make sure, _storage will not exceed _maxsize.
        Make sure, FIFO rule is being followed: the oldest examples has to be removed earlier
        """
        data = obs_t, action, reward, obs_tp1, done
        self._storage.append(data)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        batch = [random.choice(self._storage) for _ in range(batch_size)]
        s, a, r, s_next, done = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(s_next), np.array(done)


def play_and_train(env, agent, replay=None, t_max=10 ** 4, replay_batch_size=32):
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total reward
    :param replay: ReplayBuffer where agent can store and sample (s,a,r,s',done) tuples.
        If None, do not use experience replay
    """
    total_reward = 0.0
    s = env.reset()

    for t in range(t_max):
        a = agent.get_action(s)
        next_s, r, done, _ = env.step(a)
        data = s, a, r, next_s
        agent.update(*data)

        if replay:
            replay.add(*data, done)
            batch = replay.sample(replay_batch_size)
            for s, a, r, s_next, done in zip(*batch):
                agent.update(s, a, r, s_next)

        s = next_s
        total_reward += r
        if done:
            break

    return total_reward
