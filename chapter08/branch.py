import gym
import gym.utils.seeding
import numpy as np


class BranchingStatesEnv(gym.Env):
    """ Branching States environment per Chapter 8.6 """
    action_space = gym.spaces.Discrete(2)
    reward_range = (-np.inf, +np.inf)
    p_termination = 0.10

    def __init__(self, n=1_000, b=1):
        self.observation_space = gym.spaces.Discrete(n)
        self.n = n
        self.b = b

        self.np_random = None
        self.transitions = None
        self.rewards = None
        self.probs = None
        self.terminals = None
        self.start = None
        self.seed()

        self.state = None
        self.reset()

    def branches(self, state, action):
        transitions = self.transitions[state, action]
        rewards = self.rewards[state, action]
        probs = self.probs[state, action]
        terminals = self.terminals[state, action]
        return transitions, rewards, probs, terminals

    def step(self, action):
        transitions, rewards, probs, terminals = self.branches(self.state, action)
        idx = self.np_random.choice(1 + self.b, p=probs)
        self.state = transitions[idx]
        return transitions[idx], rewards[idx], terminals[idx], {}

    def reset(self):
        self.state = self.start
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)

        size = [self.observation_space.n, self.action_space.n, 1 + self.b]
        self.transitions = self.np_random.randint(low=1, high=self.observation_space.n, size=size)
        self.transitions[:, :, 0] = 0
        self.transitions[0, :, :] = 0

        self.rewards = self.np_random.normal(loc=0, scale=1, size=size)
        self.rewards[:, :, 0] = 0.0
        self.rewards[0, :, :] = 0.0

        self.probs = np.zeros(size, dtype=np.float)
        self.probs[:, :, :] = (1 - self.p_termination) / self.b
        self.probs[:, :, 0] = self.p_termination
        assert np.allclose(np.sum(self.probs, axis=2), 1)

        self.terminals = np.zeros(size, dtype=np.bool)
        self.terminals[:, :, :] = 0
        self.terminals[:, :, 0] = 1
        self.terminals[0, :, :] = 1

        self.start = self.np_random.randint(low=1, high=self.observation_space.n)
        return [seed]


gym.envs.registration.register(
    id='BranchingStates-v0',
    entry_point=lambda n, b: BranchingStatesEnv(n, b),
    nondeterministic=True,
    kwargs={'n': 1_000, 'b': 3}
)
