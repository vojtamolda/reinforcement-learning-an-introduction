import gym
import gym.utils.seeding


class BairdsCounterexample(gym.Env):
    """ Baird's counterexample MDP per Figure 11.1. """

    # Dashed line action = 0, solid line action = 1.
    action_space = gym.spaces.Discrete(2)
    # Upper states = 0-5, lower state = 6.
    observation_space = gym.spaces.Discrete(7)
    # No rewards at all.
    reward_range = (0, 0)

    def __init__(self):
        self.seed()

    def step(self, action):
        assert self.action_space.contains(action)

        state = None
        if action == 0:  # Dashed line action
            state = self.np_random.randint(low=0, high=6)
        if action == 1:  # Solid line action
            state = 6

        return state, 0.0, False, {}

    def reset(self):
        state = self.np_random.randint(self.observation_space.n)
        return state

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


gym.envs.registration.register(
    id='BairdsCounterexample-v0',
    entry_point=lambda: BairdsCounterexample(),
    nondeterministic=True,
)
