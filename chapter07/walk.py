import gym
import gym.utils.seeding


class RandomWalkEnv(gym.Env):
    """ Random Walk environment per Example 6.2 (with size=5) """
    action_space = gym.spaces.Discrete(1)
    reward_range = (-1, -1)

    def __init__(self, size=5):
        self.observation_space = gym.spaces.Discrete(size)
        self.np_random = None
        self.position = None

        self.seed()

    def step(self, action):
        assert self.action_space.contains(action)

        delta = self.np_random.choice([-1, +1])
        self.position += delta
        reward, done = 0, False

        if self.position < 0:
            self.position = 0
            reward, done = 0, True
        if self.observation_space.n <= self.position:
            self.position = self.observation_space.n - 1
            reward, done = 1, True

        assert self.observation_space.contains(self.position)
        return self.position, reward, done, {}

    def reset(self):
        self.position = self.observation_space.n // 2
        return self.position

    def render(self, mode='ansi'):
        letters = [chr(ord('A') + i) for i in range(self.observation_space.n)]
        letters[self.position] = "#"
        string = "-".join(letters)
        print(string)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


gym.envs.registration.register(
    id='RandomWalk-v0',
    entry_point=lambda size: RandomWalkEnv(size),
    nondeterministic=True,
    kwargs={'size': 5}
)
