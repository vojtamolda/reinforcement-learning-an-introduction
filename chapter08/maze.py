import gym
import gym.utils.seeding
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class ShortcutMazeEnv(gym.Env):
    """ Shortcut Maze environment per Example 8.5 """
    size = (9, 6)
    start = (3, 0)
    goal = (8, 5)
    shortcut_timestep = 3_000
    walls = [(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2)]

    actions = [(0, +1), (+1, 0), (0, -1), (-1, 0)]
    action_space = gym.spaces.Discrete(len(actions))
    observation_space = gym.spaces.MultiDiscrete(size)
    reward_range = (-1, -1)

    def __init__(self):
        self.np_random = None
        self.walls = self.walls[:]
        self.timestep = 0
        self.position = None
        self.arrow = None
        self.ax = None

    def step(self, action):
        assert self.action_space.contains(action)

        # Check whether the shortcut should be opened
        if self.timestep == self.shortcut_timestep:
            del self.walls[-1]
        self.timestep += 1

        # Calculate move vector and new position
        delta = self.actions[action]
        position = self.position + np.array(delta)

        # Check for collisions with walls
        if tuple(position) in self.walls:
            position = self.position
        else:
            position = np.clip(position, 0, self.observation_space.nvec - 1)

        # Store position for the next step and calculate arrow for rendering
        self.arrow = position - self.position
        self.position = position

        # Check for terminal state
        done = (position == self.goal).all()
        reward = int(done)

        assert self.observation_space.contains(position)
        return position, reward, done, {}

    def reset(self):
        self.position = np.array(self.start)
        self.arrow = np.array((0, 0))
        self.ax = None

        return self.position

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        if self.ax is None:
            fig = plt.figure()
            self.ax = fig.gca()

            # Show map in the background
            gridworld = np.zeros(self.size)
            gridworld[tuple(zip(*self.walls))] = 1.0
            gridworld[8, 2] = 0.5
            self.ax.imshow(gridworld.T, aspect='equal', origin='lower', cmap='Greys')

            # Annotations at start and goal positions
            self.ax.annotate("G", self.goal, size=25, color='gray', ha='center', va='center')
            self.ax.annotate("S", self.start, size=25, color='gray', ha='center', va='center')

            # Major tick marks showing wind strength
            self.ax.set_xticks([])
            self.ax.set_xticklabels([])
            self.ax.set_yticks([])
            self.ax.set_yticklabels([])

            # Thin grid lines at minor tick mark locations
            self.ax.set_xticks(np.arange(-0.5, self.size[0]), minor=True)
            self.ax.set_yticks(np.arange(-0.5, self.size[1]), minor=True)
            self.ax.grid(which='minor', color='black', linewidth=0.20)
            self.ax.tick_params(which='both', length=0)
            self.ax.set_frame_on(True)

        # Arrow pointing from the previous to the current position
        if (self.arrow == 0).all():
            patch = mpatches.Circle(self.position, radius=0.05, color='black', zorder=1)
        else:
            patch = mpatches.FancyArrow(*(self.position - self.arrow), *self.arrow, color='black',
                                        zorder=2, fill=True, width=0.05, head_width=0.25,
                                        length_includes_head=True)
        self.ax.add_patch(patch)


class MultiDiscreteObservationWrapper(gym.ObservationWrapper):
    """ Wrap n-dimentional MultiDiscrete observation space into a large Discrete space with a flat single index. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert type(self.env.observation_space) == gym.spaces.MultiDiscrete
        flat_n = np.ravel_multi_index(self.env.observation_space.nvec - 1, self.env.observation_space.nvec) + 1
        self.observation_space = gym.spaces.Discrete(flat_n)

    def observation(self, observation):
        flat_observation = np.ravel_multi_index(observation, self.env.observation_space.nvec)
        return flat_observation


gym.envs.registration.register(
    id='ShortcutMaze-v0',
    entry_point=lambda: MultiDiscreteObservationWrapper(ShortcutMazeEnv()),
)
