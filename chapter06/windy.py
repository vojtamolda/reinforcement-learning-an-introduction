import gym
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class WindyGridworldEnv(gym.Env):
    """ Windy Gridworld environment per Example 6.5 (defaults), Exercise 6.9 (king=True, stop=True) or
     Exercise 6.10 (king=True, stochastic=True) """
    size = (10, 7)
    start = (0, 3)
    goal = (7, 3)
    wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

    stop_action = [(0, 0)]
    pawns_actions = [(0, +1), (+1, 0), (0, -1), (-1, 0)]
    kings_action = [(+1, +1), (+1, -1), (-1, -1), (-1, +1)]

    observation_space = gym.spaces.MultiDiscrete(size)
    reward_range = (-1, -1)

    def __init__(self, king=False, stop=False, stochastic=False):
        self.stochastic = stochastic
        self.king = king
        self.stop = stop

        self.actions = self.pawns_actions[:]
        if self.king:
            self.actions += self.kings_action
        if self.stop:
            self.actions += self.stop_action
        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.position = None
        self.arrow = None

        self.ax = None

    def step(self, action):
        assert self.action_space.contains(action)

        # Calculate move vector and new position
        delta = self.actions[action]
        position = self.position + np.array(delta)

        # Add wind to the position
        wind = self.wind[self.position[0]]
        if self.stochastic and wind > 0:
            wind += np.random.choice([-1, 0, +1])
        position[1] += wind

        # Store position for the next step and calculate arrow for rendering
        position = np.clip(position, 0, self.observation_space.nvec - 1)
        self.arrow = position - self.position
        self.position = position

        # Check for terminal state
        done = (position == self.goal).all()
        reward = -1

        assert self.observation_space.contains(position)
        return position, reward, done, {}

    def reset(self):
        self.position = np.array(self.start)
        self.arrow = np.array((0, 0))

        self.ax = None

        return self.position

    def render(self, mode='human'):
        if self.ax is None:
            fig = plt.figure()
            self.ax = fig.gca()

            # Background colored by wind strength
            wind = np.vstack([self.wind] * self.size[1])
            self.ax.imshow(wind, aspect='equal', origin='lower', cmap='Blues')

            # Annotations at start and goal positions
            self.ax.annotate("G", self.goal, size=25, color='gray', ha='center', va='center')
            self.ax.annotate("S", self.start, size=25, color='gray', ha='center', va='center')

            # Major tick marks showing wind strength
            self.ax.set_xticks(np.arange(len(self.wind)))
            self.ax.set_xticklabels(self.wind)
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


gym.envs.registration.register(
    id='WindyGridworld-v0',
    entry_point=lambda king, stop, stochastic: WindyGridworldEnv(king, stop, stochastic),
    kwargs={'king': False, 'stop': False, 'stochastic': False},
    max_episode_steps=5_000,
)
