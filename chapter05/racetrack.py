import gym
import numpy as np
import skimage.draw

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


class RacetrackEnv(gym.Env):
    """ Racetrack environment per Exercise 5.12 """
    max_accel = 1
    max_speed = 5
    grass, road, start, finish = 0, 1, 2, 3

    action_space = gym.spaces.MultiDiscrete([2 * max_accel + 1, 2 * max_accel + 1])
    reward_range = (-1, -1)

    def __init__(self, racetrack_csv, noisy=True):
        self.racetrack = np.fliplr(np.genfromtxt(racetrack_csv, delimiter=',', dtype=np.int).T)
        self.observation_space = gym.spaces.MultiDiscrete([*self.racetrack.shape, self.max_speed, self.max_speed])
        self.state = None

        self.noisy = noisy
        self.ax = None

    def step(self, action):
        # Decompose state into car position and speed
        position, speed = self.s2ps(self.state)

        # Calculate acceleration by offsetting the action
        if self.noisy and np.random.rand() < 0.1:
            acceleration = np.array([0, 0])
        else:
            acceleration = action - self.max_accel
        assert (-self.max_accel <= acceleration).all() and (acceleration <= self.max_accel).all()

        # Solve pseudo-ODE for speed
        speed += acceleration

        # Clip speed between 0 and max_speed and don't allow 0
        speed = np.clip(speed, 0, self.max_speed - 1)
        speed = speed if (speed > 0).any() else np.array([0, 1])
        assert (0 < speed).any() and (speed < self.max_speed).all()

        # Solve pseudo-ODE for position
        position += speed

        # Calculate intersection of the speed vector and the racetrack
        xs, ys = skimage.draw.line(*(position - speed), *position)
        xs_within_track = np.clip(xs, 0, self.racetrack.shape[0] - 1)
        ys_within_track = np.clip(ys, 0, self.racetrack.shape[1] - 1)
        collisions = self.racetrack[xs_within_track, ys_within_track]

        # Check whether the car is on the road, out of track or crossing the finish line
        within_track_limits = (xs == xs_within_track).all() and (ys == ys_within_track).all()
        crossing_finish = (collisions == self.finish).any()
        on_grass = (collisions == self.grass).any()

        # Restart when on the grass or going out of racetrack limits without crossing the finish line
        if on_grass or (not within_track_limits and not crossing_finish):
            self.state = self.ps2s(position, speed)
            self.reset(hard=False)
            return self.state, -1.0, False, {}

        # Clip position to racetrack limits (applies when crossing the finish line)
        clipped = np.clip(position, 0, np.array(self.racetrack.shape) - 1)
        speed -= position - clipped
        position = clipped
        assert (0 <= position).all() and (position <= self.racetrack.shape).all()

        # Compose state from new car position and speed
        self.state = self.ps2s(position, speed)
        return self.state, -1.0, crossing_finish, {}

    def reset(self, state=None, hard=True):
        if self.ax is not None:
            reset = np.copy(self.state)
            self.render()

        if state is None:
            starts = np.where(self.racetrack == self.start)
            idx = np.random.randint(starts[0].size)
            position = np.array([starts[0][idx], starts[1][idx]])
            speed = np.array([0, 0])

            state = self.ps2s(position, speed)
            self.state = state
        else:
            self.state = state

        self.ax = None if hard else self.ax
        if self.ax is not None:
            self.render(reset=reset)

        return self.state

    def render(self, mode='human', reset=None):
        if self.ax is None:
            fig = plt.figure()
            self.ax = fig.gca()

            # Racetrack background with custom colormap
            cmap = mcolors.ListedColormap(['white', 'gray', 'red', 'green'])
            self.ax.imshow(self.racetrack.T, aspect='equal', origin='lower', cmap=cmap)

            # Major tick marks max_speed step apart
            self.ax.set_xticks(np.arange(0, self.racetrack.shape[0], self.max_speed), minor=False)
            self.ax.set_yticks(np.arange(0, self.racetrack.shape[1], self.max_speed), minor=False)

            margin = 1
            # Thin grid lines at minor tick mark locations
            self.ax.set_xticks(np.arange(-0.5 - margin, self.racetrack.shape[0] + margin, 1), minor=True)
            self.ax.set_yticks(np.arange(-0.5 - margin, self.racetrack.shape[1] + margin, 1), minor=True)
            self.ax.grid(which='minor', color='black', linewidth=0.05)
            self.ax.tick_params(which='minor', length=0)
            self.ax.set_frame_on(False)

        position, speed = self.s2ps(self.state)

        if reset is not None:
            # Reset arrow pointing from the reset position to the current car position
            reset_position, reset_speed = self.s2ps(reset)
            patch = mpatches.FancyArrow(*reset_position, *(position - reset_position), color='blue',
                                        fill=False, width=0.10, head_width=0.25, length_includes_head=True)
        else:
            # Speed arrow pointing to the the current car position
            if (speed == 0).all():
                patch = mpatches.Circle(position, radius=0.1, color='black', zorder=1)
            else:
                patch = mpatches.FancyArrow(*(position - speed), *speed, color='black',
                                            zorder=2, fill=True, width=0.05, head_width=0.25, length_includes_head=True)
        self.ax.add_patch(patch)
        return self.ax

    @classmethod
    def s2ps(cls, state):
        position, speed = np.hsplit(np.copy(state), 2)
        return position, speed

    @classmethod
    def ps2s(cls, position, speed):
        state = np.hstack([position, speed])
        return np.copy(state)


gym.envs.registration.register(
    id='Racetrack-v0',
    entry_point=lambda track, noisy: RacetrackEnv(f"racetrack/{track}.csv", noisy),
    nondeterministic=True,
    kwargs={'track': 'test', 'noisy': True}
)
