import gym
import numpy as np


def run_episode(env, policy=None, render=True):
    """ Follow policy through an environment's episode and return an array of collected rewards """
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.MultiDiscrete

    state = env.reset()
    if render:
        env.render()

    done = False
    rewards = []
    while not done:
        state_ridx = np.ravel_multi_index(state, env.observation_space.nvec)
        action = np.argmax(policy[state_ridx])
        state, reward, done, info = env.step(action)
        rewards += [reward]

        if render:
            env.render()

    if render:
        import matplotlib.pyplot as plt
        plt.show()

    return rewards


def sarsa(env, num_episodes, eps0=0.5, alpha=0.5):
    """ On-policy Sarsa algorithm per Chapter 6.4 (with exploration rate decay) """
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.MultiDiscrete

    # Number of available actions and maximal state ravel index
    n_action = env.action_space.n
    n_state_ridx = np.ravel_multi_index(env.observation_space.nvec - 1, env.observation_space.nvec) + 1

    # Initialization of action value function
    q = np.zeros([n_state_ridx, n_action], dtype=np.float)

    # Initialize policy to equal-probable random
    policy = np.ones([n_state_ridx, n_action], dtype=np.float) / n_action

    history = [0] * num_episodes
    for episode in range(num_episodes):
        # Reset the environment
        state = env.reset()
        state_ridx = np.ravel_multi_index(state, env.observation_space.nvec)
        action = np.random.choice(n_action, p=policy[state_ridx])

        done = False
        while not done:
            # Step the environment forward and check for termination
            next_state, reward, done, info = env.step(action)
            next_state_ridx = np.ravel_multi_index(next_state, env.observation_space.nvec)
            next_action = np.random.choice(n_action, p=policy[next_state_ridx])

            # Update q values
            q[state_ridx, action] += alpha * (reward + q[next_state_ridx, next_action] - q[state_ridx, action])

            # Extract eps-greedy policy from the updated q values
            eps = eps0 / (episode + 1)
            policy[state_ridx, :] = eps / n_action
            policy[state_ridx, np.argmax(q[state_ridx])] = 1 - eps + eps / n_action
            assert np.allclose(np.sum(policy, axis=1), 1)

            # Prepare the next q update
            state_ridx = next_state_ridx
            action = next_action
            history[episode] += 1

    return q, policy, history
