import gym
import numpy as np


def sample_episode(env, policy, render=False):
    """ Follow policy through an episode and return arrays of visited actions, states and returns """
    choices_ridxs = np.arange(int(np.prod(env.action_space.nvec)))
    state_ridxs = []
    action_ridxs = []
    rewards = []

    done = False
    state = env.reset()
    if render:
        env.render()

    while not done:
        state_ridx = np.ravel_multi_index(state, env.observation_space.nvec)
        state_ridxs += [state_ridx]

        # Sample action from the policy
        action_ridx = np.random.choice(choices_ridxs, p=policy[state_ridx])
        action = np.array(np.unravel_index(action_ridx, env.action_space.nvec))
        action_ridxs += [action_ridx]

        # Step the environment forward and take the sampled action
        state, reward, done, info = env.step(action)
        rewards += [reward]

        if render:
            env.render()

    # Returns without discounting
    returns = np.cumsum(rewards[::-1])[::-1]

    assert len(state_ridxs) == len(action_ridxs) == len(returns)
    return state_ridxs, action_ridxs, returns


def monte_carlo_control_eps_soft(env, num_episodes, eps=0.10, alpha=0.05):
    """ Every-visit Monte Carlo algorithm for eps-soft policies per Chapter 5.4 """
    assert type(env.action_space) == gym.spaces.MultiDiscrete
    assert type(env.observation_space) == gym.spaces.MultiDiscrete

    # Maximal state and action ravel indices
    n_action_ridx = np.ravel_multi_index(env.action_space.nvec - 1, env.action_space.nvec) + 1
    n_state_ridx = np.ravel_multi_index(env.observation_space.nvec - 1, env.observation_space.nvec) + 1

    # Optimistic (i.e. encouraging exploration) initialization of state-action values
    q = np.ones([n_state_ridx, n_action_ridx], dtype=np.float)

    # Initialize policy to eps-soft greedy wrt to random q
    policy = np.ones([n_state_ridx, n_action_ridx], dtype=np.float) / n_action_ridx

    for episode in range(num_episodes):
        # Sample an episode and collect first-visit states, actions & returns
        state_ridxs, action_ridxs, returns = sample_episode(env, policy)

        # Update the state-action values with first-visit returns
        for state_ridx, action_ridx, retrn in zip(state_ridxs, action_ridxs, returns):
            q[state_ridx, action_ridx] += alpha * (retrn - q[state_ridx, action_ridx])

        # Update policy to be eps-soft greedy wrt to updated q values
        greedy_action_ridxs = np.argmax(q[state_ridxs, :], axis=1)
        policy[state_ridxs, :] = eps / n_action_ridx
        policy[state_ridxs, greedy_action_ridxs] = 1 - eps + eps / n_action_ridx
        assert np.allclose(np.sum(policy, axis=1), 1)

        if episode % 10_000 == 0:
            print(f"Episode {episode}/{num_episodes}: #updates={len(state_ridxs)} return={min(returns)}")

    # Return deterministic greedy policy wrt q values
    greedy_action_ridxs = np.argmax(q, axis=1)
    policy[:, :] = 0
    policy[np.arange(n_state_ridx), greedy_action_ridxs] = 1
    assert np.allclose(np.sum(policy, axis=1), 1)

    return q, policy
