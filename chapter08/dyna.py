import gym
import numpy as np


def dyna_q(env, n, num_episodes, eps=0.1, alpha=0.5, gamma=0.95):
    """ Tabular Dyna-Q algorithm per Chapter 8.2 """
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.Discrete
    history = [0]

    # Number of available actions and maximal state ravel index
    n_state, n_action = env.observation_space.n, env.action_space.n

    # Initialization of action value function
    q = np.zeros([n_state, n_action], dtype=np.float)

    # Initialize policy to equal-probable random
    policy = np.ones([n_state, n_action], dtype=np.float) / n_action
    assert np.allclose(np.sum(policy, axis=1), 1)

    # Model of a deterministic environment
    model = {}

    for episode in range(num_episodes):
        state = env.reset()

        done = False
        while not done:
            # Sample action according to the current policy and step the environment forward
            action = np.random.choice(n_action, p=policy[state])
            next, reward, done, info = env.step(action)
            history += [reward]

            # Update q value with a q-learning update and reset visit counter
            q[state, action] += alpha * (reward + gamma * np.max(q[next]) - q[state, action])
            model[state, action] = next, reward

            # Planning with previously visited state-action pairs
            transitions = list(model.keys())
            for i in range(n):
                p_state, p_action = transitions[np.random.choice(len(transitions))]
                p_next, p_reward = model[p_state, p_action]
                q[p_state, p_action] += alpha * (p_reward + gamma * np.max(q[p_next]) - q[p_state, p_action])

            # Extract eps-greedy policy from the updated q values
            policy[state, :] = eps / n_action
            policy[state, np.argmax(q[state])] = 1 - eps + eps / n_action
            assert np.allclose(np.sum(policy, axis=1), 1)

            # Prepare the next q update and increase visit counter for all states
            state = next

    return q, policy, history


def dyna_q_plus(env, n, num_episodes, eps=0.1, alpha=0.5, gamma=0.95, kappa=1e-4, action_only=False):
    """ Tabular Dyna-Q+ algorithm per Chapter 8.3 (action_only=False) or Exercise 8.4 (action_only=True). """
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.Discrete
    history = [0]

    # Number of available actions and maximal state ravel index
    n_state, n_action = env.observation_space.n, env.action_space.n

    # Initialization of action value function and visit counter
    q = np.zeros([n_state, n_action], dtype=np.float)
    tau = np.zeros([n_state, n_action], dtype=np.int)

    # Initialize policy to equal-probable random
    policy = np.ones([n_state, n_action], dtype=np.float) / n_action
    assert np.allclose(np.sum(policy, axis=1), 1)

    # Model of a deterministic environment
    model = {}

    for episode in range(num_episodes):
        state = env.reset()

        done = False
        while not done:
            # Sample action according to the current policy and step the environment forward
            action = np.random.choice(n_action, p=policy[state])
            next, reward, done, info = env.step(action)
            history += [reward]

            # Update q value with a q-learning update and reset visit counter
            q[state, action] += alpha * (reward + gamma * np.max(q[next]) - q[state, action])
            model.setdefault(state, {})[action] = next, reward
            tau[state, action] = 0

            # Planning that allows taking unvisited actions from visited states
            states = list(model.keys())
            for i in range(n):
                p_state = states[np.random.choice(len(states))]
                p_action = np.random.choice(n_action)
                p_next, p_reward = model[p_state].get(p_action, (p_state, 0))
                bonus = 0 if action_only else kappa * np.sqrt(tau[p_state, p_action])
                q[p_state, p_action] += alpha * (p_reward + bonus + gamma * np.max(q[p_next]) - q[p_state, p_action])

            # Extract eps-greedy policy from the updated q values and exploration bonus
            bonus = kappa * np.sqrt(tau[state]) if action_only else 0
            policy[state, :] = eps / n_action
            policy[state, np.argmax(q[state] + bonus)] = 1 - eps + eps / n_action
            assert np.allclose(np.sum(policy, axis=1), 1)

            # Prepare the next q update and increase visit counter for all states
            state = next
            tau += 1

    return q, policy, history
