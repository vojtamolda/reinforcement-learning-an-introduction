import gym
import numpy as np


def nstep_on_policy_return(v, done, states, rewards):
    """ Calculate un-discounted n-step on-policy return per (7.1) """
    assert len(states) - 1 == len(rewards)

    if not rewards:
        # Append value of the n-th state unless in the termination phase
        return 0 if done else v[states[0]]

    sub_return = nstep_on_policy_return(v, done, states[1:], rewards[1:])
    return rewards[0] + sub_return


def td_on_policy_prediction(env, policy, n, num_episodes, alpha=0.5, tderr=False):
    """ n-step TD algorithm for on-policy value prediction per Chapter 7.1. Value function updates are
     calculated by summing TD errors per Exercise 7.2 (tderr=True) or with (7.2) (tderr=False). """
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.Discrete

    # Number of available actions and states
    n_state, n_action = env.observation_space.n, env.action_space.n,
    assert policy.shape == (n_state, n_action)

    # Initialization of value function
    v = np.ones([n_state], dtype=np.float) * 0.5

    history = []
    for episode in range(num_episodes):
        # Reset the environment and initialize n-step rewards and states
        state = env.reset()
        nstep_states = [state]
        nstep_rewards = []

        dv = np.zeros_like(v)

        done = False
        while nstep_rewards or not done:
            if not done:
                # Step the environment forward and check for termination
                action = np.random.choice(n_action, p=policy[state])
                state, reward, done, info = env.step(action)

                # Accumulate n-step rewards and states
                nstep_rewards.append(reward)
                nstep_states.append(state)

                # Keep accumulating until there's enough for the first n-step update
                if len(nstep_rewards) < n:
                    continue
                assert len(nstep_states) - 1 == len(nstep_rewards) == n

            # Calculate un-discounted n-step return per (7.1)
            v_target = nstep_on_policy_return(v, done, nstep_states, nstep_rewards)

            if tderr is True:
                # Accumulate TD errors over the episode while v is kept constant per Exercise 7.2
                dv[nstep_states[0]] += alpha * (v_target - v[nstep_states[0]])
            else:
                # Update value function toward the target per (7.2)
                v[nstep_states[0]] += alpha * (v_target - v[nstep_states[0]])

            # Remove the used n-step reward and states
            del nstep_rewards[0]
            del nstep_states[0]

            # Update value function with the sum of TD errors accumulated during the episode
            v += dv

        history += [np.copy(v)]
    return history


def nstep_off_policy_per_decision_return(v, done, states, rewards, isrs):
    """ Calculate un-discounted n-step per decision off-policy return per (7.13) """
    assert len(states) - 1 == len(rewards) == len(isrs)

    if not rewards:
        return 0 if done else v[states[0]]

    sub_return = nstep_off_policy_per_decision_return(v, done, states[1:], rewards[1:], isrs[1:])
    return isrs[0] * (rewards[0] + sub_return) + (1 - isrs[0]) * v[states[0]]


def td_off_policy_prediction(env, target, behavior, n, num_episodes, alpha=1e-3, simpler=True):
    """ n-step TD algorithm for off-policy value prediction per Chapter 7.4. Returns and value function updates
     are calculated using (7.1) and (7.9) (simpler=True) or (7.13) and (7.2) (simpler=False) """
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.Tuple

    # Number of available actions and states
    n_action = env.action_space.n
    n_state = [space.n for space in env.observation_space.spaces]
    assert target.shape == tuple(n_state + [n_action])
    assert behavior.shape == tuple(n_state + [n_action])

    # Initialization of value function
    v = np.zeros(n_state, dtype=np.float)

    history = []
    for episode in range(num_episodes):
        # Reset the environment and initialize n-step rewards and states
        state = env.reset()
        nstep_states = [state]
        nstep_rewards = []
        nstep_isrs = []

        done = False
        while nstep_rewards or not done:
            if not done:
                # Step the environment forward
                action = np.random.choice(n_action, p=behavior[state])
                state, reward, done, info = env.step(action)

                # Accumulate n-step rewards, states and importance sampling ratios
                nstep_rewards.append(reward)
                nstep_states.append(state)
                nstep_isrs.append(target[state + (action,)] / behavior[state + (action,)])

                # Keep accumulating until there's enough for the first n-step update
                if len(nstep_rewards) < n:
                    continue
                assert len(nstep_states) - 1 == len(nstep_rewards) == len(nstep_isrs) == n

            if simpler is True:
                # Calculate un-discounted n-step return per (7.1)
                v_target = nstep_on_policy_return(v, done, nstep_states, nstep_rewards)
                # Multiply n-step importance sampling ratios
                nstep_isr = np.prod(nstep_isrs)
                # Update value function toward the target per (7.9)
                v[nstep_states[0]] += alpha * nstep_isr * (v_target - v[nstep_states[0]])
            else:
                # Calculate un-discounted n-step return per (7.13)
                v_target = nstep_off_policy_per_decision_return(v, done, nstep_states, nstep_rewards, nstep_isrs)
                # Update value function toward the target per (7.2)
                v[nstep_states[0]] += alpha * (v_target - v[nstep_states[0]])

            # Remove the used n-step reward and states
            del nstep_rewards[0]
            del nstep_states[0]
            del nstep_isrs[0]

        history += [np.copy(v)]
    return history
