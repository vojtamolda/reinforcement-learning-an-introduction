import gym
import itertools
import numpy as np


def evaluate(env, q, state, depth=3):
    """ Evaluate state under the greedy policy implied by action value function q """
    if depth == 0:
        return np.max(q[state])

    action = np.argmax(q[state])
    transitions, rewards, probs, terminals = env.branches(state, action)
    v_transitions = [evaluate(env, q, transition, depth - 1) for transition in transitions]

    value = np.sum(probs * (rewards + v_transitions))
    return value


def uniform_sampling(env, num_updates=20_000):
    """ Random state and action uniform sampling per Chapter 8.6 """
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.Discrete

    q = np.zeros([env.observation_space.n, env.action_space.n])
    history = []

    states_actions = itertools.product(range(env.observation_space.n), range(env.action_space.n))
    states_actions = itertools.cycle(states_actions)

    for update in range(num_updates):
        state, action = next(states_actions)

        transitions, rewards, probs, terminals = env.branches(state, action)
        q[state, action] = np.sum(probs * (rewards + np.max(q[transitions], axis=1) * ~terminals))

        if update % 100 == 0:
            history += [(update, evaluate(env, q, env.start))]

    return zip(*history)


def trajectory_sampling(env, num_updates=20_000, eps=0.1):
    """ On-policy eps-greedy trajectory sampling per Chapter 8.6 """
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.Discrete

    q = np.zeros([env.observation_space.n, env.action_space.n])
    history = []

    state = env.reset()
    for update in range(num_updates):
        action = env.action_space.sample() if np.random.rand() < eps else np.argmax(q[state])

        transitions, rewards, probs, terminals = env.branches(state, action)
        q[state, action] = np.sum(probs * (rewards + np.max(q[transitions], axis=1) * ~terminals))

        state, reward, done, info = env.step(action)
        if done:
            state = env.reset()

        if update % 100 == 0:
            history += [(update, evaluate(env, q, env.start))]

    return zip(*history)
