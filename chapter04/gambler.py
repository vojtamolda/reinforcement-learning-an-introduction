import gym
import numpy as np

import mdp


class GamblersProblem(mdp.MDP):
    """ Gambler's Problem MDP per Example 4.3 with minimal bet of 1. """
    n = 100
    min_bet = 1

    gamma = 1
    reward_range = (0, 1)
    terminals = [np.array([0]), np.array([n])]
    action_space = gym.spaces.Discrete(n - min_bet)
    observation_space = gym.spaces.MultiDiscrete([n + 1])

    def __init__(self, p_win=0.5):
        self.p_win = p_win
        self.reset()

    def transitions(self, capital, bet):
        assert self.action_space.contains(bet)
        assert self.observation_space.contains(capital)

        transitions = np.zeros((2, 1), dtype=np.int)
        rewards = np.zeros(2, dtype=np.float)
        ps = np.zeros(2, dtype=np.float)
        terminals = np.zeros(2, dtype=np.bool)

        bet = min(capital, bet + self.min_bet)

        capital_win = min(capital + bet, self.terminals[1])
        transitions[1, 0] = capital_win
        rewards[1] = float(capital_win == self.terminals[1])
        ps[1] = self.p_win
        terminals[1] = (capital_win == self.terminals[1])

        capital_loss = max(capital - bet, self.terminals[0])
        transitions[0, 0] = capital_loss
        rewards[0] = 0.0
        ps[0] = 1 - self.p_win
        terminals[0] = (capital_loss == self.terminals[0])

        return transitions, rewards, ps, terminals

    def reset(self):
        self.state = self.observation_space.sample()
        return self.state
