from environment import MMEnvironmentBoxActions
from math import log


class AvellanedaStoikovAgent:
    def __init__(self, sigma, k, gamma=0.5):
        self.sigma = sigma
        self.k = -k
        self.gamma = gamma

    def predict(self, state):
        T_minus_t = state[0]
        Q = state[1]
        rho = Q * self.gamma * self.sigma ** 2 * T_minus_t
        bid_diff = rho + (self.gamma / 2) * self.sigma ** 2 * T_minus_t + (1 / self.gamma) * log(
            1 + self.gamma / self.k)
        ask_diff = - rho + (self.gamma / 2) * self.sigma ** 2 * T_minus_t + (1 / self.gamma) * log(
            1 + self.gamma / self.k)
        return [[bid_diff, ask_diff]]


class AvellanedaStoikovEnvironment(MMEnvironmentBoxActions):

    def state(self):
        return (self.environment_length - self.total_steps) / self.environment_length, self.position

