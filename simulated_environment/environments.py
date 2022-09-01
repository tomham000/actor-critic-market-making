from abc import ABCMeta, abstractmethod, ABC
import numpy as np
import runstats

from numpy import sqrt, exp, log, inf
from gym import Env, spaces


class BaseEnvironment(metaclass=ABCMeta):
    def __init__(self, s0=100, T=1, dt=0.005, sigma=2, A=137.45, k=1.5, beta=0.5):
        self.s0 = s0
        self.S = s0
        self.t = 0
        self.Q = 0
        self.X = 0
        self.W = 0
        self.T = T
        self.dt = dt
        self.sigma = sigma
        self.A = A
        self.k = k
        self.beta = beta
        self.positions = []
        self.number_of_buys = 0
        self.number_of_sells = 0
        self.bid = None
        self.ask = None

    @abstractmethod
    def step(self, action):
        # Define how the agent behaves at each step
        pass

    def reset(self):
        self.S = self.s0
        self.Q = 0.0
        self.t = 0.0
        self.W = 0.0
        self.X = 0.0
        self.number_of_buys = 0
        self.number_of_sells = 0

    def get_rate(self, d):
        return self.A * exp(-self.k * d)

    def update_price(self):
        z = np.random.normal(0, 1)
        self.S += self.sigma * sqrt(self.dt) * z
        self.t += self.dt

    def update_inv_and_cash(self, bid: float, ask: float):
        db = self.S - bid
        da = ask - self.S

        lb = self.get_rate(db)
        la = self.get_rate(da)

        dnb = 1 if np.random.uniform() <= lb * self.dt else 0
        self.number_of_buys += dnb
        dna = 1 if np.random.uniform() <= la * self.dt else 0
        self.number_of_sells += dna
        self.Q += dnb - dna

        self.X += -dnb * bid + dna * ask

    def get_steps_to_T(self):
        return int((self.T - self.t) / self.dt)


class AvellanedaEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__()

    def rho(self):
        return self.S - self.beta * self.sigma ** 2 * (self.T - self.t) * self.Q

    def spread(self):
        return self.beta * self.sigma ** 2 * (self.T - self.t) + 2 * log(1 + self.beta / self.k) / self.beta

    def step(self, action=None):
        self.bid = self.rho() - self.spread() / 2
        self.ask = self.rho() + self.spread() / 2
        self.update_inv_and_cash(self.bid, self.ask)
        self.update_price()
        self.W = self.X + self.Q * self.S
        self.positions.append(self.Q)


class RLEnvironment(BaseEnvironment, Env):

    def __init__(self, max_abs_dif=4, spread_resolution=0.1, displacement_actions=15, spread_actions=30, s_bins=20,
                 s_min=95, s_bin_size=0.5):
        super().__init__()
        self.kappa = 2 * self.beta

        self.max_abs_dif = max_abs_dif
        self.spread_resolution = spread_resolution
        self.displacement_actions = displacement_actions
        self.spread_actions = spread_actions

        self.s_bins = s_bins
        self.s_min = s_min
        self.s_bin_size = s_bin_size

        self.stats = runstats.ExponentialStatistics(decay=0.999)

        self.observation_space = spaces.Box(low=np.array([self.s_min, -inf, 0.0]),
                                            high=np.array(
                                                [self.s_min + self.spread_actions * self.spread_resolution, inf,
                                                 self.T]),
                                            dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([displacement_actions, spread_actions])

    def step(self, action):
        displacement = (action[0] - (self.displacement_actions - 1) / 2) * self.max_abs_dif / (
                    self.displacement_actions - 1)
        spread = action[1] * self.spread_resolution

        self.bid = self.S - displacement - spread / 2
        self.ask = self.S - displacement + spread / 2

        self.update_inv_and_cash(self.bid, self.ask)
        self.update_price()

        previous_w = self.W
        self.W = self.X + self.Q * self.S

        dw = (self.W - previous_w)

        self.stats.push(dw)
        self.positions.append(self.Q)
        reward = dw - self.kappa / 2 * (dw - self.stats.mean()) ** 2

        return self.state(), reward, self.t >= self.T, {}  # Return interface of gym environment

    def get_s_bin(self) -> float:
        excess = self.S - self.s_min
        if excess < 0:
            return self.s_min
        res = excess // self.s_bin_size
        if res > self.s_bins:
            return self.s_min + self.s_bins * self.s_bin_size
        return self.s_min + res * self.s_bin_size

    def state(self):
        return self.get_s_bin(), self.Q + 200, int((self.T - self.t) / self.dt)

    def reset(self):
        self.S = self.s0
        self.Q = 0.0
        self.t = 0.0
        self.W = 0.0
        self.X = 0.0
        return self.state()  # Return interface of gym environment

    def render(self, mode="human"):
        pass
