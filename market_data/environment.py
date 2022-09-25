from gym import Env, spaces
import pandas as pd
from random import randrange
import numpy as np
from numpy import inf, log
import random
import string
import os
import logging
from hyperparameters import hyperparameters, dir_names


class MMEnvironmentMultiDiscreteActions(Env):
    def __init__(self, data_path='./TSLA_clean.csv', debug=False, eval_env=False, log_path=None, algo_name='',
                 env_length=hyperparameters['environment_length']):
        self.env_id = algo_name + ''.join(random.choice(string.ascii_uppercase + string.digits) for i in range(10))
        print(f'Creating environment {self.env_id}')
        self.eval_env = eval_env
        self.log_path = log_path
        self.environment_length = env_length
        self.total_episodes = 0
        self.position_penalty = hyperparameters['position_penalty']
        self.rebate = hyperparameters['trade_rebate']

        self.max_spread_ticks = hyperparameters['max_spread_ticks']
        self.artificial_flow_rate = hyperparameters['artifical_flow_rate']

        self.df = pd.read_csv(data_path)
        self.lob_events = self.df.loc[self.df['type'].isin(('ORDER', 'TRADE', 'TRADE HIDDEN'))]
        self.lob_index = 0

        self.tick = 0.01
        self.half_tick = self.tick / 2

        # Reset env variables
        self.total_steps = 0
        self.wealth = 0
        self.cash = 0
        self.position = 0
        self.delta_slippage = 0
        self.current_bid = -inf
        self.current_ask = inf
        self.wealths = []
        self.positions = []
        self.rewards = []
        self.buy_trades = []
        self.sell_trades = []
        self.buys = []
        self.sells = []
        self.bids = []
        self.asks = []
        self.midprices = []

        self.max_abs_position = hyperparameters['max_abs_position']

        self.order_size = 10

        logger = logging.getLogger()
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        self.action_space = spaces.MultiDiscrete([2 * self.max_spread_ticks, 2 * self.max_spread_ticks])

        max_observed_spread = 150
        self.observation_space = spaces.MultiDiscrete(
            [2 * max_observed_spread, 2 * max_observed_spread, 2 * self.max_abs_position + 1])

    def step(self, action):
        sell_trade = False
        buy_trade = False
        lob_state = self.lob_events.iloc[self.lob_index]
        logging.debug(f"Market BBA is {lob_state['bid_0']} @ {lob_state['ask_0']}")
        self.update_market(action, lob_state)
        logging.debug(
            f"New event: {lob_state['direction']} {lob_state['type']} at {lob_state['price']} for {lob_state['volume']} lots.")
        if len(self.midprices) != 0:
            self.delta_slippage += (lob_state['midprice'] - self.midprices[-1]) * self.position
        self.midprices.append(lob_state['midprice'])
        if lob_state['direction'] == 'BUY':
            # order/trade is a buy, compare to agent's ask price
            if lob_state['price'] >= self.current_ask:
                logging.debug(f'Agent has sold')
                order_volume = lob_state['volume']
                execution_volume = min(self.order_size, order_volume)
                logging.debug(f'Agent executed {execution_volume} units at price {self.current_ask}')

                self.position -= execution_volume
                logging.debug(f'New position is {self.position}')
                self.cash += execution_volume * self.current_ask
                self.cash += self.rebate
                logging.debug(f'New cash is {self.cash}')
                self.sell_trades.append((self.current_ask, execution_volume))
                sell_trade = True
            # artificial trade
            if lob_state['ask_0'] > self.current_ask:
                difference = lob_state['ask_0'] - self.current_ask
                midprice_difference = lob_state['ask_0'] - lob_state['midprice']
                proportional_difference = difference / midprice_difference
                if np.random.uniform() < proportional_difference * self.artificial_flow_rate:
                    # artifical_trade made
                    self.position -= self.order_size
                    self.cash += self.order_size * self.current_ask
                    self.cash += self.rebate
                    self.sell_trades.append((self.current_ask, self.order_size))
                    sell_trade = True




        else:
            if lob_state['price'] <= self.current_bid:
                logging.debug(f'Agent has bought')
                order_volume = lob_state['volume']
                execution_volume = min(self.order_size, order_volume)
                logging.debug(f'Agent executed {execution_volume} units at price {self.current_bid}')

                self.position += execution_volume
                logging.debug(f'New inventory is {self.position}')
                self.cash -= execution_volume * self.current_bid
                self.cash += self.rebate
                logging.debug(f'New cash is {self.cash}')
                self.buy_trades.append((self.current_bid, execution_volume))
                buy_trade = True
            # artificial trade
            if lob_state['bid_0'] < self.current_bid:
                difference = self.current_bid - lob_state['bid_0']
                midprice_difference = lob_state['midprice'] - lob_state['bid_0']
                proportional_difference = difference / midprice_difference
                if np.random.uniform() < proportional_difference * self.artificial_flow_rate:
                    # artifical_trade made
                    self.position += self.order_size
                    self.cash -= self.order_size * self.current_bid
                    self.cash += self.rebate
                    self.buy_trades.append((self.current_bid, self.order_size))
                    buy_trade = True

        self.sells.append(int(sell_trade))
        self.buys.append(int(buy_trade))
        self.total_steps += 1
        self.lob_index += 1
        previous_wealth = self.wealth
        self.wealth = self.cash + self.position * lob_state['midprice']
        self.wealths.append(self.wealth)
        dw = self.wealth - previous_wealth

        inventory_penalty = self.position_penalty * self.position ** 2
        reward = dw - inventory_penalty

        self.rewards.append(reward)
        self.positions.append(self.position)

        if self.total_steps >= self.environment_length:
            print(
                f'Run {self.total_episodes} - wealth: {self.wealth}, reward: {sum(self.rewards)}, position: {self.position}, avg position: {sum(self.positions) / len(self.positions)}, trades: {len(self.buy_trades) + len(self.sell_trades)}\n')
            if self.eval_env and self.total_episodes % 10 == 0:
                self.save_env_data()

        return self.state(), reward, self.total_steps >= self.environment_length, {}

    def state(self):
        # state: midprice, ask0_spread, bid0_spread, position
        lob_state = self.lob_events.iloc[self.lob_index]
        midprice = lob_state['midprice']
        bid_distance = lob_state['bid_spread_0']
        ask_distance = lob_state['ask_spread_0']

        bounded_position = self.position
        if bounded_position < - self.max_abs_position:
            bounded_position = - self.max_abs_position
        elif bounded_position > self.max_abs_position:
            bounded_position = self.max_abs_position

        return (int(bid_distance / self.half_tick), int(ask_distance / self.half_tick),
                bounded_position + self.max_abs_position)

        # internal_state = np.array([self.position])
        # return np.concatenate([lob_state[['midprice', 'ask_spread_0', 'bid_spread_0']].to_numpy('float32'), internal_state], dtype='float32')

    def reset(self):
        self.total_steps = 0
        self.wealth = 0
        self.cash = 0
        self.position = 0
        self.delta_slippage = 0
        self.current_bid = -inf
        self.current_ask = inf
        self.wealths = []
        self.positions = []
        self.rewards = []
        self.buy_trades = []
        self.sell_trades = []
        self.buys = []
        self.sells = []
        self.bids = []
        self.asks = []
        self.midprices = []

        self.lob_index = randrange(len(self.lob_events) - self.environment_length)
        self.total_episodes += 1

        return self.state()

    def update_market(self, action, lob_state):
        midprice = lob_state['midprice']
        if self.position < self.max_abs_position:
            self.current_bid = midprice - (1 + action[0]) * self.half_tick
        else:
            self.current_bid = 0

        if self.position > -self.max_abs_position:
            self.current_ask = midprice + (1 + action[1]) * self.half_tick
        else:
            self.current_ask = 10_000

        self.bids.append(self.current_bid)
        self.asks.append(self.current_ask)
        logging.debug(f'Agent\s new market: {self.current_bid} @ {self.current_ask}.')

    def save_env_data(self, filename=None):
        df = pd.DataFrame({
            'wealth': self.wealths,
            'position': self.positions,
            'reward': self.rewards,
            'bid': self.bids,
            'ask': self.asks,
            'mid': self.midprices,
            'buy': self.buys,
            'sell': self.sells
        })

        os.makedirs(dir_names['env_data'] + self.env_id + '/', exist_ok=True)
        if filename is None:
            filename = f'{self.env_id}/env_{self.total_episodes}.csv'
        df.to_csv(dir_names['env_data'] + filename)


class MMEnvironmentBoxActions(Env):
    def __init__(self, data_path='./TSLA_clean.csv', debug=False, eval_env=False, log_path=None, algo_name='',
                 env_length=hyperparameters['environment_length']):
        self.env_id = algo_name + ''.join(random.choice(string.ascii_uppercase + string.digits) for i in range(10))
        print(f'Creating environment {self.env_id}')
        self.eval_env = eval_env
        self.log_path = log_path
        self.environment_length = env_length
        self.total_episodes = 0
        self.position_penalty = hyperparameters['position_penalty']
        self.rebate = hyperparameters['trade_rebate']

        self.max_spread_ticks = hyperparameters['max_spread_ticks']
        self.artificial_flow_rate = hyperparameters['artifical_flow_rate']

        self.df = pd.read_csv(data_path)
        self.lob_events = self.df.loc[self.df['type'].isin(('ORDER', 'TRADE', 'TRADE HIDDEN'))]
        self.lob_index = 0

        self.tick = 0.01
        self.half_tick = self.tick / 2

        # Reset env variables
        self.total_steps = 0
        self.wealth = 0
        self.cash = 0
        self.position = 0
        self.delta_slippage = 0
        self.current_bid = -inf
        self.current_ask = inf
        self.wealths = []
        self.positions = []
        self.rewards = []
        self.buy_trades = []
        self.sell_trades = []
        self.buys = []
        self.sells = []
        self.bids = []
        self.asks = []
        self.midprices = []

        self.max_abs_position = hyperparameters['max_abs_position']

        self.order_size = 10

        logger = logging.getLogger()
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        self.action_space = spaces.Box(low=np.array([self.half_tick, self.half_tick]), high=np.array(
            [self.max_spread_ticks * self.tick, self.max_spread_ticks * self.tick]))

        max_observed_spread = 150
        self.observation_space = spaces.MultiDiscrete(
            [2 * max_observed_spread, 2 * max_observed_spread, 2 * self.max_abs_position + 1])

    def step(self, action):
        sell_trade = False
        buy_trade = False
        lob_state = self.lob_events.iloc[self.lob_index]
        logging.debug(f"Market BBA is {lob_state['bid_0']} @ {lob_state['ask_0']}")
        self.update_market(action, lob_state)
        if len(self.midprices) != 0:
            self.delta_slippage += (lob_state['midprice'] - self.midprices[-1]) * self.position
        logging.debug(
            f"New event: {lob_state['direction']} {lob_state['type']} at {lob_state['price']} for {lob_state['volume']} lots.")
        self.midprices.append(lob_state['midprice'])
        if lob_state['direction'] == 'BUY':
            # order/trade is a buy, compare to agent's ask price
            if lob_state['price'] >= self.current_ask:
                logging.debug(f'Agent has sold')
                order_volume = lob_state['volume']
                execution_volume = min(self.order_size, order_volume)
                logging.debug(f'Agent executed {execution_volume} units at price {self.current_ask}')

                self.position -= execution_volume
                logging.debug(f'New position is {self.position}')
                self.cash += execution_volume * self.current_ask
                self.cash += self.rebate
                logging.debug(f'New cash is {self.cash}')
                self.sell_trades.append((self.current_ask, execution_volume))
                sell_trade = True
            # artificial trade
            if lob_state['ask_0'] > self.current_ask:
                difference = lob_state['ask_0'] - self.current_ask
                midprice_difference = lob_state['ask_0'] - lob_state['midprice']
                proportional_difference = difference / midprice_difference
                if np.random.uniform() < proportional_difference * self.artificial_flow_rate:
                    # artifical_trade made
                    self.position -= self.order_size
                    self.cash += self.order_size * self.current_ask
                    self.cash += self.rebate
                    self.sell_trades.append((self.current_ask, self.order_size))
                    sell_trade = True




        else:
            if lob_state['price'] <= self.current_bid:
                logging.debug(f'Agent has bought')
                order_volume = lob_state['volume']
                execution_volume = min(self.order_size, order_volume)
                logging.debug(f'Agent executed {execution_volume} units at price {self.current_bid}')

                self.position += execution_volume
                logging.debug(f'New inventory is {self.position}')
                self.cash -= execution_volume * self.current_bid
                self.cash += self.rebate
                logging.debug(f'New cash is {self.cash}')
                self.buy_trades.append((self.current_bid, execution_volume))
                buy_trade = True
            # artificial trade
            if lob_state['bid_0'] < self.current_bid:
                difference = self.current_bid - lob_state['bid_0']
                midprice_difference = lob_state['midprice'] - lob_state['bid_0']
                proportional_difference = difference / midprice_difference
                if np.random.uniform() < proportional_difference * self.artificial_flow_rate:
                    # artifical_trade made
                    self.position += self.order_size
                    self.cash -= self.order_size * self.current_bid
                    self.cash += self.rebate
                    self.buy_trades.append((self.current_bid, self.order_size))
                    buy_trade = True

        self.sells.append(int(sell_trade))
        self.buys.append(int(buy_trade))
        self.total_steps += 1
        self.lob_index += 1
        previous_wealth = self.wealth
        self.wealth = self.cash + self.position * lob_state['midprice']
        self.wealths.append(self.wealth)
        dw = self.wealth - previous_wealth

        inventory_penalty = self.position_penalty * self.position ** 2
        reward = dw - inventory_penalty

        self.rewards.append(reward)
        self.positions.append(self.position)

        if self.total_steps >= self.environment_length:
            print(
                f'Run {self.total_episodes} - wealth: {self.wealth}, reward: {sum(self.rewards)}, position: {self.position}, avg position: {sum(self.positions) / len(self.positions)}, trades: {len(self.buy_trades) + len(self.sell_trades)}\n')
            if self.eval_env and self.total_episodes % 10 == 0:
                self.save_env_data()

        return self.state(), reward, self.total_steps >= self.environment_length, {}

    def state(self):
        # state: midprice, ask0_spread, bid0_spread, position
        lob_state = self.lob_events.iloc[self.lob_index]
        midprice = lob_state['midprice']
        bid_distance = lob_state['bid_spread_0']
        ask_distance = lob_state['ask_spread_0']

        bounded_position = self.position
        if bounded_position < - self.max_abs_position:
            bounded_position = - self.max_abs_position
        elif bounded_position > self.max_abs_position:
            bounded_position = self.max_abs_position

        return (int(bid_distance / self.half_tick), int(ask_distance / self.half_tick),
                bounded_position + self.max_abs_position)

        # internal_state = np.array([self.position])
        # return np.concatenate([lob_state[['midprice', 'ask_spread_0', 'bid_spread_0']].to_numpy('float32'), internal_state], dtype='float32')

    def reset(self):
        self.total_steps = 0
        self.wealth = 0
        self.cash = 0
        self.position = 0
        self.delta_slippage = 0
        self.current_bid = -inf
        self.current_ask = inf
        self.wealths = []
        self.positions = []
        self.rewards = []
        self.buy_trades = []
        self.sell_trades = []
        self.buys = []
        self.sells = []
        self.bids = []
        self.asks = []
        self.midprices = []

        self.lob_index = randrange(len(self.lob_events) - self.environment_length)
        self.total_episodes += 1

        return self.state()

    def update_market(self, action, lob_state):
        midprice = lob_state['midprice']
        if self.position < self.max_abs_position:
            self.current_bid = midprice - action[0]
        else:
            self.current_bid = 0

        if self.position > -self.max_abs_position:
            self.current_ask = midprice + action[1]
        else:
            self.current_ask = 10_000

        self.bids.append(self.current_bid)
        self.asks.append(self.current_ask)
        logging.debug(f'Agent\s new market: {self.current_bid} @ {self.current_ask}.')

    def save_env_data(self, filename=None):
        df = pd.DataFrame({
            'wealth': self.wealths,
            'position': self.positions,
            'reward': self.rewards,
            'bid': self.bids,
            'ask': self.asks,
            'mid': self.midprices,
            'buy': self.buys,
            'sell': self.sells
        })

        os.makedirs(dir_names['env_data'] + self.env_id + '/', exist_ok=True)

        if filename is None:
            filename = f'{self.env_id}/env_{self.total_episodes}.csv'
        df.to_csv(dir_names['env_data'] + filename)
