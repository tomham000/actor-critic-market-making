from stable_baselines3 import A2C
from environments import AvellanedaEnvironment, RLEnvironment
from simulated_environment.agents.q_learning.q_learning import QLearningAgent


def run_avellaneda_agent_simulation(env: AvellanedaEnvironment):
    wealths = []
    while env.t < env.T:
        wealths.append(env.W)
        env.step()
    return {'terminal_wealth': env.W,
            'wealth_path': wealths,
            'average_position': sum(env.positions) / len(env.positions),
            'total_trades': env.number_of_buys + env.number_of_sells}


def run_q_agent_simulation(env: RLEnvironment, agent: QLearningAgent):
    wealths = []
    bids = []
    asks = []
    prices = []
    buys = []
    sells = []

    while env.t < env.T:
        old_buys = env.number_of_buys
        old_sells = env.number_of_sells
        prices.append(env.S)
        state = env.state()
        action = agent.choose_action(state, ignore_exploration=True)
        env.step(action)
        buys.append(env.number_of_buys - old_buys)
        sells.append(env.number_of_sells - old_sells)
        bids.append(env.bid)
        asks.append(env.ask)
        wealths.append(env.W)
    return {'terminal_wealth': env.W,
            'wealth_path': wealths,
            'average_position': sum(env.positions) / len(env.positions),
            'total_trades': env.number_of_buys + env.number_of_sells,
            'bids': bids,
            'asks': asks,
            'prices': prices,
            'buys': buys,
            'sells': sells
            }


def run_a2c_agent_simulation(env: RLEnvironment, agent: A2C):
    wealths = []
    bids = []
    asks = []
    prices = []
    buys = []
    sells = []
    while env.t < env.T:
        old_buys = env.number_of_buys
        old_sells = env.number_of_sells
        prices.append(env.S)
        state = env.state()
        action = agent.predict(state)
        env.step(action[0])
        buys.append(env.number_of_buys - old_buys)
        sells.append(env.number_of_sells - old_sells)
        bids.append(env.bid)
        asks.append(env.ask)
        wealths.append(env.W)
    return {'terminal_wealth': env.W,
            'wealth_path': wealths,
            'average_position': sum(env.positions) / len(env.positions),
            'total_trades': env.number_of_buys + env.number_of_sells,
            'bids': bids,
            'asks': asks,
            'prices': prices,
            'buys': buys,
            'sells': sells
            }


def run_total_simulation(a2c_agent: A2C, q_agent: QLearningAgent, q_env=RLEnvironment(), a2c_env=RLEnvironment(), n_sims=1000):
    q_wealths = []
    a2c_wealths = []
    avellaneda_wealths = []

    q_positions = []
    a2c_positions = []
    avellaneda_positions = []

    q_trades = []
    a2c_trades = []
    avellaneda_trades = []

    avellaneda_env = AvellanedaEnvironment()

    for i in range(n_sims):
        if i % 100 == 0:
            print(f'simulation {i}')
        avellaneda_env.reset()
        q_env.reset()
        a2c_env.reset()
        q_env.number_of_buys = 0
        q_env.number_of_sells = 0
        a2c_env.number_of_buys = 0
        a2c_env.number_of_sells = 0

        avellaneda_results = run_avellaneda_agent_simulation(avellaneda_env)
        avellaneda_wealths.append(avellaneda_results['terminal_wealth'])
        avellaneda_positions.append(avellaneda_results['average'])
        avellaneda_trades.append(avellaneda_results['total_trades'])

        q_results = run_q_agent_simulation(q_env, q_agent)
        q_wealths.append(q_results['terminal_wealth'])
        q_positions.append(q_results['average'])
        q_trades.append(q_results['total_trades'])

        a2c_results = run_a2c_agent_simulation(a2c_env, a2c_agent)
        a2c_wealths.append(a2c_results['terminal_wealth'])
        a2c_positions.append(a2c_results['average'])
        a2c_trades.append(a2c_results['total_trades'])
    return {'q_agent':
                {'wealths': q_wealths, 'positions': q_positions, 'trades': q_trades},
            'a2c_agent':
                {'wealths': a2c_wealths, 'positions': a2c_positions, 'trades': a2c_trades},
            'avellaneda_agent':
                {'wealths': avellaneda_wealths, 'positions': avellaneda_positions, 'trades': avellaneda_trades}
            }
