import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3 import A2C

from config import a2c_dir, q_learning_dir
from environments import RLEnvironment
from simulation import run_total_simulation, run_q_agent_simulation

a2c_agent = A2C.load(a2c_dir)
a2c_env = RLEnvironment()
q_agent = pd.read_pickle(q_learning_dir)
q_env = RLEnvironment()

simulation_results = run_total_simulation(a2c_agent, q_agent, n_sims=1000)

print(f'Q trades: {sum(simulation_results["q_agent"]["trades"]) / len(simulation_results["q_agent"]["trades"])}')
print(f'A2C trades: {sum(simulation_results["a2c_agent"]["trades"]) / len(simulation_results["a2c_agent"]["trades"])}')
print(
    f'Avellaneda trades: {sum(simulation_results["avellaneda_agent"]["trades"]) / len(simulation_results["avellaneda_agent"]["trades"])}')

print(f'Q wealth: {sum(simulation_results["q_agent"]["wealths"]) / len(simulation_results["q_agent"]["wealths"])}')
print(
    f'A2C wealth: {sum(simulation_results["a2c_agent"]["wealths"]) / len(simulation_results["a2c_agent"]["wealths"])}')
print(
    f'Avellaneda wealth: {sum(simulation_results["avellaneda_agent"]["wealths"]) / len(simulation_results["avellaneda_agent"]["wealths"])}')

print(f'Q wealth STD: {np.std(simulation_results["q_agent"]["wealths"])}')
print(f'A2C wealth STD: {np.std(simulation_results["a2c_agent"]["wealths"])}')
print(f'Avellaneda wealth STD: {np.std(simulation_results["avellaneda_agent"]["wealths"])}')

fig = go.Figure()
fig.add_trace(go.Histogram(x=simulation_results["q_agent"]["wealths"], xbins=dict(size=1), name="Tabular Q Learning"))
fig.add_trace(
    go.Histogram(x=simulation_results["avellaneda_agent"]["wealths"], xbins=dict(size=1), name="Avellaneda-Stoikov"))
fig.add_trace(go.Histogram(x=simulation_results["a2c_agent"]["wealths"], xbins=dict(size=1), name="A2C"))

fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.update_layout(title_text="Comparison of Terminal Wealth Distributions",
                  xaxis_title="Terminal Wealth",
                  yaxis_title="Count",
                  title_font_size=25,
                  width=900,
                  height=500, )
fig.show()

q_single_realisation = run_q_agent_simulation(q_env, q_agent)

df = pd.DataFrame({
    'bid': q_single_realisation['bids'],
    'ask': q_single_realisation['asks'],
    'price': q_single_realisation['prices'],
    'buys': q_single_realisation['buys'],
    'sells': q_single_realisation['sells'],
    'wealth': q_single_realisation['wealth_path']
})

df['buy_price'] = df['bid'] * df['buys']
df['sell_price'] = df['ask'] * df['sells']

buys = df[df['buy_price'] != 0]
sells = df[df['sell_price'] != 0]

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Scatter(y=df['bid'], mode='lines', name='bid', showlegend=True, line=dict(
    color='lightblue',
    width=1
)),
              secondary_y=False, )

fig.add_trace(go.Scatter(y=df['ask'], mode='lines', name='ask', showlegend=True, line=dict(
    color='lightblue',
    width=1)),
              secondary_y=False, )

fig.add_trace(go.Scatter(y=df['price'], mode='lines', name='midprice', showlegend=True, line=dict(
    color='orange',
    width=1)),
              secondary_y=False, )

fig.add_trace(
    go.Scatter(
        x=buys.index, y=buys['buy_price'], mode="markers", name="Buy trade", marker=dict(
            color='darkgreen',
            size=10,
        ), marker_symbol='triangle-up'
    ),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        x=sells.index, y=sells['sell_price'], mode="markers", name="Sell trade", marker=dict(
            color='darkred',
            size=10,
        ), marker_symbol='triangle-down'
    ),
    secondary_y=False,
)


fig.update_layout(title_text="Example Realisation of Q-Learning MM Strategy",
                  xaxis_title="Timestep",
                  yaxis_title="Price",
                  title_font_size=25,
                  width=900,
                  height=500, )

fig.show()


