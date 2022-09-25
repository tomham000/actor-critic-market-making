from plotly.subplots import make_subplots
from stable_baselines3 import PPO, DDPG, SAC, TD3
from hyperparameters import dir_names
import pickle
import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import trim_mean

from statsmodels.api import qqplot

from environment import MMEnvironmentMultiDiscreteActions, MMEnvironmentBoxActions
from avellaneda_stoikov import AvellanedaStoikovAgent, AvellanedaStoikovEnvironment

warnings.filterwarnings("ignore")


model_ids = {
    'PPO': 'ppoPU58QXA0EL',
    'DDPG': 'ddpgRM6TKUQ08N',
    'SAC': 'sacSK7LGUTU6H',
    'AS': ''
}

gamma = 0.8

with open(f'{dir_names["parameters"]}avellaneda_stoikov_fit_parameters.pkl', 'rb') as handle:
    params = pickle.load(handle)

ppo_model = PPO.load(dir_names['ppo_logs'] + 'best_models/' + model_ids['PPO'] + '/best_model.zip')
ddpg_model = DDPG.load(dir_names['ddpg_logs'] + 'best_models/' + model_ids['DDPG'] + '/best_model.zip')
sac_model = SAC.load(dir_names['sac_logs'] + 'best_models/' + model_ids['SAC'] + '/best_model.zip')
as_model = AvellanedaStoikovAgent(params['sigma'], params['k'], gamma=1)

ppo_df = pd.DataFrame(columns=['wealth', 'pos', 'trades'])
sac_df = pd.DataFrame(columns=['wealth', 'pos', 'trades'])
ddpg_df = pd.DataFrame(columns=['wealth', 'pos', 'trades'])
as_df = pd.DataFrame(columns=['wealth', 'pos', 'trades'])
dfs = {
    'PPO': ppo_df,
    'DDPG': ddpg_df,
    'SAC': sac_df,
    'AS': as_df
}

env_length = 10_000

discrete_env = MMEnvironmentMultiDiscreteActions(env_length=env_length,
                                                 data_path=f'./TSLA_clean_test_data.csv')
box_env = MMEnvironmentBoxActions(env_length=env_length, data_path=f'./TSLA_clean_test_data.csv')
as_env = AvellanedaStoikovEnvironment(env_length=env_length,
                                      data_path=f'./TSLA_clean_test_data.csv')

models_and_envs = {
    'PPO': (ppo_model, discrete_env),
    'DDPG': (ddpg_model, box_env),
    'SAC': (sac_model, box_env),
    'AS': (as_model, as_env)
}

sims = 1000

for i in range(sims):
    for name, model_env_pair in models_and_envs.items():
        print(f"================{name}===============")
        model = model_env_pair[0]
        env = model_env_pair[1]
        env.reset()
        while env.total_steps < env.environment_length:
            action = model.predict(env.state())[0]
            env.step(action)

        dfs[name] = dfs[name].append({'wealth': env.wealth, 'pos': sum(env.positions) / len(env.positions),
                                      'trades': len(env.buy_trades) + len(env.sell_trades), 'slippage': env.delta_slippage}, ignore_index=True,)
        print(f'mean wealth: {dfs[name]["wealth"].mean()}')
        print(f'slippage: {dfs[name]["slippage"].mean()}')


for algo, df in dfs.items():
    fig = go.Figure()

    fig.add_trace(go.Histogram(x=df['wealth'], xbins=dict(size=8), name=algo))

    fig.add_trace(go.Histogram(x=dfs['AS']['wealth'], xbins=dict(size=8), name='Avellaneda', marker_color='turquoise'))

    fig.add_trace(go.Scatter(x=[df['wealth'].median(), df['wealth'].median()],
                             y=[0, 80],
                             mode='lines',
                             line=dict(color='red', width=2, dash='dash'),
                             name=f'{algo} median'))

    fig.add_trace(go.Scatter(x=[dfs['AS']['wealth'].median(), dfs['AS']['wealth'].median()],
                             y=[0, 80],
                             mode='lines',
                             line=dict(color='black', width=2, dash='dash'),
                             name=f'Avellaneda median'))

    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)
    fig.update_layout(title_text=f"{algo} Terminal Wealth Distribution",
                      xaxis_title="Terminal Wealth",
                      yaxis_title="Density",
                      title_font_size=25,
                      width=900,
                      height=500, )
    fig.show()



print('mean')

for algo, df in dfs.items():
    print(f'{algo}: {df["wealth"].mean()}')

print('trimmed mean')

for algo, df in dfs.items():
    print(f'{algo}: {trim_mean(df["wealth"], 0.1)}')

print('median')

for algo, df in dfs.items():
    print(f'{algo}: {df["wealth"].median()}')

print('std')
for algo, df in dfs.items():
    print(f'{algo}: {df["wealth"].std()}')

print('position')
for algo, df in dfs.items():
    print(f'{algo}: {df["pos"].mean()}')

print('position')
for algo, df in dfs.items():
    print(f'{algo}: {df["pos"].mean()}')

plt.style.use('seaborn')

fig, (ax1) = plt.subplots(1, 1)
ax1.set_title('DDPG Wealth Q-Q Plot')


qqplot(np.array(dfs['DDPG']['wealth']),
       ax=ax1,
       line='s')
fig.savefig('ddpg_qq.png', dpi=100)
plt.show()


from scipy.stats import ttest_1samp, ttest_ind, median_test

for name, df in dfs.items():
    print(name)
    print(ttest_1samp(df['wealth'], 0, alternative='greater'))

print(ttest_ind(dfs['AS']['wealth'], dfs['DDPG']['wealth'], equal_var=False))
print(median_test(dfs['AS']['wealth'], dfs['DDPG']['wealth']))
print(median_test(dfs['AS']['wealth'], dfs['PPO']['wealth']))
print(median_test(dfs['AS']['wealth'], dfs['SAC']['wealth']))

wealth_paths = []

for i in range(4):
    ddpg_bids = []
    ddpg_asks = []
    ddpg_prices = []
    ddpg_buys = []
    ddpg_sells = []
    ddpg_wealths = []

    box_env.reset()
    while box_env.total_steps < box_env.environment_length:

        old_buys = len(box_env.buy_trades)
        old_sells = len(box_env.sell_trades)
        ddpg_prices.append(box_env.lob_events.iloc[box_env.lob_index]['midprice'])
        action = ddpg_model.predict(box_env.state())[0]
        box_env.step(action)
        ddpg_buys.append(len(box_env.buy_trades) - old_buys)
        ddpg_sells.append(len(box_env.sell_trades) - old_sells)
        if box_env.current_bid == 0:
            ddpg_bids.append(np.nan)
        else:
            ddpg_bids.append(box_env.current_bid)
        if box_env.current_ask == 10_000:
            ddpg_asks.append(np.nan)
        else:
            ddpg_asks.append(box_env.current_ask)
        ddpg_wealths.append(box_env.wealth)

    df = pd.DataFrame({
        'bid': ddpg_bids,
        'ask': ddpg_asks,
        'price': ddpg_prices,
        'buys': ddpg_buys,
        'sells': ddpg_sells,
        'wealth': ddpg_wealths
    })

    df['buy_price'] = df['bid'] * df['buys']
    df['sell_price'] = df['ask'] * df['sells']

    df['buy_price'] = df['buy_price'].apply(lambda x: 0 if x > 600 else x)
    df['sell_price'] = df['sell_price'].apply(lambda x: 0 if x > 600 else x)

    buys = df[df['buy_price'] != 0]
    sells = df[df['sell_price'] != 0]

    wealth_paths.append(ddpg_wealths)


fig = make_subplots(rows=2, cols=2)

fig.add_trace(
    go.Scatter(x=list(range(200)), y=wealth_paths[0], mode='lines', name=f'Path {i}', showlegend=True, line=dict(
        color='black', width=1)),
    secondary_y=False, row=1, col=1)

fig.add_trace(
    go.Scatter(x=list(range(10_000)), y=wealth_paths[1], mode='lines', name=f'Path {i}', showlegend=True, line=dict(
        color='black', width=1)),
    secondary_y=False, row=1, col=2)

fig.add_trace(
    go.Scatter(x=list(range(10_000)), y=wealth_paths[2], mode='lines', name=f'Path {i}', showlegend=True, line=dict(
        color='black', width=1)),
    secondary_y=False, row=2, col=1)

fig.add_trace(
    go.Scatter(x=list(range(10_000)), y=wealth_paths[3], mode='lines', name=f'Path {i}', showlegend=True, line=dict(
        color='black', width=1)),
    secondary_y=False, row=2, col=2)

fig.update_xaxes(title_text="Timestep", row=1, col=1)
fig.update_xaxes(title_text="Timestep", row=1, col=2)
fig.update_xaxes(title_text="Timestep", row=2, col=1)
fig.update_xaxes(title_text="Timestep", row=2, col=2)

# Update yaxis properties
fig.update_yaxes(title_text="Wealth", row=1, col=1)
fig.update_yaxes(title_text="Wealth", row=1, col=2)
fig.update_yaxes(title_text="Wealth", row=2, col=1)
fig.update_yaxes(title_text="Wealth", row=2, col=2)

fig.update_layout(title_text="Example Realisations of DDPG MM Strategy",
                  title_font_size=25,
                  width=900,
                  height=500, )

fig.show()
