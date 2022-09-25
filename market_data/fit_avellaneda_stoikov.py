import pandas as pd
import numpy as np
from math import sqrt
import pickle

from market_data.hyperparameters import dir_names

df = pd.read_csv('TSLA_data/TSLA_clean.csv')
df_all = df
df = df.loc[df['type'].isin(('ORDER', 'TRADE', 'TRADE HIDDEN'))]

# vol
df_ts = df[['time', 'price']]
df_ts = df_ts.groupby('time').mean().reset_index()
df_ts['time'] = df_ts['time'] - df_ts['time'].loc[0]
df_ts['dt'] = df_ts['time'].diff()
df_ts['dS'] = df_ts['price'].diff()
df_ts = df_ts.dropna()

standardised = (df_ts['price'] - df_ts['price'].iloc[0]) / np.sqrt(df_ts['time'])
sigma = np.sqrt(standardised.var())

S = df_ts['price'].iloc[0]
simulated_prices = [S]
time_diffs = df_ts['time'].diff().dropna()
for index, dt in time_diffs.items():
    z = np.random.normal(0, 1)
    S += sigma * sqrt(dt) * z
    simulated_prices.append(S)
df_ts['sim'] = simulated_prices

df_fills = df_all.loc[df_all['type'].isin(('ORDER', 'TRADE'))]
df_buy_side = df_fills.loc[df_fills['direction'] == 'BUY'].reset_index(
    drop=True)  # buy limit orders, sell market orders
df_sell_side = df_fills.loc[df_fills['direction'] == 'SELL'].reset_index(
    drop=True)  # sell limit orders, buy market orders

df_buy_side = df_buy_side.head(50_000)
order_fill_indices = np.full(df_buy_side.shape[0], np.nan)
for index, row in df_buy_side[['type', 'price']].iterrows():
    if index % 100 == 0:
        print(index)
    if row['type'] != 'ORDER':
        continue
    for index_2, row_2 in df_buy_side[['type', 'price']].iloc[index:, ].iterrows():
        if row_2['type'] != 'TRADE':
            continue
        if row_2['price'] <= row['price']:
            # order filled
            order_fill_indices[index] = index_2
            break

df_buy_side['order_fill_index'] = order_fill_indices

df_sell_side = df_sell_side.head(50_000)
order_fill_indices = np.full(df_sell_side.shape[0], np.nan)
for index, row in df_sell_side[['type', 'price']].iterrows():
    if index % 100 == 0:
        print(index)
    if row['type'] != 'ORDER':
        continue
    for index_2, row_2 in df_sell_side[['type', 'price']].iloc[index:, ].iterrows():
        if row_2['type'] != 'TRADE':
            continue
        if row_2['price'] >= row['price']:
            # order filled
            order_fill_indices[index] = index_2
            break

df_sell_side['order_fill_index'] = order_fill_indices

df_buy_side['order_fill_timedelta'] = df_buy_side.apply(
    lambda x: np.nan if pd.isnull(x['order_fill_index']) else df_buy_side.iloc[int(x['order_fill_index'])]['time'] - x[
        'time'], axis=1)
df_sell_side['order_fill_timedelta'] = df_sell_side.apply(
    lambda x: np.nan if pd.isnull(x['order_fill_index']) else df_sell_side.iloc[int(x['order_fill_index'])]['time'] - x[
        'time'], axis=1)

df_buy_side['midprice_diff'] = df_buy_side['midprice'] - df_buy_side['price']
df_sell_side['midprice_diff'] = df_sell_side['price'] - df_sell_side['midprice']

df_order_fills = df_buy_side.append(df_sell_side)

df_order_fills = df_order_fills[(df_order_fills['type'] == 'ORDER')]
df_order_fills = df_order_fills[df_order_fills['order_fill_timedelta'].notnull()]

df_order_fills = df_order_fills[df_order_fills['order_fill_timedelta'] <= 1]

max_rate = 0.25
all_deltas = sorted(df_order_fills['midprice_diff'].loc[df_order_fills['midprice_diff'] <= max_rate].apply(
    lambda x: round(x, 2)).unique().tolist())

delta_lambda_pairs = {}
for delta in all_deltas:
    delta_orders = df_order_fills.loc[np.isclose(df_order_fills['midprice_diff'], delta)]
    mean = delta_orders['order_fill_timedelta'].mean()
    if mean == 0 or pd.isnull(mean):
        continue
    l = 1 / mean
    delta_lambda_pairs[delta] = l

from scipy import optimize

params = optimize.curve_fit(lambda t, a, b: a * np.exp(b * t), list(delta_lambda_pairs.keys()),
                            (list(delta_lambda_pairs.values())), p0=(2, -1))[0]
A = params[0]
k = params[1]

x = np.linspace(0.01, max_rate, 100)
y = A * np.exp(k * x)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(specs=[[{"secondary_y": False}]])

fig.add_trace(go.Scatter(x=list(delta_lambda_pairs.keys()), y=list(delta_lambda_pairs.values()), mode='lines',
                         name='$\\text{Observed fill rate}$', showlegend=True, line=dict(
        color='darkblue',
    )),
              secondary_y=False, )

fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='$\\text{Fitted fill rate (}\hat{A}e^{-\hat{k}\delta}\\text{)}$',
                         showlegend=True, line=dict(
        color='red',
    )),
              secondary_y=False, )

fig.update_layout(title_text="Fitted vs Observed Fill Rates",
                  xaxis_title="$\\text{Spread, } \delta$",
                  yaxis_title='$\\text{Fill rate, } \lambda(\delta)$',
                  title_font_size=25,
                  width=900,
                  height=500, )

fig.show()

as_params = {'sigma': sigma, 'A': A, 'k': k}

with open(f'{dir_names}avellaneda_stoikov_fit_parameters.pkl', 'wb') as handle:
    pickle.dump(as_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
