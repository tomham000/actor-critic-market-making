import pandas as pd
from hyperparameters import hyperparameters

ticker = 'TSLA'
data_dir = 'tsla_data'
max_book_depth = hyperparameters['max_book_depth']
total_book_depth = hyperparameters['total_book_depth']
lookback_steps = hyperparameters['event_lookback_steps']

events_cols = ['time', 'type', 'id', 'volume', 'price', 'direction']
orderbook_cols = []
for i in range(total_book_depth):
    orderbook_cols.append(f'ask_{i}')
    orderbook_cols.append(f'ask_vol_{i}')
    orderbook_cols.append(f'bid_{i}')
    orderbook_cols.append(f'bid_vol_{i}')

events_df = pd.DataFrame()
orderbook_df = pd.DataFrame()

start_day = 9
start_month = 11
total_days = 1

for i in range(total_days):
    print(f'day: {start_day + i}')
    events = pd.read_csv(f'./{data_dir}/TSLA_2020-{start_month}-0{start_day + i}_34200000_57600000_message_5.csv',
                         header=None, names=events_cols, usecols=range(6))
    orderbook = pd.read_csv(f'./{data_dir}/TSLA_2020-{start_month}-0{start_day + i}_34200000_57600000_orderbook_5.csv',
                            header=None, names=orderbook_cols)
    events_df = events_df.append(events)
    orderbook_df = orderbook_df.append(orderbook)

df = pd.concat([events_df, orderbook_df], axis=1)


event_type_map = {1: 'ORDER', 2: 'PARTIAL CANCEL', 3: 'DELETE' , 4: 'TRADE', 5: 'TRADE HIDDEN', 6: 'UNKOWN', 7: 'HALT'}
event_direction_map = {-1: 'SELL', 1: 'BUY'}


df['type'] = events_df.apply(lambda x: event_type_map[int(x['type'])], axis=1)
df['direction'] = events_df.apply(lambda x: event_direction_map[int(x['direction'])], axis=1)

df['price'] /= 10_000

for i in range(max_book_depth):
    df[f'ask_{i}'] /= 10_000
    df[f'bid_{i}'] /= 10_000


df['midprice'] = (df['ask_0'] + df['bid_0'])/2
for i in range(max_book_depth):
    df[f'ask_spread_{i}'] = df[f'ask_{i}'] - df['midprice']
    df[f'bid_spread_{i}'] = df['midprice'] - df[f'bid_{i}']

df[orderbook_cols] = df[orderbook_cols].shift(1) # Each event should be linked with its PREVIOUS order book state
df = df.dropna()

# Drop unecessary LOB levels
df = df.drop([f"{side}_{str(level)}" for side in ['bid', 'ask','bid_vol', 'ask_vol'] for level in range(max_book_depth, total_book_depth)], axis=1)

df['timedelta'] = df['time'].diff()
df = df.dropna()

df['microprice'] = (df['ask_0'] * df['bid_vol_0'] + df['bid_0'] * df['ask_vol_0']) / (df['bid_vol_0'] + df['ask_vol_0'])

for i in range(max_book_depth):
    df[f'spread_{i}'] = df[f'ask_{i}'] - df[f'bid_{i}']

df = df.drop('id', axis=1)

df.to_csv(f'./{ticker}_data.csv')
