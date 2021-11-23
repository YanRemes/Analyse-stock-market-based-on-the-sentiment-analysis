from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import datetime as dt

discussion = pd.read_csv('/home/yan/PycharmProjects/pythonProject/wsb_comments.csv', error_bad_lines=False, index_col=False, dtype='unicode')
discussion = discussion.drop_duplicates('text')
discussion = discussion[discussion['text'].notnull()]

API_key = 'your key here'
top = ['SRNE', 'CRSR', 'GME', 'AMC', 'TSLA', 'MVIS', 'SPCE', 'CLNE', 'AAPL', 'WKHS', 'RKT', 'CLF', 'NVDA', 'AMZN', 'VIAC', 'ASO', 'TH', 'DTE', 'ATH']
ts = TimeSeries(key=API_key, output_format='csv')
csvreader, _ = ts.get_daily(symbol=top)

df = None
flag = False
for tickers in top:
    csvreader, meta = ts.get_daily(symbol=tickers)
    if not flag:
        df = pd.DataFrame(csvreader)
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
        flag = True
        df = df.iloc[:, 0::4]
    else:
        df_t = pd.DataFrame(csvreader)
        df_t.columns = df_t.iloc[0]
        df_t = df_t.iloc[1:].reset_index(drop=True)
        df_t = df_t.iloc[:, 0::4]
        # df = pd.merge(df, df_t, on="timestamp")
        # pd.concat([df, df_t])
        df = df.merge(df_t, on='timestamp')
df.columns = ['timestamp', 'SRNE', 'CRSR', 'GME', 'AMC', 'TSLA', 'MVIS', 'SPCE', 'CLNE', 'AAPL', 'WKHS', 'RKT', 'CLF', 'NVDA', 'AMZN', 'VIAC', 'ASO', 'TH', 'DTE', 'ATH']
df = df.iloc[::-1]

discussion['timestamp'] = pd.to_datetime(discussion['dt']).dt.strftime('%Y-%m-%d')
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
df = df[df['timestamp'].isin(discussion['timestamp'])]

df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
df.to_csv('top_20_tickers.csv', index=False)
