from collections import OrderedDict
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import string
import re
# pd.set_option('display.max_rows', None)
import pandas as pd
from PIL.ImageChops import offset
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from oauthlib.uri_validate import host

df = pd.read_csv('/home/yan/PycharmProjects/pythonProject/NLP_ML.csv')
df = df.drop_duplicates('text')
df['text'] = df['text'].map(lambda x: re.sub('[^a-zA-Z.\d\s]', '', str(x)))
top_tickers = pd.read_csv('/home/yan/PycharmProjects/pythonProject/top_20_tickers.csv')
names = list(top_tickers.columns)
names.pop(0)
item = pd.DataFrame(df)
pat = '|'.join(r"\b{}\b".format(x) for x in names)
df_ticker = df[df['text'].str.contains(pat,case=False,na=True)]

df_ticker['text'] = df_ticker['text'].str.upper()

def dedup(sentence, to_dedup):
    for word in to_dedup:
        while sentence.split().count(word) > 3:
            sentence = ''.join(sentence.rsplit(word, 1)).replace('  ', ' ')
    return sentence

def foo(row):
    global names
    sentence = row['text']
    return dedup(sentence, names)
df_ticker['text'] = df_ticker.apply(foo, axis=1)

pat = '|'.join(r"\b{}\b".format(x) for x in names)


bullish = pd.DataFrame(df_ticker)
bullish = bullish[bullish.eq(1).any(1)]
bullish_comments_df = bullish.set_index('dt')['text'].str.extractall('(' + pat + ')')[0].reset_index(name='tickers')
bullish_comments_df1 = pd.crosstab(bullish_comments_df['dt'], bullish_comments_df['tickers'])
bullish_comments_df1.reset_index(inplace=True)
bullish_comments_df1.index = pd.to_datetime(bullish_comments_df1.pop('dt'))
bullish_comments_df1 = bullish_comments_df1.resample('1D').sum().fillna(value=0)
bullish_comments_df1.reset_index(inplace=True)

bearish = pd.DataFrame(df_ticker)
bearish = bearish[bearish.eq(-1).any(1)]
bearish_comments_df = bearish.set_index('dt')['text'].str.extractall('(' + pat + ')')[0].reset_index(name='tickers')
bearish_comments_df1 = pd.crosstab(bearish_comments_df['dt'], bearish_comments_df['tickers'])
bearish_comments_df1.reset_index(inplace=True)
bearish_comments_df1.index = pd.to_datetime(bearish_comments_df1.pop('dt'))
bearish_comments_df1 = bearish_comments_df1.resample('1D').sum().fillna(value=0)
bearish_comments_df1.reset_index(inplace=True)

neutral = pd.DataFrame(df_ticker)
neutral = neutral[neutral.eq(0).any(1)]
neutral_comments_df = neutral.set_index('dt')['text'].str.extractall('(' + pat + ')')[0].reset_index(name='tickers')
neutral_comments_df1 = pd.crosstab(neutral_comments_df['dt'], neutral_comments_df['tickers'])
neutral_comments_df1.reset_index(inplace=True)
neutral_comments_df1.index = pd.to_datetime(neutral_comments_df1.pop('dt'))
neutral_comments_df1 = neutral_comments_df1.resample('1D').sum().fillna(value=0)
neutral_comments_df1.reset_index(inplace=True)

bullish_comments_df1 = bullish_comments_df1[bullish_comments_df1.columns.intersection(bearish_comments_df1.columns)]
bearish_comments_df1 = bearish_comments_df1[bearish_comments_df1.columns.intersection(bullish_comments_df1.columns)]
neutral_comments_df1 = neutral_comments_df1[neutral_comments_df1.columns.intersection(bullish_comments_df1.columns)]

bullish_bearish_neutral = pd.concat([bearish_comments_df1, bullish_comments_df1, neutral_comments_df1]).groupby(['dt']).sum().reset_index()

bullish_ratio_df1 = bullish_comments_df1.set_index('dt').div(bullish_bearish_neutral.set_index('dt')).reset_index().fillna(value=0)
bearish_ratio_df1 = bearish_comments_df1.set_index('dt').div(bullish_bearish_neutral.set_index('dt')).reset_index().fillna(value=0)
neutral_ratio_df1 = neutral_comments_df1.set_index('dt').div(bullish_bearish_neutral.set_index('dt')).reset_index().fillna(value=0)

bullish_comments_df1['dt'] = bullish_comments_df1['dt'].dt.strftime('%Y-%m-%d')
bearish_comments_df1['dt'] = bearish_comments_df1['dt'].dt.strftime('%Y-%m-%d')

bullish_comments_df1 = bullish_comments_df1.set_index('dt')
bearish_comments_df1 = bearish_comments_df1.set_index('dt')
mask = bearish_comments_df1.lt(10) |

bullish_comments_df1.lt(10)
bearish_comments_df1 = bearish_comments_df1.mask(mask)
bullish_comments_df1 = bullish_comments_df1.mask(mask)
bearish_comments_df1 = bearish_comments_df1.fillna(0)
bullish_comments_df1 = bullish_comments_df1.fillna(0)
bearish_comments_df1.reset_index(inplace=True)
bullish_comments_df1.reset_index(inplace=True)

bullish_comments_df1.to_csv('bullish_comments.csv', index=False)
bearish_comments_df1.to_csv('bearish_comments.csv', index=False)

bullish_ratio_df1['dt'] = bullish_ratio_df1['dt'].dt.strftime('%Y-%m-%d')
bearish_ratio_df1['dt'] = bearish_ratio_df1['dt'].dt.strftime('%Y-%m-%d')

bearish_ratio_df1.loc[:, bearish_ratio_df1.columns != 'dt'] *= -1
bearish_comments_df1.loc[:,bearish_comments_df1.columns != 'dt'] *= -1

bullish_comments_df1[bullish_comments_df1.eq(0)] = np.nan
bearish_comments_df1[bearish_comments_df1.eq(0)] = np.nan
bullish_ratio_df1[bullish_ratio_df1.eq(0)] = np.nan
bearish_ratio_df1[bearish_ratio_df1.eq(0)] = np.nan

bullish_ratio_df1.iloc[:,1:] = bullish_ratio_df1.iloc[:,1:] * 100
bearish_ratio_df1.iloc[:,1:] = bearish_ratio_df1.iloc[:,1:] * 100

bullish_ratio_df1 = bullish_ratio_df1.set_index('dt')
bearish_ratio_df1 = bearish_ratio_df1.set_index('dt')
bearish_ratio_df1 = bearish_ratio_df1.mask(mask)
bullish_ratio_df1 = bullish_ratio_df1.mask(mask)
bearish_ratio_df1 = bearish_ratio_df1.fillna(0)
bullish_ratio_df1 = bullish_ratio_df1.fillna(0)
bearish_ratio_df1.reset_index(inplace=True)
bullish_ratio_df1.reset_index(inplace=True)

bullish_ratio_df1.to_csv('bullish_ratio.csv', index=False)
bearish_ratio_df1.to_csv('bearish_ratio.csv', index=False)

bullish_comments_df1 = bullish_comments_df1.rename(columns={"dt": "timestamp"})
top_tickers = top_tickers[top_tickers.columns.intersection(bullish_comments_df1.columns)]
top_tickers = top_tickers[bullish_comments_df1.columns]

plot_cols = top_tickers.columns[1:]

for i, col in enumerate(plot_cols):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.bar(bullish_comments_df1.timestamp, bullish_comments_df1[col], color='green', alpha=0.25)
    ax.bar_label(ax.containers[0], rotation = 90)
    ax.bar(bearish_comments_df1.dt, bearish_comments_df1[col], color='red', alpha=0.25)
    ax.bar_label(ax.containers[1], rotation = 90)
    ax2 = fig.add_subplot(frame_on=False)
    ax3 = fig.add_subplot(frame_on=False)
    ax2.bar(bullish_ratio_df1.dt, bullish_ratio_df1[col], color='darkorange', alpha=0.25)
    # ax2.bar_label(ax2.containers[0], rotation = 90)
    ax2.bar(bearish_ratio_df1.dt, bearish_ratio_df1[col], color='peru', alpha=0.25)
    # ax2.bar_label(ax2.containers[1], rotation = 90)
    ax3.plot(top_tickers.timestamp, top_tickers[col], color='blue')
    ax.yaxis.tick_right()
    ax2.yaxis.tick_right()
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    # ax2.set_yticks(10)
    # ax3.axes.xaxis.set_visible(False)
    ax2.axes.xaxis.set_visible(False)
    ax3.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(True)
    # ax.set_xticks(bullish_comments_df1.timestamp)
    ax.set_xticklabels(bullish_comments_df1.timestamp, rotation=90, ha='left')
    ax2.get_shared_y_axes().join(ax2, ax)
    ax.get_shared_x_axes().join(ax, ax2)
    plt.xticks(rotation=90)
    ticker = mpatches.Patch(color='white', label=col)
    blue_patch = mpatches.Patch(color='blue', label='price')
    green_patch = mpatches.Patch(color='green', label='bullish amount')
    red_patch = mpatches.Patch(color='red', label='bearish amount')
    darkorange_patch = mpatches.Patch(color='darkorange', label='bullish ratio')
    peru_patch = mpatches.Patch(color='peru', label='bearish ratio')
    plt.legend(handles=[ticker, blue_patch, red_patch, green_patch, darkorange_patch, peru_patch])
    fig.set_size_inches(20, 10.5)
    plt.savefig('ticker_{}.png'.format(col))
