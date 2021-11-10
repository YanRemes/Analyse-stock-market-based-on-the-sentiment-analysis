# import matplotlib.pyplot as plt
import re
from datetime import timedelta

import ax as ax
# import eastern as eastern
import matplotlib
# import self as self
# import swifter
import pandas as pd
import tkinter
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
# import sns as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as mpatches

price = pd.read_csv('/home/yan/PycharmProjects/pythonProject/top_20_tickers.csv')
names = price.columns.tolist()
names.remove('timestamp')
discussion = pd.read_csv('/home/yan/PycharmProjects/pythonProject/wsb_comments.csv', error_bad_lines=False, index_col=False, dtype='unicode')
discussion = discussion.drop_duplicates('text')
discussion = discussion[discussion['text'].notnull()]
discussion['text'] = discussion['text'].str.lower()
discussion['timestamp'] = pd.to_datetime(discussion['dt']).dt.strftime('%Y-%m-%d')
price['timestamp'] = pd.to_datetime(price['timestamp']).dt.strftime('%Y-%m-%d')
discussion = discussion[discussion['timestamp'].isin(price['timestamp'])]

def dedup(sentence, to_dedup):
    for word in to_dedup:
        while sentence.split().count(word) > 3:
            sentence = ''.join(sentence.rsplit(word, 1)).replace('  ', ' ')
    return sentence

def foo(row):
    global names
    sentence = row['text']
    return dedup(sentence, names)
discussion['text'] = discussion.apply(foo, axis=1)

bearish_vocab = ['decline', 'decrease', 'drop', 'fall', 'down', 'plummet', 'plunge', 'dip', 'slump', 'sell',
                 'selling', 'red', 'short', 'sells', 'declines', 'decreases', 'drops', 'dropped', 'puts', 'bears', 'bear']
bullish_vocab = ['rise', 'bump', 'climb', 'up', 'grow', 'increase', 'jump', 'rocket', 'buy', 'up', 'buying', 'buys',
                 'green', 'bought', 'climbs', 'grew', 'grows', 'rises', 'calls', 'long', 'call', 'moon', 'bulls', '🚀', 'bull']

pat = '|'.join(r"\b{}\b".format(x) for x in bullish_vocab)
discussion['contains_bullish'] = discussion['text'].str.findall(pat)
discussion['contains_bullish'] = discussion['contains_bullish'].apply(set).apply(list).apply(len)

pat = '|'.join(r"\b{}\b".format(x) for x in bearish_vocab)
discussion['contains_bearish'] = discussion['text'].str.findall(pat)
discussion['contains_bearish'] = discussion['contains_bearish'].apply(set).apply(list).apply(len)

discussion['bullish'] = np.where((discussion['contains_bullish'] > discussion['contains_bearish']), 1, np.nan)

discussion['bearish'] = np.where((discussion['contains_bullish'] < discussion['contains_bearish']), -1, np.nan)

discussion["compare"] = discussion["bullish"].fillna(0) + discussion["bearish"].fillna(0)

del discussion['contains_bearish']
del discussion['contains_bullish']
del discussion['bearish']
del discussion['bullish']

# price.rename(columns={'timestamp':'dt'}, inplace=True)
# discussion = discussion[discussion['dt'] == price['dt']]
discussion.to_csv('comments_binary.csv', index=False)
