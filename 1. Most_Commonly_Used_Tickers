import enchant
import pandas as pd
import string
from collections import Counter

ticker_bin = Counter()

def clean_word(word):
    res = []
    [res.append(c) for c in word if c not in string.punctuation and c.isalpha()]
    return ''.join(res)

def ticker_extractor(text):
    global ticker_bin
    words = text.split()
    words = set([clean_word(word) for word in words])
    words = words & tickers | words & set([ticker.upper() for ticker in tickers])
    words = [word for word in words if not d.check(word)
             or (d.check(word) and word.isupper() and not ['A', 'IM'])]
    words = [word.upper() for word in words]
    ticker_bin += Counter(words)
    print(ticker_bin)

price = pd.read_parquet('rus_price_150321.parquet', engine='pyarrow')
discussion = pd.read_csv('wsb_comments.csv', error_bad_lines=False, index_col=False, dtype='unicode')
discussion = discussion.drop_duplicates('text')
tickers = price.columns[1::2]
tickers = [item.split(',') for item in tickers]
tickers = set((item for sublist in tickers for item in sublist))
discussion['text'] = discussion['text']
d = enchant.Dict("en_US")

discussion = discussion[discussion['text'].notnull()]
discussion.text.apply(lambda x: ticker_extractor(x))
ticker_bin = [word for word, cnt in ticker_bin.most_common(20)]
print('end')
print(ticker_bin)
