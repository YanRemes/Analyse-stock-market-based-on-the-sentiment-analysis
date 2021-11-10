import enchant
import pandas as pd
import string
from collections import Counter
from nltk.corpus import stopwords
from stopwords import res

discussion = pd.read_csv('/home/yan/PycharmProjects/pythonProject/wsb_comments.csv', error_bad_lines=False, index_col=False, dtype='unicode')
discussion = discussion.drop_duplicates('id')
discussion = discussion[discussion['text'].notnull()]
stop = stopwords.words('english')

discussion['text'] = discussion['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
word_bin = Counter()

for string in discussion["text"].values:
    word_bin.update(string.split(" "))
print(word_bin.most_common(200))
