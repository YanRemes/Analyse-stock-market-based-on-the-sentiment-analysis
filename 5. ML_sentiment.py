import re
from nltk.corpus import movie_reviews
from random import shuffle
from nltk import ngrams
from nltk.corpus import stopwords
import string
from nltk import NaiveBayesClassifier
from nltk import classify
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np

df = pd.read_csv('/home/yan/PycharmProjects/pythonProject/comments_binary.csv')
df = df.drop_duplicates('text')
df['text'] = df['text'].map(lambda x: re.sub('[^a-zA-Z.\d\s]', '', x))

top_tickers = pd.read_csv('/home/yan/PycharmProjects/pythonProject/top_20_tickers.csv').iloc[::-1]
names = list(top_tickers.columns)
names.pop(0)
item = pd.DataFrame(df)
pat = '|'.join(r"\b{}\b".format(x) for x in names)
df_ticker = df[df['text'].str.contains(pat, case=False, na=True)]

stopwords_english = stopwords.words('english')


def clean_words(words, stopwords_english):
    words_clean = []
    for word in words:
        word = word.lower()
        if word not in stopwords_english and word not in string.punctuation:
            words_clean.append(word)
    return words_clean


# feature extractor function for unigram
def bag_of_words(words):
    words_dictionary = dict([word, True] for word in words)
    return words_dictionary


# feature extractor function for ngrams (bigram)
def bag_of_ngrams(words, n=2):
    words_ng = []
    for item in iter(ngrams(words, n)):
        words_ng.append(item)
    words_dictionary = dict([word, True] for word in words_ng)
    return words_dictionary


text = "It was a very good movie."
words = word_tokenize(text.lower())

words_clean = clean_words(words, stopwords_english)
important_words = ['above', 'below', 'off', 'over', 'under', 'more', 'most', 'such', 'no', 'nor', 'not', 'only', 'so',
                   'than', 'too', 'very', 'just', 'but']
stopwords_english_for_bigrams = set(stopwords_english) - set(important_words)
words_clean_for_bigrams = clean_words(words, stopwords_english_for_bigrams)
unigram_features = bag_of_words(words_clean)
bigram_features = bag_of_ngrams(words_clean_for_bigrams)
all_features = unigram_features.copy()
all_features.update(bigram_features)


def bag_of_all_words(words, n=2):
    words_clean = clean_words(words, stopwords_english)
    words_clean_for_bigrams = clean_words(words, stopwords_english_for_bigrams)

    unigram_features = bag_of_words(words_clean)
    bigram_features = bag_of_ngrams(words_clean_for_bigrams)

    all_features = unigram_features.copy()
    all_features.update(bigram_features)

    return all_features


pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append(words)


neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append(words)

pos_reviews_set = []
for words in pos_reviews:
    pos_reviews_set.append((bag_of_all_words(words), 'pos'))

# negative reviews feature set
neg_reviews_set = []
for words in neg_reviews:
    neg_reviews_set.append((bag_of_all_words(words), 'neg'))

# shuffle(pos_reviews_set)
# shuffle(neg_reviews_set)

test_set = pos_reviews_set[:200] + neg_reviews_set[:200]
train_set = pos_reviews_set[200:] + neg_reviews_set[200:]

classifier = NaiveBayesClassifier.train(train_set)

accuracy = classify.accuracy(classifier, test_set)

def getPositivity(my_text):
    custom_review_tokens = word_tokenize(my_text)
    custom_review_set = bag_of_all_words(custom_review_tokens)
    prob_result = classifier.prob_classify(custom_review_set)
    return (prob_result.prob("pos"))

df_ticker['prediction'] = df_ticker['text'].apply(lambda text: getPositivity(text))

df_ticker['compare'] = np.select([df_ticker['prediction'].between(0, 0.4, inclusive='left'),
                       df_ticker['prediction'].between(0.4, 0.6, inclusive='left'),
                       df_ticker['prediction'].between(0.6, 1)],
                      choicelist=[-1, 0, 1])

df_ticker.to_csv('NLP_ML.csv', index=False)
