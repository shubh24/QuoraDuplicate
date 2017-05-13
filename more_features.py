from __future__ import division
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.metrics import log_loss
import pandas as pd 
import numpy as np
import xgboost as xgb
import math
import nltk
from nltk import ngrams
from sklearn.cross_validation import train_test_split
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from simhash import Simhash
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from matplotlib import pyplot
from sklearn.manifold import TSNE

import spacy
nlp = spacy.load('en')

from explore import *

def run_xgb(x_train, x_valid, y_train, y_valid):

    # x_train = pd.concat([pos_train, neg_train]) #Concat positive and negative
    # y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist() #Putting in 1 and 0

    # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

    # Set our parameters for xgboost
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.05
    params['max_depth'] = 6
    params['silent'] = 1

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    bst = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, verbose_eval=50)

    # d_test = xgb.DMatrix(x_test_feat)
    # p_test = bst.predict(d_test)

    xgb.plot_importance(bst)
    pyplot.show()

    # return p_test

def q1_sents(row):

    try:
        return len(nltk.tokenize.sent_tokenize(str(row.question1)))
    except:
        return 0

def q2_sents(row):

    try:
        return len(nltk.tokenize.sent_tokenize(str(row.question2)))
    except:
        return 0

def q1_exclaim(row):

    try:
        return sum([1 for i in str(row.question1) if i == "!"])
    except:
        return 0


def q2_exclaim(row):

    try:
        return sum([1 for i in str(row.question2) if i == "!"])
    except:
        return 0

def q1_question(row):

    try:
        return sum([1 for i in str(row.question1) if i == "?"])
    except:
        return 0


def q2_question(row):

    try:
        return sum([1 for i in str(row.question2) if i == "?"])
    except:
        return 0

def common_chunk_score(row):

    q1 = nlp(unicode(str(row["question1"]), "utf-8"))
    q2 = nlp(unicode(str(row["question2"]), "utf-8"))

    q1_nc = q1.noun_chunks
    q2_nc = q2.noun_chunks

    q1_nc = set([str(i) for i in q1_nc])
    q2_nc = set([str(i) for i in q2_nc])

    common_nc = len(q1_nc.intersection(q2_nc))

    if len(q1_nc) + len(q2_nc) == 0:
        return 0
    else:
       return common_nc/(len(q1_nc) + len(q2_nc) - common_nc)

def common_ne_score(row):

    q1 = nlp(unicode(str(row["question1"]), "utf-8"))
    q2 = nlp(unicode(str(row["question2"]), "utf-8"))

    q1_ne = q1.ents
    q2_ne = q2.ents

    q1_ne = set([str(i) for i in q1_ne])
    q2_ne = set([str(i) for i in q2_ne])

    common_ne = len(q1_ne.intersection(q2_ne))

    if len(q1_ne) + len(q2_ne) == 0:
        return 0
    else:
       return common_ne/(len(q1_ne) + len(q2_ne) - common_ne)

def count_q1_ne(row):

    q1 = nlp(unicode(str(row["question1"]), "utf-8"))
    q1_ne = q1.ents

    if len(q1_ne) == 0:
        return 0
    else:
        return len(q1_ne)/len(row["question1"].split())

def count_q2_ne(row):

    q2 = nlp(unicode(str(row["question2"]), "utf-8"))
    q2_ne = q2.ents

    if len(q2_ne) == 0:
        return 0
    else:
        return len(q2_ne)/len(row["question2"].split())

def get_features(x_train_feat):

    # x_train_feat["q1_sents"] = x_train_feat.apply(q1_sents, axis = 1)
    # x_train_feat["q2_sents"] = x_train_feat.apply(q2_sents, axis = 1)
    # x_train_feat["q1_exclaim"] = x_train_feat.apply(q1_sents, axis = 1)
    # x_train_feat["q2_exclaim"] = x_train_feat.apply(q2_sents, axis = 1)
    # x_train_feat["q1_question"] = x_train_feat.apply(q1_sents, axis = 1)
    # x_train_feat["q2_question"] = x_train_feat.apply(q2_sents, axis = 1)
    # x_train_feat["ne_score"] = x_train_feat.apply(common_ne_score, axis = 1)
    # x_train_feat["q1_ne"] = x_train_feat.apply(count_q1_ne, axis = 1)
    # x_train_feat["q2_ne"] = x_train_feat.apply(count_q2_ne, axis = 1)
    # x_train_feat["nc_score"] = x_train_feat.apply(common_chunk_score, axis = 1)

    return x_train_feat

def generate_hash_freq(row):

    hash_key1 = hash(row["question1"].lower())
    hash_key2 = hash(row["question2"].lower())

    if hash_key1 not in hash_table:
        hash_table[hash_key1] = 1
    else:
        hash_table[hash_key1] += 1

    if hash_key2 not in hash_table:
        hash_table[hash_key2] = 1
    else:
        hash_table[hash_key2] += 1

if __name__ == '__main__':

    x_train_feat = pd.read_csv('./x_train_feat.csv').fillna("")
    x_train_feat['spacy_sim'] = pd.to_numeric(pd.Series(x_train_feat['spacy_sim']), errors = "coerce")

    # df_train = pd.read_csv('./train.csv').fillna("")
    # x_train_feat = pd.concat([df_train, x_train_feat], axis = 1)

    # train_qs = pd.Series(x_train_feat['question1'].tolist() + x_train_feat['question2'].tolist()).astype(str)

    # words = (" ".join(train_qs)).lower().split()
    # counts = Counter(words)
    # weights = {word: get_inverse_freq(1/(10000 + int(count)), count) for word, count in counts.items()}

    # stops = set(stopwords.words("english"))

    # hash_table = {}

    # x_train_feat.apply(generate_hash_freq, axis = 1)

    x_train_feat = get_features(x_train_feat)
    x_train_feat.to_csv("x_train_feat.csv", index = False)

    x_label = x_train_feat.pop("is_duplicate")
    x_train_feat = x_train_feat.iloc[:, range(5, 42)]

    x_train, x_valid, y_train, y_valid = train_test_split(x_train_feat, x_label, test_size=0.2, random_state=4242, stratify = x_label)

    p = run_xgb(x_train, x_valid, y_train, y_valid)
