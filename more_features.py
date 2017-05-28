from __future__ import division
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.metrics import log_loss
import pandas as pd 
import numpy as np
import xgboost as xgb
import math
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
import pickle
import spacy
nlp = spacy.load('en')
from tqdm import tqdm, tqdm_pandas
tqdm.pandas()
from explore import *

def run_xgb_val(x_train, x_valid, y_train, y_valid):

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

    bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=50)

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

def quotes_q1(row):

    try:
        return len(re.findall(r'\"(.+?)\"', row["question1"]))
    except:
        return 0

def quotes_q2(row):

    try:
        return len(re.findall(r'\"(.+?)\"', row["question2"]))
    except:
        return 0

def get_features(x_train_feat):

    # x_train_feat["q1_sents"] = x_train_feat.apply(q1_sents, axis = 1)
    # x_train_feat["q2_sents"] = x_train_feat.apply(q2_sents, axis = 1)
    # x_train_feat["q1_exclaim"] = x_train_feat.apply(q1_sents, axis = 1)
    # x_train_feat["q2_exclaim"] = x_train_feat.apply(q2_sents, axis = 1)
    # x_train_feat["q1_question"] = x_train_feat.apply(q1_sents, axis = 1)
    # x_train_feat["q2_question"] = x_train_feat.apply(q2_sents, axis = 1)
    # x_train_feat["ne_score"] = x_train_feat.apply(common_ne_score, axis = 1)
    # x_train_feat["q1_ne_ratio"] = x_train_feat.apply(count_q1_ne, axis = 1)
    # x_train_feat["q2_ne_ratio"] = x_train_feat.apply(count_q2_ne, axis = 1)
    ##change names of q1_ne to q1_ne_ratio, q2_ne to q2_ne_ratio
    # x_train_feat["nc_score"] = x_train_feat.apply(common_chunk_score, axis = 1)
    # x_train_feat["quotes_q1"] = x_train_feat.apply(quotes_q1, axis = 1)
    # x_train_feat["quotes_q2"] = x_train_feat.apply(quotes_q2, axis = 1)
    ## x_train_feat["cluster_sim"] = x_train_feat.progress_apply(get_cluster_sim, axis = 1)
    # x_train_feat["q1_ne_hash_freq"] = x_train_feat.progress_apply(q1_ne_hash_freq, axis = 1)
    # x_train_feat["q2_ne_hash_freq"] = x_train_feat.progress_apply(q2_ne_hash_freq, axis = 1)

    # x_train_feat["sents_diff"] = abs(x_train_feat["q1_sents"] - x_train_feat["q2_sents"])
    # x_train_feat["exclaim_diff"] = abs(x_train_feat["q1_exclaim"] - x_train_feat["q2_exclaim"]) 
    # x_train_feat["question_diff"] = abs(x_train_feat["q1_question"] - x_train_feat["q2_question"]) 
    # x_train_feat["ne_diff"] = abs(x_train_feat["q1_ne_ratio"] - x_train_feat["q2_ne_ratio"]) 
    # x_train_feat["quotes_diff"] = abs(x_train_feat["quotes_q1"] - x_train_feat["quotes_q2"]) 
    # x_train_feat["chunk_hash_diff"] = abs(x_train_feat["q1_ne_hash_freq"] - x_train_feat["q2_ne_hash_freq"]) 

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


def cluster(to_be_clustered):

    cluster_hash = {}

    for index, row in to_be_clustered.iterrows():
        print index
        q1 = nlp(unicode(str(row["question1"]), "utf-8"))
        q2 = nlp(unicode(str(row["question2"]), "utf-8"))

        q1_ne = q1.ents
        q2_ne = q2.ents

        q1_ne = set([str(i) for i in q1_ne])
        q2_ne = set([str(i) for i in q2_ne])

        for i in q1_ne:
   
            if i not in cluster_hash:
                cluster_hash[i] = [j for j in q1_ne]
            else:
                cluster_hash[i] = list(set(cluster_hash[i] + [j for j in q1_ne]))

        for i in q2_ne:
   
            if i not in cluster_hash:
                cluster_hash[i] = [j for j in q2_ne]
            else:
                cluster_hash[i] = list(set(cluster_hash[i] + [j for j in q2_ne]))

    return cluster_hash

def get_cluster_sim(row):

    q1 = nlp(unicode(str(row["question1"]), "utf-8"))
    q2 = nlp(unicode(str(row["question2"]), "utf-8"))

    q1_ne = q1.ents
    q2_ne = q2.ents

    q1_ne = set([str(i) for i in q1_ne])
    q2_ne = set([str(i) for i in q2_ne])

    q1_combined_ne = []
    q2_combined_ne = []

    for i in q1_ne:
        if i not in cluster_hash:
            continue
        else:
            q1_combined_ne = list(set(q1_combined_ne + cluster_hash[i]))

    for i in q2_ne:
        if i not in cluster_hash:
            continue
        else:
            q2_combined_ne = list(set(q2_combined_ne + cluster_hash[i]))

    common_ne = len(set(q1_combined_ne).intersection(set(q2_combined_ne)))

    if common_ne == 0:
        return 0
    else:
       return common_ne/(len(q1_combined_ne) + len(q2_combined_ne) - common_ne)

def q1_ne_hash_freq(row):

    q1 = nlp(unicode(str(row["question1"]), "utf-8"))
    q1_ne = q1.ents

    if len(q1_ne) == 0:
        return 1

    q1_ne = "-".join(set([str(i).lower() for i in q1_ne]))
    hash_key1 = hash(q1_ne)

    if hash_key1 not in hash_table:
        return 1
    else:
        return hash_table[hash_key1]

def q2_ne_hash_freq(row):

    q2 = nlp(unicode(str(row["question2"]), "utf-8"))
    q2_ne = q2.ents

    if len(q2_ne) == 0:
        return 1

    q2_ne = "-".join(set([str(i).lower() for i in q2_ne]))
    hash_key2 = hash(q2_ne)

    if hash_key2 not in hash_table:
        return 1
    else:
        return hash_table[hash_key2]

def train_to_qid(row):
    
    global qid_hash

    if row["question1"] not in qid_hash:
        qid_hash[row["question1"]] = row["qid1"]    

    if row["question2"] not in qid_hash:
        qid_hash[row["question2"]] = row["qid2"]    

def test_to_qid(row):

    #cant take absolute qids, not seen in train. 
    #Difference magnitude means anything?
    #Proximity of qid, or frequency or something
    # instances_in_train_q1 = df_train[df_train["question1"] == row["question1"]]
    # instances_in_train_q2 = df_train[df_train["question2"] == row["question1"]]

    # if len(instances_in_train_q1) > 0:
    #     qid1 = instances_in_train_q1.iloc[0].qid1
    # elif len(instances_in_train_q2) > 0:
    #     qid1 = instances_in_train_q2.iloc[0].qid2
    # else:
    #     qid1 = 537934 + int(row["test_id"])

    # instances_in_train_q1 = df_train[df_train["question1"] == row["question2"]]
    # instances_in_train_q2 = df_train[df_train["question2"] == row["question2"]]

    # if len(instances_in_train_q1) > 0:
    #     qid2 = instances_in_train_q1.iloc[0].qid1
    # elif len(instances_in_train_q2) > 0:
    #     qid2 = instances_in_train_q2.iloc[0].qid2
    # else:
    #     qid2 = 537934 + int(row["test_id"]) + 1

    # return pd.Series({"qid1": qid1, "qid2": qid2})

    global qid_hash
    global max_qid

    if row["question1"] not in qid_hash:
        max_qid += 1

        qid_hash[row["question1"]] = max_qid

        q1_qid = max_qid
    else:
        print "dupli", row["question1"]
        q1_qid = qid_hash[row["question1"]]

    if row["question2"] not in qid_hash:
        max_qid += 1

        qid_hash[row["question2"]] = max_qid

        q2_qid = max_qid
    else:
        print "dupli", row["question1"]
        q2_qid = qid_hash[row["question2"]]

    return max(q1_qid, q2_qid)


if __name__ == '__main__':

    x_train_feat = pd.read_csv('./x_train_feat.csv').fillna("")
    x_train_feat['spacy_sim'] = pd.to_numeric(pd.Series(x_train_feat['spacy_sim']), errors = "coerce")

    # with open('cluster_hash.pickle', 'rb') as handle:
    #     cluster_hash = pickle.load(handle)

    with open('hash_table_ne.pickle', 'rb') as handle:
        hash_table_ne = pickle.load(handle)

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
    x_train_feat = x_train_feat.iloc[:, range(5, 53)]

    x_train, x_valid, y_train, y_valid = train_test_split(x_train_feat, x_label, test_size=0.2, random_state=4242, stratify = x_label)

    p = run_xgb_val(x_train, x_valid, y_train, y_valid)
