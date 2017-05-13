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

question_types = ["what", "how", "why", "is", "which", "can", "i", "who", "do", "where", "if", "does", "are", "when", "should", "will", "did", "has", "would", "have", "was", "could"]

def submit(p_test):

    sub = pd.DataFrame()

    sub['test_id'] = df_test['test_id']
    sub['is_duplicate'] = p_test

    sub.to_csv('simple_xgb.csv', index=False)   

def get_inverse_freq(inverse_freq, count, min_count=2):

    if count < min_count:   
        return 0
    else:
        return inverse_freq

def get_tf(text):

    tf = {}

    for word in text:
        tf[word] = text.count(word)/len(text)

    return tf


def tuple_similarity(q1_words, q2_words):

    if len(q1_words) == 0 or len(q2_words) == 0:
        return 0

    common_words = len(set(q1_words).intersection(set(q2_words)))
    all_words = len(set(q1_words).union(set(q2_words)))

    return common_words/all_words

def basic_nlp(row):

    # q1_tf = get_tf(q1_words)
    # q2_tf = get_tf(q2_words)
    
    q1_words = str(row.question1).lower().split()
    q2_words = str(row.question2).lower().split()

    #modify this!
    if len(q1_words) == 0 or len(q2_words) == 0:
        return 0

    common_words = list(set(q1_words).intersection(q2_words))
    
    common_words_score = np.sum([weights.get(w, 0) for w in common_words])
    all_words_score = np.sum([weights.get(w, 0) for w in q1_words]) + np.sum([weights.get(w, 0) for w in q2_words]) - common_words_score

    hamming_score = sum(1 for i in zip(q1_words, q2_words) if i[0]==i[1])/max(len(q1_words), len(q2_words))

    jacard_score =  len(common_words)/(len(q1_words) + len(q2_words) - len(common_words))  
    cosine_score = len(common_words)/(pow(len(q1_words),0.5)*pow(len(q2_words),0.5))

    bigrams_q1 = set(ngrams(q1_words, 2))
    bigrams_q2 = set(ngrams(q2_words, 2))
    common_bigrams = len(bigrams_q1.intersection(bigrams_q2))
    if common_bigrams == 0:
        bigram_score = 0
    else:
        bigram_score = common_bigrams/(len(bigrams_q1.union(bigrams_q2)))    

    trigrams_q1 = set(ngrams(q1_words, 3))
    trigrams_q2 = set(ngrams(q2_words, 3))
    common_trigrams = len(trigrams_q1.intersection(trigrams_q2))
    if common_trigrams == 0:
        trigram_score = 0
    else:
        trigram_score = common_trigrams/(len(trigrams_q1.union(trigrams_q2)))    

    pos_tag1 = nltk.pos_tag(q1_words)
    pos_tag2 = nltk.pos_tag(q2_words)
    pos_hash = {}
    common_pos = []
    
    for tag in pos_tag1:
        if tag[1] not in pos_hash:
            pos_hash.update({tag[1]:[tag[0]]})
        else:
            pos_hash[tag[1]].append(tag[0])
    for tag in pos_tag2:
        if tag[1] not in pos_hash:
            continue
        if tag[0] in pos_hash[tag[1]]:
            common_pos.append(tag[0])

    common_pos_score = np.sum([weights.get(w, 0) for w in common_pos])
    all_pos_score = np.sum([weights.get(w, 0) for w in q1_words]) + np.sum([weights.get(w, 0) for w in q2_words]) - common_pos_score

    sequence1 = get_word_bigrams(q1_words)
    sequence2 = get_word_bigrams(q2_words)

    try:
        simhash_diff = Simhash(sequence1).distance(Simhash(sequence2))/64
    except:
        simhash_diff = 0.5

    fw_q1 = q1_words[0]
    fw_q2 = q2_words[0]

    if fw_q1 == fw_q2 and fw_q1 in question_types:
        question_type_same = 1
    else:
        question_type_same = 0

    return pd.Series({

        "weighted_word_match_ratio" : common_words_score/all_words_score,
        "weighted_word_match_diff": all_words_score - common_words_score, 
        "weighted_word_match_sum": common_words_score,
        "jacard_score": jacard_score,
        "hamming_score": hamming_score,
        "cosine_score": cosine_score,
        "bigram_score": bigram_score,
        "trigram_score": trigram_score,
        "pos_score": common_pos_score/all_pos_score,
        "simhash_diff": simhash_diff,
        "question_type_same": question_type_same,
        "q1_stops": len(set(q1_words).intersection(stops))/len(q1_words),
        "q2_stops": len(set(q2_words).intersection(stops))/len(q2_words),
        "q1_len": len(str(row.question1)),
        "q2_len": len(str(row.question2)),
        "len_diff": abs(len(str(row.question1)) - len(str(row.question2))),
        "len_avg": (len(str(row.question1)) + len(str(row.question2)))/2,
        "q1_sents": len(nltk.tokenize.sent_tokenize(row.question1)),
        "q2_sents": len(nltk.tokenize.sent_tokenize(row.question2)),
        "q1_words": len(q1_words),
        "q2_words": len(q2_words),
        "words_diff": abs(len(q1_words) - len(q2_words)),
        "words_avg": (len(q1_words) + len(q2_words))/2,
        "q1_caps_count": sum([1 for i in str(row.question1) if i.isupper()]),
        "q2_caps_count": sum([1 for i in str(row.question2) if i.isupper()]),
        "q1_exclaim": sum([1 for i in str(row.question1) if i == "!"]),
        "q2_exclaim": sum([1 for i in str(row.question2) if i == "!"]),
        "q1_question": sum([1 for i in str(row.question1) if i == "?"]),
        "q2_question": sum([1 for i in str(row.question2) if i == "?"]),

    })

def get_word_bigrams(words):

    ngrams = []

    for i in range(0, len(words)):
        if i > 0:
            ngrams.append("%s %s"%(words[i-1], words[i]))

    return ngrams

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

def q1_hash_freq(row):

    hash_key1 = hash(str(row["question1"]).lower())
    return hash_table[hash_key1]

def q2_hash_freq(row):

    hash_key2 = hash(str(row["question2"]).lower())
    return hash_table[hash_key2]

def spacy_sim(row):

    q1 = nlp(unicode(str(row["question1"]), "utf-8"))
    q2 = nlp(unicode(str(row["question2"]), "utf-8"))

    return q1.similarity(q2)

def common_ne_score(row):

    q1 = nlp(unicode(str(row["question1"]), "utf-8"))
    q2 = nlp(unicode(str(row["question2"]), "utf-8"))

    q1_ne = q1.ents
    q2_ne = q2.ents

    common_ne = len(list(set(q1_ne).intersection(q2_ne)))

    if len(q1_ne) + len(q2_ne) == 0:
        return 0
    else:
       return common_ne/(len(q1_ne) + len(q2_ne) - common_ne)


def cluster(to_be_clustered):

    cluster_hash = {}
    inverse_hash = {}
    cluster_counter = 1

    for index, row in to_be_clustered.iterrows():
        q1_words = tuple(row[0])
        q2_words = tuple(row[1])
        is_duplicate = row[2]

        if is_duplicate == 1:
            if q1_words not in cluster_hash and q2_words not in cluster_hash:
                cluster_hash[q1_words] = cluster_counter
                cluster_hash[q2_words] = cluster_counter
                inverse_hash[cluster_counter] = [q1_words, q2_words]
                cluster_counter += 1

            elif q1_words in cluster_hash and q2_words not in cluster_hash:
                cluster_hash[q2_words] = cluster_hash[q1_words]
                inverse_hash[cluster_hash[q1_words]].append(q2_words)

            elif q2_words in cluster_hash and q1_words not in cluster_hash:
                cluster_hash[q1_words] = cluster_hash[q2_words]
                inverse_hash[cluster_hash[q2_words]].append(q1_words)                

        elif is_duplicate == 0:
            if q1_words not in cluster_hash:
                cluster_hash[q1_words] = cluster_counter
                inverse_hash[cluster_counter] = [q1_words]
                cluster_counter += 1

            if q2_words not in cluster_hash:
                cluster_hash[q2_words] = cluster_counter
                inverse_hash[cluster_counter] = [q2_words]
                cluster_counter += 1

    for i in inverse_hash:
        tuple_sum = tuple()
        for j in inverse_hash[i]:
            tuple_sum += j
        inverse_hash[i] = tuple(set(tuple_sum))

    return cluster_hash, inverse_hash

def get_cluster_sim(row):

    q1 = tuple(row["q1_words"])
    q2 = tuple(row["q2_words"])

    if q1 in cluster_hash and q2 in cluster_hash:
        if cluster_hash[q1] == cluster_hash[q2]:
            return 1
        else:
            return 0
    else:
        return 0
    #     else:
    #         return tuple_similarity(inverse_hash[cluster_hash[q1]], inverse_hash[cluster_hash[q2]])

    # elif q1 in cluster_hash and q2 not in cluster_hash:
    #     return tuple_similarity(inverse_hash[cluster_hash[q1]], q2)

    # elif q2 in cluster_hash and q1 not in cluster_hash:
    #     return tuple_similarity(inverse_hash[cluster_hash[q2]], q1)

    # else:
    #     return tuple_similarity(q1, q2)

def get_features(x_train, x_test):
    
    x_train_feat = x_train.apply(basic_nlp, axis=1)
    x_test_feat = x_test.apply(basic_nlp, axis=1)

    x_train_feat["q1_freq"] = x_train.apply(q1_hash_freq, axis = 1)
    x_train_feat["q2_freq"] = x_train.apply(q2_hash_freq, axis = 1)
    x_train_feat["q_freq_avg"] = (x_train_feat["q1_freq"] + x_train_feat["q2_freq"])/2

    x_test_feat["q1_freq"] = x_test.apply(q1_hash_freq, axis = 1)
    x_test_feat["q2_freq"] = x_test.apply(q2_hash_freq, axis = 1)
    x_test_feat["q_freq_avg"] = (x_test_feat["q1_freq"] + x_test_feat["q2_freq"])/2

    x_train_feat["spacy_sim"] = x_train.apply(spacy_sim, axis = 1)
    x_test_feat["spacy_sim"] = x_test.apply(spacy_sim, axis = 1)

    # x_train_feat["common_ne_score"] = x_train.apply(common_ne_score, axis = 1)
    # x_test_feat["common_ne_score"] = x_test.apply(common_ne_score, axis = 1)

    return x_train_feat, x_test_feat

def oversample(x_train, y_train):

    pos_train = x_train[y_train == 1]
    neg_train = x_train[y_train == 0]

    #Oversampling negative class
    p = 0.165
    scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1 #How much times greater is the train ratio than actual

    while scale > 1:
        neg_train = pd.concat([neg_train, neg_train])
        scale -=1
    neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])

    return pos_train, neg_train

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

    d_test = xgb.DMatrix(x_test_feat)
    p_test = bst.predict(d_test)

    xgb.plot_importance(bst)
    pyplot.show()

    return p_test

def run_tsne(pos_train, neg_train, x_test_feat):

    x_train = pd.concat([pos_train, neg_train]) #Concat positive and negative
    y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist() #Putting in 1 and 0

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

    df_subsampled = x_train[0:3000]
    X = MinMaxScaler().fit_transform(df_subsampled[['z_len1', 'z_len2', 'z_words1', 'z_words2', 'word_match']])
    # y = y_train['is_duplicate'].values

    tsne = TSNE(
        n_components=3,
        init='random', # pca
        random_state=101,
        method='barnes_hut',
        n_iter=200,
        verbose=2,
        angle=0.5
    ).fit_transform(X)

    trace1 = go.Scatter3d(
        x=tsne[:,0],
        y=tsne[:,1],
        z=tsne[:,2],
        mode='markers',
        marker=dict(
            sizemode='diameter',
            color = y_train,
            colorscale = 'Portland',
            colorbar = dict(title = 'duplicate'),
            line=dict(color='rgb(255, 255, 255)'),
            opacity=0.75
        )
    )

    data=[trace1]
    layout=dict(height=800, width=800, title='3d embedding with engineered features')
    fig=dict(data=data, layout=layout)
    py.plot(data, filename='3d_bubble')

def validate(training):

    training_res = training.pop("is_duplicate")
    x_train, x_valid, y_train, y_valid = train_test_split(training, training_res, test_size=0.2, random_state=4242, stratify = training_res)

    return(x_train, x_valid, y_train, y_valid)

def controller(x_train, x_valid, y_train, y_valid):

    # x_train, x_test_feat = get_features(x_train, x_valid)

    # pos_train, neg_train = oversample(x_train, y_train) #Taking lite for now

    return run_xgb(x_train, x_valid, y_train, y_valid)

    # run_tsne(pos_train, neg_train, x_test_feat)

if __name__ == '__main__':
    
    df_train = pd.read_csv('./train.csv').fillna("")
    df_test = pd.read_csv('./test.csv').fillna("")

    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
    test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

    # tfidf = TfidfVectorizer(max_features = 256, stop_words='english', ngram_range=(1, 1))
    # tfidf.fit_transform(train_qs[0:2500])

    words = (" ".join(train_qs)).lower().split()
    counts = Counter(words)
    weights = {word: get_inverse_freq(1/(10000 + int(count)), count) for word, count in counts.items()}

    stops = set(stopwords.words("english"))

    hash_table = {}

    df_train.apply(generate_hash_freq, axis = 1)
    df_test.apply(generate_hash_freq, axis = 1)

    x_train, x_test, y_train, y_valid = validate(df_train)
    x_train_feat, x_test_feat = get_features(x_train, x_test)
    final_train = pd.concat([x_train_feat, x_test_feat])

    final_test = get_features_test(df_test)

    res = run_xgb(x_train_feat, x_test_feat, y_train, y_valid)

    # res = controller(x_train, x_valid, y_train, y_valid)

    #Compare res & y_valid

    # submit(res)
