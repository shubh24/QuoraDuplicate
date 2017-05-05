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

def clean(text):

    text = ''.join([c for c in text if c not in punctuation])
    # text = text.encode('ascii', 'ignore').decode('ascii')

    text = text.lower().split()
    text = [w for w in text if not w in stops]

    # stemmer = SnowballStemmer('english')
    # cleaned_words = [stemmer.stem(word) for word in text]

    return text

def clean_master(row):

    q1_words = clean(str(row['question1']))
    q2_words = clean(str(row['question2']))

    return pd.Series({"q1_words" : q1_words, "q2_words": q2_words})

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

def word_match_share(row):

    q1_words = row[0]
    q2_words = row[1]

    if len(q1_words) == 0 or len(q2_words) == 0:
        return 0

    common_words = len(list(set(q1_words).intersection(q2_words)))
    all_words = len(q1_words) + len(q2_words) - common_words

    return common_words/all_words

def weighted_word_match_share(row):

    # q1_tf = get_tf(q1_words)
    # q2_tf = get_tf(q2_words)

    q1_words = row[0]
    q2_words = row[1]

    if len(q1_words) == 0 or len(q2_words) == 0:
        return 0

    common_words = list(set(q1_words).intersection(q2_words))
    
    common_words_score = np.sum([weights.get(w, 0) for w in common_words])
    all_words_score = np.sum([weights.get(w, 0) for w in q1_words]) + np.sum([weights.get(w, 0) for w in q2_words]) - common_words_score
    
    return pd.Series({"weighted_word_match_ratio" : common_words_score/all_words_score, "weighted_word_match_diff": all_words_score - common_words_score, "weighted_word_match_sum": common_words_score})

def get_bigrams(row):

    q1_words = str(row["question1"]).split(" ")
    q2_words = str(row["question2"]).split(" ")

    bigrams_q1 = set(ngrams(q1_words, 2))
    bigrams_q2 = set(ngrams(q2_words, 2))

    common_bigrams = len(bigrams_q1.intersection(bigrams_q2))

    if common_bigrams == 0:
        return 0
    else:
        return common_bigrams/(len(bigrams_q1.union(bigrams_q2)))    

def get_trigrams(row):

    q1_words = str(row["question1"]).split(" ")
    q2_words = str(row["question2"]).split(" ")

    trigrams_q1 = set(ngrams(q1_words, 3))
    trigrams_q2 = set(ngrams(q2_words, 3))

    common_trigrams = len(trigrams_q1.intersection(trigrams_q2))

    if common_trigrams == 0:
        return 0
    else:
        return common_trigrams/(len(trigrams_q1.union(trigrams_q2)))    

def pos_match(row):

    q1 = str(row["question1"])
    q2 = str(row["question2"])

    q1 = ''.join([c.lower() for c in q1 if c not in punctuation])
    q2 = ''.join([c.lower() for c in q2 if c not in punctuation])

    pos_tag1 = nltk.pos_tag(nltk.word_tokenize(q1))
    pos_tag2 = nltk.pos_tag(nltk.word_tokenize(q2))

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
    all_pos_score = np.sum([weights.get(w, 0) for w in nltk.word_tokenize(q1)]) + np.sum([weights.get(w, 0) for w in nltk.word_tokenize(q2)]) - common_pos_score

    return common_pos_score/all_pos_score

def generate_hash_freq(row):

    hash_key1 = hash(row["question1"])
    hash_key2 = hash(row["question2"])

    if hash_key1 not in hash_table:
        hash_table[hash_key1] = 1
    else:
        hash_table[hash_key1] += 1

    if hash_key2 not in hash_table:
        hash_table[hash_key2] = 1
    else:
        hash_table[hash_key2] += 1

def q1_hash_freq(row):

    hash_key1 = hash(str(row["question1"]))
    return hash_table[hash_key1]

def q2_hash_freq(row):

    hash_key2 = hash(str(row["question2"]))
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

def stopwords_ratio_q1(row):

    q1 = str(row["question1"])

    q1 = ''.join([c.lower() for c in q1 if c not in punctuation]).split(" ")

    return len(set(q1).intersection(stops))/len(q1)

def stopwords_ratio_q2(row):

    q2 = str(row["question2"])

    q2 = ''.join([c.lower() for c in q2 if c not in punctuation]).split(" ")

    return len(set(q2).intersection(stops))/len(q2)

def question_type(row):

    fw_q1 = str(row["question1"]).lower().split(" ")[0]
    fw_q2 = str(row["question2"]).lower().split(" ")[0]

    if fw_q1 == fw_q2 and fw_q1 in question_types:
        return 1
    else:
        return 0

def cluster(to_be_clustered):

    cluster_hash = {}
    cluster_counter = 1

    for index, row in to_be_clustered.iterrows():
        q1_words = tuple(row[0])
        q2_words = tuple(row[1])
        is_duplicate = row[2]

        if is_duplicate == 1:
            if q1_words not in cluster_hash and q2_words not in cluster_hash:
                cluster_hash[q1_words] = cluster_counter
                cluster_hash[q2_words] = cluster_counter
                cluster_counter += 1

            elif q1_words in cluster_hash and q2_words not in cluster_hash:
                cluster_hash[q2_words] = cluster_hash[q1_words]

            elif q2_words in cluster_hash and q1_words not in cluster_hash:
                cluster_hash[q1_words] = cluster_hash[q2_words]

        elif is_duplicate == 0:
            if q1_words not in cluster_hash:
                cluster_hash[q1_words] = cluster_counter
                cluster_counter += 1

            if q2_words not in cluster_hash:
                cluster_hash[q2_words] = cluster_counter
                cluster_counter += 1

    return cluster_hash

def get_cluster_size_q1(row):

    q1 = tuple(row["q1_words"])

    if q1 not in cluster_hash:
        return 0
    else:
        return inverse_cluster[cluster_hash[q1]]

def get_cluster_size_q2(row):

    q2 = tuple(row["q2_words"])

    if q2 not in cluster_hash:
        return 0
    else:
        return inverse_cluster[cluster_hash[q2]]

def get_features(x_train, x_test):
    
    cleaned_train = x_train.apply(clean_master, axis=1, raw=True)
    cleaned_test = x_test.apply(clean_master, axis=1, raw=True)

    x_train_feat = cleaned_train.apply(weighted_word_match_share, axis=1)
    x_train_feat['word_match'] = cleaned_train.apply(word_match_share, axis=1)    
    # x_train_feat['z_tfidf_sum1'] = x_train.question1.map(lambda x:  np.sum(tfidf.transform([str(x)]).data))
    # x_train_feat['z_tfidf_sum2'] = x_train.question2.map(lambda x: np.sum(tfidf.transform([str(x)]).data))

    x_test_feat = cleaned_test.apply(weighted_word_match_share, axis=1)
    x_test_feat['word_match'] = cleaned_test.apply(word_match_share, axis=1)
    # x_test_feat['z_tfidf_sum1'] = x_test.question1.map(lambda x: np.sum(tfidf.transform([str(x)]).data))
    # x_test_feat['z_tfidf_sum2'] = x_test.question2.map(lambda x: np.sum(tfidf.transform([str(x)]).data))

    cluster_hash = cluster(pd.concat([pd.concat([cleaned_train, cleaned_test]), pd.concat([y_train, y_valid])], axis = 1))

    inverse_cluster = {}
    for i in cluster_hash.values():
        if i not in inverse_cluster:
            inverse_cluster[i] = 1
        else:
            inverse_cluster[i] += 1

    x_train_feat["cluster_count_q1"] = cleaned_train.apply(get_cluster_size_q1, axis = 1)
    x_train_feat["cluster_count_q2"] = cleaned_train.apply(get_cluster_size_q2, axis = 1)

    x_test_feat["cluster_count_q1"] = cleaned_test.apply(get_cluster, axis = 1)
    x_test_feat["cluster_count_q2"] = cleaned_test.apply(get_cluster, axis = 1)

    x_train_feat['bigram_match'] = x_train.apply(get_bigrams, axis = 1)
    x_test_feat['bigram_match'] = x_test.apply(get_bigrams, axis = 1)

    x_train_feat['trigram_match'] = x_train.apply(get_trigrams, axis = 1)
    x_test_feat['trigram_match'] = x_test.apply(get_trigrams, axis = 1)

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

    x_train_feat['pos_match_ratio'] = x_train.apply(pos_match, axis = 1)
    x_test_feat['pos_match_ratio'] = x_test.apply(pos_match, axis = 1)

    x_train_feat['question_type'] = x_train.apply(question_type, axis = 1)
    x_test_feat['question_type'] = x_test.apply(question_type, axis = 1)

    x_train_feat['stopwords_ratio_q1'] = x_train.apply(stopwords_ratio_q1, axis = 1)    
    x_train_feat['stopwords_ratio_q2'] = x_train.apply(stopwords_ratio_q2, axis = 1)    

    x_test_feat['stopwords_ratio_q1'] = x_test.apply(stopwords_ratio_q1, axis = 1)    
    x_test_feat['stopwords_ratio_q2'] = x_test.apply(stopwords_ratio_q2, axis = 1)    
    
    x_train_feat['z_len1'] = cleaned_train.q1_words.map(lambda x: len(str(x)))    
    x_train_feat['z_len2'] = cleaned_train.q2_words.map(lambda x: len(str(x)))
    x_train_feat['len_diff'] = abs(x_train_feat['z_len1'] - x_train_feat['z_len2'])
    x_train_feat['z_avg_len'] = (x_train_feat['z_len1'] + x_train_feat['z_len2'])/2

    x_test_feat['z_len1'] = cleaned_test.q1_words.map(lambda x: len(str(x)))    
    x_test_feat['z_len2'] = cleaned_test.q2_words.map(lambda x: len(str(x)))
    x_test_feat['len_diff'] = abs(x_test_feat['z_len1'] - x_test_feat['z_len2'])
    x_test_feat['z_avg_len'] = (x_test_feat['z_len1'] + x_test_feat['z_len2'])/2

    x_train_feat['z_words1'] = cleaned_train.q1_words.map(lambda row: len(str(row).split(" ")))    
    x_train_feat['z_words2'] = cleaned_train.q2_words.map(lambda row: len(str(row).split(" ")))
    x_train_feat['words_diff'] = abs(x_train_feat['z_words1'] - x_train_feat['z_words2'])
    x_train_feat['z_avg_words'] = (x_train_feat['z_words1'] + x_train_feat['z_words2'])/2

    x_test_feat['z_words1'] = cleaned_test.q1_words.map(lambda row: len(str(row).split(" ")))    
    x_test_feat['z_words2'] = cleaned_test.q2_words.map(lambda row: len(str(row).split(" ")))
    x_test_feat['words_diff'] = abs(x_test_feat['z_words1'] - x_test_feat['z_words2'])
    x_test_feat['z_avg_words'] = (x_test_feat['z_words1'] + x_test_feat['z_words2'])/2

    # y_train = x_train['is_duplicate'].values

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

    x_train, x_valid, y_train, y_valid = validate(df_train)
    
    x_train_feat, x_valid_feat = get_features(x_train, x_valid)
    res = run_xgb(x_train_feat, x_valid_feat, y_train, y_valid)

    # res = controller(x_train, x_valid, y_train, y_valid)

    #Compare res & y_valid

    # submit(res)
