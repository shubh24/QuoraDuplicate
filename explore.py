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

    # sequence1 = get_word_bigrams(q1_words)
    # sequence2 = get_word_bigrams(q2_words)

    # try:
    #     simhash_diff = Simhash(sequence1).distance(Simhash(sequence2))/64
    # except:
    #     simhash_diff = 0.5

    q1 = nlp(unicode(str(row["question1"]), "utf-8"))
    q2 = nlp(unicode(str(row["question2"]), "utf-8"))

    q1_ne = q1.ents
    q2_ne = q2.ents

    q1_ne = set([str(i) for i in q1_ne])
    q2_ne = set([str(i) for i in q2_ne])

    if len(q1_ne) == 0:
        q1_ne_ratio = 0
    else:
        q1_ne_ratio = len(q1_ne)/len(row["question1"].split())

    if len(q2_ne) == 0:
        q2_ne_ratio = 0
    else:
        q2_ne_ratio = len(q2_ne)/len(row["question2"].split())

    common_ne = len(q1_ne.intersection(q2_ne))

    if len(q1_ne) + len(q2_ne) == 0:
        common_ne_score = 0
    else:
       common_ne_score = common_ne/(len(q1_ne) + len(q2_ne) - common_ne)

    pos_hash = {}
    common_pos = []

    for word in q1:
        if word.tag_ not in pos_hash:
            pos_hash.update({word.tag_ : [word.text]})
        else:
            pos_hash[word.tag_].append(word.text)

    for word in q2:
        if word.tag_ not in pos_hash:
            continue
        if word.text in pos_hash[word.tag_]:
            common_pos.append(word.text)

    common_pos_score = np.sum([weights.get(w, 0) for w in common_pos])
    all_pos_score = np.sum([weights.get(w, 0) for w in q1_words]) + np.sum([weights.get(w, 0) for w in q2_words]) - common_pos_score

    q1_pronouns_count = 0
    q2_pronouns_count = 0

    for word in q1:
        if str(word.tag_) == "PRP":
            q1_pronouns_count += 1

    for word in q2:
        if str(word.tag_) == "PRP":
            q2_pronouns_count += 1

    pronouns_diff = abs(q1_pronouns_count - q2_pronouns_count)

    q1_nc = q1.noun_chunks
    q2_nc = q2.noun_chunks

    q1_nc = set([str(i) for i in q1_nc])
    q2_nc = set([str(i) for i in q2_nc])

    common_nc = len(q1_nc.intersection(q2_nc))

    if len(q1_nc) + len(q2_nc) == 0:
        common_nc_score = 0
    else:
       common_nc_score = common_nc/(len(q1_nc) + len(q2_nc) - common_nc)

    fw_q1 = q1_words[0]
    fw_q2 = q2_words[0]

    if fw_q1 == fw_q2 and fw_q1 in question_types:
        question_type_same = 1
    else:
        question_type_same = 0

    try:
        q1_quotes = len(re.findall(r'\"(.+?)\"', row["question1"]))
    except:
        q1_quotes = 0

    try:
        q2_quotes = len(re.findall(r'\"(.+?)\"', row["question2"]))
    except:
        q2_quotes = 0

    # with open('hash_table_ne.pickle', 'rb') as handle:
    #     hash_table_ne = pickle.load(handle)

    if len(q1_ne) == 0:
        q1_ne_hash_freq = 1
    else:
        hash_key1 = hash("-".join(set([str(i).lower() for i in q1_ne])))

        if hash_key1 not in hash_table_ne:
            q1_ne_hash_freq = 1
        else:
            q1_ne_hash_freq = hash_table_ne[hash_key1]

    if len(q2_ne) == 0:
        q2_ne_hash_freq = 1
    else:
        hash_key2 = hash("-".join(set([str(i).lower() for i in q2_ne])))

        if hash_key2 not in hash_table_ne:
            q2_ne_hash_freq = 1
        else:
            q2_ne_hash_freq = hash_table_ne[hash_key2]

    try:
        q1_sents = len(nltk.tokenize.sent_tokenize(row.question1))
    except:
        q1_sents = 1
    try:
        q2_sents = len(nltk.tokenize.sent_tokenize(row.question2))
    except:
        q2_sents = 1

    q1_exclaim = sum([1 for i in str(row.question1) if i == "!"])
    q2_exclaim = sum([1 for i in str(row.question2) if i == "!"])

    q1_question = sum([1 for i in str(row.question1) if i == "?"])
    q2_question = sum([1 for i in str(row.question2) if i == "?"])

    hash_key1 = hash(str(row["question1"]).lower())
    if hash_key1 in hash_table:
        q1_hash_freq = hash_table[hash_key1]
    else:
        q1_hash_freq = 1

    hash_key2 = hash(str(row["question2"]).lower())
    if hash_key2 in hash_table:
        q2_hash_freq = hash_table[hash_key2]
    else:
        q2_hash_freq = 1

    spacy_sim = q1.similarity(q2)

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
        # "simhash_diff": simhash_diff,
        "question_type_same": question_type_same,
        "q1_stops": len(set(q1_words).intersection(stops))/len(q1_words),
        "q2_stops": len(set(q2_words).intersection(stops))/len(q2_words),
        "q1_len": len(str(row.question1)),
        "q2_len": len(str(row.question2)),
        "len_diff": abs(len(str(row.question1)) - len(str(row.question2))),
        "len_avg": (len(str(row.question1)) + len(str(row.question2)))/2,
        "q1_sents": q1_sents,
        "q2_sents": q2_sents,
        "sents_diff": abs(q1_sents - q2_sents),
        "q1_words": len(q1_words),
        "q2_words": len(q2_words),
        "words_diff": abs(len(q1_words) - len(q2_words)),
        "words_avg": (len(q1_words) + len(q2_words))/2,
        "q1_caps_count": sum([1 for i in str(row.question1) if i.isupper()]),
        "q2_caps_count": sum([1 for i in str(row.question2) if i.isupper()]),
        "q1_exclaim": q1_exclaim,
        "q2_exclaim": q2_exclaim,
        "exclaim_diff": abs(q1_exclaim - q2_exclaim),
        "q1_question": q1_question,
        "q2_question": q2_question,
        "question_diff": abs(q1_question - q2_question),
        "ne_score": common_ne_score,
        "nc_score": common_nc_score,
        "q1_ne_ratio": q1_ne_ratio,
        "q2_ne_ratio": q2_ne_ratio,
        "ne_diff": abs(q1_ne_ratio - q2_ne_ratio),
        "q1_quotes": q1_quotes,
        "q2_quotes": q2_quotes,
        "quotes_diff": abs(q1_quotes - q2_quotes),
        "q1_ne_hash_freq": q1_ne_hash_freq,
        "q2_ne_hash_freq": q2_ne_hash_freq,
        "chunk_hash_diff": abs(q1_ne_hash_freq - q2_ne_hash_freq),
        "q1_hash_freq": q1_hash_freq,
        "q2_hash_freq": q2_hash_freq,
        "q_freq_avg": (q1_hash_freq + q2_hash_freq)/2,
        "freq_diff": abs(q1_hash_freq - q2_hash_freq),
        "spacy_sim": spacy_sim,
        "q1_pronouns_count": q1_pronouns_count,
        "q2_pronouns_count": q2_pronouns_count,
        "pronouns_diff": pronouns_diff
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

    # d_test = xgb.DMatrix(x_test_feat)
    # p_test = bst.predict(d_test)

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
    # df_test.apply(generate_hash_freq, axis = 1)
    with open('hash_table.pickle', 'wb') as handle:
        pickle.dump(hash_table, handle)

    with open('hash_table.pickle', 'rb') as handle:
        hash_table = pickle.load(handle)

    with open('hash_table_ne.pickle', 'rb') as handle:
        hash_table_ne = pickle.load(handle)

    #Validation
    x_train, x_test, y_train, y_valid = validate(df_train)
    x_train_feat = x_train.apply(basic_nlp, axis = 1)
    x_test_feat = x_test.apply(basic_nlp, axis = 1)
    res = run_xgb(x_train_feat, x_test_feat, y_train, y_valid)

    # res = controller(x_train, x_valid, y_train, y_valid)

    #Compare res & y_valid

    #Real Testing
    x_train = df_train.apply(basic_nlp, axis = 1)
    x_train.to_csv('x_train.csv', index=False)
    %reset_selective x_train 

    x_test_1 = df_test[0:390000].apply(basic_nlp, axis = 1)
    x_test_1.to_csv('x_test_1.csv', index=False)   
    %reset_selective x_test_1   

    x_test_2 = df_test[390000:780000].apply(basic_nlp, axis = 1)
    x_test_2.to_csv('x_test_2.csv', index=False)
    %reset_selective x_test_2   

    x_test_3 = df_test[780000:1170000].apply(basic_nlp, axis = 1)
    x_test_3.to_csv('x_test_3.csv', index=False)   
    %reset_selective x_test_3   

    x_test_4 = df_test[1170000:1560000].apply(basic_nlp, axis = 1)
    x_test_4.to_csv('x_test_4.csv', index=False)   
    %reset_selective x_test_4   

    x_test_5 = df_test[1560000:1950000].apply(basic_nlp, axis = 1)
    x_test_5.to_csv('x_test_5.csv', index=False)   
    %reset_selective x_test_5   

    x_test_6 = df_test[1950000:2345796].apply(basic_nlp, axis = 1)
    x_test_6.to_csv('x_test_6.csv', index=False)   
    %reset_selective x_test_6   

    # submit(res)
