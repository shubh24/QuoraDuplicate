from __future__ import division
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.metrics import log_loss
import pandas as pd 
import numpy as np
import xgboost as xgb
import math
from sklearn.cross_validation import train_test_split
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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

def get_inverse_freq(inverse_freq, count, min_count=2):

    if count < min_count:
        return 0
    else:
        return math.log(inverse_freq)


def get_tf(text):

    tf = {}

    for word in text:
        tf[word] = text.count(word)/len(text)

    return tf

def word_match_share(row):

    q1_words = clean(str(row['question1']))
    q2_words = clean(str(row['question2']))

    if len(q1_words) == 0 or len(q2_words) == 0:
        return 0

    common_words = list(set(q1_words).intersection(q2_words))
    
    return len(common_words)/(len(q1_words) + len(q2_words) - len(common_words))

def weighted_word_match_share(row):

    q1_words = clean(str(row['question1']))
    q2_words = clean(str(row['question2']))

    q1_tf = get_tf(q1_words)
    q2_tf = get_tf(q2_words)

    if len(q1_words) == 0 or len(q2_words) == 0:
        return 0

    common_words = list(set(q1_words).intersection(q2_words))
    
    common_words_score = np.sum([weights.get(w, 0)*(q1_tf[w] + q2_tf[w])/2 for w in common_words])
    all_words_score = np.sum([weights.get(w, 0)*q1_tf[w] for w in q1_words]) + np.sum([weights.get(w, 0)*q2_tf[w] for w in q2_words]) - common_words_score
    
    return common_words_score/all_words_score

def get_features():

    x_train = pd.DataFrame()
    x_test = pd.DataFrame()

    x_train['word_match'] = df_train.apply(word_match_share, axis=1, raw=True)
    # x_train['tfidf_word_match'] = df_train.apply(weighted_word_match_share, axis=1, raw=True)
    x_train['z_tfidf_sum1'] = df_train.question1.map(lambda x: np.sum(tfidf.transform([str(x)]).data))
    x_train['z_tfidf_sum2'] = df_train.question2.map(lambda x: np.sum(tfidf.transform([str(x)]).data))

    x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)
    # x_test['tfidf_word_match'] = df_test.apply(weighted_word_match_share, axis=1, raw=True)
    x_test['z_tfidf_sum1'] = df_test.question1.map(lambda x: np.sum(tfidf.transform([str(x)]).data))
    x_test['z_tfidf_sum2'] = df_test.question2.map(lambda x: np.sum(tfidf.transform([str(x)]).data))

    x_train['z_len1'] = x_train.question1.map(lambda x: len(str(x)))    
    x_train['z_len2'] = x_train.question2.map(lambda x: len(str(x)))

    x_test['z_len1'] = x_test.question1.map(lambda x: len(str(x)))    
    x_test['z_len2'] = x_test.question2.map(lambda x: len(str(x)))

    y_train = df_train['is_duplicate'].values

    return x_train, x_test, y_train

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

def run_xgb(pos_train, neg_train, x_test):

    x_train = pd.concat([pos_train, neg_train]) #Concat positive and negative
    y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist() #Putting in 1 and 0

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

    # Set our parameters for xgboost
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['max_depth'] = 4

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

    d_test = xgb.DMatrix(x_test)
    p_test = bst.predict(d_test)

    submit(p_test)

if __name__ == '__main__':
    
    df_train = pd.read_csv('./train.csv')
    df_test = pd.read_csv('./test.csv')

    # p = df_train['is_duplicate'].mean() # Our predicted probability

    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
    test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
    tfidf.fit_transform(train_qs)

    words = (" ".join(train_qs)).lower().split()
    counts = Counter(words)
    weights = {word: get_inverse_freq(1/(10000 + count), count) for word, count in counts.items()}

    stops = set(stopwords.words("english"))

    x_train, x_test, y_train = get_features()

    pos_train, neg_train = oversample(x_train, y_train)

    run_xgb(pos_train, neg_train, x_test)