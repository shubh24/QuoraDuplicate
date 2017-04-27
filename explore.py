from __future__ import division
from collections import Counter
from nltk.corpus import stopwords
from sklearn.metrics import log_loss
import pandas as pd 
import numpy as np

df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')

# print df_train.head()

p = df_train['is_duplicate'].mean() # Our predicted probability

def submit():

	print('Predicted score:', log_loss(df_train['is_duplicate'], np.zeros_like(df_train['is_duplicate']) + p))

	sub = pd.DataFrame({'test_id': df_test['test_id'], 'is_duplicate': p})
	sub.to_csv('naive_submission.csv', index=False)
	
	return sub

stops = set(stopwords.words("english"))

def word_match_share(row):

	q1_words = str(row['question1']).lower().split()
	q1_words = [x for x in q1_words if x not in stops]

	q2_words = str(row['question2']).lower().split()
	q2_words = [x for x in q2_words if x not in stops]

	if len(q1_words) == 0 or len(q2_words) == 0:
		return 0

	common_words = list(set(q1_words).intersection(q2_words))
	
	return len(common_words)/(len(q1_words) + len(q2_words) - len(common_words))

def get_weight(count, eps=10000, min_count=2):

    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

def weighted_word_match_share(row):

	q1_words = str(row['question1']).lower().split()
	q1_words = [x for x in q1_words if x not in stops]

	q2_words = str(row['question2']).lower().split()
	q2_words = [x for x in q2_words if x not in stops]

	if len(q1_words) == 0 or len(q2_words) == 0:
		return 0

	common_words = list(set(q1_words).intersection(q2_words))

	common_words_score = np.sum([weights.get(w, 0) for w in common_words])
	all_words_score = np.sum([weights.get(w, 0) for w in q1_words]) + np.sum([weights.get(w, 0) for w in q2_words]) - common_words_score
	
	return common_words_score/all_words_score

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

print word_match_share(df_train.iloc[5])
print weighted_word_match_share(df_train.iloc[5])
