import pandas as pd

from nltk.corpus import wordnet as wn

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

import math

df_train = pd.read_csv('train.csv')

def get_terms(sentence):

    return [i for i in sentence.lower().split() if i not in stop]


def iterate(df):

    correct_count = 0
    wrong_count = 0
    logloss = 0

    for index, row in df.iterrows():
        
        res = row["is_duplicate"]
        terms1 = get_terms(row["question1"])
        terms2 = get_terms(row["question2"])

        sims = []

        for word1 in terms1:
            word1_sim = []

            try:
                syn1 = wn.synsets(word1)[0]
            except:
                sims.append([0 for i in range(0, len(terms2))])
                continue


            for word2 in terms2:

                try:
                    syn2 = wn.synsets(word2)[0]
                except:
                    word1_sim.append(0)
                    continue

                word_similarity = syn1.wup_similarity(syn2)
                word1_sim.append(word_similarity)
            
            sims.append(word1_sim)

        # print sims

        word1_score = 0

        for i in range(0, len(terms1), 1):
            try:
                word1_score += max(sims[i])
            except:
                continue
        word1_score /= len(terms1)

        word2_score = 0

        for i in range(0, len(terms2), 1):
            try:
                word2_score += max([j[i] for j in sims])
            except:
                continue
        word2_score /= len(terms2)

        pair_score = (word1_score + word2_score)/2

        if res == 1:
            logloss += math.log(pair_score)

        if (pair_score > 0.5):
            pred = 1
        else:
            pred = 0

        if pred == res:
            correct_count += 1
        else:
            wrong_count += 1

        if index%100 == 0:
            print correct_count, wrong_count
            print logloss/(correct_count + wrong_count)

if __name__ == '__main__':
    iterate(df_train)
