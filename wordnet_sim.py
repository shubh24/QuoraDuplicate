import pandas as pd

from nltk.corpus import wordnet as wn

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

df_train = pd.read_csv('train.csv')

def get_terms(sentence):

    return [i for i in sentence.lower().split() if i not in stop]

def iterate(df):

    for index, row in df.iterrows():
        print index, row["question1"], row["question2"]

        terms1 = get_terms(row["question1"])
        terms2 = get_terms(row["question2"])
        print terms1, terms2

        sims = []

        for word1 in terms1:
            word1_sim = []

            for word2 in terms2:

                try:
                    syn1 = wn.synsets(word1)[0]
                    syn2 = wn.synsets(word2)[0]
                except:
                    continue

                word_similarity = syn1.wup_similarity(syn2)
                word1_sim.append(word_similarity)
            
            sims.append(word1_sim)

        print sims

if __name__ == '__main__':
    iterate(df_train)
