import gensim
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)

df_train = pd.read_csv("./data/clean_test.csv")
df_train.fillna(value="empty", inplace = True) # fill all nan data
sample_sentence_list = df_train.question1[:]
sample_sentence_list2 = df_train.question2[:]

def compute_vector(sentence, sentence2):
    words = sentence.split()
    words2 = sentence2.split()

    i = 0
    vector = np.zeros((300,))
    for word in words:
        try:
            vector += model[word]
        except KeyError:
            vector += np.zeros((300,))
            continue
        i+=1

    #vector = vector / i

    i = 0
    vector2 = np.zeros((300,))
    for word in words2:
        try:
            vector2 += model[word]
        except KeyError:
            vector2 += np.zeros((300,))
            continue
        i += 1
    #vector2 = vector2 / i
    return vector.reshape(1, -1), vector2.reshape(1, -1)

with open("./score/test_out_word2vec_sum.csv", "wt") as fout:
    fout.write('test_id,is_duplicate\n')
    for i in range(len(sample_sentence_list)):
        sample_sentence = sample_sentence_list[i]
        sample_sentence2 = sample_sentence_list2[i]

        #print(sample_sentence)
        #print(sample_sentence2)
        v1, v2 = compute_vector(sample_sentence, sample_sentence2)
        #print(v1, v2)
        print(i, sample_sentence_list[i])
        print(i, sample_sentence_list2[i])
        try:
            fout.write(str(i) +','+ str(cosine_similarity(v1, v2)[0][0])+'\n')
        except ValueError:
            fout.write(str(i) +',0.0\n')

        if i % 100000 == 0:
            progress = i/len(sample_sentence_list)*100
            print("{}% completed".format(round(progress, 1)))

