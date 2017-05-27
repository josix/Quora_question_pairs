import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#from memory_profiler import profile

from collections import defaultdict
import math

from Preprocessor import Preprocessor

def build_trains_question_tfidf(dataframe):
    question_list = dataframe["question1"].tolist() + dataframe["question2"].tolist()
    print("question_list_length:", len(question_list))
    vectorizer = CountVectorizer()
    word_count_matrix = vectorizer.fit_transform(question_list)
    #print(word_count_matrix._shape)
    ''' use for compute idf dictionary
    word = vectorizer.get_feature_names()
    bool_word_count_matrix = word_count_matrix.astype(bool).astype(int)
    col_sum_matrix = bool_word_count_matrix.sum(axis = 0)
    #print(type(col_sum_matrix), col_sum_matrix, col_sum_matrix.item(0))
    idf = defaultdict(int)
    total_doc = word_count_matrix.shape[0]
    for term_i in range(col_sum_matrix.shape[1]):
        idf[word[term_i]] = math.log10(total_doc/col_sum_matrix.item(term_i))
        #print(word[term_i],":", col_sum_matrix.item(term_i))

    #print(word)
    #print(word_count_matrix.toarray())
    '''

    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(word_count_matrix)

    return tfidf


def build_test_question_tfidf(question1, question2, corpus_word = None, idf = None):
    if corpus_word is None and idf is None:
        tfidf_vectorizer = TfidfVectorizer(preprocessor = Preprocessor)
        tfidf = tfidf_vectorizer.fit_transform([question1, question2])
        return tfidf
    if question1 is not "" and question2 is not "":
        test_question_list = [question1 , question2]
        tfidf = []
        for question in test_question_list:
            vectorizer = CountVectorizer()
            question_word_count_matrix = vectorizer.fit_transform([question]).toarray()
            word = vectorizer.get_feature_names()
            index = { value:i for i, value in enumerate(word)}
            question_tfidf = []
            for term_i in range(len(corpus_word)):
                if corpus_word[term_i] in set(word):
                    question_tfidf.append(idf[corpus_word[term_i]]*question_word_count_matrix[0][index[corpus_word[term_i]]])
                else:
                    question_tfidf.append(0)
            tfidf.append(question_tfidf)
        return tfidf
    else:
        return [[1], [1]]

if __name__ == "__main__":
    df_train = pd.read_csv("./test.csv")
    print("test_shape:(ori)", df_train.shape)
    print( df_train.dtypes)
    df_train.fillna(value="empty", inplace = True) # fill all nan data
    question_total = df_train.shape[0] # get length of row
    print("test_shape:(dropped)", df_train.shape)
    with open("test_out.csv", "wt") as fout:
        fout.write("test_id,is_duplicate\n")
        for index, row in df_train.iterrows():
            if index == 1346464:
                fout.write(str(index)+","+ str(1.0)+"\n")
                continue
            print(index, row["question1"], row["question2"])
            #print(build_test_question_tfidf(row["question1"], row["question2"]).toarray())
            vector_1, vector_2 = build_test_question_tfidf(row["question1"], row["question2"])
            #print(str(index)+","+ str(cosine_similarity(vector_1, vector_2)))
            fout.write(str(index)+","+ str(cosine_similarity(vector_1, vector_2)[0][0])+"\n")
#    tfidf = build_trains_question_tfidf(df_train)
#
#    del df_train
#    question1_tfidf = tfidf[:question_total]
#    question2_tfidf = tfidf[question_total:]
#    print(question1_tfidf.shape)
#    print(question2_tfidf.T.shape)
#    print(type(question1_tfidf))
#    print(question1_tfidf.dtype)
#    df_output = pd.DataFrame(columns = ["is_duplicate"], dtype=float)
#    for row in range(question1_tfidf.shape[0]):
#        df_output.loc[row] = [cosine_similarity(question1_tfidf[row], question2_tfidf[row])[0][0]]
#        print(row, cosine_similarity(question1_tfidf[row], question2_tfidf[row])[0][0])
#        #print(row, "1:", question1_tfidf[row])
#        #print(row, "2:", question2_tfidf[row])
#    print(df_output.head())
#    df_output.to_csv("out.csv", encoding='utf-8')

    #print(len(similarity))
    #print(average)
