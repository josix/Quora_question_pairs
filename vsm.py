import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix

from collections import defaultdict
import math

from Similarity import cosine

def build_trains_question_tfidf(dataframe):
    question_list = dataframe["question1"].tolist() + dataframe["question2"].tolist()
    print("question_list_length:", len(question_list))
    vectorizer = CountVectorizer()
    word_count_matrix = vectorizer.fit_transform(question_list)
    #print(word_count_matrix._shape)
    '''
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

    return tfidf.toarray()

def build_test_question_tfidf(question1, question2, corpus_word, idf):
    #print("corpus_word: ", corpus_word)
    if question1 is not None and question2 is not None:
        test_question_list = [question1 , question2]
        tfidf = []
        for question in test_question_list:
            vectorizer = CountVectorizer()
            question_word_count_matrix = vectorizer.fit_transform([question]).toarray()
            word = vectorizer.get_feature_names()
            #print(set(word))
            #print(question_word_count_matrix)
            index = { value:i for i, value in enumerate(word)}
            #print(index)
            question_tfidf = []
            for term_i in range(len(corpus_word)):
                if corpus_word[term_i] in set(word):
                    #print( "in_word:", corpus_word[term_i])
                    #print(index[corpus_word[term_i]])
                    #print("question_array:", question_word_count_matrix[0])
                    #print("first:", * question_word_count_matrix[0][index[corpus_word[term_i]]])
                    question_tfidf.append(idf[corpus_word[term_i]]*question_word_count_matrix[0][index[corpus_word[term_i]]])
                else:
                    #print("not_in:", corpus_word[term_i])
                    question_tfidf.append(0)
            tfidf.append(question_tfidf)
        #print(len(tfidf[0]))
        #print(len(tfidf[1]))
        return tfidf
    else:
        return None

if __name__ == "__main__":
    df_train = pd.read_csv("./train.csv")
    print("test_shape:(ori)", df_train.shape)
    df_train = df_train.dropna(axis = 0, how = "any")
    question_total = df_train.shape[0]
    print("test_shape:(dropped)", df_train.shape)
    tfidf = build_trains_question_tfidf(df_train)
    question1_tfidf = tfidf[:question_total]
    question2_tfidf = tfidf[question_total:]
    print(question1_tfidf.shape)
    print(question2_tfidf.T.shape)
    print(type(question1_tfidf))
    print(csr_matrix(question1_tfidf).multiply(csr_matrix(question2_tfidf).T))
    #similarity = defaultdict(int)
    #for i in range(len(tfidf)-1, question_total-1, -1):
        #print(i, i-question_total)
        #similarity[i - question_total] = cosine(tfidf[i], tfidf[i - question_total])

