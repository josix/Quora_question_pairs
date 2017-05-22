import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix

from collections import defaultdict
import math

def build_trains_question_tfidf():
    df_train = pd.read_csv("./train.csv")
    trains_question_list = df_train["question1"].tolist() + df_train["question2"].tolist()
    #print(len(trains_question_list))
    index = 0
    for i in trains_question_list:
        if i is np.nan:
            del trains_question_list[index]
        index += 1
    vectorizer = CountVectorizer()
    word_count_matrix = vectorizer.fit_transform(trains_question_list)
    #print(word_count_matrix._shape)
    word = vectorizer.get_feature_names()
    bool_word_count_matrix = word_count_matrix.astype(bool).astype(int)
    col_sum_matrix = bool_word_count_matrix.sum(axis = 0)
    #print(type(col_sum_matrix), col_sum_matrix, col_sum_matrix.item(0))
    idf = defaultdict(int)
    total_doc = word_count_matrix.shape[0]
    for term_i in range(col_sum_matrix.shape[1]):
        #print(word[term_i],":", col_sum_matrix.item(term_i))
        idf[word[term_i]] = math.log10(total_doc/col_sum_matrix.item(term_i))

    #print(word)
    #print(word_count_matrix.toarray())
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(word_count_matrix)
    #print(tfidf.toarray())
    return tfidf.toarray(), word, idf

def build_test_question_tfidf(question1, question2, idf):
    if question1 is not None and question2 is not None:
        test_question_list = [question1 , question2]
        tfidf = []
        for question in test_question_list:
            vectorizer = CountVectorizer()
            question_word_count_matrix = vectorizer.fit_transform([question]).toarray()
            word = vectorizer.get_feature_names()
            print(word)
            print(question_word_count_matrix)
            question_tfidf = defaultdict(int)
            for term_i in range(len(word)):
                print(question_word_count_matrix[0][term_i], idf[word[term_i]])
                question_tfidf[word[term_i]] = idf[word[term_i]]*question_word_count_matrix[0][term_i]
            tfidf.append(question_tfidf)
        return tfidf
    else:
        return None

if __name__ == "__main__":
    corpus_tfidf, word, idf = build_trains_question_tfidf()
    #for i in range(len(corpus_tfidf)):
        #for j in range(len(word)):
            #if corpus_tfidf[i][j] > 0:
                #print(word[j], corpus_tfidf[i][j])
    #print(idf)
    question1 = "How does the Surface Pro himself 4 compare with iPad Pro?"
    question2 = "Why did Microsoft choose core m3 and not core i3 home Surface Pro 4?"
    question_tfidf = build_test_question_tfidf(question1, question2, idf)



