import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jellyfish
#from memory_profiler import profile

from collections import defaultdict
import math

from Parser import Preprocessor

def build_question_tfidf_okapi(question1, question2):
    doclen = [len(question1),  len(question2)]
    avgdoclen = sum(doclen)/len(doclen)
    vectorizer = CountVectorizer(preprocessor = Preprocessor)
    word_count_matrix = vectorizer.fit_transform([question1, question2])
    #print(word_count_matrix._shape)
    word = vectorizer.get_feature_names()
    bool_word_count_matrix = word_count_matrix.astype(bool).astype(int)
    col_sum_matrix = bool_word_count_matrix.sum(axis = 0)
    #print(type(col_sum_matrix), col_sum_matrix, col_sum_matrix.item(0))
    idf = defaultdict(lambda:1)
    total_doc = word_count_matrix.shape[0]
    for term_i in range(col_sum_matrix.shape[1]):
        idf[word[term_i]] = math.log10((total_doc+1)/(1+col_sum_matrix.item(term_i))) + 1
        #print(word[term_i],":", col_sum_matrix.item(term_i))

    tfidf_matrix = []
    for doc_index, doc_vector in enumerate( word_count_matrix.toarray()):
        tfidf_vector = []
        #print("doc_index:", doc_index,"doc_vector:",  doc_vector)
        for index, tf in enumerate(doc_vector):
            #print(index, tf)
            tf_okapi = tf / (tf + 0.5 + 1.5 * doclen[doc_index] / avgdoclen)
            #print(word[index], idf[word[index]])
            tfidf_vector.append(tf_okapi*idf[word[index]])
        tfidf_matrix.append(np.array(tfidf_vector).reshape(1, -1))

    #print(word_count_matrix.toarray())

    return tfidf_matrix


def build_question_tfidf(question1, question2):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform([question1, question2])
    return tfidf

if __name__ == "__main__":
    df_train = pd.read_csv("./data/clean_train.csv")
    print("test_shape:(ori)", df_train.shape)
    print( df_train.dtypes)
    df_train.fillna(value="empty", inplace = True) # fill all nan data
    print("test_shape:(dropped)", df_train.shape)
    error_out = open("value_error_string_train.csv", "wt")
    with open("train_out_vsm.csv", "wt") as fout:
        fout.write("test_id,is_duplicate\n")
        for index, row in df_train.iterrows():
            if index == 1346464:
                fout.write(str(index)+","+ str(1.0)+"\n")
                continue
            #print(index, row["question1"], row["question2"])
            #print(build_test_question_tfidf(row["question1"], row["question2"]).toarray())
            try:
                vector_1, vector_2 = build_question_tfidf(row["question1"], row["question2"]) # raw tf weighting
            except ValueError:
                similarity = jellyfish.jaro_winkler(row["question1"], row["question2"])
                error_out.write(str(index)+","+row["question1"]+","+row["question2"]+","+str(similarity)+"\n")
                fout.write(str(index)+","+ str(similarity)+"\n")
                continue
            #vector_1, vector_2 = build_question_tfidf_okapi(row["question1"] , row["question2"]) # okapi tf weighting
            #print(str(index)+","+ str(cosine_similarity(vector_1, vector_2)))
            fout.write(str(index)+","+ str(cosine_similarity(vector_1, vector_2)[0][0])+"\n")
    error_out.close()
