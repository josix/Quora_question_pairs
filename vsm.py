import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def build_trains_question_tfidf():
    df_train = pd.read_csv("./train.csv")
    trains_question_list = df_train["question1"].tolist() + df_train["question2"].tolist()
    index = 0
    for i in trains_question_list:
        if i is np.nan:
            del trains_question_list[index]
        index += 1
    vectorizer = CountVectorizer()
    word_count_matrix = vectorizer.fit_transform(trains_question_list)
    word = vectorizer.get_feature_names()
    #print(word)
    #print(word_count_matrix.toarray())
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(word_count_matrix)
    #print(tfidf.toarray())
    return tfidf.toarray(), word

if __name__ == "__main__":
    tfidf, word = build_trains_question_tfidf()
    for i in range(len(tfidf)):
        for j in range(len(word)):
            if tfidf[i][j] > 0:
                print(word[j], tfidf[i][j])


