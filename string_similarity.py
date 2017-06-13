import pandas as pd
import numpy as np

import jellyfish

if __name__ == "__main__":
    df_train = pd.read_csv("./data/clean_train.csv")
    df_train.fillna(value="", inplace = True) # fill all nan data
    sample_sentence_list = df_train.question1[:]
    sample_sentence_list2 = df_train.question2[:]
    with open("train_out_edit_distance.csv", "wt") as fout:
        fout.write('test_id,is_duplicate\n')
        for i in range(len(sample_sentence_list)):
            sample_sentence = sample_sentence_list[i]
            sample_sentence2 = sample_sentence_list2[i]
            #print(sample_sentence, sample_sentence2)
            score = jellyfish.jaro_winkler(sample_sentence, sample_sentence2)
            #print(score)
            fout.write(str(i) +','+ str(score)+'\n')

            if i % 100000 == 0:
                progress = i/len(sample_sentence_list)*100
                print("{}% completed".format(round(progress, 1)))


