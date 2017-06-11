import pandas as pd
import numpy as np
import sys


#loading files into df
df_train =  pd.read_csv("../training_data/raw/train.csv")
df_test  =  pd.read_csv("../training_data/raw/test.csv")

df_train = df_train.fillna('empty')
df_test  = df_test.fillna('empty')


def comp(question1, question2):
	if(question1 == question2):
		return 1
	else:
		return 0


def pair_process_questions(score_list, pair_list, score_list_name, dataframe):
	'''transform questions and display progress'''
	for pair in pair_list:
		q1 = pair[0].split(' ')
		q2 = pair[1].split(' ')
		score_list.append(comp(q1[0], q2[0]))
		if len(score_list) % 100000 == 0:
			progress = len(score_list)/len(dataframe) * 100
			print("{} is {}% complete.".format(score_list_name, round(progress, 1)))

#train_output
train_out = "train_samebeg.csv"

train_score = []
pair_process_questions(train_score, zip(df_train['question1'].tolist(), df_train['question2'].tolist()), 'train_score', df_train)
if len(df_train) != len(train_score):
	print("Length didnt matched!\n")
	sys.exit()	


with open(train_out, "wt") as fout:
	fout.write('"test_id","score"\n')
	for i in range(len(train_score)):
		fout.write("%d,%f\n" % (i, train_score[i]))
		if i % 100000 == 0:
			progress = i/len(train_score)*100
			print("train >>> {}% written.".format(round(progress, 1)))

#test_output
test_out = "test_samebeg.csv"

test_score = []
pair_process_questions(test_score, zip(df_test['question1'].tolist(), df_test['question2'].tolist()), 'train_score', df_test)
if len(df_test) != len(test_score):
	print("Length didnt matched!\n")
	sys.exit()


with open(test_out, "wt") as fout:
	fout.write('"test_id","score"\n')
	for i in range(len(test_score)):
		fout.write("%d,%f\n" % (i, test_score[i]))
		if i % 100000 == 0:
			progress = i/len(test_score)*100
			print("test >>> {}% written.".format(round(progress, 1)))


