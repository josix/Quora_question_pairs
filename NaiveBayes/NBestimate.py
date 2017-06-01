import pandas as pd
import numpy as np

'''
load training data
'''

file_loc_train   = "./train.csv"
file_loc_JM      = "./train_JM.csv"
file_loc_tfidf   = "./train_tfidf.csv"
file_loc_laplace = "./train_laplace.csv"
file_loc_key 	 = "./train_key.csv"

'''
build dataframe
'''

df_train 		= pd.read_csv(file_loc_train)
df_JM 			= pd.read_csv(file_loc_JM)
df_tfidf 		= pd.read_csv(file_loc_tfidf)
df_laplace 		= pd.read_csv(file_loc_laplace)
df_key 			= pd.read_csv(file_loc_key)

"""
test dataframe building
"""

print("________training data frame________")
print("train  ", df_train.head(1))
print("JM     ", df_JM.head(1))
print("tfidf  ", df_tfidf.head(1))
print("laplace", df_laplace.head(1))
print("key    ", df_key.head(1))
print("___________________________________\n\n")


"""
transform into lists
"""

ls_label 	 		= df_train['is_duplicate'].tolist()
ls_JMscore  		= df_JM['is_duplicate'].tolist()
ls_tfidfscore  		= df_tfidf['is_duplicate'].tolist()
ls_laplacescore  	= df_laplace['is_duplicate'].tolist()
ls_keyscore  		= df_key['is_duplicate'].tolist()
ls_traindata   		= [[ls_JMscore[i], ls_tfidfscore[i], ls_laplacescore[i], ls_keyscore[i]] for i in range(len(ls_label))]

"""
test lsits building
"""

print("________training data lists________")
print("label  ", len(ls_label), 		ls_label[:2])
print("JM     ", len(ls_JMscore),		ls_JMscore[:2])
print("tfidf  ", len(ls_tfidfscore), 	ls_tfidfscore[:2])
print("laplace", len(ls_laplacescore), ls_laplacescore[:2])
print("key    ", len(ls_keyscore), 	ls_keyscore[:2])
print("train  ", len(ls_traindata), 	ls_traindata[:2])
print("___________________________________\n\n")

"""
build GaussianNB model
"""

from sklearn.naive_bayes import GaussianNB

#turn list to array
arr_traindata 		= np.array(ls_traindata)
arr_label 			= np.array(ls_label)

GNBmodel = GaussianNB()
GNBmodel.fit(arr_traindata, arr_label)

"""
accuracy eval
"""

print("________evaluation in training data________")
from sklearn.metrics import accuracy_score
predict = GNBmodel.predict(ls_traindata)
print("predict:", predict[:10], "\ntruth:", ls_label[:10])
print("ACscore:", accuracy_score(ls_label, predict))

"""
log loss eval
"""

from sklearn.model_selection import cross_val_score
print("neg_log_loss:", cross_val_score(GNBmodel, arr_traindata, arr_label, scoring='neg_log_loss'))
print("___________________________________________\n\n")

########################
#	build test model   
########################
'''
load training data
'''

file_loc_JM      = "./test_JM.csv"
file_loc_tfidf   = "./test_tfidf.csv"
file_loc_laplace = "./test_laplace.csv"
file_loc_key 	 = "./test_key.csv"

'''
build dataframe
'''

df_JM 			= pd.read_csv(file_loc_JM)
df_tfidf 		= pd.read_csv(file_loc_tfidf)
df_laplace 		= pd.read_csv(file_loc_laplace)
df_key 			= pd.read_csv(file_loc_key)

"""
test dataframe building
"""

print("________testing data frame________")
print("JM     ", df_JM.head(1))
print("tfidf  ", df_tfidf.head(1))
print("laplace", df_laplace.head(1))
print("key    ", df_key.head(1))
print("___________________________________\n\n")

"""
transform into lists
"""

ls_JMscore  		= df_JM['is_duplicate'].tolist()
ls_tfidfscore  		= df_tfidf['is_duplicate'].tolist()
ls_laplacescore  	= df_laplace['is_duplicate'].tolist()
ls_keyscore  		= df_key['is_duplicate'].tolist()
ls_testdata   		= [[ls_JMscore[i], ls_tfidfscore[i], ls_laplacescore[i], ls_keyscore[i]] for i in range(len(ls_keyscore))]

"""
test lsits building
"""

print("________testing data lists________")
print("JM     ", len(ls_JMscore),		ls_JMscore[:2])
print("tfidf  ", len(ls_tfidfscore), 	ls_tfidfscore[:2])
print("laplace", len(ls_laplacescore), ls_laplacescore[:2])
print("key    ", len(ls_keyscore), 	ls_keyscore[:2])
print("test   ", len(ls_testdata), 	ls_testdata[:2])
print("___________________________________\n\n")

"""
build prediction list
"""

ls_predict_proba = GNBmodel.predict_proba(ls_testdata)
ls_predict = GNBmodel.predict(ls_testdata)
print("test_predict_proba:", ls_predict_proba[:10])
print("test_predict:", ls_predict[:10])

"""
output result
"""


print("\n\nOpening file...")
f_out = open('re_test.csv', 'w')
print("Start writing result file")
f_out.write("test_id,is_duplicate\n")
for i in range(len(ls_predict)):
	if(i%1000 == 0):
		print(i)
	f_out.write("%d,%f\n" % (i, ls_predict_proba[i][1]))

f_out.close()
print("File closed.")








