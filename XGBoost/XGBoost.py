import pandas as pd
import numpy as np



'''
load training data
'''

file_loc_train   = "../training_data/raw/train.csv"
file_loc_numcomp = "../FeatureAnalyzer/train_numcomp.csv"
file_loc_samebeg = "../FeatureAnalyzer/train_samebeg.csv"
file_loc_tfidf   = "../training_data/raw/train_tfidf.csv"
file_loc_key 	 = "../training_data/raw/train_key.csv"
file_loc_test 	 = "../training_data/raw/test.csv"

'''
reading files
'''

df_train 		= pd.read_csv(file_loc_train)
df_numcomp 		= pd.read_csv(file_loc_numcomp)
df_samebeg 		= pd.read_csv(file_loc_samebeg)
df_tfidf 		= pd.read_csv(file_loc_tfidf)
df_key 			= pd.read_csv(file_loc_key)
df_test 		= pd.read_csv(file_loc_test)

"""
test dataframe building
"""

print("________training data frame________")
print("train  ", df_train.head(1))
print("numcomp", df_numcomp.head(1))
print("tfidf  ", df_tfidf.head(1))
print("key    ", df_key.head(1))
print("___________________________________\n\n")

"""

"""
x_train 			= pd.DataFrame()

#ls_JMscore  		= df_JM['is_duplicate'].tolist()
#ls_tfidfscore  		= df_tfidf['is_duplicate'].tolist()
#ls_laplacescore  	= df_laplace['is_duplicate'].tolist()
#ls_keyscore  		= df_key['is_duplicate'].tolist()
#ls_traindata   		= [[ls_JMscore[i], ls_tfidfscore[i], ls_laplacescore[i], ls_keyscore[i]] for i in range(len(ls_label))]

x_train['TFIDF']	= df_tfidf['is_duplicate']
x_train['key']		= df_key['is_duplicate']
x_train['numcomp']  = df_numcomp['score']
x_train['samebeg']  = df_samebeg['score']
y_train 	 		= df_train['is_duplicate'].values


"""
test lsits building
"""

print("________training data lists________")
print("y_train", len(y_train), 		y_train[:2])
print("x_train\n", x_train.head())
print("___________________________________\n\n")

"""
Rebalanceing data
"""


pos_train = x_train[y_train == 1]
neg_train = x_train[y_train == 0]

p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -= 1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
x_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train


from sklearn.cross_validation import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

"""
XGBoost
"""

import xgboost as xgb
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

'''
load test data
'''

file_loc_tfidf   = "../training_data/raw/test_tfidf.csv"
file_loc_key 	 = "../training_data/raw/test_key.csv"
file_loc_numcomp = "../FeatureAnalyzer/test_numcomp.csv"
file_loc_samebeg = "../FeatureAnalyzer/test_samebeg.csv"

'''
build dataframe
'''
x_test = pd.DataFrame()
df_tfidf 		= pd.read_csv(file_loc_tfidf)
df_key 			= pd.read_csv(file_loc_key)
df_numcomp 		= pd.read_csv(file_loc_numcomp)
df_samebeg 		= pd.read_csv(file_loc_samebeg)

x_test['TFIDF'] = df_tfidf['is_duplicate']
x_test['key']	= df_key['is_duplicate']
x_test['numcomp']=df_numcomp['score']
x_test['samebeg']=df_samebeg['score']


d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)

f_out = open('xgb_result.csv', 'w')
f_out.write("test_id,is_duplicate\n")
for i in range(len(p_test)):
	if(i%100000 == 0):
		print("%.2f%% Writen." % (i/len(p_test)*100))
	f_out.write("%d,%f\n" % (i, p_test[i]))

f_out.close()

'''
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('simple_xgb.csv', index=False)
'''



