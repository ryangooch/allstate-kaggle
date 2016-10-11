"""
Exploration of the features and data.

Ryan Gooch, Apr 2016
"""

import numpy as np
import pandas as pd

from sklearn import cross_validation, preprocessing, metrics

# also import data import and export functions
from dataio import getdata, writesub

from sklearn.feature_selection import VarianceThreshold

# Get the data in, skip header row
# train = np.genfromtxt('train.csv',delimiter=',',skip_header=1)
trainpath = 'data/train_sub.csv'
testpath = '/data/holdout.csv'

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
y = df_train['TARGET']
X = df_train.drop(['ID','TARGET'], axis=1)
print(X.shape)

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
print(X_new.shape)

X_IDs = df_test['ID']
X_test = df_test.drop(['ID'],axis=1)
X_test = model.transform(X_test)

"""
L1-SVM dropped features from 307 to 65. Pretty steep
"""

# Standardize, ignore numerical warning for now
X = preprocessing.scale(X_new)
X_test = preprocessing.scale(X_test)

# Random state for repeatability, split into training and validation sets
rs = 19683
X_train, X_val, y_train, y_val = \
		cross_validation.train_test_split(X, y, \
			test_size=0.25, random_state=rs)

import xgboost as xgb

### load data in do training
dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_val,label=y_val)
param = {'max_depth':8, 'eta':0.5, 'silent':1, 'objective':'binary:logistic',\
	'eval_metric':'auc'}
watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 3
bst = xgb.train(param, dtrain, num_round, watchlist)

print ('start testing prediction from first n trees')
### predict using first 1 tree
label = y_val
ypred1 = bst.predict(dtest, ntree_limit=1)
# by default, we predict using all the trees
ypred2 = bst.predict(dtest)
print ('error of ypred1=%f' % (np.sum((ypred1>0.5)!=label) /float(len(label))))
print ('error of ypred2=%f' % (np.sum((ypred2>0.5)!=label) /float(len(label))))

roc = metrics.roc_auc_score(y_val, ypred1)
print("ROC, ypred1: %.6f"%roc)
roc = metrics.roc_auc_score(y_val, ypred2)
print("ROC, ypred2: %.6f"%roc)
roc = metrics.roc_auc_score(y_val, np.mean([ypred1,ypred2],0))
print("ROC, averaged ypreds: %.6f"%roc)
cc = np.corrcoef([ypred1,ypred2])
print(cc)
"""
('ROC, ypred1: ', 0.7829)
('ROC, ypred2: ', 0.8008)

So, this is good. Highly correlated, so just use ypred2 if in ensemble
"""

"""
('ROC, ypred1: ', 0.7902)
('ROC, ypred2: ', 0.8103)
('ROC, ypreds: ', 0.8115) <----> Kaggle Test set ROC: 0.8121, val = consistent

for first 8 trees. Also these two are uncorrelated! Go ahead and make a 
submission because why not.
"""

 # Make predictions on test set
### load data in do training
dtrain = xgb.DMatrix(X,label=y)
dtest = xgb.DMatrix(X_test)
watchlist  = [(dtrain,'train')]
num_round = 3
bst = xgb.train(param, dtrain, num_round)

print ('start testing prediction from first n trees')
### predict using first 1 tree
ypred1 = bst.predict(dtest, ntree_limit=1)
# by default, we predict using all the trees
ypred2 = bst.predict(dtest)
ypred3 = np.genfromtxt('xgb1.03-22.ys.csv',delimiter=',')

ypred3 = ypred3[1:,1]
writesub(X_IDs, np.mean([ypred1,ypred2],0), sub = "XGB.auc-obj.2ensemble.2016.04.03.csv")
writesub(X_IDs, np.mean([ypred1,ypred2,ypred3],0), sub = "XGB.3round.3ensemble.2016.04.01.csv")
# 0.003 shy of improvement... getting better!