import pandas
import shap
import numpy
import scipy.stats
from minepy import MINE
mine = MINE(alpha=0.6, c=15, est="mic_approx")
shap.initjs()
import warnings
warnings.filterwarnings('ignore')


from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from scipy.stats import spearmanr

import sys
inPath = r''
sys.path.append(inPath)
from subfunction import *


### load data for real test dataset
train_dataframe = pandas.read_excel("train.xlsx")
train = train_dataframe.values
name = train_dataframe.keys()
temp_count = train.shape[1]
label_train = list(train[:,(temp_count-1)])
X_name = name[1:(temp_count-1)]
X_train = train[:,1:(temp_count-1)]
sample_count1 = X_train.shape[0]

test_BraTS = pandas.read_excel("test_BraTS.xlsx")
test_BraTS  = test_BraTS[name]
test = test_BraTS.values
temp_count = test.shape[1]
label_test = list(test[:,(temp_count-1)])
X_test = test[:,1:(temp_count-1)]
sample_count2 = X_train.shape[0]


### load data for simulated test dataset
train_dataframe = pandas.read_excel("train.xlsx")
train = train_dataframe.values
name = train_dataframe.keys()
temp_count = train.shape[1]
label_train = list(train[:,(temp_count-1)])
X_name = name[1:(temp_count-1)]
X_train = train[:,1:(temp_count-1)]
sample_count1 = X_train.shape[0]

test_lvyGAP = pandas.read_excel("test_lvyGAP.xlsx")
test_lvyGAP = test_lvyGAP[name]
test = test_lvyGAP.values
temp_count = test.shape[1]
label_test = list(test[:,(temp_count-1)])
X_test = test[:,1:(temp_count-1)]


test_LGG_10 = pandas.read_excel("test_LGG_10.xlsx")
test_LGG_10 = test_LGG_10[name]
test10 = test_LGG_10.values
temp_count = test10.shape[1]
label_test_10 = list(test10[:,(temp_count-1)])
X_test_10 = test10[:,1:(temp_count-1)]


### apply SMOTE to a new simulated test dataset
from imblearn.over_sampling import SMOTE
X_test = numpy.vstack((X_test, X_test_10))
label_test = numpy.append(label_test,label_test_10)
oversampler=SMOTE()
X_test,label_test = oversampler.fit_resample(X_test ,label_test)
sample_count2 = X_test.shape[0]


### Cho HH method

train_sel_mrmr = mi_mRMR(X_train,label_train,5,1,1)
train_sel_set = X_train[:,train_sel_mrmr]
test_sel_set = X_test[:,train_sel_mrmr]
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier()
RF_model.fit(train_sel_set,label_train)
pre_train = RF_model.predict(train_sel_set)
pre_test = RF_model.predict(test_sel_set)


### Gao M method

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
norm_train = scaler.transform(X_train)
norm_test = scaler.transform(X_test)

from sklearn.feature_selection import SelectKBest, chi2
cc = SelectKBest(chi2, k=15)
cc.fit(norm_train , label_train)
dd = cc.get_support()

test_sel_set = norm_test[:,dd]
train_sel_set = norm_train[:,dd]

from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier()
RF_model.fit(train_sel_set,label_train)
pre_train = RF_model.predict(train_sel_set)
pre_test = RF_model.predict(test_sel_set)

### Cui G method

sel_one = []
for i in range(X_train.shape[1]):
    aa = X_train[:,i]
    HGG = list(aa[0:102])
    LGG = list(aa[102:])
    t, pval = scipy.stats.ttest_ind(HGG,LGG)
    if pval<0.05:
        sel_one.append(i)
train_sel_set = X_train[:,sel_one]
test_sel_set = X_test[:,sel_one]
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_sel_set,label_train)
sel_two = []
for i in range(len(lasso.coef_)):
    if lasso.coef_[i] != 0:
        sel_two.append(i)
train_sel_set = train_sel_set[:,sel_two]
test_sel_set = test_sel_set[:,sel_two]
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier()
RF_model.fit(train_sel_set,label_train)
pre_train = RF_model.predict(train_sel_set)
pre_test = RF_model.predict(test_sel_set)

from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier()
RF_model.fit(train_sel_set,label_train)
pre_train = RF_model.predict(train_sel_set)
pre_test = RF_model.predict(test_sel_set)

### Kha Q H method

for i in range(len(label_train)):
    if label_train[i] == -1:
        label_train[i] = 0
for i in range(len(label_test)):
    if label_test[i] == -1:
        label_test[i] = 0

sel_one = []
for i in range(X_train.shape[1]):
    aa = X_train[:,i]
    scc = scipy.stats.spearmanr(aa,label_train)[0]
    if scc>0.8:
        sel_one.append(i)
train_sel_set = X_train[:,sel_one]
test_sel_set = X_test[:,sel_one]

sel_two = []
import xgboost as xgb
xg = xgb.XGBClassifier(booster='gbtree', colsample_bytree=0.4, gamma=1,
              learning_rate=0.9, max_depth=4, min_child_weight=10, n_estimators=700, 
              subsample=0.8)
xg.fit(train_sel_set, label_train)
explainer = shap.TreeExplainer(xg)
shap_values = explainer.shap_values(train_sel_set)
importances = numpy.abs(shap_values).mean(0)
indices_2 = numpy.argsort(importances)[::-1]
sel_two = list(indices_2[0:7])
train_sel_set = train_sel_set[:,sel_two]
test_sel_set = test_sel_set[:,sel_two]

xg = xgb.XGBClassifier(booster='gbtree', colsample_bytree=0.4, gamma=1,
              learning_rate=0.9, max_depth=4, min_child_weight=10, n_estimators=700, 
              subsample=0.8)
xg.fit(train_sel_set, label_train)
pre_train = xg.predict(train_sel_set)
pre_test = xg.predict(test_sel_set)


### compute result

acc_train = 0
acc_test = 0

fpr, tpr, thresholds  =  roc_curve(label_train, pre_train, drop_intermediate=False)
Youden = list(tpr - fpr)
temp_index = Youden.index(max(Youden))
train_auc = auc(fpr, tpr)
train_tpr = tpr[temp_index]
train_fpr = fpr[temp_index]
for i in range(sample_count1):
    if (pre_train[i] >= thresholds[temp_index]) & (i < 102):
        acc_train += 1
    if (pre_train[i] < thresholds[temp_index]) & (i >= 102) :
        acc_train += 1
train_acc = acc_train/sample_count1

tr= []
tr.append(train_auc)
tr.append(train_acc)
tr.append(1-train_fpr)
tr.append(train_tpr)

fpr2, tpr2, thresholds2  =  roc_curve(label_test, pre_test, drop_intermediate=False)
Youden2 = list(tpr2 - fpr2)
temp_index2 = Youden2.index(max(Youden2))
test_auc = auc(fpr2, tpr2)
test_tpr = tpr2[temp_index2]
test_fpr = fpr2[temp_index2]
for i in range(sample_count2):
    if (pre_test[i] >= thresholds[temp_index]) & (i < 34):
        acc_test += 1
    if (pre_test[i] < thresholds[temp_index]) & (i >= 34):
        acc_test += 1
test_acc = acc_test/sample_count2

te = []
te.append(test_auc)
te.append(test_acc)
te.append(1-test_fpr)
te.append(test_tpr)
