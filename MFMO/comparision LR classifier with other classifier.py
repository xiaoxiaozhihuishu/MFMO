import pandas
import numpy
import time
from sklearn.metrics import roc_curve, auc


### load data
train = pandas.read_excel("train_biomarker.xlsx")
train = train.values
temp_count = train.shape[1]
sample_count1 = train.shape[0]
label_train = list(train[:,(temp_count-1)])
X_train = train[:,1:(temp_count-1)]

test = pandas.read_excel("test_biomarker.xlsx")
test = test.values
temp_count = test.shape[1]
sample_count2 = test.shape[0]
label_test = list(test[:,(temp_count-1)])
X_test = test[:,1:(temp_count-1)]


### LR
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(X_train,label_train)
pre_train = regr.predict(X_train)
pre_test = regr.predict(X_test)

### SVM
from sklearn.svm import SVC
regr = make_pipeline(StandardScaler(), SVC(gamma='auto'))
regr.fit(X_train,label_train)
pre_train = regr.predict(X_train)
pre_test = regr.predict(X_test)

### RF
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestClassifier()
regr.fit(X_train,label_train)
pre_train = regr.predict(X_train)
pre_test = regr.predict(X_test)

### XGBoost
import xgboost as xgb
regr = xgb.XGBRegressor()
regr.fit(X_train,label_train)
pre_train = regr.predict(X_train)
pre_test = regr.predict(X_test)

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
    if (pre_train[i] >= thresholds[temp_index]) & (i<102):
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
    if (pre_test[i] >= thresholds[temp_index]) & (i<20):
        acc_test += 1
    if (pre_test[i] < thresholds[temp_index]) & (i>=20):
        acc_test += 1
test_acc = acc_test/sample_count2

te = []
te.append(test_auc)
te.append(test_acc)
te.append(1-test_fpr)
te.append(test_tpr)
