import pandas
import shap
import numpy
from minepy import MINE
shap.initjs()
import warnings
warnings.filterwarnings('ignore')

from sklearn import linear_model
from sklearn.linear_model import LinearRegression

import sys
inPath = r''
sys.path.append(inPath)
from subfunction import *

top5filter_train = pandas.read_excel("top5filter_train.xlsx")
fea_name = top5filter_train.keys()
train = top5filter_train.values
train_sample_count = train.shape[0]
temp_count = train.shape[1]
label_train = list(train[:,(temp_count-1)])
X_train = train[:,1:(temp_count-1)]

name = fea_name[1:(temp_count-1)]

ddd = top5filter_train[name]
dl = top5filter_train["label"]

top5filter_test = pandas.read_excel("top5filter_test.xlsx")
test = top5filter_test.values
test_sample_count = test.shape[0]
temp_count = test.shape[1]
label_test = list(test[:,(temp_count-1)])
X_test = test[:,1:(temp_count-1)]


ggg = top5filter_test[name]
gl = top5filter_test["label"]

indices_sort = []

for i in range(5):
    c = range(i*10,(i+1)*10,1)
    temp_X_train = X_train[:,c]
    X_test_2 = X_test[:,c]
    Linear_reg = LinearRegression().fit(temp_X_train,label_train)
    explainer = shap.LinearExplainer(Linear_reg,temp_X_train,feature_dependence="independent")
    shap_values = explainer.shap_values(temp_X_train)
    importances = numpy.abs(shap_values).mean(0)
    indices = numpy.argsort(importances)[::-1]
    print(indices)
    indices_sort = indices_sort + list(numpy.array(indices)[[0,1,2]]+10*(i))

X_train_2 = X_train[:,indices_sort]
X_test_2 = X_test[:,indices_sort]
a15_name = name[indices_sort]

d = Linear_check_wu(X_train_2,label_train,X_test_2,label_test)

Linear_reg = LinearRegression().fit(X_train_2,label_train)
explainer = shap.LinearExplainer(Linear_reg,X_train_2,feature_dependence="independent")
shap_values = explainer.shap_values(X_train_2)
importances = numpy.abs(shap_values).mean(0)
indices_2 = numpy.argsort(importances)[::-1]

indices_10 = indices_2[0:10]

X_train_3 = X_train_2[:,indices_10]
X_test_3 = X_test_2[:,indices_10]
b10_name = a15_name[indices_10]

result = Linear_check(X_train_3,label_train,X_test_3,label_test)

print(result)
print(b10_name)
