import pandas
import numpy
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve, auc


import sys
inPath = r''
sys.path.append(inPath)
from subfunction import *

train = pandas.read_excel(".\\filter_train.xlsx")
name = train.keys()
train = train.values
temp_count = train.shape[1]
label_train = list(train[:,(temp_count-1)])
X_train = train[:,1:(temp_count-1)]

name = name[1:(temp_count-1)]

test = pandas.read_excel(".\\filter_test.xlsx")
test = test.values
temp_count = test.shape[1]
label_test = list(test[:,(temp_count-1)])
X_test = test[:,1:(temp_count-1)]

NSGA_sel = pandas.read_excel(".\\NSGA_sel.xlsx",header = None)
NSGA_sel = NSGA_sel.values

r = NSGA_sel.shape[0]
c = NSGA_sel.shape[1]

auc_train = []
auc_test = []

for i in range(r):
    
    sel_feat = []
    temp = NSGA_sel[i,:]
    
    for j in range(c):
        if NSGA_sel[i,j]==1:
            sel_feat.append(j)

        temp_len = len(sel_feat)

    if temp_len >= 10:
        nn = 10
    else:
        nn = temp_len

    train_sel_set = X_train[:,sel_feat]
    test_sel_set = X_test[:,sel_feat]
    train_sel_mrmr = mRMR(train_sel_set,label_train,nn,1,1)
    train_sel_set = train_sel_set[:,train_sel_mrmr]
    test_sel_set = test_sel_set[:,train_sel_mrmr]

    b = Linear_check(train_sel_set,label_train,test_sel_set,label_test)
    auc_train.append(b[0][nn-1])
    auc_test.append(b[2][nn-1])

    print(b)
    print(i)
    print(train_sel_mrmr)

print(max(auc_train))
print(auc_train.index(max(auc_train)))
#print(auc_test[auc_train.index(max(auc_train))])
#print(max(auc_test))
#print(auc_test.index(max(auc_test)))


sel_feat = []
for j in range(c):
    if NSGA_sel[auc_train.index(max(auc_train)),j]==1:
        sel_feat.append(j)
sel_feat = numpy.array(sel_feat)
train_sel_set = X_train[:,sel_feat]
test_sel_set = X_test[:,sel_feat]
train_sel_mrmr = mRMR(train_sel_set,label_train,nn,1,1)

print(name[sel_feat[train_sel_mrmr]])
