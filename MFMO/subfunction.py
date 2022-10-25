import numpy
from minepy import MINE
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

mine = MINE(alpha=0.6, c=15, est="mic_approx")


def feature_to_set_MIC(X,y):
    X = numpy.array(X)
    count = X.shape[1]
    MI = numpy.zeros(count)
    for i in range(count):
        mine.compute_score(X[:,i],y)
        MI[i] = mine.mic()
        #print(MI[i])
    return numpy.mean(MI)


def mRMR(X,y,n,lamda,sita):
    #k = 0
    X = numpy.array(X)
    feature_num = X.shape[1]
    index_D = []
    index_R = list(range(feature_num))
    
    for i in range(n):
        
        mRMR_value = []
        mi_D = numpy.zeros(feature_num)
        mi_R = numpy.zeros(feature_num)

        if index_D == []:
            
            for j in index_R:
                mine.compute_score(X[:,j],y)
                mRMR_value.append(mine.mic())
                
            indices = mRMR_value.index(max(mRMR_value))
            
            index_D.append(indices)
            index_R.remove(indices)
            
            continue

        for j in index_R:
            mine.compute_score(X[:,j],y)
            mi_D[j] = mine.mic()
            mi_R[j] = feature_to_set_MIC(X[:,index_D],X[:,j])
            mRMR_value.append( lamda * mi_D[j] - sita * mi_R[j])
        
        indices = mRMR_value.index(max(mRMR_value))

        index_D.append(index_R[indices])
        index_R.remove(index_R[indices])
        #k = k+1
        #print(k)

    return index_D

def feature_to_set_MI(X,y):
    X = numpy.array(X)
    count = X.shape[1]
    MI = numpy.zeros(count)
    for i in range(count):
        MI[i] = metrics.normalized_mutual_info_score(X[:,i],y)
        #print(MI[i])
    return numpy.mean(MI)


def mi_mRMR(X,y,n,lamda,sita):
    #k = 0
    X = numpy.array(X)
    feature_num = X.shape[1]
    index_D = []
    index_R = list(range(feature_num))
    
    for i in range(n):
        
        mRMR_value = []
        mi_D = numpy.zeros(feature_num)
        mi_R = numpy.zeros(feature_num)

        if index_D == []:
            
            for j in index_R:
                mRMR_value.append(metrics.normalized_mutual_info_score(X[:,j],y))
                
            indices = mRMR_value.index(max(mRMR_value))
            
            index_D.append(indices)
            index_R.remove(indices)
            
            continue

        for j in index_R:

            mi_D[j] = metrics.normalized_mutual_info_score(X[:,j],y)
            mi_R[j] = feature_to_set_MI(X[:,index_D],X[:,j])
            mRMR_value.append( lamda * mi_D[j] - sita * mi_R[j])
        
        indices = mRMR_value.index(max(mRMR_value))

        index_D.append(index_R[indices])
        index_R.remove(index_R[indices])
        #k = k+1
        #print(k)

    return index_D


def Linear_auc(X_train,y_train,X_test,y_test):

    Linear_reg = LinearRegression().fit(X_train,y_train)
            
    temp1 = Linear_reg.predict(X_train)
    temp2 = Linear_reg.predict(X_test)
    #temp1 = regr.predict(X_test)

    fpr, tpr, thresholds  =  roc_curve(y_train, temp1, drop_intermediate=False)
    Youden = list(tpr - fpr)
    temp_index = Youden.index(max(Youden))
    train_auc = auc(fpr, tpr)

    fpr, tpr, thresholds  =  roc_curve(y_test, temp2, drop_intermediate=False)
    Youden = list(tpr - fpr)
    temp_index = Youden.index(max(Youden))
    test_auc = auc(fpr, tpr)

    return train_auc,test_auc


def Linear_check(X_train,y_train,X_test,y_test):

    X_train = numpy.array(X_train)
    feature_num = X_train.shape[1]

    AUC_train_linear = []
    AUC_test_linear = []

    for i in range(feature_num):

        if i == 0:
            temp_train = X_train[:,0]
            temp_train = temp_train[:,numpy.newaxis]
            temp_test = X_test[:,0]
            temp_test = temp_test[:,numpy.newaxis]
            temp_auc = Linear_auc(temp_train,y_train,temp_test,y_test)
            AUC_train_linear.append(temp_auc[0])
            AUC_test_linear.append(temp_auc[1])
            continue

        # 
        temp_index = range(i+1)
        temp_train = X_train[:,temp_index]
        temp_test = X_test[:,temp_index]
        temp_auc = Linear_auc(temp_train,y_train,temp_test,y_test)
        AUC_train_linear.append(temp_auc[0])
        AUC_test_linear.append(temp_auc[1])

        '''if i >= 1:
            if AUC_linear[i-1] >AUC_linear[i] :
                return AUC_linear,i-1
                break'''
    train_index = AUC_train_linear.index(max(AUC_train_linear))
    test_index = AUC_test_linear.index(max(AUC_test_linear))
    
    return AUC_train_linear,train_index,AUC_test_linear,test_index

