#coding = utf-8

import os, sys 
import numpy  as np 
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

#---------load the package for classification start---------------------------
sys.path.append(r'D:/GoogleDriver/GFA-MachineLearning/ClassificationPackage/')
#sys.path.append(r'/home/yh527/project/ClassificationPackage/')
from datacorrelation   import data_correlation
from traintestsplit    import design_train_test
from modeloptimization import model_optimization 
#---------load the package for classification end----------------------------

scoring = 'roc_auc'

def readata(datafile = '../../finaldata_all.csv'):
    """ overview data and return clean data """

    data = pd.read_csv(datafile)
    shuffled = np.random.permutation(data.shape[0])
    data = data.reindex(index = shuffled)
    
    #-----------remove elements from features-----------
    # usedfeatures = [i for i in data.columns if len(i) > 2]
    # data = data[usedfeatures]
    # print ('After removing elements: ', data.shape)    
    
    #-----------select features want to use-------------
    # topfeatures = pd.read_csv('../machinelearning/topfeatures/featureimportance.csv').iloc[:, 0].tolist()
    # m = 10
    # data = data[topfeatures[:m] + ['label']]
    # print ('Used Features (' + str(m) + '):\n' + '\n'.join(topfeatures[:m]))
    # print ('Total Data: ', data.shape)

    return data

def train_classifiers(scoring):
    """
    data preparation and choose model to train then tune hyperparameters
    """

    #-----------------get train and test sets--------------------
    data = readata()
    train_set, test_set = design_train_test(data, test_size = 0.30)
    train_set.to_csv('traindata.csv', index = False)
    test_set.to_csv('testdata.csv', index = False)
    print ('Train Set Shape: ', train_set.shape)
    print ('Test Set Shape: ', test_set.shape)

    #-------------------data exploration-------------------------
    # corr_matrix = train_set.corr()
    # data_correlation(corr_matrix)

    #-------------------data preparation--------------------------
    X_train = train_set.drop('label', axis = 'columns')
    y_train = train_set['label'].copy()
    X_test  = test_set.drop('label', axis = 'columns')
    y_test  = test_set['label'].copy()

    #-------------------feature scaling---------------------------
    #only use transform for test dataset
    from sklearn.preprocessing import StandardScaler  #be normal distribution
    from sklearn.preprocessing import Imputer
    from sklearn.pipeline      import Pipeline

    num_pipeline = Pipeline([('imputer', Imputer(strategy = 'mean')),
                             ('std_scaler', StandardScaler())]) 

    X_train_scaled   = num_pipeline.fit_transform(X_train) #transform returns numpy array
    X_train_prepared = pd.DataFrame(X_train_scaled, columns = X_train.columns)
    X_test_scaled    = num_pipeline.transform(X_test)
    X_test_prepared  = pd.DataFrame(X_test_scaled, columns = X_test.columns)
    
    #---------------------try one algorithm----------------------------------
    #try_one(X_train_prepared, y_train)

    #---------------------Grid Search to tune hyperparameters----------------
    model_optimization(X_train_prepared, y_train, X_test_prepared, y_test, scoring, m = 0, n = 4, path = './')

train_classifiers(scoring)