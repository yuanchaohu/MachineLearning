#coding = utf-8

import os, sys 
import numpy  as np 
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

#---------load the package for classification start---------------------------
#sys.path.append(r'D:/GoogleDriver/GFA-MachineLearning/ClassificationPackage/')
sys.path.append(r'/home/yh527/project/ClassificationPackage/')
from datacorrelation   import data_correlation
from modeloptimization import model_optimization 
#---------load the package for classification end----------------------------

scoring = 'roc_auc'

def get_test_alloy(testn = 10, file1 = '../../usedelements.dat', datafile = '../../finaldata_all3.csv'):
    """ get the dataset of one alloy to be tested """

    data = pd.read_csv(datafile)
    with open(file1, 'r') as f:
        usedelements = [i.strip('\n') for i in f.readlines()[2:]]

    import random
    random.shuffle(usedelements)    
    conditions = (data[usedelements[:testn]].any(axis = 'columns'))
    data[conditions].to_csv('testdata.csv', index = False)
    data[~conditions].to_csv('traindata.csv', index = False)

    with open('test_elements.dat', 'w') as f:
        f.write('\n'.join(usedelements[:testn]))

    with open('test_elements_all.dat', 'w') as f:
        f.write('\n'.join([i for i in data[conditions].columns if len(i) <= 2]))

get_test_alloy()

def choose_fraction(datafile = '../finaldata.csv', alloysfile = '../alloys.csv', frac = 0.20):
    """ Choose a fraction of data from each system for training """

    data   = pd.read_csv(datafile)
    alloys = pd.read_csv(alloysfile, header = None)
    traindata = []
    testdata  = []
    for i in alloys.index:
        testalloy  = [j for j in alloys.loc[i] if str(j) != 'nan']
        onesystem  = data[(data[testalloy].all(axis = 'columns'))]
        nums       = onesystem.shape[0]
        onesystem.reset_index(drop = True, inplace = True)    
        shuffled   = np.random.permutation(nums)
        onesystem  = onesystem.reindex(index = shuffled)
        traindata.append(onesystem[:int(nums * frac)])
        testdata.append(onesystem[int(nums * frac):])

    pd.concat(traindata, axis = 'index').to_csv('traindata.csv', index = False)
    pd.concat(testdata, axis = 'index').to_csv('testdata.csv', index = False)

#choose_fraction()

def readata(datafile = './traindata.csv'):
    """ overview data and return clean data """

    data = pd.read_csv(datafile)
    shuffled = np.random.permutation(data.shape[0])
    data = data.reindex(index = shuffled)

    #-----------remove elements from features-----------
    usedfeatures = [i for i in data.columns if len(i) > 2]
    data = data[usedfeatures]
    print ('After removing elements: ', data.shape)    

    #------only use top features--------------
    # topfeatures = pd.read_csv('../machinelearning/topfeatures/featureimportance.csv').iloc[:, 0].tolist()
    # m = 10
    # data = data[topfeatures[:m] + ['label']]
    # print ('Used Features (' + str(m) + '):\n' + '\n'.join(topfeatures[:m]))
    # print (data.shape)

    return data

def train_classifiers(scoring):
    """
    data preparation and choose model to train then tune hyperparameters
    """

    #-----------------get train and test set--------------------
    train_set = readata(datafile = 'traindata.csv')
    test_set  = readata(datafile = 'testdata.csv')
    test_set  = test_set.reindex(columns = train_set.columns)
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
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.pipeline      import Pipeline

    X_train_num = X_train.drop('method', axis = 'columns')
    X_train_cat = X_train['method']
    X_test_num  = X_test.drop('method', axis = 'columns')
    X_test_cat  = X_test['method']

    num_pipeline = Pipeline([('imputer', Imputer(strategy = 'mean')),
                             ('std_scaler', StandardScaler())]) 
    encoder      = LabelBinarizer()

    X_train_scaled_num = num_pipeline.fit_transform(X_train_num) #transform returns numpy array
    X_train_scaled_cat = encoder.fit_transform(X_train_cat)
    X_train_prepared   = np.c_[X_train_scaled_num, X_train_scaled_cat]
    X_train_prepared   = pd.DataFrame(X_train_prepared, columns = X_train_num.columns.tolist() + list(encoder.classes_))
    print (X_train_prepared.shape)

    X_test_scaled_num  = num_pipeline.transform(X_test_num) #transform returns numpy array
    X_test_scaled_cat  = encoder.transform(X_test_cat)
    X_test_prepared    = np.c_[X_test_scaled_num, X_test_scaled_cat]
    X_test_prepared    = pd.DataFrame(X_test_prepared, columns = X_test_num.columns.tolist() + list(encoder.classes_))
    print (X_test_prepared.shape)

    #---------------------try one algorithm----------------------------------
     #try_one(X_train_prepared, y_train)

    #---------------------Grid Search to tune hyperparameters----------------
    model_optimization(X_train_prepared, y_train, X_test_prepared, y_test, scoring, m = 0, n = 4, path = './')
    
train_classifiers(scoring)
