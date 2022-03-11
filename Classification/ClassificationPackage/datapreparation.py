#coding = utf-8

import pandas as pd 
import numpy as np 

def train_test_preparation(train_set, test_set):

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

    if 'method' in X_train.columns:
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

        X_test_scaled_num  = num_pipeline.transform(X_test_num) #transform returns numpy array
        X_test_scaled_cat  = encoder.transform(X_test_cat)
        X_test_prepared    = np.c_[X_test_scaled_num, X_test_scaled_cat]
        X_test_prepared    = pd.DataFrame(X_test_prepared, columns = X_test_num.columns.tolist() + list(encoder.classes_))

    else:
        num_pipeline = Pipeline([('imputer', Imputer(strategy = 'mean')),
                                 ('std_scaler', StandardScaler())]) 

        X_train_scaled     = num_pipeline.fit_transform(X_train) #transform returns numpy array
        X_train_prepared   = pd.DataFrame(X_train_scaled, columns = X_train.columns)

        X_test_scaled     = num_pipeline.transform(X_test) #transform returns numpy array
        X_test_prepared   = pd.DataFrame(X_test_scaled, columns = X_train.columns)

    return (X_train_prepared, y_train, X_test_prepared, y_test)