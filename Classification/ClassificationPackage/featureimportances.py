#coding = utf-8

import os, sys 
import numpy  as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 


def get_feature_importance(data, scoring):
    """ Get Feature importances for some models based on the whole dataset """

    #---------get the training dataset ready-----------------
    #data     = readata()
    shuffled = np.random.permutation(data.shape[0])
    data     = data.reindex(index = shuffled)
    X_train  = data.drop('label', axis = 'columns')
    y_train  = data['label'].copy()
    print ('X_train shape: ', X_train.shape)

    #-----------feature scaling-----------------------------
    from sklearn.preprocessing import StandardScaler

    std_scaler = StandardScaler()
    X_train_scaled   = std_scaler.fit_transform(X_train)
    X_train_prepared = pd.DataFrame(X_train_scaled, columns = X_train.columns)

    #----------best model and feature importance------------------
    from sklearn.ensemble       import RandomForestClassifier
    from sklearn.ensemble       import AdaBoostClassifier
    from sklearn.neural_network import MLPClassifier

    clf_RF = RandomForestClassifier(n_estimators = 65, max_features = 'auto', min_impurity_decrease = 0.0001, 
             min_samples_leaf = 2, min_samples_split =2, oob_score = True, random_state = 42)
    clf_Ad = AdaBoostClassifier(learning_rate = 0.1, n_estimators = 445, random_state = 42)
    clf_NN = MLPClassifier(activation = 'tanh', alpha = 1e-6, hidden_layer_sizes = (500,), learning_rate_init = 0.01
                           , max_iter = 1000000, tol = 1e-4, random_state = 42, early_stopping = True)
    
    clfs = [('RandomForest', clf_RF), ('Adaboost', clf_Ad)]
    for name, clf in clfs:
        clf.fit(X_train_prepared, y_train)
        featureimportances = {m:n for m, n in zip(X_train_prepared.columns.tolist(), clf.feature_importances_)}
        pd.Series(featureimportances).sort_values(ascending = False).to_csv(scoring + '/' + name + '_' + scoring + '_featureimportance.csv'
                                                                            , float_format = '%.4f')

    clf_NN.fit(X_train_prepared, y_train)
    featureimportances = {m:n for m, n in zip(X_train_prepared.columns.tolist(), clf_NN.coefs_[1])}
    pd.Series(featureimportances).sort_values(ascending = False).to_csv(scoring + '/NeuralNetwork_' + scoring + '_featureimportance1.csv', float_format = '%.4f')
