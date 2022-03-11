#coding = utf-8

import os, sys 
import numpy  as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 


def Multi_classifiers():
    """ Designing different classification algorithms for gridsearch """

    from sklearn.linear_model          import SGDClassifier, LinearRegression
    from sklearn.neural_network        import MLPClassifier
    from sklearn.neighbors             import KNeighborsClassifier
    from sklearn.svm                   import SVC, LinearSVC 
    from sklearn.gaussian_process      import GaussianProcessClassifier
    from sklearn.tree                  import DecisionTreeClassifier 
    from sklearn.ensemble              import RandomForestClassifier, AdaBoostClassifier 
    from sklearn.naive_bayes           import GaussianNB, BernoulliNB 
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    classifiers = [('SGD', SGDClassifier(max_iter = 1000000, tol = 1e-4, random_state = 42)),
                   ('LinearSVC', LinearSVC(tol = 1e-4, random_state = 42)),
                   ('SVM', SVC(tol = 1e-4, random_state = 42)),
                   ('RandomForest', RandomForestClassifier(oob_score = False, random_state = 42)),
                   ('NeuralNetwork', MLPClassifier(max_iter = 1000000, tol = 1e-4, random_state = 42, early_stopping = True)),
                   ('GaussianProcess', GaussianProcessClassifier(random_state = 42)), #('DecisionTree', DecisionTreeClassifier()),
                   ('Adaboost', AdaBoostClassifier(random_state = 42)), #use decision tree
                   ('KNN', KNeighborsClassifier())]
                   # ('BernoulliNB', BernoulliNB())]
                   # ('GaussianNB', GaussianNB()),
                   # ('QDA', QuadraticDiscriminantAnalysis(tol = 1e-5))]

    paramsall = []
    
    params_SGD = [{'loss': ['log', 'modified_huber', 'perceptron'], 'penalty': ['l1', 'l2', 'elasticnet'], 'alpha': np.logspace(-6, -1, 6)}]
    paramsall.append(params_SGD)

    params_LinearSVC = [{'loss': ['hinge', 'squared_hinge'], 'C': np.logspace(-8, 8, 17)}]
    paramsall.append(params_LinearSVC)

    params_SVM = [{'C': np.logspace(-8, 5, 14), 'kernel': ['rbf'], 'gamma': np.logspace(-8, 2, 11)}]
    paramsall.append(params_SVM)

    params_RandomForest = [{'n_estimators': np.arange(5, 500, 20), 'max_features': ['auto', 'log2'], 'min_samples_split': np.arange(2, 42, 10), 
                            'min_samples_leaf': np.arange(2, 22, 10), 'min_impurity_decrease': [0.0001, 0.001, 0.01]}]
    paramsall.append(params_RandomForest)

    params_NeuralNetwork = [{'hidden_layer_sizes': [(10,), (50,), (100,), (250,), (500,), (1000,)], 'activation': ['logistic', 'tanh', 'relu'], 
                            'alpha': np.logspace(-6, -1, 6), 'learning_rate_init': [0.0001, 0.001, 0.01, 0.1]}]
    paramsall.append(params_NeuralNetwork)

    params_GaussianProcess = [{'max_iter_predict': [100, 500, 1000, 2000]}]
    paramsall.append(params_GaussianProcess)

    params_AdaBoost= [{'n_estimators': np.arange(5, 500, 20), 'learning_rate': np.logspace(-4, 1, 6)}]
    paramsall.append(params_AdaBoost)

    params_KNN = [{'n_neighbors': np.arange(3, 20, 2), 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                   'leaf_size': np.arange(10, 70, 10), 'p': [1, 2]}]
    paramsall.append(params_KNN)

    # params_BernulliNB = [{'alpha': [0.001, 0.01, 0.1, 1.0, 10]}]
    # paramsall.append(BernoulliNB)

    # params_GassianNB = []
    # paramsall.append(params_GassianNB)

    # params_QDA = []
    # paramsall.append(params_QDA)

    return classifiers, paramsall
