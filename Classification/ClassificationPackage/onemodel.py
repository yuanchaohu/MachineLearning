#coding = utf-8

import os, sys 
import numpy  as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 

def try_one(X_train_prepared, y_train):
    """ try one algorithm for test """

    from sklearn.linear_model      import SGDClassifier
    from sklearn.model_selection   import cross_val_score, cross_val_predict
    from sklearn.metrics           import confusion_matrix
    from sklearn.metrics           import precision_score, recall_score, f1_score
    from sklearn.metrics           import roc_auc_score, log_loss
    from sklearn.metrics           import precision_recall_curve, roc_curve

    sgd_clf = SGDClassifier(max_iter = 100, tol = 1e-6, random_state = 42)
    name = 'SGD'
    #sgd_clf.fit(X_train_prepared, y_train)

    #------------------cross validation try-----------------------
    print (X_train_prepared.shape)
    crossvalscores =cross_val_score(sgd_clf, X_train_prepared, y_train, scoring = 'accuracy', cv = 10)
    print (crossvalscores.mean(), crossvalscores.std())
    #the 'accuracy' is not accurate for skewed datasets

    #-----------------performance measurement---------------------
    y_train_predict = cross_val_predict(sgd_clf, X_train_prepared, y_train, cv = 10)
    print ('confusion matrix:\n ', confusion_matrix(y_train, y_train_predict))
    print ('precision score: ', precision_score(y_train, y_train_predict))
    print ('recall score: ', recall_score(y_train, y_train_predict))
    print ('F1 score: ', f1_score(y_train, y_train_predict))

    y_train_scores = cross_val_predict(sgd_clf, X_train_prepared, y_train, cv = 10, method = 'decision_function')

    precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_scores)
    plot_precision_recall(precisions, recalls, thresholds)
    plt.savefig(name + '_PR.png', dpi = 600)
    plt.close()

    falsepos, truepos, thresholds = roc_curve(y_train, y_train_scores)
    areaucurve = roc_auc_score(y_train, y_train_scores)
    print ('AUC: ', areaucurve)
    plot_roc_curve(falsepos, truepos, areaucurve)
    plt.savefig(name + '_ROC.png', dpi = 600)
    plt.close()
