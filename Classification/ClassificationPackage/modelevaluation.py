#coding = utf-8

import os, sys 
import numpy  as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 
from plots import plot_precision_recall
from plots import plot_roc_curve

def model_evaluation(y_test, y_test_predict, y_test_score, name, path = './'):
    """model performance evaluation"""

    from sklearn.metrics               import confusion_matrix
    from sklearn.metrics               import precision_score, recall_score, f1_score
    from sklearn.metrics               import roc_auc_score, log_loss
    from sklearn.metrics               import precision_recall_curve, roc_curve

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_score)
    PR = np.column_stack((np.column_stack((thresholds, precisions[:-1])), recalls[:-1]))
    pd.DataFrame(PR, columns = ['thresholds', 'precisions', 'recalls']).to_csv(path + name + '_PR.csv', index = False)
    plot_precision_recall(precisions, recalls, thresholds)
    plt.savefig(path + name + '_PR.png', dpi = 600)
    plt.close()

    falsepositive, truepositive, thresholds = roc_curve(y_test, y_test_score)
    ROC = np.column_stack((np.column_stack((falsepositive, truepositive)), thresholds))
    pd.DataFrame(ROC, columns = ['False Positive', 'True Positive', 'Thresholds']).to_csv(path + name + '_ROC.csv', index = False)
    areaucurve = roc_auc_score(y_test, y_test_score)
    plot_roc_curve(falsepositive, truepositive, areaucurve, label = name)
    plt.savefig(path + name + '_ROC.png', dpi = 600)
    plt.close()

    measurements = [name, str(confusion_matrix(y_test, y_test_predict)), precision_score(y_test, y_test_predict),
                           recall_score(y_test, y_test_predict), f1_score(y_test, y_test_predict), areaucurve]
    pd.Series(measurements, index = [name, 'confusion matrix', 'precision', 'recall', 'f1', 'AUC']).to_csv(path + name + '_scores.csv')

    return measurements
