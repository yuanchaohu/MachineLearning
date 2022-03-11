#coding = utf-8

import os, sys 
import numpy  as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 

def plot_precision_recall(precisions, recalls, thresholds):
    """ plot precision-recall-thresholds curve for classification """

    plt.figure(figsize = (12, 6))
    plt.subplot(121)
    plt.plot(thresholds, precisions[:-1], 'r--', label = 'precision')
    plt.plot(thresholds, recalls[:-1], 'b-', label = 'recall')
    plt.ylim(0, 1)
    plt.xticks(size = 18)
    plt.yticks(size = 18)
    plt.xlabel('Threshold', size = 20)
    plt.legend(loc = 'best', fontsize = 18)

    plt.subplot(122)
    plt.plot(recalls, precisions, 'r')
    plt.xlabel('Recall', size = 20)
    plt.ylabel('Precision', size = 20)
    plt.xticks(size = 18)
    plt.yticks(size = 18)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()

def plot_roc_curve(falsepos, truepos, areaucurve, ncol = 1, label = ''):
    """ plot roc curve for classification """

    plt.figure(figsize = (6, 6))
    plt.plot(falsepos, truepos, 'b-', linewidth = 2, label = label + ' (' + format(areaucurve, '.3f') + ')')
    plt.plot([0, 1], [0, 1], 'k--', label = 'Random Guessing')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(size = 18)
    plt.yticks(size = 18)
    plt.xlabel('False Positive Ratio', size = 20)
    plt.ylabel('True Positive Ratio', size = 20)
    plt.legend(loc = 'lower right', ncol = ncol, fontsize = 18)
    plt.tight_layout()

def plot_learning_curve(estimator, X, y, scoring, cv = 10, train_size = np.linspace(0.1, 1.0, 10)):
    """plot learning curve of a model"""

    from sklearn.model_selection import learning_curve

    train_size, train_scores, valid_scores = learning_curve(estimator, X, y, train_sizes = train_size,
                                             scoring = scoring, cv = cv, shuffle = True, random_state = 42)

    plt.figure(figsize = (6, 6))
    train_scores_mean = train_scores.mean(axis = 1)
    train_scores_std  = train_scores.std(axis = 1)
    valid_scores_mean = valid_scores.mean(axis = 1)
    valid_scores_std  = valid_scores.std(axis = 1)
    plt.plot(train_size, train_scores_mean, 'o-', color = 'r', label = 'Training score')
    plt.plot(train_size, valid_scores_mean, 's-', color = 'g', label = 'Cross-validation score')
    plt.fill_between(train_size, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = .1, color = 'r')
    plt.fill_between(train_size, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha = .1, color = 'g')
    plt.xlabel('Training examples', size = 20)
    plt.ylabel('Score', size = 20)
    plt.xticks(size = 18)
    plt.yticks(size = 18)
    #plt.ylim(0.7, 1.01)
    plt.legend(loc = 'lower right', fontsize = 18)
    plt.grid()
    plt.tight_layout()
