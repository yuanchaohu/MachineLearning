#coding = utf-8

import os, sys 
import numpy  as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 

from modelevaluation import model_evaluation
from multiplemodels  import Multi_classifiers  
from plots           import plot_learning_curve


def model_optimization(X_train_prepared, y_train, X_test_prepared, y_test, scoring, m = 0, n = 4, path = './'):
    """Use Grid Search to optimize the models

    m and n is used to use the specified model
    """

    from sklearn.model_selection import GridSearchCV
    
    classifiers, paramsall = Multi_classifiers()
    allmodels = '  '.join([i[0] for i in classifiers])
    print ('All models: \n', allmodels)

    bestperformances = pd.DataFrame(columns = ['name', 'Best Score', 'Best Hyperparameters', 'Best Estimator'])
    measurements = pd.DataFrame(columns = ['name', 'Confusion Matrix', 'Precision Score', 'Recall Score', 'F1 Score', 'AUC'])
    for i, (name, clf) in enumerate(classifiers[m:n]):
        print (i+m, name)
        grid_search = GridSearchCV(clf, paramsall[i+m], cv = 10, scoring = scoring, return_train_score = True)
        grid_search.fit(X_train_prepared, y_train)

        (pd.DataFrame.from_dict(grid_search.cv_results_)).to_csv(path + name + '_gridsearchresults.csv', index = False)
        bestperformances.loc[i] = [name, str(grid_search.best_score_), str(grid_search.best_params_), str(grid_search.best_estimator_)]
        bestperformances.loc[i].to_csv(path + name + '_bestparameters.csv')

        #---------------plot learning curve------------------------
        plot_learning_curve(grid_search.best_estimator_, X_train_prepared, y_train, scoring = scoring)
        plt.savefig(path + name + '_learningcurve.png', dpi = 600)
        plt.close()

        #--------------model performance evaluation-----------------
        y_test_predict = grid_search.predict(X_test_prepared)
        if hasattr(grid_search, 'decision_function'):
            y_test_score = grid_search.decision_function(X_test_prepared)
        else:
            y_test_score = grid_search.predict_proba(X_test_prepared)
            y_test_score = y_test_score[:, 1]

        measurements.loc[i] = model_evaluation(y_test, y_test_predict, y_test_score, name, path)

        #------------feature importance or weights-------------------
        modelattributes = vars(grid_search.best_estimator_)
        if 'coef_' in modelattributes.keys():
            feature_importances = modelattributes['coef_'].ravel()
        elif 'coefs_' in modelattributes.keys():
            feature_importances = modelattributes['coefs_'][0].ravel()
        elif hasattr(grid_search.best_estimator_, 'feature_importances_'):
            feature_importances = grid_search.best_estimator_.feature_importances_
        else:
            feature_importances = np.zeros(X_train_prepared.shape[1])

        if feature_importances.any():
            featureimportances = {m:n for m, n in zip(X_train_prepared.columns.tolist(), feature_importances)}
            pd.Series(featureimportances).sort_values(ascending = False).to_csv(path + name + '_featureimportance.csv', float_format = '%.4f')

    bestperformances.to_csv(path + 'gridsearch_besthyperparameters.csv', index = False)
    measurements.to_csv(path + 'gridsearch_modelperformance.csv', index = False)
