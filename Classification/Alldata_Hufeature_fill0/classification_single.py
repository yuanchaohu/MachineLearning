#coding = utf-8

import os, sys 
import numpy  as np 
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

#---------load the package for classification start---------------------------
#sys.path.append(r'D:/GoogleDriver/GFA-MachineLearning/ClassificationPackage/')
sys.path.append(r'/home/yh527/project/soft/ClassificationPackage/')
from datacorrelation import data_correlation
from traintestsplit  import design_train_test
from plots           import plot_precision_recall
from plots           import plot_roc_curve
from plots           import plot_learning_curve
from modelevaluation import model_evaluation
from onemodel        import try_one
from multiplemodels  import Multi_classifiers  
#---------load the package for classification end----------------------------

scoring = 'roc_auc'
testalloy = ['V', 'Ni', 'Nb']

def get_test_alloy(testalloy, datafile = '../../finaldata_all.csv'):
    """ get the dataset of one alloy to be tested """

    data = pd.read_csv(datafile)
    conditions = (data[testalloy].all(axis = 'columns'))
    data[conditions].to_csv('testdata.csv', index = False)
    data[~conditions].to_csv('traindata.csv', index = False)
    print ((data[conditions].columns != data[~conditions].columns).sum())

get_test_alloy(testalloy)

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

def readata(datafile = './traindata.csv', output = 'data_information_train.dat'):
    """ overview data and return clean data """

    data = pd.read_csv(datafile)
    data.drop(['XRDname'], axis = 'columns', inplace = True)
    print ('Null data: ', data.isnull().values.sum())
    
    f = open(output, 'w')
    f.write('instances: %d;  features:  %d\n' % (data.shape[0], data.shape[1] - 1))
    f.write('\n')
    medium = data['components'].value_counts()
    f.write('components \n' + pd.concat([medium.to_frame(), (medium / medium.sum()).to_frame()], axis = 1).to_string(header = ['num', 'frac']) + '\n')
    f.write('\n')
    medium = data['label'].value_counts()
    f.write('labels \n' + pd.concat([medium.to_frame(), (medium / medium.sum()).to_frame()], axis = 1).to_string(header = ['num', 'frac']) + '\n')
    f.write('\n')

    data = data[(data['label'] == 1) | (data['label'] == 3)]
    data['label'].where(data['label'] == 1, 0, inplace = True)
    medium = data['label'].value_counts()
    f.write('used labels \n' + pd.concat([medium.to_frame(), (medium / medium.sum()).to_frame()], axis = 1).to_string(header = ['num', 'frac']) + '\n')
    f.write('1: amorphous\n0: crystallized\n')
    f.close()

    data.reset_index(drop = True, inplace = True)    
    shuffled = np.random.permutation(data.shape[0])
    data = data.reindex(index = shuffled)

    #------only use top features--------------
    #topfeatures = pd.read_csv('../machinelearning/topfeatures/featureimportance.csv').iloc[:, 0].tolist()
    #m = 20
    #data = data[topfeatures[:m] + ['label']]
    #print ('Used Features (' + str(m) + '):\n' + '\n'.join(topfeatures[:m]))
    #print (data.shape)

    return data

def train_classifiers(scoring):
    """
    data preparation and choose model to train then tune hyperparameters
    """

    #-----------------get train and test set--------------------
    train_set = readata(datafile = 'traindata.csv', output = 'data_information_train.dat')
    test_set  = readata(datafile = 'testdata.csv', output = 'data_information_test.dat')

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
    from sklearn.preprocessing import MinMaxScaler    #to [0, 1]
    
    std_scaler       = StandardScaler()
    X_train_scaled   = std_scaler.fit_transform(X_train) #transform returns numpy array
    X_train_prepared = pd.DataFrame(X_train_scaled, columns = X_train.columns)
    X_test_scaled    = std_scaler.transform(X_test)
    X_test_prepared  = pd.DataFrame(X_test_scaled, columns = X_test.columns)
    
    #---------------------try one algorithm----------------------------------
    #try_one(X_train_prepared, y_train)

    #---------------------Grid Search to tune hyperparameters----------------
    from sklearn.model_selection import GridSearchCV
    
    classifiers, paramsall = Multi_classifiers()
    bestperformances = pd.DataFrame(columns = ['name', 'Best Score', 'Best Hyperparameters', 'Best Estimator'])
    measurements = pd.DataFrame(columns = ['name', 'Confusion Matrix', 'Precision Score', 'Recall Score', 'F1 Score', 'AUC'])
    for i, (name, clf) in enumerate(classifiers[:4]):
        print (i, name)
        grid_search = GridSearchCV(clf, paramsall[i], cv = 10, scoring = scoring, return_train_score = True)
        grid_search.fit(X_train_prepared, y_train)

        (pd.DataFrame.from_dict(grid_search.cv_results_)).to_csv(name + '_gridsearchresults.csv', index = False)
        bestperformances.loc[i] = [name, str(grid_search.best_score_), str(grid_search.best_params_), str(grid_search.best_estimator_)]
        bestperformances.loc[i].to_csv(name + '_bestparameters.csv')

        #---------------plot learning curve------------------------
        plot_learning_curve(grid_search.best_estimator_, X_train_prepared, y_train, scoring = scoring)
        plt.savefig(name + '_learningcurve.png', dpi = 600)
        plt.close()

        #--------------model performance evaluation-----------------
        y_test_predict = grid_search.predict(X_test_prepared)
        if hasattr(grid_search, 'decision_function'):
            y_test_score = grid_search.decision_function(X_test_prepared)
        else:
            y_test_score = grid_search.predict_proba(X_test_prepared)
            y_test_score = y_test_score[:, 1]

        measurements.loc[i] = model_evaluation(y_test, y_test_predict, y_test_score, name)

        #------------feature importance or weights-------------------
        modelattributes = vars(grid_search.best_estimator_)
        if 'coef_' in modelattributes.keys():
            feature_importances = modelattributes['coef_'].ravel()
        elif 'coefs_' in modelattributes.keys():
            feature_importances = modelattributes['coefs_'][0].ravel()
        elif hasattr(grid_search.best_estimator_, 'feature_importances_'):
            feature_importances = grid_search.best_estimator_.feature_importances_
        else:
            feature_importances = np.zeros(X_train.shape[1])

        if feature_importances.any():
            featureimportances = {m:n for m, n in zip(X_train.columns.tolist(), feature_importances)}
            pd.Series(featureimportances).sort_values(ascending = False).to_csv(name + '_featureimportance.csv', float_format = '%.4f')

    bestperformances.to_csv('gridsearch_besthyperparameters.csv', index = False)
    measurements.to_csv('gridsearch_modelperformance.csv', index = False)

train_classifiers(scoring)
