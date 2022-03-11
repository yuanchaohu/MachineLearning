#coding = utf-8

import os, sys 
import numpy  as np 
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

#scoring = sys.argv[1] 
scoring = 'roc_auc'
#testalloy = ['V', 'Ni', 'Nb']

def get_test_alloy(testalloy, datafile = '../finaldata.csv'):
    """ get the dataset of the alloy to be tested """

    data = pd.read_csv(datafile)
    conditions = (data[testalloy].all(axis = 'columns'))
    data[conditions].to_csv('testdata.csv', index = False)
    data[~conditions].to_csv('traindata.csv', index = False)
    print ((data[conditions].columns != data[~conditions].columns).sum())

#get_test_alloy(testalloy)

def choose_fraction(datafile = '../finaldata.csv', alloysfile = '../alloys.csv', frac = 0.20):
    """ Choose a fraction of data from each system """

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

choose_fraction()

def readata(datafile = './finaldata_others.csv', output = 'data_information.dat'):
    """
    overview data and return clean data
    """

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
    topfeatures = pd.read_csv('../featureimportance.csv').iloc[:, 0].tolist()
    m = 20
    data = data[topfeatures[:m] + ['label']]
    print ('Used Features (' + str(m) + '):\n' + '\n'.join(topfeatures[:m]))
    print (data.shape)

    return data

def design_train_test(test_size = 0.30):
    """
    split train and test sets from the database
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import train_test_split

    data = readata()
    split = StratifiedShuffleSplit(n_splits = 10, test_size = test_size, random_state = 42)
    for train_index, test_index in split.split(data, data['label']):
        train_set = data.loc[train_index]
        test_set  = data.loc[test_index]

    choosetrain = train_set['label'].value_counts() / train_set.shape[0]
    print ('train set\n', choosetrain)
    choosetest  = test_set['label'].value_counts() / test_set.shape[0]
    print ('test set\n', choosetest)

    return train_set, test_set

def data_correlation(corr_matrix):
    """ Get the pearson correlation coefficients between features and label """

    correlations = corr_matrix['label'].sort_values(ascending = False)
    correlations.to_csv('correlations.csv')
    correlations.drop('label', inplace = True)
    correlations.rename(index = lambda x: x.replace(' (kJ/mol)', '\n(kJ/mol)'), inplace = True)
    correlations.rename(index = lambda x: x.capitalize(), inplace = True)

    correlations[:10].sort_values(ascending = True).plot.barh(color = 'b', alpha = 0.7) #abs().sort_values(ascending = False)
    plt.xlabel('Correlation Coefficient')
    plt.title('Correlation with "Label"')
    plt.subplots_adjust(left = 0.35)
    plt.tight_layout()
    plt.savefig('correlation_matrix_positive.png', dpi = 600)
    plt.close()
    correlations[-10:].sort_values(ascending = False).plot.barh(color = 'r', alpha = 0.7) #abs().sort_values(ascending = False)
    plt.xlabel('Correlation Coefficient')
    plt.title('Correlation with "Label"')
    plt.subplots_adjust(left = 0.35)
    plt.tight_layout()
    plt.savefig('correlation_matrix_negative.png', dpi = 600)
    plt.close()

def plot_precision_recall(precisions, recalls, thresholds):
    """ plot precision-recall-thresholds curve for classification """

    plt.figure(figsize = (12, 6))
    plt.subplot(121)
    plt.plot(thresholds, precisions[:-1], 'r--', label = 'precision')
    plt.plot(thresholds, recalls[:-1], 'b-', label = 'recall')
    plt.ylim(0, 1)
    plt.xticks(size = 14)
    plt.yticks(size = 14)
    plt.xlabel('Threshold', size = 16)
    plt.legend(loc = 'best', fontsize = 14)

    plt.subplot(122)
    plt.plot(recalls, precisions, 'r')
    plt.xlabel('Recall', size = 16)
    plt.ylabel('Precision', size = 16)
    plt.xticks(size = 14)
    plt.yticks(size = 14)
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
    plt.xticks(size = 14)
    plt.yticks(size = 14)
    plt.xlabel('False Positive Ratio', size = 16)
    plt.ylabel('True Positive Ratio', size = 16)
    plt.legend(loc = 'lower right', ncol = ncol, fontsize = 14)
    plt.tight_layout()

def plot_learning_curve(estimator, X, y, scoring, cv = 50, train_size = np.linspace(0.1, 1.0, 10)):
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
    plt.xlabel('Training examples', size = 16)
    plt.ylabel('Score', size = 16)
    plt.xticks(size = 14)
    plt.yticks(size = 14)
    #plt.ylim(0.7, 1.01)
    plt.legend(loc = 'lower right', fontsize = 14)
    plt.grid()
    plt.tight_layout()

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
                   ('RandomForest', RandomForestClassifier(oob_score = True, random_state = 42)),
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

    params_SVM = [{'C': np.logspace(-8, 8, 17), 'kernel': ['poly', 'rbf', 'sigmoid'], 'gamma': np.logspace(-8, 2, 11)}]
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

def train_classifiers(scoring):
    """
    data preparation and choose model to train then tune hyperparameters
    """

    #-----------------get train and test set--------------------
#    train_set, test_set = design_train_test()
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
    #use Imputer method to deal with NULL numerical data
    #use LabelBinarizer method for Text and Categorical info

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
        grid_search = GridSearchCV(clf, paramsall[i], cv = 50, scoring = scoring, return_train_score = True)
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

def get_feature_importance(scoring):
    """ Get Feature importances for some models based on the whole dataset """

    #---------get the training dataset ready-----------------
    data     = readata()
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

#get_feature_importance(scoring)


def final_model(scoring, testalloy):
    """ Use optimized hyerparameters for different algorithms """

    #---------get the training dataset ready-----------------
    data     = readata(datafile = './finaldata_others.csv')
    X_train  = data.drop('label', axis = 'columns')
    y_train  = data['label'].copy()
    print ('X_train shape: ', X_train.shape)

    #---------get prediction set ready-----------------------
    datapred = readata(datafile = './finaldata_' + ''.join(testalloy) + '.csv')
    X_pred   = datapred.drop('label', axis = 'columns')
    y_pred   = datapred['label'].copy()
    print ('X_predict shape: ', X_pred.shape)
    print ('Prediction set:\n', y_pred.value_counts())

    #-----------feature scaling-----------------------------
    from sklearn.preprocessing import StandardScaler

    std_scaler       = StandardScaler()
    X_train_scaled   = std_scaler.fit_transform(X_train)
    X_train_prepared = pd.DataFrame(X_train_scaled, columns = X_train.columns)
    
    X_pred_scaled    = std_scaler.transform(X_pred)
    X_pred_prepared  = pd.DataFrame(X_pred_scaled, columns = X_pred.columns)
    
    #------------machine learning---------------------------
    from sklearn.svm       import SVC 

    clf = SVC(C = 2636.6508987303555, gamma = 0.01, kernel = 'rbf', tol = 1e-4, random_state = 42)
    clf.fit(X_train_prepared, y_train)

    #----------model prediction capability measurement-----------------
    path = './prediction/'
    if not os.path.exists(path):
        os.makedirs(path)

    y_pred_predict = clf.predict(X_pred_prepared)
    if hasattr(clf, 'decision_function'):
        y_pred_score = clf.decision_function(X_pred_prepared)
    else:
        y_pred_score = clf.predict_proba(X_pred_prepared)
        y_pred_score = y_pred_score[:, 1]

    model_evaluation(y_pred, y_pred_predict, y_pred_score, name = 'SVC_prediction', path = path)

#final_model(scoring, testalloy)
