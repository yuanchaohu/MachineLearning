#coding = utf-8

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#----------read data---------
data = pd.read_csv('../../fulldata.dat', sep = '\s+')
data = data[data['logRc'] > -6]
print (data.shape)

data['epsilon1'] = (data['eBB'] - data['eAA']) / (data['eBB'] + data['eAA'])
data['epsilon2'] = 2 * data['eAB'] / (data['eBB'] + data['eAA'])
data['sigma'] = (data['rBB'] - data['rAA']) / data['rAB']

#------design training and test set---------
features = ['epsilon1', 'epsilon2', 'fB'] #, 'sigma'
condition = data['sigma'] == 0 
x_train = data.loc[condition, features].copy()
y_train = data.loc[condition, 'logRc'].copy()

testset = pd.read_csv('testset.dat', sep = '\s+')
x_test = testset[features]
print (x_train.shape, x_test.shape)

#------data processing-----------------
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()
x_train_scaled   = scaler.fit_transform(x_train)
x_train_prepared = pd.DataFrame(x_train_scaled, columns = x_train.columns)
x_test_scaled    = scaler.transform(x_test)
x_test_prepared  = pd.DataFrame(x_test_scaled, columns = x_test.columns)
#x_train_prepared['fB'] = x_train['fB'].values
#x_test_prepared['fB']  = x_test['fB'].values

#-----training-------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

def cross_validation():
    regs = [LinearRegression(), Ridge(max_iter=10000), Lasso(max_iter=100000), ElasticNet(max_iter=10000)]
    names = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']
    markers = ['o', 's', '*', 'X']
    degrees = [1, 2, 3, 4, 5, 6, 7]
    scoring = 'neg_mean_squared_error'
    plt.figure()
    for n, i in enumerate(regs):
        output = []
        for j in degrees:
            poly = PolynomialFeatures(j)
            x_train_used = poly.fit_transform(x_train_prepared)
            scores = cross_val_score(i, x_train_used, y_train, scoring = scoring, cv = 10)
            real_scores = np.sqrt(-scores)
            #print ('model: ' + str(i))
            print ('degree: %d ; CV score: %.3f' %(j, real_scores.mean()))
            output.append(real_scores.mean())
        print ()
        plt.plot(degrees, output, '-' + markers[n], label = names[n])
    
    plt.xlabel('Polynomial Degree', size = 20)
    plt.ylabel('RMSE', size = 20)
    plt.xticks(size = 18)
    plt.yticks(size = 18)
    plt.ylim(0.2, 0.7)
    plt.legend(fontsize = 14)
    plt.tight_layout()
    plt.savefig('cross_validation.score.png', dpi = 600)
    plt.show()
    plt.close()

#cross_validation()

def predictions():
    model = make_pipeline(PolynomialFeatures(6), Ridge())
    model.fit(x_train_prepared, y_train)
    y_predict = model.predict(x_test_prepared)
    print (model.steps[1][1].coef_, model.steps[1][1].coef_.shape)
    print (model.steps[1][1].intercept_)
    #----save to files-----------
    medium = model.steps[1][1].coef_
    a = np.argsort(medium)[::-1]
    medium = np.column_stack((a, medium[a]))
    np.savetxt('coeff.dat', medium, fmt = '%d %.6f', comments = '', header = 'i coeff')
    x_test1['Measured'] = y_test
    x_test1['Predicted'] = y_predict
    x_test1.to_csv('predictions.csv')
    medium = (x_test1['group'] == 3) & (x_test1['fB'] == 0.5)
    x_test1[medium].to_csv('predictions.group3.csv')

    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    r2 = r2_score(y_test, y_predict)
    label = 'RMSE=%.3f \n' %rmse
    label = label + r'$R^2=%.3f$' %r2
    plt.scatter(y_test, y_predict, edgecolor = 'gray', label = label)
    plt.plot([-5.5, -1.7], [-5.5, -1.7], '--', lw = 2, c = 'orange')
    plt.xlabel('Measured', size = 20)
    plt.ylabel('Prediction', size = 20)
    plt.xticks(size = 18)
    plt.yticks(size = 18)
    plt.legend(fontsize = 14)
    plt.tight_layout()
    plt.savefig('predictions.png', dpi = 600)
    plt.show()
    plt.close()

#predictions()

def plot_learning_curve():
    from sklearn.model_selection import learning_curve

    model = make_pipeline(PolynomialFeatures(6), Ridge())
    train_size = np.linspace(0.2, 1.0, 9)
    scoring = 'neg_mean_squared_error'
    train_size, train_scores, valid_scores = learning_curve(model, x_train_prepared, y_train, train_sizes = train_size,
                                             scoring = scoring, cv = 10, shuffle = True, random_state = 42)

    #plt.figure(figsize = (6, 6))
    train_scores_mean = np.sqrt(-train_scores).mean(axis = 1)
    train_scores_std  = np.sqrt(-train_scores).std(axis = 1)
    valid_scores_mean = np.sqrt(-valid_scores).mean(axis = 1)
    valid_scores_std  = np.sqrt(-valid_scores).std(axis = 1)
    plt.plot(train_size, train_scores_mean, 'o-', color = 'r', label = 'Training score', markersize = 12)
    plt.plot(train_size, valid_scores_mean, 's-', color = 'g', label = 'Cross-validation score', markersize = 12)
    plt.fill_between(train_size, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = .1, color = 'r')
    plt.fill_between(train_size, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha = .1, color = 'g')
    plt.xlabel('Training Size', size = 20)
    plt.ylabel('RMSE', size = 20)
    plt.xticks(size = 18)
    plt.yticks(size = 18)
    plt.legend(loc = 'upper right', fontsize = 18)
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi = 600)
    plt.show()
    plt.close()

#plot_learning_curve()

def get_feature():
    poly = PolynomialFeatures(6).fit(x_train_prepared)
    medium = poly.get_feature_names(x_train_prepared.columns)

    data = pd.read_csv('coeff.dat', sep = '\s+')
    data['abs'] = data['coeff'].abs()
    data.sort_values(by = 'abs', inplace = True, ascending = False)
    data['term'] = 'NAN'
    print (data, data.index)
    for i in data.index:
        j = int(data.loc[i, 'i'])
        data.loc[i, 'term'] = medium[j]

    data.to_csv('features.weights.terms.csv')
    print (medium)

#get_feature().

model = make_pipeline(PolynomialFeatures(6), Ridge())
model.fit(x_train_prepared, y_train)
y_predict = model.predict(x_test_prepared)
x_test['predicted'] = y_predict
x_test.to_csv('testset.predictions.csv', index = False)