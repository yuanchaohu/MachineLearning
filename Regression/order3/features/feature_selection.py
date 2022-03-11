#coding = utf-8

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#----------read data---------
data = pd.read_csv('../../../fulldata.dat', sep = '\s+')
data = data[data['logRc'] > -6]
print (data.columns)
print (data.shape)

data['epsilon1'] = (data['eBB'] - data['eAA']) / (data['eBB'] + data['eAA'])
data['epsilon2'] = 2 * data['eAB'] / (data['eBB'] + data['eAA'])
data['sigma'] = (data['rBB'] - data['rAA']) / data['rAB']

features = ['epsilon1', 'epsilon2', 'sigma', 'fB']
fullx = data[features].copy()
fully = data['logRc'].copy()

#------design training and test set---------
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(fullx, fully, test_size = 0.3, random_state = 42)
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

#-----feature selection-------------
from sklearn.preprocessing import PolynomialFeatures

features = np.loadtxt('../coeff.dat', skiprows = 1)[:, 0].astype(np.int)
poly = PolynomialFeatures(3)
x_train_used = poly.fit_transform(x_train_prepared)
x_test_used  = poly.transform(x_test_prepared)
print (x_train_used.shape, x_test_used.shape)

#-----training-------------
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

def iterative():
    f = open('performance.dat', 'w')
    f.write('n RMSE R2\n')

    for i in range(features.shape[0]):
        model = Ridge()
        selected = features[:i + 1]
        model.fit(x_train_used[:, selected], y_train)
        y_predict = model.predict(x_test_used[:, selected])
        rmse = np.sqrt(mean_squared_error(y_test, y_predict))
        r2 = r2_score(y_test, y_predict)
        print (i+1, rmse, r2)
        f.write('%d %.6f %.4f\n' %(i+1, rmse, r2))
    f.close()

iterative()

def predictions():
    model = make_pipeline(PolynomialFeatures(6), Ridge())
    model.fit(x_train_prepared, y_train)
    y_predict = model.predict(x_test_prepared)
    print (model.steps[1][1].coef_, model.steps[1][1].coef_.shape)
    print (model.steps[1][1].intercept_)

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