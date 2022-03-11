#coding = utf-8 

import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt 

models = ['SGD', 'LinearSVC', 'SVM']
variables = [1, 2, 3, 5, 10, 15, 20]

def combine_scores(models, variables):
    """ get the same scores from different situations """

    path = './scores/'
    if not os.path.exists(path):
        os.makedirs(path)

    for i in models:
        results = pd.DataFrame(index = np.arange(len(variables)) , columns = ['vars', 'ConfusionMatrix', 'Precision', 'Recall', 'F1', 'ROC_AUC'])
        for m, j in enumerate(variables):
            results.iloc[m, 0] = j 
            datapath = './top' + str(j) + '/'
            data = pd.read_csv(datapath + i + '_scores.csv')
            results.iloc[m, 1:] = data.iloc[:, 1].tolist()
        results.to_csv(path + i + '_scores_all.csv', index = False)

#combine_scores(models, variables)

def plot_scores(models):
    """Plot different scores of a model vs. different variables"""

    path = './scores/'
    for i in models:
        data = pd.read_csv(path + i + '_scores_all.csv')

        plt.figure(figsize = (6, 6))
        scores = ['Precision', 'Recall', 'F1', 'ROC_AUC']
        colors = ['r', 'orange', 'b', 'g']
        for m, j in enumerate(scores):
            plt.plot(data.loc[:, 'vars'], data.loc[:, j], '-o', markersize = 8, c = colors[m], label = j)

        plt.legend(loc = 'lower right', fontsize = 12)
        plt.xlabel('Number of Features', size = 12)
        plt.ylabel('Scores', size = 12)
        plt.xticks(range(0,21,5), fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.tight_layout()
        plt.savefig(path + i + '.png', dpi = 600)
        #plt.show()
        plt.close()

#plot_scores(models)
