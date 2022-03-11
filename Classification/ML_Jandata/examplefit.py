#coding = utf - 8

import os, sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def examplefit():
    xdata = np.linspace(0, 20, 20)
    ydata = 0.5 * xdata + np.random.rand(xdata.shape[0]) * 5
    plt.scatter(xdata, ydata, s = 80, c = 'orange')
    plt.plot([-10, 30], [-2.5, 17.5], '--' , c = 'b', linewidth = 3)
    plt.xlim(-10, 30)
    plt.xticks(())
    plt.yticks(())
    plt.axvspan(0, 20, color = 'lightgreen', alpha = 0.3)
    plt.axvspan(-10, 0, color = 'red', alpha = 0.2)
    plt.axvspan(20, 30, color = 'red', alpha = 0.2)
    plt.savefig('examplefit.png', dpi = 300)
    plt.show()

#examplefit()

def heatofmixing(datafile = './finaldata.csv'):

    data = pd.read_csv(datafile)
    data = data[['heatofmixing', 'components', 'label']]

    plt.figure(figsize = (6, 6))
    condition = (data['label'] == 1)
    plt.plot(data['components'][condition], data['heatofmixing'][condition], 'o', markersize = 10, c = 'r', label = 'amorphous')
    plt.plot(data['components'][~condition], data['heatofmixing'][~condition], 's', c = 'g', label = 'crystal')
    plt.legend(loc = 'upper right', fontsize = 12)
    plt.xlabel('Components', size = 12)
    plt.ylabel('Heat of Mixing (kJ/mol)', size = 12)
    plt.tight_layout()
    plt.savefig('heatofmixing_components.png', dpi = 600)
    plt.show()
    plt.close()

#heatofmixing()

topfeatures = """boiling point polydisperity
Covalent Radius polydisperity
Pauling Electronegativity polydisperity
Ionic Radius secondM
p valences
Ionic Radius mean
radius polydisperity
Covalent Radius sixthM
s valences
d valences
period sixthM
Melting Temperature (K) mean
components
Covalent Radius fourthM
s mean
Melting Temperature (K) secondM
group polydisperity
boiling point secondM
group secondM
boiling point mean"""
topfeatures = topfeatures.split('\n')

def explore(topfeatures):
    """ plot different combinations of features """
    path = './featurecombinations/'
    if not os.path.exists(path):
        os.makedirs(path)

    data  = pd.read_csv('./finaldata.csv')
    data1 = data[data['label'] == 1]
    data2 = data[data['label'] == 3]
    i = 0
    for x in topfeatures:
        for y in topfeatures:
            if x != y:
                plt.figure(figsize = (6, 6))
                plt.plot(data1[x], data1[y], 'o', markerfacecolor = 'yellow', markeredgecolor = 'red', alpha = 0.9, label = 'amorphous')
                plt.plot(data2[x], data2[y], 's', markerfacecolor = "blue", markeredgecolor = 'lightgreen', alpha = 0.2, label = 'crystal')
                #plt.plot([data2[x].min(), data2[x].max()])
                plt.xlabel(x, size = 12)
                plt.ylabel(y, size = 12)
                plt.legend(loc = 'upper left', fontsize = 12)
                plt.tight_layout()
                plt.savefig(path + 'exploration_' + str(i) + '.png', dpi = 600)
                plt.close()
                #plt.show()
                #print ((data2[y] >= 8).sum())
                i += 1

#explore(topfeatures)

def dimensionreduction(topfeatures):
    """ Use PCA to reduce dimensions for visualization """
    path = 'pca/'
    if not os.path.exists(path):
        os.makedirs(path)

    from sklearn.decomposition import PCA 
    pca = PCA(n_components = 2)

    data  = pd.read_csv('./finaldata.csv')
    data.drop(columns = ['XRDname'], inplace = True)
    pca_data = pca.fit_transform(data.drop(columns = ['label'])) #(data[topfeatures])
    print (pca.explained_variance_ratio_)
    data1 = pca_data[data['label'] == 1]
    data2 = pca_data[data['label'] == 3]

    plt.figure(figsize = (6, 6))
    plt.plot(data1[:, 0], data1[:, 1], 'o', markerfacecolor = 'yellow', markeredgecolor = 'red', alpha = 0.9, label = 'amorphous')
    plt.plot(data2[:, 0], data2[:, 1], 's', markerfacecolor = "blue", markeredgecolor = 'lightgreen', alpha = 0.2, label = 'crystal')
    plt.xlabel('Principal Component #1', size = 12)
    plt.ylabel('Principal Component #2', size = 12)
    plt.legend(loc = 'upper left', fontsize = 12)
    plt.title('explained variance ratio ' + str(pca.explained_variance_ratio_))
    plt.tight_layout()
    plt.savefig(path + 'pca_2_all.png', dpi = 600)
    plt.close()

#dimensionreduction(topfeatures)