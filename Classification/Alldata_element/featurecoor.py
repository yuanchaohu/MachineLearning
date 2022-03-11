#coding = utf-8

#to explore the correlation between features and label 

import os, sys 
import numpy  as np 
import pandas as pd 
import matplotlib.pyplot as plt 

data = pd.read_csv('finaldata_all3.csv')
print (data.shape)
features = [i for i in data.columns if len(i) > 2]
data = data[features]
print (data.shape)

#ax = data.hist(bins = 50, figsize = (60, 60), xlabelsize = 4, ylabelsize = 4)
#[x.title.set_size(10) for x in ax.ravel()]
#plt.tight_layout()
#plt.show()
#plt.close()


corr_matrix = data.corr()
print (corr_matrix.shape)
#print (corr_matrix.head())

def corr_single(corr_matrix):
    fig, ax = plt.subplots(9, 9, figsize = (60, 60))
    m = 0 
    for i in range(9):
        for j in range(9):
            ax[i, j].plot(np.arange(81), corr_matrix.iloc[:, m], '-o', markersize = 4, label = corr_matrix.columns[m])
            ax[i, j].legend(loc = 'upper right')
            ax[i, j].set_ylim(-1, 1)
            ax[i, j].set_xlim(0, 81)
            ax[i, j].set_xticks(np.arange(0, 81, 10))
            ax[i, j]
            m += 1

    plt.tight_layout()
    #plt.savefig('All_correlation.png', dpi = 600)
    #plt.show()
    plt.close()

    plt.plot(np.arange(81), corr_matrix['label'], '-o')
    plt.savefig('label_correlation.png', dpi = 600)
    plt.close()

#corr_single(corr_matrix)

def plot_network(corr, name1):
    import networkx as nx 
    features = corr.columns.tolist()
    num_features = dict(zip(features, range(1, len(features) + 1)))
    #print (num_features)
    with open('num_features.dat', 'w') as f:
        for i, j in num_features.items():
            f.write(str(j) + ':   ' + i + '\n')

    links = corr.stack().reset_index()
    links.columns = ['var1', 'var2', 'value']
    print (links.shape)

    links['var1r'] = links['var1'].map(num_features)
    links['var2r'] = links['var2'].map(num_features)
    print (links.shape)
    links.drop(columns = ['var1', 'var2'], inplace = True)

    cutoff = 0.5

    links_filtered = links.loc[(links['value'].abs() > cutoff) & (links['var1r'] != links['var2r'])]
    print (links_filtered.shape)


    G=nx.from_pandas_edgelist(links_filtered, 'var1r', 'var2r')
    nx.draw(G, with_labels = True, node_color = 'orange', node_size = 30, edge_color = 'green', linewidth = 1, font_size = 5)
    #plt.tight_layout()
    #plt.title('correlation  coefficient > 0.80')
    plt.savefig(name1 + '_network_' + str(cutoff) + '1.png', dpi = 2400)
    plt.show()

#plot_network(corr_matrix)

with open('usedelements1.dat', 'r') as f:
    data1 = [i.strip('\n') for i in f.readlines()[2:]]

data2 = pd.read_csv('elements_full.csv', index_col = 0)
data2 = data2.loc[data1]
data2.drop(columns = ['HeatFusion'], inplace = True)
print (data2.shape)
corr_matrix = data2.corr()
print (corr_matrix.shape)
plot_network(corr_matrix, name1 = 'elements_full')