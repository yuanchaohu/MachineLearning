#coding = utf-8

#to explore the correlation between features and label 

import os, sys 
import numpy  as np 
import pandas as pd 
import matplotlib.pyplot as plt 

data = pd.read_csv('finaldata_all.csv')
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

def plot_network(corr):
    import networkx as nx 
    features = corr.columns.tolist()
    #print (features)

    #plt.figure(figsize = (60, 60))

    links = corr.stack().reset_index()
    links.columns = ['var1', 'var2', 'value']
    print (links.shape)

    links_filtered = links.loc[(links['value'] > 0.8) & (links['var1'] != links['var2'])]
    print (links_filtered.shape)

    G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
    nx.draw(G, with_labels = True, node_color = 'orange', node_size = 20, edge_color = 'red', linewidth = 1, font_size = 3)
    #plt.tight_layout()
    #plt.title('correlation  coefficient > 0.80')
    plt.savefig('network.png', dpi = 1200)
    plt.show()

plot_network(corr_matrix)