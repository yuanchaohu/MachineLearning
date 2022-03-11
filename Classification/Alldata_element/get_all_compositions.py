#coding = utf-8

import pandas as pd 

data = pd.read_csv('../Alldata_combine/finaldata_all.csv')
compositions = [i for i in data.columns if len(i) <= 2]
data1 = data[compositions + ['label', 'method']]
print (data1.shape)
print (data1.columns)

data1 = data1[data1[compositions].astype(bool).sum(axis = 'columns') == 3]

data1 = data1.loc[:, (data1 != 0).any(axis = 0)]

data1.to_csv('compositions_3.csv', index = False)
print (data1.shape)