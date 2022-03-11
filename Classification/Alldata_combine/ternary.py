#coding = utf-8

import os, sys 
import numpy  as np 
import pandas as pd 

data = pd.read_csv('finaldata_all.csv')
print (data.shape)

compositions = [i for i in data.columns if len(i) <= 2]
data = data[data[compositions].astype(bool).sum(axis = 'columns') == 3]
data = data.loc[:, (data != 0).any(axis = 0)]
print (data.shape)

data.dropna(axis = 'index', inplace = True)
data.drop(columns = 'components', inplace = True)
print (data.shape)

data.to_csv('finaldata_ternary.csv', index = False)