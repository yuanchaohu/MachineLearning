#coding = utf-8

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

data1 = pd.read_csv('predict.binary.csv')

data2 = pd.read_csv('../../binaries/LJ/parameters.dat', sep = '\s+')
data2.rename(columns = {'elementA': 'A', 'elementB': 'B'}, inplace = True)
data2 = data2[['A', 'B', 'GFA']]
print (data2.head())

data = data1.merge(data2)
data.sort_values(by = 'P_GFA', ascending = False, inplace = True)
print (data.head())
#data.to_csv('GFA.ranks.predicted.csv', index = False)

condition = (data['A'] == 'B') & (data['B'] == 'Fe')
print (data[condition])