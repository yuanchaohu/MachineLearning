#coding = utf-8

import os, sys 
import numpy  as np 
import pandas as pd 

sys.path.append(r'D:/GoogleDriver/GFA-MachineLearning/ClassificationPackage/')
from features import features_elements

alloys = pd.read_csv('./compositions_3.csv')
print (alloys.shape)

data = features_elements(alloys, n = 3, file1 = './elements_full.csv')
print (data.shape)
data.to_csv('finaldata_all3raw.csv', index = False)

data.dropna(axis = 'index', inplace = True)
print (data.shape)

data = data.loc[:, (data != 0).any(axis = 0)]

usedelements = [i for i in data.columns if len(i) <= 2]
with open('usedelements.dat', 'w') as f:
    f.write('Element Num: %d\n\n' %len(usedelements))
    f.write('\n'.join(usedelements))

data.to_csv('finaldata_all3.csv', index = False)
print (data.shape)