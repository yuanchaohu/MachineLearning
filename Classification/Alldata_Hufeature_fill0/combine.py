#coding = utf-8

import os 
import numpy as np 
import pandas as pd

fileoutput = open('Output.dat', 'w') 

#------------read first database-------------------
data1 = pd.read_csv('../ML_JANdata/finaldata.csv')
data1['XRDname'] = 'JanGroup'
data1 = data1[(data1['label'] == 1) | (data1['label'] == 3)]
features = [i for i in data1.columns.tolist() if len(i) > 2]
print ('Jan data: ', data1.shape, file = fileoutput)
with open('features.dat', 'w') as f:
    f.write('Num: ' + str(len(features)) + '\n' * 3)
    f.write('\n'.join(features))

#------------read second database-------------------
data2 = pd.read_csv('../hudata/Hu_final.csv')
data2['XRDname'] = 'HuCollection'
print ('Hu data: ', data2.shape, file = fileoutput)

#------------read third database-------------------
data3 = pd.read_csv('../LoganMG/useddata/combined.csv')
#data3['XRDname'] = 'Logan'
print ('Logan data: ', data3.shape, file = fileoutput)

#------------combine databases-------------------
data = pd.concat([pd.concat([data1, data2], axis = 'index'), data3], axis = 'index')
elements = [i for i in data.columns if len(i) <= 2]
data[elements] = data[elements].fillna(0)
print ('concat data', data.shape, file = fileoutput)
with open('elements.dat', 'w') as f:
    f.write('Num: ' + str(len(elements)) + '\n' * 3)
    f.write('\n'.join(elements))

#------------select features-------------------
usedfeatures = elements + features
data = data[usedfeatures]
print ('After choosing features: ', data.shape, file = fileoutput)

print ('duplicates: ', data.duplicated(subset = elements).sum(), file = fileoutput)
data.drop_duplicates(subset = elements, inplace = True)
print ('After Removing duplicates: ', data.shape, file = fileoutput)

print (data['XRDname'].value_counts() / data.shape[0], file = fileoutput)
print (data['label'].value_counts() / data.shape[0], file = fileoutput)


data.drop(['XRDname'], axis = 'columns', inplace = True)
data['label'].where(data['label'] == 1, 0, inplace = True)

f = open('data_information.dat', 'w')
medium = data['label'].value_counts()
f.write('used labels \n' + pd.concat([medium.to_frame(), (medium / medium.sum()).to_frame()], axis = 1).to_string(header = ['num', 'frac']) + '\n')
f.write('1: amorphous\n0: crystallized\n')
f.write('\n')
f.write('instances: %d;  features:  %d\n' % (data.shape[0], data.shape[1] - 1))
f.write('\n')
medium = data['components'].value_counts()
f.write('components \n' + pd.concat([medium.to_frame(), (medium / medium.sum()).to_frame()], axis = 1).to_string(header = ['num', 'frac']) + '\n')
f.write('\n')
f.close()

data.reset_index(drop = True, inplace = True)   
#------remove columns with too many missing values-------
num_null = data.isnull().sum(axis = 'index')
null_index = num_null[num_null > data.shape[0] * 0.10].index.tolist()
print ('\nFeatures with too many NULL:\n', '\n'.join(null_index), file = fileoutput)
usedfeatures = [i for i in data.columns if i not in null_index]
data = data[usedfeatures]
print ('\nFinal Data Shape: ', data.shape, file = fileoutput)

data.fillna(0, inplace = True)

data.to_csv('finaldata_all.csv', index = False)
fileoutput.close()