#coding = utf-8

import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def cleanstatistics():
    datadir = './Hu_category/'
    datadirall = os.listdir(datadir)
    if 'desktop.ini' in datadirall:
        datadirall.remove('desktop.ini')

    appended_data = []
    for i in datadirall:
        if i.endswith('dat'):
            data = pd.read_table(datadir + i, sep = '\s+')
            data.rename(columns = {'Amorphous=1;PartCrystal=2;Crystal=3;NotSure=0': 'label'}, inplace = True)
        elif i.endswith('csv'):
            data = pd.read_csv(datadir + i)
            data.rename(columns = {'Unnamed: 0': 'XRDname', 'classification': 'label', 'X': 'x', 'Y': 'y'}, inplace = True)

        if 'rx' in data.columns:
            data.drop(['rx', 'ry'], axis = 'columns', inplace = True)

        counts = data['label'].value_counts()
        counts.sort_index(inplace = True)

        counts['system'] = i.split('_')[0]
        numelement = len(data.columns) - 4
        counts['numelement'] = numelement
        counts['elements'] = ''.join(data.columns[3:-1])
        appended_data.append(counts.to_frame().T)

    appended_data = pd.concat(appended_data, axis = 'index')
    appended_data.fillna(0, inplace = True)
    appended_data.reindex(appended_data['elements'].str.len().sort_values().index)
    appended_data.to_csv('statistics.csv', index = False)

cleanstatistics()

def counts():
    f = open('statistics.dat', 'w')
    data = pd.read_csv('statistics.csv')
    print (data.columns)
    totalnum = data.iloc[:, :4].sum()
    validnum = data.iloc[:, :4][data['elements'] != 'AlCuFe'].sum()
    invalid  = data.iloc[:, :4][data['elements'] == 'AlCuFe'].sum()
    print ()
    f.write('total num = %6d' %totalnum.sum() + '\n')
    f.write('valid num = %6d , %6.3f' %(validnum.sum(), validnum.sum() / totalnum.sum()) + '\n')
    f.write('invalid num = %6d, %6.3f' %(invalid.sum(), invalid.sum() / totalnum.sum()) + '\n')
    f.write('\n')
    totalfrac = totalnum /totalnum.sum()    
    f.write('total: \n' + pd.concat([totalnum.to_frame(), totalfrac.to_frame()], axis = 1).to_string(header = ['num', 'frac']) + '\n')

    f.write('\n')
    validfrac = validnum / validnum.sum()
    f.write('valid: \n' + pd.concat([validnum.to_frame(), validfrac.to_frame()], axis = 1).to_string(header = ['num', 'frac']) + '\n')

    f.write('\n')
    invalidfrac = invalid / invalid.sum()
    f.write('invalid: \n' + pd.concat([invalid.to_frame(), invalidfrac.to_frame()], axis = 1).to_string(header = ['num', 'frac']) + '\n')

    #f.write('\nbinary\n')
    #print (data.iloc[:, :4][(data['elements'] != 'AlCuFe') & (data['numelement'] == 2)].sum())

    f.close()

counts()