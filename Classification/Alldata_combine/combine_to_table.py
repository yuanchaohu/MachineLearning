#coding = utf-8

import numpy as np 
import pandas as pd 
import os, sys

def get_table(datadir = './lookup-data/selected/'):
    """ Combined the elemental properties as a table """

    fulldata = []
    datadirall = os.listdir(datadir)
    for i in datadirall:
        data = pd.read_table(datadir + i, sep = '\s+', header = None, names = [i.split('.t')[0]])
        fulldata.append(data)

    finaldata = pd.concat(fulldata, axis = 'columns')
    finaldata.rename(columns = {'Abbreviation': 'ElementName'}, inplace = True)
    finaldata.replace({'Missing': np.nan}, inplace = True)
    finaldata.dropna(how = 'all', inplace = True)
    print (finaldata.columns)
    finaldata.to_csv('elements_full.csv', index = False)
    print (finaldata.shape)

get_table()