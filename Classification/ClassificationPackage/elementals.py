#coding = utf-8

import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def elementsclean(path):
    """Clean elemental properties of all possible elements"""

    data1 = pd.read_csv(path + 'radius_elements.csv')
    print (data1.head(5))

    data2 = pd.read_excel(path + 'elementproperties.xlsx')
    data2.rename(columns = {'Atomic Symbol': 'ElementName', 'Atomic Number': 'ElmentNum', 'Ionic Radius?': 'Ionic Radius'}, inplace = True)

    data = pd.merge(data1, data2, how = 'right')
    #data.dropna(axis = 'columns', inplace = True)
    print(data.columns)
    data.drop(['Metallic Radius (Å)', 'Crystalline Structure', 'Magnetic Properties', 'Metallic Classification', 
               'Electronic shell structure?'], axis = 'columns', inplace = True)
    data.to_csv(path + 'elements_full.csv', index = False)
    return data

#elementsclean()