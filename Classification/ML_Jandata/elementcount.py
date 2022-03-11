#coding = utf - 8

import os, sys
import subprocess
import numpy as np 
import pandas as pd 

elements = ['Ag', 'Al', 'Au', 'Co', 'Cr', 'Cu', 'Fe', 'Ge', 'Mg', 'Mn', 'Nb', 'Ni', 'Pd', 'Si', 'V', 'W', 'Y']
data = pd.read_csv('finaldata.csv')
#f = open('elementscount.dat', 'w')
#f.write('element  fraction\n')
fraction = []
for i in elements:
    frac = data[i].nonzero()[0].shape[0] / data.shape[0]
    #f.write(i + ' %.4f\n' %frac)
    fraction.append(frac)
#f.close()
pd.Series(dict(zip(elements, fraction))).sort_values(ascending = False).to_csv('elementscount.csv')
