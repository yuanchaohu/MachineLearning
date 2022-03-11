#coding = utf-8
import os 
import numpy as np 
import pandas as pd 


dir1 = r'D:/GoogleDriver/Yuanchao & Sebastian - MGML/combined/'
dir2 = './features/'

homedirall  = os.listdir(dir2)
results = 0
for i in homedirall:
    if i.endswith('csv'):
        data1 = pd.read_csv(dir1 + i)
        data2 = pd.read_csv(dir2 + i)
        print ((data1 != data2).values.sum())
        results += (data1 != data2).values.sum()
print ('total: ', results)
