#coding = utf-8

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

data = pd.read_csv('performance.dat', sep = '\s+')
print (data.head())

plt.figure(figsize = (8, 6))

plt.subplot(121)
plt.plot(data['n'], data['RMSE'], '-o')
plt.ylabel('RMSE')

plt.subplot(122)
plt.plot(data['n'], data['R2'], '-s')
plt.ylabel('R2')

plt.tight_layout()
plt.show()
plt.close()