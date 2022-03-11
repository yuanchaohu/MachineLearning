#coding = utf-8

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

slopes = [-0.22, 0.03, 0.1, 0.13, 0.15, 0.19, 0.22, 0.26, 0.26]
frac = np.linspace(0, 1, 11).round(1)[1:-1]
values = dict(zip(frac, slopes))
print (values)

data = pd.read_csv('testset.predictions.csv')
print (data.shape, data.head())
xdata = data['epsilon1'] - data['epsilon2'] * data['fB'].map(values)
ydata = data['predicted']

plt.scatter(xdata, ydata, edgecolor = 'red', alpha = 0.5)
xlabel = r'$\frac{\epsilon_{BB} - \epsilon_{AA} - 2\epsilon_{AB}k(f_B)}{\epsilon_{AA} + \epsilon_{BB}}$'

#-----------
data = pd.read_csv('../../fulldata.dat', sep = '\s+')
data = data[data['logRc'] > -6]
data = data[data['group'] == 1]
data['epsilon1'] = (data['eBB'] - data['eAA']) / (data['eBB'] + data['eAA'])
data['epsilon2'] = 2 * data['eAB'] / (data['eBB'] + data['eAA'])
xdata = data['epsilon1'] - data['epsilon2'] * data['fB'].map(values)
ydata = data['logRc']
plt.scatter(xdata, ydata, c = 'red')

plt.xlabel(xlabel, size = 20)
plt.ylabel(r'$Predicted log_{10}R_c$', size = 20)
plt.xticks([-1.4, -1.0, -0.5, 0, 0.5], size = 18)
plt.yticks(Size = 18)
plt.xlim(-1.4, 0.7)
plt.ylim(-15, 0)
plt.tight_layout()
plt.savefig('predictions.png', dpi = 600)
plt.show()
plt.close()