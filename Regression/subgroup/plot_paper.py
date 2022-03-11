#coding = utf-8

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

slopes = [-0.22, 0.03, 0.1, 0.13, 0.15, 0.19, 0.22, 0.26, 0.26]
frac = np.linspace(0, 1, 11).round(1)[1:-1]
values = dict(zip(frac, slopes))
print (values)

data = pd.read_csv('predictions_n6.csv')
print (data.head())

xdata = data['epsilon1'] - data['epsilon2'] * data['fB'].map(values)
ydata = data['Predicted']
plt.scatter(xdata, ydata, label = 'ML model (6)')

ydata = -2 * xdata**2 - 2
plt.scatter(xdata, ydata, label = 'formula')

ydata = data['Measured']
plt.scatter(xdata, ydata, label = 'real')

xlabel = r'$\frac{\epsilon_{BB} - \epsilon_{AA} - 2\epsilon_{AB}k(f_B)}{\epsilon_{AA} + \epsilon_{BB}}$'
plt.xlabel(xlabel, size = 20)
plt.ylabel(r'$log_{10}R_c$', size = 20)
plt.xticks([-1, -0.5, 0, 0.5], size = 18)
plt.yticks(Size = 18)

plt.legend()
plt.tight_layout()
plt.savefig('compare.predictions.png', dpi = 600)
plt.show()
plt.close()