#coding = utf-8

import numpy as np 
import matplotlib.pyplot as plt 

data = np.loadtxt('output.dat', skiprows = 10)
data = data.ravel()[:-1]
print (data)

def plot():
    xdata = range(data.shape[0])
    ydata = np.sort(np.abs(data))
    plt.plot(xdata, ydata, 'o')
    plt.yscale('log')
    plt.show()
    plt.close()

    condition = ydata > 0.1
    print (condition.sum())

plot()

data = np.abs(data)
a = data.argsort()[::-1]
new = np.column_stack((a, data[a]))
np.savetxt('coeff.dat', new, fmt = '%d %.6f', header = 'i c')