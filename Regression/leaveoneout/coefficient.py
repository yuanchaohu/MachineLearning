#coding = utf-8

import numpy as np 
import matplotlib.pyplot as plt 

data = np.loadtxt('output.dat', skiprows = 59)
new = [1.65641245e-02, -2.31061241e-03]
data = np.hstack((data.ravel(), np.array(new)))
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

#plot()

data = np.abs(data)
a = data.argsort()[::-1]
new = np.column_stack((a, data[a]))
np.savetxt('coeff.dat', new, fmt = '%d %.6f', header = 'i c')