#coding = utf-8

import numpy as np 

def create_xtest():
    #get known data input
    data = np.loadtxt('voidsnumber.dat', skiprows = 1)
    print (data.shape)

    e1 = np.linspace(-0.9, 0, 100).round(3)
    e2 = np.linspace(1.0, 5, 100).round(3)
    fb = np.linspace(0.1, 0.9, 9).round(1)
    data = np.zeros((100*100*9, 3))
    n = 0
    for i in fb:
        for j in e1:
            for k in e2:
                data[n, 0] = j 
                data[n, 1] = k
                data[n, 2] = i 
                n += 1

    np.savetxt('testset.dat', data, fmt = '%.3f %.3f %.2f', header = 'epsilon1 epsilon2 fB', comments = '')
create_xtest()