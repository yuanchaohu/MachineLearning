#coding = utf-8

import pandas as pd 
import matplotlib.pyplot as plt 

xlabel = 'Measured'
ylabel = 'Predicted'
data1 = pd.read_csv('predictions.csv')
label = 'RMSE= 0.212\n'
label = label + r'$R^2=0.892$'
plt.scatter(data1[xlabel], data1[ylabel], marker = 'o', edgecolor = 'gray', label = label)

data2 = pd.read_csv('predictions.group3.csv')
selected = [4, 53, 20, 11]
markers = ['s', 'v', '<', '>']
full = []
for i, j in enumerate(selected):
    plt.scatter(data2.loc[j-2, xlabel], data2.loc[j-2, ylabel], marker = markers[i], edgecolor = 'red', s = 120)
    full.append(data2.loc[j-2])

plt.plot([-5.5, -1.7], [-5.5, -1.7], '--', lw = 2, c = 'orange')
plt.xlabel('Measured', size = 20)
plt.ylabel('Prediction', size = 20)
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.legend(fontsize = 14)
plt.tight_layout()
plt.savefig('predictions.example.png', dpi = 600)
plt.show()
plt.close()

a = pd.concat(full, axis = 'columns')
a.to_csv('predictions.selected.csv')