#coding = utf-8

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 


slopes = [-0.22, 0.03, 0.1, 0.13, 0.15, 0.19, 0.22, 0.26, 0.26]
frac = np.linspace(0, 1, 11).round(1)[1:-1]
values = dict(zip(frac, slopes))
print (values)

markers = ['s', 'v', 'X']
j = 0
for i in [6]:
    filename = 'predictions_n' + str(i) + '.csv'
    data = pd.read_csv(filename)
    plt.scatter(data['Measured'], data['Predicted'], label = 'n=' + str(i), marker = markers[j])
    j += 1
    if i == 6:
        print (data['fB'].map(values))
        data['model'] = data['epsilon1'] - data['fB'].map(values) * data['epsilon2']
        data['model'] = -2 * data['model'] ** 2 - 2
        plt.scatter(data['Measured'], data['model'], label = 'formula', marker = markers[j], alpha = 0.75)
        j += 1

plt.xlabel('Measured', size = 20)
plt.ylabel('Prediction', size = 20)
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.legend(fontsize = 14)
plt.tight_layout()
plt.savefig('comparison.png', dpi = 600)
plt.show()
plt.close()
