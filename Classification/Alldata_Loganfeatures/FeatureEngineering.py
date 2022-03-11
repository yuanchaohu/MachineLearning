#coding = utf-8

import os, sys 
import numpy  as np 
import pandas as pd 

sys.path.append(r'D:/GoogleDriver/GFA-MachineLearning/ClassificationPackage/')
from features import features_Hu
from features import features_Logan

def train_features(datadir = '../Hu_category/'):
    """Design features for train set"""

    fulldata = []
    tabledir = './features/'
    if not os.path.exists(tabledir):
        os.makedirs(tabledir)

    datadirall = os.listdir(datadir)
    if 'desktop.ini' in datadirall:
        datadirall.remove('desktop.ini')

    for i in datadirall:
        print (i)
#-------------read composition and label----------------
        if i.endswith('dat'):
            data = pd.read_table(datadir + i, sep = '\s+')
            data.rename(columns = {'Amorphous=1;PartCrystal=2;Crystal=3;NotSure=0': 'label'}, inplace = True)
        elif i.endswith('csv'):
            data = pd.read_csv(datadir + i)
            data.rename(columns = {'Unnamed: 0': 'XRDname', 'classification': 'label', 'X': 'x', 'Y': 'y'}, inplace = True)

        if 'rx' in data.columns: data.drop(['rx', 'ry'], axis = 'columns', inplace = True)
        data = data[data['x']**2 + data['y']**2 < 45**2]
        data.drop(['x', 'y'], axis = 'columns', inplace = True)

        compositions = [x for x in data.columns if x not in ['XRDname', 'label']]
        data[compositions] = (data[compositions].values) / (data[compositions].values).sum(axis = 1)[:, np.newaxis]
        
        #***************************************************************
        #------------------use Hu features--------------------------
        # file1 = '../elements_full.csv'
        # file2 = '../modified_heatofmixing.csv'
        # data1 = features_Hu(data, compositions, file1, file2)
        #-----------------use Logan features------------------------
        file1 = './elements_full.csv'
        data1 = features_Logan(data, compositions, file1)
        #***************************************************************

        data1.to_csv(tabledir + i.split('_')[0] + '.csv', index = False)
        fulldata.append(data1)

    finaldata = pd.concat(fulldata, axis = 'index')
    allelement = [i for i in finaldata.columns if len(i) <= 2]
    finaldata[allelement] = finaldata[allelement].fillna(0)
    finaldata.to_csv(tabledir + 'finaldata.csv', index = False)

    features = [x for x in finaldata.columns.tolist() if x not in allelement + ['label', 'XRDname']]
    f = open(tabledir + 'features.dat', 'w', encoding = 'utf-8')
    f.write('%8d   dimensions\n' % len(features))
    f.write(str('\n'.join(features)))
    f.close()

train_features()


def prediction_features(path = './machinelearning/', elementspath = '../'):
    """
    Create features for machine learning predictions on new instances

    The features should be the same as the training set
    """

    num = 300
    compt1 = np.linspace(0.05, 0.95, num)[np.random.permutation(num)]
    compt2 = np.linspace(0.05, 0.95, num)[np.random.permutation(num)]
    compt3 = 1 - compt1 - compt2
    alloys = {'Cu': compt1, 'Zr': compt2, 'Al': compt3}
    alloys = pd.DataFrame.from_dict(alloys)
    compositions = alloys.columns.tolist()
    alloys[compositions] = np.abs(alloys.values) / np.abs(alloys.values).sum(axis = 1)[:, np.newaxis]
    predictiondata, *rest = feature_engineering(alloys, compositions, elementspath)

    trainingdata = pd.read_csv('./features/finaldata.csv')
    trainingdata.drop(columns = ['XRDname', 'label'], inplace = True)
    print ((trainingdata.columns != predictiondata.columns).sum())

    predictiondata.to_csv(path + 'predictionfeatures_' + ''.join(compositions) + '.csv', index = False)

#prediction_features()