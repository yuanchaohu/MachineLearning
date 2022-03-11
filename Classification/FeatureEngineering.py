#coding = utf-8

import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def elementsclean():
    """Clean elemental properties of all possible elements"""

    data1 = pd.read_csv('radius_elements.csv')
    print (data1.head(5))

    data2 = pd.read_excel('elementproperties.xlsx')
    data2.rename(columns = {'Atomic Symbol': 'ElementName', 'Atomic Number': 'ElmentNum', 'Ionic Radius?': 'Ionic Radius'}, inplace = True)

    data = pd.merge(data1, data2, how = 'right')
    #data.dropna(axis = 'columns', inplace = True)
    print(data.columns)
    data.drop(['Metallic Radius (Å)', 'Crystalline Structure', 'Magnetic Properties', 'Metallic Classification', 
               'Electronic shell structure?'], axis = 'columns', inplace = True)
    data.to_csv('elements_full.csv', index = False)
    return data

#elementsclean()

def feature_engineering(alloys, compositions):
    """ Feature engineering for machine learning by giving 

    alloys with compositions as pandas DataFrame 
    compositions is a list of elements from alloys like ['Cu', 'Zr', 'Al']
    """

#------------read-in elemental properties--------------
    elements = pd.read_csv('elements_full.csv')
    elements.set_index('ElementName', inplace = True)

#------------combine elements and compositions---------
    usedelements = elements.index.tolist()
    data1 = pd.concat([alloys, pd.DataFrame(columns = usedelements)], ignore_index=True)
    data1.fillna(0, inplace = True)

#----------------heat of mixing-------------------------
    heatofmixing = pd.read_csv('modified_heatofmixing.csv', index_col = 0)
    for p in data1.index.tolist():
        medium = 0
        for m in compositions:
            for n in compositions:
                medium += data1.loc[p, m] * data1.loc[p, n] * heatofmixing.loc[m, n]
        data1.loc[p, 'heatofmixing'] = medium / 2

#--------------different moments------------------------
    obitals = ['s', 'p', 'd', 'f']
    properties = [x for x in elements.columns if x not in obitals]
    for m in properties:
        medium = data1[compositions].values * ((elements.loc[compositions, m].values)[np.newaxis, :]) #.astype(np.float64)
        ave = medium.sum(axis = 1)
        data1[m + ' mean'] = ave

        medium = data1[compositions].values * np.power((elements.loc[compositions, m].values)[np.newaxis, :], 2) #.astype(np.float64)
        data1[m + ' secondM'] = medium.sum(axis = 1)
        data1[m + ' polydisperity'] = np.sqrt(np.abs((medium.sum(axis = 1) - ave**2) / ave))
        
        medium = data1[compositions].values * np.power((elements.loc[compositions, m].values)[np.newaxis, :], 3) #.astype(np.float64)
        data1[m + ' thirdM'] = medium.sum(axis = 1)

        medium = data1[compositions].values * np.power((elements.loc[compositions, m].values)[np.newaxis, :], 4) #.astype(np.float64)
        data1[m + ' fourthM'] = medium.sum(axis = 1)

        medium = data1[compositions].values * np.power((elements.loc[compositions, m].values)[np.newaxis, :], 5) #.astype(np.float64)
        data1[m + ' fifthM'] = medium.sum(axis = 1)

        medium = data1[compositions].values * np.power((elements.loc[compositions, m].values)[np.newaxis, :], 6) #.astype(np.float64)
        data1[m + ' sixthM'] = medium.sum(axis = 1)

#--------------s p d f valence structure------------------------------
    for m in obitals:
        medium = data1[compositions].values * ((elements.loc[compositions, m].values)[np.newaxis, :]) #.astype(np.float64)
        data1[m + ' mean']   = medium.sum(axis = 1)
        data1[m + ' valences'] = elements.loc[compositions, m].sum()

#-------------component number----------------------------------------
    data1['components'] = len(compositions)

    return data1 


def train_features(datadir = './Hu_category/'):
    """Design features for train set"""

    fulldata = []
    allelement = []
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

        if 'rx' in data.columns:
            data.drop(['rx', 'ry'], axis = 'columns', inplace = True)

        data = data[data['x']**2 + data['y']**2 < 45**2]
        data.drop(['x', 'y'], axis = 'columns', inplace = True)
        compositions = [x for x in data.columns if x not in ['XRDname', 'label']]
        data[compositions] = (data[compositions].values) / (data[compositions].values).sum(axis = 1)[:, np.newaxis]

        data1 = feature_engineering(data, compositions)
        data1.loc[:, (data1 != 0).any(axis = 0)].to_csv(tabledir + i.split('_')[0] + '.csv', index = False)
        fulldata.append(data1)
        allelement.append(compositions)

    finaldata = pd.concat(fulldata, axis = 'index')
    allelement = list(map(str.strip, set(sum(allelement, []))))
    finaldata.dropna(axis = 'columns', inplace = True)
    finaldata = finaldata.loc[:, (finaldata != 0).any(axis = 0)]
    finaldata.to_csv(tabledir + 'finaldata.csv', index = False)
    usedproperties = [x.split(' mea')[0] for x in finaldata.columns.tolist() if x.endswith('mean')] + ['heatofmixing', 'components']
    features = [x for x in finaldata.columns.tolist() if x not in allelement + ['label', 'XRDname']]
    f = open(tabledir + 'features.dat', 'w', encoding = 'utf-8')
    
    f.write(str('\n'.join(usedproperties)) + '\n')
    f.write('\n')
    f.write('%8d   dimensions\n' % len(features))
    f.write(str('\n'.join(features)))
    f.close()

#train_features()

func = lambda x: '%.5f' %x 
data1 = pd.read_csv('features/finaldata.csv')
data2 = pd.read_csv('combined/finaldata0.csv')
#print (data1.columns)
#print (data1[data1['heatofmixing'].map(func) != data2['heatofmixing'].map(func)].index)
print ((data1['Covalent Radius polydisperity'].map(func).values != data2['Covalent Radius mismatch'].map(func).values).sum())
#print ((data1['components'] != data2['components']).sum())


def prediction_features(path = './machinelearning/'):
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
    predictiondata, *rest = feature_engineering(alloys, compositions)

    trainingdata = pd.read_csv('./features/finaldata.csv')
    trainingdata.drop(columns = ['XRDname', 'label'], inplace = True)
    print ((trainingdata.columns != predictiondata.columns).sum())

    predictiondata.to_csv(path + 'predictionfeatures_' + ''.join(compositions) + '.csv', index = False)

#prediction_features()