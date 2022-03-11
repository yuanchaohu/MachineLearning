#coding = utf-8

import os, math 
import numpy as np 
import pandas as pd 

def features_Hu(alloys, compositions, file1 = './elements_full.csv', file2 = './modified_heatofmixing.csv'):
    """ Feature engineering for machine learning by giving 

    alloys with compositions as pandas DataFrame 
    compositions is a list of elements from alloys like ['Cu', 'Zr', 'Al']
    """

#------------read-in elemental properties--------------
    elements = pd.read_csv(file1)
    elements.set_index('ElementName', inplace = True)

#------------combine elements and compositions---------
    # usedelements = elements.index.tolist()
    # data1 = pd.concat([alloys, pd.DataFrame(columns = usedelements)], ignore_index=True)
    # data1.fillna(0, inplace = True)
    data1 = alloys
#----------------heat of mixing-------------------------
    heatofmixing = pd.read_csv(file2, index_col = 0)
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
        if ave != 0:
            data1[m + ' polydisperity'] = np.sqrt(np.abs((medium.sum(axis = 1) - ave**2) / ave).astype(np.float64))
        else:
            data1[m + ' polydisperity'] = np.nan
            
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



def norms(data, n):
    """Compute different norms"""

    if data.shape[1] > 1:
        return np.power(np.power(np.abs(data), n).sum(axis = 1), 1./n)
    else:
        return np.power(np.power(np.abs(data), n).sum(), 1./n)

def features_Logan(alloys, compositions, file1 = './element_full.csv'):
    """ Design features from Logan's Paper """

    #---------------read in elemental properties-------------------
    elements = pd.read_csv(file1)
    elements.set_index('ElementName', inplace = True)

    #---------------combine elements and compositions--------------
    # usedelements = elements.index.tolist()
    # data1 = pd.concat([alloys, pd.DataFrame(columns = usedelements)], ignore_index=True)
    # data1.fillna(0, inplace = True)
    data1 = alloys
    #----------stoichiometric attributes---------
    data1['components'] = len(compositions)
    for i in [2, 3, 5, 7, 10]:
        data1['norm_' + str(i)] = norms(data1[compositions], i)

    #----------elemental property statistics-------
    properties = [i for i in elements.columns if not (i.endswith('Valence') | i.endswith('Unfilled'))]
    for m in properties:
        medium = elements.loc[compositions, m].values
        data1[m + ' min']   = medium.min()
        data1[m + ' max']   = medium.max()
        data1[m + ' range'] = medium.max() - medium.min()
        data1[m + ' mode']  = medium[data1[compositions].values.argmax(axis = 1)]

        #-------statistical average and deviation-------------------
        ave    = (data1[compositions].values * medium[np.newaxis, :]).sum(axis = 1)
        data1[m + ' mean'] = ave
        avedev = (data1[compositions].values * np.abs(medium[np.newaxis, :] - ave[:, np.newaxis])).sum(axis = 1)
        data1[m + ' Ave_Dev'] = avedev

    #----------valence shell attributes-------------
    for j in ['Valence', 'Unfilled']:
        for m in ['s', 'p', 'd', 'f']:
            name = 'N' + m + j 
            data1[name + ' sum'] = elements.loc[compositions, name].sum()
        data1[j + ' sum'] = elements.loc[compositions, 'N' + j].sum()

    for m in ['s', 'p', 'd', 'f']:
        medium1 = data1[compositions].values * (elements.loc[compositions, 'N' + m + 'Valence'].values)[np.newaxis, :]
        medium2 = data1[compositions].values * (elements.loc[compositions, 'NValence'].values)[np.newaxis, :]
        data1['frac_' + m] = medium1.sum(axis = 1) / medium2.sum(axis = 1)

    #----------ionicity attributes------------------
    for p in data1.index:
        medium1 = 0
        medium2 = []
        for m in compositions:
            for n in compositions:
                if m != n:
                    ionic_character = 1 - np.exp(-0.25 * np.square(elements.loc[m, 'Electronegativity'] - elements.loc[n, 'Electronegativity']))
                    medium2.append(ionic_character)
                    medium1 += data1.loc[p, m] * data1.loc[p, n] * ionic_character

        if len(medium2) > 0: data1.loc[p, 'maxionic']  = max(medium2)
        data1.loc[p, 'meanionic'] = medium1 / 2.

    return data1


def features_combine(alloys, compositions, file1 = './element_full.csv', file2 = './modified_heatofmixing.csv'):
    """ Design features by combining Hu and Logan features 

    alloys with compositions as pandas DataFrame 
    compositions is a list of elements from alloys like ['Cu', 'Zr', 'Al']
    """

    #---------------read in elemental properties-------------------
    elements = pd.read_csv(file1)
    elements.set_index('ElementName', inplace = True)

    #------------combine elements and compositions---------
    # usedelements = elements.index.tolist()
    # data1 = pd.concat([alloys, pd.DataFrame(columns = usedelements)], ignore_index=True)
    # data1.fillna(0, inplace = True)
    data1 = alloys

    #----------stoichiometric attributes---------
    data1['components'] = len(compositions)
    for i in [2, 3, 4]:
        data1['norm_' + str(i)] = norms(data1[compositions], i)

    #----------------heat of mixing-------------------------
    heatofmixing = pd.read_csv(file2, index_col = 0)
    for p in data1.index.tolist():
        medium = 0
        for m in compositions:
            for n in compositions:
                medium += data1.loc[p, m] * data1.loc[p, n] * heatofmixing.loc[m, n]
        data1.loc[p, 'heatofmixing'] = medium / 2

    #----------elemental property statistics-------
    properties = [i for i in elements.columns if not (i.endswith('Valence') | i.startswith('IsM'))]
    for m in properties:
        medium  = elements.loc[compositions, m].values.astype(np.float64)
        
        ave     = (data1[compositions].values * medium[np.newaxis, :]).sum(axis = 1)
        data1[m + ' mean'] = ave

        avedev  = (data1[compositions].values * np.abs(medium[np.newaxis, :] - ave[:, np.newaxis])).sum(axis = 1)
        data1[m + ' Ave_Dev'] = avedev

        medium2 = (data1[compositions].values * np.power(medium[np.newaxis, :], 2)).sum(axis = 1)
        data1[m + ' mean2'] = medium2
        data1[m + ' polydispersity'] = np.nan
        if len(ave) == 1:
            if ave != 0: data1.loc[:, m + ' polydispersity'] = math.sqrt(abs(medium2 - ave**2)) / ave
        else:
            data1.loc[:, m + ' polydispersity'][ave != 0] = np.sqrt(np.abs(medium2[ave != 0] - ave[ave != 0]**2).astype(np.float64)) / ave[ave != 0] #if ave != 0 else np.nan

    #----------valence shell attributes-------------
    for j in ['Valence']:
        for m in ['s', 'p', 'd', 'f']:
            name = 'N' + m + j 
            data1[name + ' sum'] = elements.loc[compositions, name].sum()
        data1[j + ' sum'] = elements.loc[compositions, 'N' + j].sum()

    for m in ['s', 'p', 'd', 'f']:
        medium1 = data1[compositions].values * (elements.loc[compositions, 'N' + m + 'Valence'].values)[np.newaxis, :]
        medium2 = data1[compositions].values * (elements.loc[compositions, 'NValence'].values)[np.newaxis, :]
        data1['frac_' + m] = medium1.sum(axis = 1) / medium2.sum(axis = 1)

    #----------ionicity attributes------------------
    for p in data1.index:
        medium1 = 0
        medium2 = []
        for m in compositions:
            for n in compositions:
                if m != n:
                    ionic_character = 1 - np.exp(-0.25 * np.square(elements.loc[m, 'Electronegativity'] - elements.loc[n, 'Electronegativity']))
                    medium2.append(ionic_character)
                    medium1 += data1.loc[p, m] * data1.loc[p, n] * ionic_character

        if len(medium2) > 0: data1.loc[p, 'maxionic']  = max(medium2)
        data1.loc[p, 'meanionic'] = medium1 / 2.

    #---------fraction of metal and metalloid in composition------------
    for m in ['IsMetal', 'IsMetalloid']:
        data1['frac_' + m] = (data1[compositions].values * (elements.loc[compositions, m].values)[np.newaxis, :]).sum(axis = 1) #.astype(np.float64)

    return data1