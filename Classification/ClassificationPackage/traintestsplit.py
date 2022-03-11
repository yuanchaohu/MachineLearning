#coding = utf-8

import os, sys 
import numpy  as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 

def design_train_test(data, test_size = 0.30):
    """
    split train and test sets from the database
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import train_test_split

    split = StratifiedShuffleSplit(n_splits = 10, test_size = test_size, random_state = 42)
    for train_index, test_index in split.split(data, data['label']):
        train_set = data.loc[train_index]
        test_set  = data.loc[test_index]

    choosetrain = train_set['label'].value_counts() / train_set.shape[0]
    print ('train set\n', choosetrain)
    choosetest  = test_set['label'].value_counts() / test_set.shape[0]
    print ('test set\n', choosetest)

    return train_set, test_set

