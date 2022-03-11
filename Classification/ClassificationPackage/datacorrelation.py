#coding = utf-8

import os, sys 
import numpy  as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 


def data_correlation(corr_matrix):
    """ Get the pearson correlation coefficients between features and label """

    correlations = corr_matrix['label'].sort_values(ascending = False)
    correlations.to_csv('correlations.csv')
    correlations.drop('label', inplace = True)
    correlations.rename(index = lambda x: x.replace(' (kJ/mol)', '\n(kJ/mol)'), inplace = True)
    correlations.rename(index = lambda x: x.capitalize(), inplace = True)

    correlations[:10].sort_values(ascending = True).plot.barh(color = 'b', alpha = 0.7) #abs().sort_values(ascending = False)
    plt.xlabel('Correlation Coefficient')
    plt.title('Correlation with "Label"')
    plt.subplots_adjust(left = 0.35)
    plt.tight_layout()
    plt.savefig('correlation_matrix_positive.png', dpi = 600)
    plt.close()
    correlations[-10:].sort_values(ascending = False).plot.barh(color = 'r', alpha = 0.7) #abs().sort_values(ascending = False)
    plt.xlabel('Correlation Coefficient')
    plt.title('Correlation with "Label"')
    plt.subplots_adjust(left = 0.35)
    plt.tight_layout()
    plt.savefig('correlation_matrix_negative.png', dpi = 600)
    plt.close()
