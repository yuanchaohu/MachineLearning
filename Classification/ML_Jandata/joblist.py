#coding = utf - 8

import os, sys
import subprocess
import numpy as np 

scorings = ['roc_auc', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'recall_macro', 'f1_macro']

def workdirs(scorings):
    for i in scorings:
        path = i + '/' 
        if not os.path.exists(path):
            os.makedirs(path)

        cmdline = 'cp classification.py ' + path
        subprocess.run(cmdline, shell = True)

workdirs(scorings)

def joblist(scorings, njobs):
    f = open('joblist', 'w')
    for i in scorings:
        path = i + '/'
        cmdline1 = 'cd ' + path + ' ; '
        cmdline2 = 'xvfb-run python classification.py ' + i + ' ' + str(j)
        cmdline  = cmdline1 + cmdline2
        f.write(cmdline + '\n')

    f.close()

#joblist(scorings, njobs)
