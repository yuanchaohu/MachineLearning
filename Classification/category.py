#!/usr/bin/python
# coding = utf-8

docstr = """
        This program is designed to classify the metallic alloys into 
        the following categories based the XRD data
        ------------------------Amorphous: 1-----------------------
        --------------------Partially Crystalline: 2---------------
        ------------------------Crystalline: 3---------------------
        -----------Yuanchao Hu: ychu0213@gmail.com-----------------
        """

import os, time  
import numpy  as np 
import pandas as pd 
import matplotlib.pyplot as plt 


num_file = 4     #number of files
ras_file = 1    #set ras_file = 1 if the file format is ras, or else 0
filedir  = r'./Sample Data/'  #target file dir, do not forget the last /
outdir   = r'./category/'     #output file dir
outfile  = 'test.dat' #better to be named with number in order
if not os.path.exists(outdir):
    os.makedirs(outdir)

output   = np.zeros(num_file) #get the category results


#--------------read ras file---------------
if ras_file == 1:
    headers = 354   #headers of the file
    footers = 2     #footers of the file
    sharename =  '20170323no1_001' #the common part of the files
    for i in range(num_file):
        filenum  = str(i + 7).zfill(3) #in the format of 001
        filename = sharename + filenum + '.ras'
        data     = np.genfromtxt(filedir + filename, skip_header = headers, skip_footer = footers)
        plt.plot(data[:, 0], data[:, 1], color = 'red', linewidth = 2, label = 'No.' + filenum)
        plt.xlabel('theta')
        plt.ylabel('intensity')
        plt.xlim(0, 90)
        plt.legend(loc = 1, fontsize = 16)
        plt.title(filename)
        plt.show()
        plt.close()

        result = int(input('please input the category for ' + filenum + ':'))
        if 0 < result < 4: 
            output[i] = result
        else:
            print ('********Please correct your classification********')
            time.sleep(5) #pause 5 secends
            plt.plot(data[:, 0], data[:, 1], color = 'blue', linewidth = 2, label = 'No.' + filenum)
            plt.xlim(0, 90)
            plt.legend(loc = 1, fontsize = 16)
            plt.title(filename)
            plt.show()
            plt.close()
            output[i] = int(input('please correct the category for ' + filenum + ':'))

    names = 'Amorphous=1; PartCrystal=2; Crystal=3'
    np.savetxt(outdir + outfile, output[:, np.newaxis], fmt = '%1d', header = names, comments = '')


#--------------read csv file---------------
else:
    sharename = '20170922No2_t30_0'
    for i in range(num_file):
        filenum  = str(i + 1).zfill(3) #in the format of 001
        filename = sharename + filenum + '_1D.csv' 
        data     = pd.read_csv(filedir + filename, header = None) 
        data     = np.array(data)
        plt.plot(data[:, 0], data[:, 1], color = 'red', linewidth = 2, label = 'No.' + filenum)
        plt.xlabel('theta')
        plt.ylabel('intensity')
        plt.xlim(0, 90)
        plt.legend(loc = 1, fontsize = 16)
        plt.title(filename)
        plt.show()
        plt.close()

        result = int(input('please input the category for ' + filenum + ':'))
        if 0 < result < 4: 
            output[i] = result
        else:
            print ('********Please correct your classification********')
            time.sleep(5) #pause 5 secends
            plt.plot(data[:, 0], data[:, 1], color = 'blue', linewidth = 2, label = 'No.' + filenum)
            plt.xlim(0, 90)
            plt.legend(loc = 1, fontsize = 16)
            plt.title(filename)
            plt.show()
            plt.close()
            output[i] = int(input('please correct the category for ' + filenum + ':'))

    names = 'Amorphous=1; PartCrystal=2; Crystal=3'
    np.savetxt(outdir + outfile, output[:, np.newaxis], fmt = '%1d', header = names, comments = '')

print ('Congratulations! All work done')