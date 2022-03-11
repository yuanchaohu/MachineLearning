#!/usr/bin/python
# coding = utf-8

docstr = """
        This program is designed to classify the metallic alloys into 
        the following categories based the XRD data

        Amorphous: 1
        Partially Crystalline: 2
        Crystalline: 3
        Not sure: 0

        """
print (docstr)

import os, time  
import numpy  as np 
import pandas as pd 
import matplotlib.pyplot as plt 

outputdir  = './category/'
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

homedir     = './DownloadedRawData/'
homedirall  = os.listdir(homedir)
if 'desktop.ini' in homedirall:
    homedirall.remove('desktop.ini')
print ('Folder Number: ', len(homedirall))
#print (homedirall)
xrdfilenum = 0

for i in homedirall[:1]:
#------get the composition of each XRD pattern--------------
    edxdatafile = homedir + i + '/EDXdata.csv'
    edxdataall  = pd.read_csv(edxdatafile, index_col = 0)
    elements    = '  '.join(list(edxdataall.columns[:-1]))
    print ('Folder: ', i)
    print ('Elements: ', elements)

#------analyze the XRD patterns-----------------------------
    xrddatadir  = homedir + i + '/XRDdata/'
    xrdfileall  = os.listdir(xrddatadir)
    if 'desktop.ini' in xrdfileall:
        xrdfileall.remove('desktop.ini')
    print ('XRD pattern Number in "' + i + '" Folder is ', len(xrdfileall))
    xrdfilenum += len(xrdfileall)
    #print (xrdfileall)
    print ('---------------------------------------')

    outputfile = i + '_classify.dat'
    f = open(outputdir + outputfile, 'w')
    f.write('XRDname | ' + elements + ' | Amorphous=1;PartCrystal=2;Crystal=3;NotSure=0\n')
    if len(xrdfileall) != 177:
        for j in xrdfileall[:3]:
            xrddata = pd.read_csv(xrddatadir + j, index_col = 0)
            plt.plot(xrddata.iloc[:, 0], xrddata.iloc[:, 1], color = 'red', linewidth = 1, label = j)
            plt.xlabel(r'$2\theta$')
            plt.ylabel('intensity (a.u.)')
            plt.xlim(0, 90)
            plt.legend(loc = 1, fontsize = 12)
            plt.title('From folder: ' + i, size = 12)
            #plt.savefig(homedir + i + '/' + j + '.png', dpi = 300)
            plt.show()
            plt.close()

            result = input('please input the category for ' + j + ' : ')
            xrdname = j.split('.')[0]
            composition = edxdataall[edxdataall['xrdId'] == xrdname].iloc[:, :-1]
            composition = composition.to_string(header = False, index = False)
            if 0 <= int(result) < 4: 
                f.write(xrdname + '   ' + composition + '   ' + result + '\n')
            else:
                print ('********Please correct your classification********')
                time.sleep(5) #pause 5 secends
                plt.plot(xrddata.iloc[:, 0], xrddata.iloc[:, 1], color = 'blue', linewidth = 1, label = j)
                plt.xlabel(r'$2\theta$')
                plt.ylabel('intensity (a.u.)')
                plt.xlim(0, 90)
                plt.legend(loc = 1, fontsize = 12)
                plt.title('From folder: ' + i, size = 12)
                #plt.savefig(homedir + i + '/' + j + '.png', dpi = 300)
                plt.show()
                plt.close()
                result = input('please correct the category for ' + j + ' : ')
                f.write(xrdname + '   ' + composition + '   ' + result + '\n')

    f.close()
    print ('-----relax 10 seconds THEN move to the next folder------')
    print (' ')
    time.sleep(10)

print ('Congratulations! All work done')