# %%
# Load raw files and perform feature extraction

import numpy as np
import scipy as sp
import pandas as pd

from scipy import signal, arange, fft, fromstring, roll
from scipy.signal import butter, lfilter, ricker
import os
import glob
import re

from scipy.stats import stats
from utilityFunctions import featureExtraction, ghostFeatures, ghostHeap, ghostVector



print('Setting parameters...')
N = 4  # value for N-fold cross-validation
SAMPLE_RATE = 250  # Hz (subject to change)
fs=SAMPLE_RATE
NUM_WINDOWS = 2  # Dependent on number of samples of phonemes
lowcut=1
highcut=(np.floor(SAMPLE_RATE/2))
pcti = 99.95
phoLimit=int(44)
triLimit=int(4)
triMin=int(1)
# initialize script

featureList=list()
phonemeList=list()


subNames=list(['GTP000','GTP003','GTP004','GTP007','GTP044','GTP045','GTP047','GTP104','GTP223','GTP303','GTP308','GTP455','GTP545','GTP556','GTP762','GTP765'])
subName='GTP007'

#subName='GTP000'
dirSub='./StimPres/gtData/'+subName +'/'


for phoNum in range(0,phoLimit):
    for triNum in range(triMin,triLimit):
        if subName=='GTP000':
        	nameFile=dirSub+'DLR' + '_' + str(int(phoNum)) + '_' + str(int(triNum)) + '.txt'
        else:
        	nameFile=dirSub+subName + '_' + str(int(phoNum)) + '_' + str(int(triNum)) + '.txt'

        if subName=='GTP007':
        	nameFile=dirSub+'GT007' + '_' + str(int(phoNum)) + '_' + str(int(triNum)) + '.txt'
        else:
        	nameFile=dirSub+subName + '_' + str(int(phoNum)) + '_' + str(int(triNum)) + '.txt'
        print(nameFile)
#rawData = np.genfromtxt(nameFile, delimiter='/t')



       # print(totalL)
        try:
        	df = pd.read_csv(nameFile, sep='\t', header=None)
        	rawData=df.to_numpy()
        	lastRow=np.squeeze(rawData[:,31])
        	h0=np.unique(lastRow)
        	totalL=np.shape(h0)[0]
        	totalL=int(totalL)
        	for aa in range(1,totalL):
        		indVal=np.where(lastRow==(aa))
        		featureVector=ghostHeap(rawData, indVal, fs, lowcut, highcut, pcti, NUM_WINDOWS)
        		print(aa)
        		featureList.append(featureVector)
        		featureVector=list()
        		phonemeList.append(str(phoNum))
        except:
        	print('No epochs.')
      #  h1=np.where(lastRow==1)
      #  h2=np.where(lastRow==2)
      #  h3=np.where(lastRow==3)
      #  h4=np.where(lastRow==4)
      #  h5=np.where(lastRow==5)

     #   featureVector=ghostVector(rawData, fs, lowcut, highcut, pcti, h1, h2, h3, h4, h5, NUM_WINDOWS)
     #   featureList.append(featureVector)
     #   phonemeList.append(str(phoNum))
       # print(np.shape(featureVector))



labels_vector = np.array(phonemeList)
df = pd.DataFrame(featureList)
outNamData=subName+'_Data.csv'
outNamLabels=subName+'_Labels.csv'

df.to_csv(outNamData,sep=',')

np.savetxt(outNamLabels, labels_vector, delimiter=",", fmt='%s')
# %%
