# %%
# Imports
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal, arange, fft, fromstring, roll
from scipy.signal import butter, lfilter, ricker
import os
import glob
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn import svm

from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, cross_validate, train_test_split
from sklearn import metrics, linear_model, preprocessing
from sklearn.cluster import DBSCAN
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer, classification_report
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from scipy.stats import stats
from utilityFunctions import featureExtraction, ghostFeatures, ghostHeap, ghostVector

# %%
# TODO: Preprocessing:
# Try DB2 Transform
# Subtract weighted welch of each background from signal
# Implement HighPass Filter
# Use markers
# After those:
# Try Periodogram
# Try More windows

# TODO: Feature Selection:
# Try L1 Feature Selection
# Try Genetic Alg. For Feature Selection
# Try LaRocco's Feature Selection Alg.
# Compare to see which does best

# TODO: Model and Analysis:
# Implement One Vs One Model
#
# Print performance stats (which phoneme does best and which does worst)

print('Setting parameters...')
N = 4  # value for N-fold cross-validation
SAMPLE_RATE = 250  # Hz (subject to change)
fs=SAMPLE_RATE
NUM_WINDOWS = 2  # Dependent on number of samples of phonemes
lowcut=1
highcut=(np.floor(SAMPLE_RATE/2))
pcti = 99.95
phoLimit=int(44)
triLimit=int(7)
triMin=int(1)
# initialize script

featureList=list()
phonemeList=list()

subName='DLR'


dirSub='./dlrTests/dlrData/'
for phoNum in range(0,phoLimit):
    for triNum in range(triMin,triLimit):
        nameFile=dirSub+subName + '_' + str(int(phoNum)) + '_' + str(int(triNum)) + '.txt'
        print(nameFile)
#rawData = np.genfromtxt(nameFile, delimiter='/t')


        df = pd.read_csv(nameFile, sep='\t', header=None)
        rawData=df.to_numpy()
        lastRow=np.squeeze(rawData[:,31])

        h1=np.where(lastRow==1)
        h2=np.where(lastRow==2)
        h3=np.where(lastRow==3)
        h4=np.where(lastRow==4)
        h5=np.where(lastRow==5)

        featureVector=ghostVector(rawData, fs, lowcut, highcut, pcti, h1, h2, h3, h4, h5)
        featureList.append(featureVector)
        phonemeList.append(str(phoNum))
        print(np.shape(featureVector))


labels_vector = np.array(phonemeList)
df = pd.DataFrame(featureList)

df.to_csv('dlrData1.csv',sep=',')

np.savetxt("dlrLabels1.csv", labels_vector, delimiter=",", fmt='%s')