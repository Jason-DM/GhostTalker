# %%
# Imports
# this is Dr. Larocco's feature selection utilizing script to run the new CSV data on our models. Surprisingly good results! 

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
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, make_scorer, classification_report
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from scipy.stats import stats
from utilityFunctions import eegFeatureReducer, balancedMatrix, featureSelect, speedClass, dirClass, dualClass, fsClass
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn import svm

from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# for N-fold cross validation
N=4

X = np.genfromtxt('dlrData2.csv', delimiter=',')

y = np.genfromtxt('dlrLabels2.csv', delimiter=',')
print(np.shape(X))
print(np.shape(y))
X = X.astype(float)
y = y.astype(int)

X[np.isnan(X)] = 0
X[np.isinf(X)] = 0

y[np.isnan(y)] = 0
y[np.isinf(y)] = 0

lims=np.shape(X)
x0=int(lims[0])
x1=int(lims[1])

X=np.squeeze(X[1:x0,1:x1])
runCats=np.squeeze(np.unique(y))
featureNumber=int(3)
catToSearch = 2

clf = QuadraticDiscriminantAnalysis()
print('QDA/LDA Results: ')
finalAcc,finalF1,finalFeatures,finalLength=dualClass(N,clf,X,y,featureNumber)
print('No FS Accuracy: ')
print(finalAcc)
print('No FS F1: ')
print(finalF1)
print(' ')
fsAcc,fsF1,finalFeatures,finalLength=fsClass(N,clf,X,y,featureNumber)
print('FS Accuracy: ')
print(fsAcc)
print('FS F1: ')
print(fsF1)
print(' ')

clf = GaussianNB()
print('Naive Bayes Results: ')
finalAcc,finalF1,finalFeatures,finalLength=dualClass(N,clf,X,y,featureNumber)
print('No FS Accuracy: ')
print(finalAcc)
print('No FS F1: ')
print(finalF1)
print(' ')
fsAcc,fsF1,finalFeatures,finalLength=fsClass(N,clf,X,y,featureNumber)
print('FS Accuracy: ')
print(fsAcc)
print('FS F1: ')
print(fsF1)
print(' ')

clf = SVC(gamma=2, C=1)
print('Linear SVM Results: ')
finalAcc,finalF1,finalFeatures,finalLength=dualClass(N,clf,X,y,featureNumber)
print('No FS Accuracy: ')
print(finalAcc)
print('No FS F1: ')
print(finalF1)
print(' ')
fsAcc,fsF1,finalFeatures,finalLength=fsClass(N,clf,X,y,featureNumber)
print('FS Accuracy: ')
print(fsAcc)
print('FS F1: ')
print(fsF1)
print(' ')

clf = KNeighborsClassifier(n_neighbors=3)
print('KNN Results: ')
finalAcc,finalF1,finalFeatures,finalLength=dualClass(N,clf,X,y,featureNumber)
print('No FS Accuracy: ')
print(finalAcc)
print('No FS F1: ')
print(finalF1)
print(' ')
fsAcc,fsF1,finalFeatures,finalLength=fsClass(N,clf,X,y,featureNumber)
print('FS Accuracy: ')
print(fsAcc)
print('FS F1: ')
print(fsF1)
print(' ')

