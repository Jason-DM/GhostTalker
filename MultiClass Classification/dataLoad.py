# %%
#Introduction and Imports
#!/usr/bin/env python3
# coding:utf-8

# Brain-Computer Interface
# v.1.0
# GhostTalker Team
# 1 Janurary 2023


# Usage:
# This script loads the raw EEG elements,
# prepares the preprocessed feature and label matricies,
# and trains an ensemble of ML models

# implemented with Python3 on Anaconda

# import basic modules
import numpy as np
import scipy as sp
from scipy import fft
from scipy.fft import fft, fftfreq
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import os  # used for File Loading
import csv  # used to Change from txt to csv

from supportFunctions import featureExtraction, eegFeatureExtraction, eegFeatureReducer, balancedMatrix

from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# %%
# Set parameters

# TODO: I probably will use different parameters...
print('Setting parameters...')
# pcti = 99.95  # amplitude percent
# lowcut = 1  # filter lower bound (Hz)
# highcut = (np.floor(fs/2))  # filter upper bound (Hz)
# featureNumber = 3  # number of features to retain
N = 4  # value for N-fold cross-validation
SAMPLE_RATE = 250  # Hz (subject to change)
NUM_WINDOWS = 2  # Dependent on number of samples of phonemes

# %%
# Processing all the files into the feature and label matricies

# TODO: Need to read all the file names using a for loop from another folder

# nameFile = 'SL5_3.txt'
# filePath = "D:\Stim_Pres_Data" # Change to dedicated File Path
# output_file = 0
# input_file = 0
# # load data-file path-Convert to CSV!!!
# for filename in os.listdir(filePath):
#     if filename.endswith(".txt"):  # Open File as txt
#         file_path = os.path.join(filePath, filename)
#         with open(file_path, "r") as input_file:


# TODO: write fxn to return the following
#
# filenameList = ['test_data_11_29_22.txt'....]
# background = 'background.txt'
# phoneme = [1....]


# load data
nameFile = 'SL5_3.txt'
df = pd.read_csv(nameFile, sep='\t')
N_SAMPLES = df.shape[0]
df.columns = [i for i in range(32)]
# Only using the 16 channles as features
df = df[df.columns[1:16]]


feature_vector = []
for col in df.columns:
    subsections = np.array_split(df[col], NUM_WINDOWS)
    means = [sub.mean() for sub in subsections]
    feature_vector.extend(means)

    # TODO: Write a function for frequency filtering
    # Build tunable FFT
    # yf = fft(np.array(df[df.columns[2]]))
    # xf = fftfreq(N_SAMPLES, 1/SAMPLE_RATE)
    # plt.plot(xf, np.abs(yf))
    # plt.show()

    # TODO: Talk to Dr. Schniter for the best way to organize this and do hyperparamter tuning
    # TODO: Run this matrix through classifier
    # TODO: Save the matrix as csv
    # TODO: Just start getting all of the hyperparamters
    # TODO: start adding all of the features you want to test to the feature matrix, make it huge

    # %%
    # # perform feature extraction
    # print('Extracting features...')
    # featureMatrixA = eegFeatureExtraction(df1, fs, lowcut, highcut, pcti)
    # #featureMatrixB = eegFeatureExtraction(df2, fs, lowcut, highcut, pcti)

    # # perform feature selection
    # print('Selecting features...')
    # topFeatures = eegFeatureReducer(featureMatrixA, featureMatrixB, featureNumber)

    # featureMatrixA = np.squeeze(featureMatrixA[:, topFeatures])
    # featureMatrixB = np.squeeze(featureMatrixB[:, topFeatures])

    # t0 = np.zeros(np.shape(featureMatrixA)[0])
    # t1 = np.ones(np.shape(featureMatrixB)[0])

    # totalLength = np.array([len(t0), len(t1)])

    # # prepare data for classification
    # print('Preparing for classification...')
    # s0 = balancedMatrix(featureMatrixA, totalLength)
    # s1 = balancedMatrix(featureMatrixB, totalLength)

    # X = np.vstack([s0, s1])
    # t0 = 0*np.ones([1, len(s0)])
    # t1 = 1*np.ones([1, len(s1)])

    # targets = np.hstack([t0, t1])
    # y = np.transpose(np.ravel(targets))

    # # compare classifiers
    # print('Running classifiers...')
    # clf = QuadraticDiscriminantAnalysis()
    # print('QDA/LDA Results: ')
    # scores = cross_val_score(clf, X, y, cv=N)
    # print("Accuracy: %0.2f (+/- %0.2f)" %
    #       (scores.mean()-.01, scores.std()+.01 * 2))
    # scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
    # print("F1 Score: %0.2f (+/- %0.2f)" %
    #       (scores.mean()-.01, scores.std()+.01 * 2))
    # #clf.fit(X, y)

    # clf = LogisticRegression(random_state=0)
    # print('Logistic Regression Results: ')
    # scores = cross_val_score(clf, X, y, cv=N)
    # print("Accuracy: %0.2f (+/- %0.2f)" %
    #       (scores.mean()-.01, scores.std()+.01 * 2))
    # scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
    # print("F1 Score: %0.2f (+/- %0.2f)" %
    #       (scores.mean()-.01, scores.std()+.01 * 2))
    # #clf.fit(X, y)

    # clf = GaussianNB()
    # print('Naive Bayes Results: ')
    # scores = cross_val_score(clf, X, y, cv=N)
    # print("Accuracy: %0.2f (+/- %0.2f)" %
    #       (scores.mean()-.01, scores.std()+.01 * 2))
    # scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
    # print("F1 Score: %0.2f (+/- %0.2f)" %
    #       (scores.mean()-.01, scores.std()+.01 * 2))
    # #clf.fit(X, y)

    # clf = SVC(gamma=2, C=1)
    # print('Linear SVM Results: ')
    # scores = cross_val_score(clf, X, y, cv=N)
    # print("Accuracy: %0.2f (+/- %0.2f)" %
    #       (scores.mean()-.01, scores.std()+.01 * 2))
    # scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
    # print("F1 Score: %0.2f (+/- %0.2f)" %
    #       (scores.mean()-.01, scores.std()+.01 * 2))
    # #clf.fit(X, y)

    # clf = AdaBoostClassifier(n_estimators=1000, random_state=0)
    # print('AdaBoost Results: ')
    # scores = cross_val_score(clf, X, y, cv=N)
    # print("Accuracy: %0.2f (+/- %0.2f)" %
    #       (scores.mean()-.01, scores.std()+.01 * 2))
    # scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
    # print("F1 Score: %0.2f (+/- %0.2f)" %
    #       (scores.mean()-.01, scores.std()+.01 * 2))
    # #clf.fit(X, y)

    # clf = MLPClassifier(alpha=2, max_iter=100)
    # print('MLP Results: ')
    # scores = cross_val_score(clf, X, y, cv=N)
    # print("Accuracy: %0.2f (+/- %0.2f)" %
    #       (scores.mean()-.01, scores.std()+.01 * 2))
    # scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
    # print("F1 Score: %0.2f (+/- %0.2f)" %
    #       (scores.mean()-.01, scores.std()+.01 * 2))
    # #clf.fit(X, y)
