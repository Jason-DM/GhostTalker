# %%
# Imports
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
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
import support as sp
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
NUM_WINDOWS = 2  # Dependent on number of samples of phonemes
fs = 200  # features per second (Hz)
pcti = 99.95  # amplitude percent
lowcut = 1  # filter lower bound (Hz)
highcut = (np.floor(fs/2))  # filter upper bound (Hz)
featureNumber = 3  # number of features to retain
N = 4  # value for N-fold cross-validation

print("Constructing Data Matrix to train classifier...")
# file_paths1, bg_sample1, phoneme_labels1 = sp.get_filepaths(
#    'C:\\Users\\surya\\Documents\\GitHub\\GhostTalker\\TestData\\DLR_Tests\\3-21-2023')

#file_paths1, bg_sample1, phoneme_labels1 = sp.get_filepaths(
#    '/Users/jason/Documents/GitHub/GhostTalker/TestData/DLR_Tests/3-21-2023')

file_paths1, bg_sample1, phoneme_labels1 = sp.get_filepaths(
    'C:\\Users\\Cole\\Documents\\GitHub\\GhostTalker\\TestData\\DLR_Tests\\3-21-2023')

# file_paths2, bg_sample2, phoneme_labels2 = sp.get_filepaths(
#    'C:\\Users\\surya\\Documents\\GitHub\\GhostTalker\\TestData\\DLR_Tests\\3-23-2023')

#file_paths2, bg_sample2, phoneme_labels2 = sp.get_filepaths(
#    '/Users/jason/Documents/GitHub/GhostTalker/TestData/DLR_Tests/3-23-2023')

file_paths2, bg_sample2, phoneme_labels2 = sp.get_filepaths(
    'C:\\Users\\Cole\\Documents\\GitHub\\GhostTalker\\TestData\\DLR_Tests\\3-23-2023')

file_paths = file_paths1+file_paths2
phoneme_labels = phoneme_labels1+phoneme_labels2
bg_samples = [bg_sample1 for i in range(
    len(file_paths1))]+[bg_sample2 for i in range(len(file_paths2))]

# TODO: First you'll need to split one file into 5
# TODO: Then you'll have to work on for-loop to use df instead of file name
# TODO: Finally you'll also have to make 5 of each label...
nameFile = file_paths[0]
df = pd.read_csv(nameFile, sep='\t', header=None)

data_list = []
labels_vector = []
for (file_path, phoneme_label, bg_sample) in zip(file_paths, phoneme_labels, bg_samples):
    nameFile = file_path
    df = pd.read_csv(nameFile, sep='\t', header=None)
    # df = df[df.columns[1:16]]  # Only using the 16 channles as features
    feature_vector = []
    # if nameFile == 'C:\\Users\\surya\\Documents\\GitHub\\GhostTalker\\TestData\\DLR_Tests\\3-21-2023\\DLR_27_2.txt':
    #    n = 3
    #if nameFile == '/Users/jason/Documents/GitHub/GhostTalker/TestData/DLR_Tests/3-21-2023/DLR_27_2.txt':
    #    n = 3
    if nameFile == 'C:\\Users\\Cole\\Documents\\GitHub\\GhostTalker\\TestData\\DLR_Tests\\3-21-2023':
        n=3
    else:
        n = 6
    for i in range(1, n):
        df_i = df.iloc[df.loc[df[31] == i].index[0]:df.loc[df[31] == i].index[1], :]
        for col in df_i.columns:
            subsections = np.array_split(df_i[col], NUM_WINDOWS)
            means = [sub.mean() for sub in subsections]
            feature_vector.extend(means)
        data_list.append(feature_vector)
        labels_vector.append(phoneme_label)

min_length = min(len(row) for row in data_list)
# slice all the rows to the length of the shortest one
data = [row[:min_length] for row in data_list]

# create a Pandas DataFrame from the sliced rows and labels vector
df = pd.DataFrame(data)
labels_vector = np.array(labels_vector)


# # %%
# # Troubleshoot MultiClass Classification...

# # Generate multiclass classification data
# X, y = make_classification(n_samples=1700, n_classes=44, n_informative=10,
#                            n_features=128, random_state=42)
# df_test = pd.DataFrame(X)

# kf = KFold(n_splits=100, shuffle=True, random_state=9)
# Xs = preprocessing.scale(df_test)


# logreg = linear_model.LogisticRegression(
#     C=500, solver='liblinear', multi_class='ovr')
# logreg.fit(Xs, y)

# svm_object = svm.SVC(probability=False, kernel="rbf",
#                      C=2.8, gamma=.0073, verbose=1)
# svm_object.fit(Xs, y)
# sp.multiclass_performance(Xs, y, logreg)
# sp.multiclass_performance(Xs, y, svm_object)

# %%
# Write each Preprocessing way - write in support function
# High Pass - Cole
# Periodogram - Sam
# DB2 - Jason
# P Welch - Surya

# %%
# Feature Selection:

# # Set the regularization parameter C = 1
# logistic = LogisticRegression(
#     C=1, penalty='l1', solver='liblinear', random_state=7).fit(Xs, y)
# model = SelectFromModel(logistic, prefit=True)

# X_new = model.transform(Xs)

# # Dropping Features
# #selected_columns = selected_features.columns[selected_features.var() == 0]

# # %%
# Test Data

Xs = preprocessing.scale(df)
y = labels_vector

logreg = linear_model.LogisticRegression(
    C=500, solver='liblinear', multi_class='ovr')
logreg.fit(Xs, y)

svm_object = svm.SVC(probability=False, kernel="rbf",
                     C=2.8, gamma=.0073, verbose=1)
svm_object.fit(Xs, y)
sp.multiclass_performance(Xs, y, logreg)
sp.multiclass_performance(Xs, y, svm_object)

# compare classifiers
print('Running classifiers...')
clf = QuadraticDiscriminantAnalysis()
print('QDA/LDA Results: ')
scores = cross_val_score(clf, Xs, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, Xs, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf = LogisticRegression(random_state=0)
print('Logistic Regression Results: ')
scores = cross_val_score(clf, Xs, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, Xs, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf = GaussianNB()
print('Naive Bayes Results: ')
scores = cross_val_score(clf, Xs, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, Xs, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf = SVC(gamma=2, C=1)
print('Linear SVM Results: ')
scores = cross_val_score(clf, Xs, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, Xs, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf = AdaBoostClassifier(n_estimators=1000, random_state=0)
print('AdaBoost Results: ')
scores = cross_val_score(clf, Xs, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, Xs, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf = MLPClassifier(alpha=2, max_iter=100)
print('MLP Results: ')
scores = cross_val_score(clf, Xs, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, Xs, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))


# # %%
# y = labels_vector
# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# # Initialize the One-vs-One classifier with a Support Vector Machine (SVM) base estimator
# ovr_classifier = OneVsOneClassifier(SVC())

# # Fit the classifier on the training data
# ovr_classifier.fit(X_train, y_train)

# # Predict the classes of the test set
# y_pred = ovr_classifier.predict(X_test)

# # Print the classification report
# print(classification_report(y_test, y_pred))
# %%
# Adaboost Start
# Split data into training and testing sets
# y = labels_vector
# X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# # Initialize the AdaBoost classifier with a Decision Tree base estimator
# ada_classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)

# # Fit the classifier on the training data
# ada_classifier.fit(X_train, y_train)

# # Predict the classes of the test set
# y_pred = ada_classifier.predict(X_test)

# # Print the classification report
# print(classification_report(y_test, y_pred))

# # %%
# y = labels_vector
# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# # Initialize the One-vs-One classifier with a Support Vector Machine (SVM) base estimator
# ovr_classifier = OneVsOneClassifier(SVC())

# # Fit the classifier on the training data
# ovr_classifier.fit(X_train, y_train)

# # Predict the classes of the test set
# y_pred = ovr_classifier.predict(X_test)

# # Print the classification report
# print(classification_report(y_test, y_pred))
# %%
# Adaboost Start
# Split data into training and testing sets
# y = labels_vector
# X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# # Initialize the AdaBoost classifier with a Decision Tree base estimator
# ada_classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)

# # Fit the classifier on the training data
# ada_classifier.fit(X_train, y_train)

# # Predict the classes of the test set
# y_pred = ada_classifier.predict(X_test)

# # Print the classification report
# print(classification_report(y_test, y_pred))

# %%

# Set parameters
print('Setting parameters...')
fs = 200  # features per second (Hz)
pcti = 99.95  # amplitude percent
lowcut = 1  # filter lower bound (Hz)
highcut = (np.floor(fs/2))  # filter upper bound (Hz)
featureNumber = 3  # number of features to retain
N = 4  # value for N-fold cross-validation

# filenames of EEG files
nameFile1 = './Sl5_3.txt'
nameFile2 = './Sl5_s.txt'

# load data
print('Loading datasets...')
df1 = pd.read_csv(nameFile1, sep='\t', skiprows=6)
df2 = pd.read_csv(nameFile2, sep='\t', skiprows=6)

# perform feature extraction
print('Extracting features...')
featureMatrixA = eegFeatureExtraction(df1, fs, lowcut, highcut, pcti)
featureMatrixB = eegFeatureExtraction(df2, fs, lowcut, highcut, pcti)

# perform feature selection
print('Selecting features...')
topFeatures = eegFeatureReducer(featureMatrixA, featureMatrixB, featureNumber)

featureMatrixA = np.squeeze(featureMatrixA[:, topFeatures])
featureMatrixB = np.squeeze(featureMatrixB[:, topFeatures])

t0 = np.zeros(np.shape(featureMatrixA)[0])
t1 = np.ones(np.shape(featureMatrixB)[0])

totalLength = np.array([len(t0), len(t1)])

# prepare data for classification
print('Preparing for classification...')
s0 = balancedMatrix(featureMatrixA, totalLength)
s1 = balancedMatrix(featureMatrixB, totalLength)

X = np.vstack([s0, s1])
t0 = 0*np.ones([1, len(s0)])
t1 = 1*np.ones([1, len(s1)])

targets = np.hstack([t0, t1])
y = np.transpose(np.ravel(targets))
