# %%
# Imports
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, cross_validate
from sklearn import metrics, linear_model, preprocessing
from sklearn.cluster import DBSCAN
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer
from scipy.stats import stats
import supportFunctions as sp

# %%
# TODO: Preprocessing:
# Try DB2 Transform
# Subtract weighted welch opf each background from signal
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

print("Constructing Data Matrix to train classifier...")
# file_paths1, bg_sample1, phoneme_labels1 = sp.get_filepaths(
#    'C:\\Users\\surya\\Documents\\GitHub\\GhostTalker\\TestData\\DLR_Tests\\3-21-2023')

file_paths1, bg_sample1, phoneme_labels1 = sp.get_filepaths(
    'C:\\Users\\surya\\Documents\\GitHub\\GhostTalker\\TestData\\DLR_Tests\\3-21-2023')

# file_paths2, bg_sample2, phoneme_labels2 = sp.get_filepaths(
#    'C:\\Users\\surya\\Documents\\GitHub\\GhostTalker\\TestData\\DLR_Tests\\3-23-2023')

file_paths2, bg_sample2, phoneme_labels2 = sp.get_filepaths(
    'C:\\Users\\surya\\Documents\\GitHub\\GhostTalker\\TestData\\DLR_Tests\\3-23-2023')

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
    if nameFile == 'C:\\Users\\surya\\Documents\\GitHub\\GhostTalker\\TestData\\DLR_Tests\\3-21-2023\\DLR_27_2.txt':
        n = 3
    else:
        n = 6
    for i in range(1, n):
        df_i = df.iloc[df.loc[df[31] == i].index[0]                       :df.loc[df[31] == i].index[1], :]
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


# %%
# Troubleshoot MultiClass Classification...

# Generate multiclass classification data
X, y = make_classification(n_samples=1000, n_classes=5, n_informative=10,
                           n_features=20, random_state=42)
df_test = pd.DataFrame(X)

kf = KFold(n_splits=100, shuffle=True, random_state=9)
Xs = preprocessing.scale(df_test)


logreg = linear_model.LogisticRegression(
    C=500, solver='liblinear', multi_class='ovr')
logreg.fit(Xs, y)

svm_object = svm.SVC(probability=False, kernel="rbf",
                     C=2.8, gamma=.0073, verbose=1)
svm_object.fit(Xs, y)
sp.multiclass_performance(Xs, y, logreg)
sp.multiclass_performance(Xs, y, svm_object)

# %%
# Write each Preprocessing way - write in support function
# High Pass - Cole
# Periodogram - Sam
# DB2 - Jason
# P Welch - Surya

# %%
# Feature Selection:

# Set the regularization parameter C = 1
logistic = LogisticRegression(
    C=1, penalty='l1', solver='liblinear', random_state=7).fit(Xs, y)
model = SelectFromModel(logistic, prefit=True)

X_new = model.transform(Xs)

# Dropping Features
#selected_columns = selected_features.columns[selected_features.var() == 0]

# %%
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


# %%
