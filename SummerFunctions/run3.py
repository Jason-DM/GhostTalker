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
import supportFunctions as sp

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

print("Constructing Data Matrix to train classifier...")
file_paths1, bg_sample1, phoneme_labels1 = sp.get_filepaths(
    './dlrTests/3-21-2023')

file_paths2, bg_sample2, phoneme_labels2 = sp.get_filepaths(
    './dlrTests/3-23-2023')

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
print(nameFile)
for (file_path, phoneme_label, bg_sample) in zip(file_paths, phoneme_labels, bg_samples):
    nameFile = file_path
    print(file_path)
    df = pd.read_csv(nameFile, sep='\t', header=None)
    # df = df[df.columns[1:16]]  # Only using the 16 channles as features
    feature_vector = []
    #if nameFile == 'C:\\Users\\surya\\Documents\\GitHub\\GhostTalker\\TestData\\DLR_Tests\\3-21-2023\\DLR_27_2.txt':
    #    n = 3
    if nameFile == './TestData/DLR_Tests/3-21-2023/DLR_27_2.txt':
        n = 3
    else:
        n = 6
    for i in range(1, n):
        print(i)

        df_i = df.iloc[df.loc[df[31] == i].index[0]                       :df.loc[df[31] == i].index[1], :]
        for col in df_i.columns:
            subsections = np.array_split(df_i[col], NUM_WINDOWS)
            means = [sub.mean() for sub in subsections]
            feature_vector.extend(means)
        data_list.append(feature_vector)
        labels_vector.append(str(phoneme_label))

min_length = min(len(row) for row in data_list)
# slice all the rows to the length of the shortest one
data = [row[:min_length] for row in data_list]

# create a Pandas DataFrame from the sliced rows and labels vector
df = pd.DataFrame(data)
labels_vector = np.array(labels_vector)
print(labels_vector)
print(np.shape(df))

df.to_csv('dlrData0.csv',sep=',')

np.savetxt("dlrLabels0.csv", labels_vector, delimiter=",", fmt='%s')
