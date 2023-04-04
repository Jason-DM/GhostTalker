# %%
# Imports
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import scipy as sp
import pandas as pd
from scipy import signal, arange, fft, fromstring, roll
from scipy.signal import butter, lfilter, ricker
import os
import glob
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from scipy.stats import stats
import supportFunctions as sp

# %%
# TODO:
# Assuming mean for feature extraction
# Assuming no frequency preprocessing
# Assuming not using markers
print('Setting parameters...')
N = 4  # value for N-fold cross-validation
SAMPLE_RATE = 250  # Hz (subject to change)
NUM_WINDOWS = 2  # Dependent on number of samples of phonemes

print("Constructing Data Matrix to train classifier...")
file_paths1, bg_sample1, phoneme_labels1 = sp.get_filepaths(
    'C:\\Users\\surya\\Documents\\GitHub\\GhostTalker\\TestData\\DLR_Tests\\3-21-2023')

file_paths2, bg_sample2, phoneme_labels2 = sp.get_filepaths(
    'C:\\Users\\surya\\Documents\\GitHub\\GhostTalker\\TestData\\DLR_Tests\\3-23-2023')

file_paths = file_paths1+file_paths2
phoneme_labels = phoneme_labels1+phoneme_labels2
bg_samples = [bg_sample1 for i in range(
    len(file_paths1))]+[bg_sample2 for i in range(len(file_paths2))]

data_list = []
labels_vector = []
for (file_path, phoneme_label, bg_sample) in zip(file_paths, phoneme_labels, bg_samples):
    nameFile = file_path
    df = pd.read_csv(nameFile, sep='\t')
    N_SAMPLES = df.shape[0]
    df.columns = [i for i in range(32)]
    df = df[df.columns[1:16]]  # Only using the 16 channles as features
    feature_vector = []
    for col in df.columns:
        subsections = np.array_split(df[col], NUM_WINDOWS)
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
