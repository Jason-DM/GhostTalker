# %%
# Imports
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
from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, cross_validate
from sklearn import metrics, linear_model, preprocessing
from sklearn.cluster import DBSCAN
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer
from scipy.stats import stats
import supportFunctions as sp

# TODO:
# Assuming mean for feature extraction
# Assuming no frequency preprocessing
# Assuming not using markers
print('Setting parameters...')
N = 4  # value for N-fold cross-validation
SAMPLE_RATE = 250  # Hz (subject to change)
NUM_WINDOWS = 2  # Dependent on number of samples of phonemes

print("Constructing Data Matrix to train classifier...")
#file_paths1, bg_sample1, phoneme_labels1 = sp.get_filepaths(
#    'C:\\Users\\surya\\Documents\\GitHub\\GhostTalker\\TestData\\DLR_Tests\\3-21-2023')

file_paths1, bg_sample1, phoneme_labels1 = sp.get_filepaths(
    '/Users/jason/Documents/GitHub/GhostTalker/TestData/DLR_Tests/3-21-2023')    

#file_paths2, bg_sample2, phoneme_labels2 = sp.get_filepaths(
#    'C:\\Users\\surya\\Documents\\GitHub\\GhostTalker\\TestData\\DLR_Tests\\3-23-2023')

file_paths2, bg_sample2, phoneme_labels2 = sp.get_filepaths(
    '/Users/jason/Documents/GitHub/GhostTalker/TestData/DLR_Tests/3-23-2023')

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
print(df.shape)
# %%

kf = KFold(n_splits=10,shuffle=True,random_state=9)

print("df shape: ",df.shape)
print(labels_vector)

Xs = preprocessing.scale(df)

logreg = linear_model.LogisticRegression(C=1, solver='liblinear', multi_class='ovr')
logreg.fit(Xs,labels_vector)

cv_results = cross_validate(logreg, Xs, labels_vector, cv=kf, scoring=('precision', 'recall', 'f1', 'accuracy'))

yhat = cross_val_predict(logreg,Xs,labels_vector, cv=kf)
C = confusion_matrix(labels_vector,yhat,normalize='true') 
'''
scorers = {
            'f1_score': make_scorer(f1_score, average='micro'),
            'precision_score': make_scorer(precision_score, average='micro'),
            'recall_score': make_scorer(recall_score, average='micro'),
            'accuracy_score': make_scorer(accuracy_score)
          }

cv_results = cross_validate(logreg, Xs, labels_vector, cv=kf, scoring=scorers)

print(cv_results)
'''

# Extract test metrics
prec = cv_results['test_precision_score']
rec = cv_results['test_recall_score']
f1 = cv_results['test_f1_score']
acc = cv_results['test_accuracy_score']

# Take average values of the metrics
precm_unreg = np.mean(prec)
recm_unreg = np.mean(rec)
f1m_unreg = np.mean(f1)
accm_unreg= np.mean(acc)

# Compute the standard errors
prec_se_unreg = np.std(prec,ddof=1)/np.sqrt(10)
rec_se_unreg = np.std(rec,ddof=1)/np.sqrt(10)
f1_se_unreg = np.std(f1,ddof=1)/np.sqrt(10)
acc_se_unreg = np.std(acc,ddof=1)/np.sqrt(10)

print('Precision = {0:.4f}, SE={1:.4f}'.format(precm_unreg,prec_se_unreg))
print('Recall =    {0:.4f}, SE={1:.4f}'.format(recm_unreg, rec_se_unreg))
print('f1 =        {0:.4f}, SE={1:.4f}'.format(f1m_unreg, f1_se_unreg))
print('Accuracy =  {0:.4f}, SE={1:.4f}'.format(accm_unreg, acc_se_unreg))

disp = ConfusionMatrixDisplay(confusion_matrix=C)
disp.plot()

# %%
