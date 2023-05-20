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


from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

#df1 = pd.read_csv('dlrData0.csv', skiprows=0)
N=8
X = np.genfromtxt('dlrData0.csv', delimiter=',')
y = np.genfromtxt('dlrLabels0.csv', delimiter=',')
#y = np.transpose(np.ravel(y))

print(np.shape(X))
print(np.shape(y))
X=np.squeeze(X[1:1541,1:321])
print(np.shape(X))
# compare classifiers
print('Running classifiers...')
clf = QuadraticDiscriminantAnalysis()
print('QDA/LDA Results: ')
scores = cross_val_score(clf, X, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf = LogisticRegression(random_state=0)
print('Logistic Regression Results: ')
scores = cross_val_score(clf, X, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf = GaussianNB()
print('Naive Bayes Results: ')
scores = cross_val_score(clf, X, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf = SVC(gamma=2, C=1)
print('Linear SVM Results: ')
scores = cross_val_score(clf, X, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf = AdaBoostClassifier(n_estimators=1000, random_state=0)
print('AdaBoost Results: ')
scores = cross_val_score(clf, X, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

clf = MLPClassifier(alpha=2, max_iter=100)
print('MLP Results: ')
scores = cross_val_score(clf, X, y, cv=N)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
scores = cross_val_score(clf, X, y, cv=N, scoring='f1_macro')
print("F1 Score: %0.2f (+/- %0.2f)" %
      (scores.mean()-.01, scores.std()+.01 * 2))
#clf.fit(X, y)

# %%
svm = SVC(gamma=2, C=1)
# Fitting Model
svm.fit(X, y)
y_pred = svm.predict(X)

# %%