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

from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, cross_validate
from sklearn.cluster import DBSCAN
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer
'''
DLR's code.
def eegFeatureExtraction(df, fs, lowcut, highcut, pcti):
    chan1 = df.iloc[:, 2]
    chan2 = df.iloc[:, 3]
    chan3 = df.iloc[:, 4]
    chan4 = df.iloc[:, 5]
    

    # rotating the vectors to array
    c1 = np.real(np.asarray(chan1))
    c2 = np.real(np.asarray(chan2))
    c3 = np.real(np.asarray(chan3))
    c4 = np.real(np.asarray(chan4))

    # Normalizing these arrays
    c1 = c1-np.mean(c1)
    c2 = c2-np.mean(c2)
    c3 = c1-np.mean(c3)
    c4 = c4-np.mean(c4)

    c1 = c1[fs::]

    f1 = featureExtraction(c1, fs, lowcut, highcut, pcti)
    features = np.squeeze(np.shape(f1))

    c2 = c2[fs::]
    c3 = c3[fs::]
    c4 = c4[fs::]

    lengthFile = np.floor(np.squeeze(np.shape(c1))/np.float(4*fs))
    lbnds = np.arange(0, (lengthFile-1))
    ubnds = np.arange(1, (lengthFile))
    capper = np.min([len(lbnds), len(ubnds)])
    lbnds = 4*fs*lbnds[0:capper]
    ubnds = 4*fs*ubnds[0:capper]
    featureMatrix = np.zeros((capper, (4*features)))

    for ix in range(0, capper):
        s1 = featureExtraction(
            c1[int(lbnds[ix]):int(ubnds[ix])], fs, lowcut, highcut, pcti)
        s2 = featureExtraction(
            c2[int(lbnds[ix]):int(ubnds[ix])], fs, lowcut, highcut, pcti)
        s3 = featureExtraction(
            c3[int(lbnds[ix]):int(ubnds[ix])], fs, lowcut, highcut, pcti)
        s4 = featureExtraction(
            c4[int(lbnds[ix]):int(ubnds[ix])], fs, lowcut, highcut, pcti)
        featall = np.concatenate((s1, s2, s3, s4), axis=0)
        featall = np.squeeze(featall[0:(4*features)])
        featureMatrix[int(ix), :] = featall
    return (featureMatrix)
'''
# adjusted code to account for 16 channels. Should work?


def eegFeatureExtraction(df, fs, lowcut, highcut, pcti):
    # Select the 16 channels
    channels = []
    for i in range(16):
        channels.append(df.iloc[:, i+2])

    # Convert each channel to a numpy array
    c = []
    for i in range(16):
        c.append(np.real(np.asarray(channels[i])))

    # Normalize each array
    for i in range(16):
        c[i] = c[i] - np.mean(c[i])

    # Shift each array by fs samples
    for i in range(16):
        c[i] = c[i][fs:]

    # Extract features from the first channel
    f = featureExtraction(c[0], fs, lowcut, highcut, pcti)
    features = np.squeeze(np.shape(f))

    # Initialize the feature matrix with the first channel's features
    featureMatrix = np.zeros((len(c[0])//(4*fs), (16*features)))
    featureMatrix[:, :features] = f

    # Iterate over each segment and extract features from each channel
    for i in range(featureMatrix.shape[0]):
        lbnds = 4*fs*i
        ubnds = 4*fs*(i+1)
        for j in range(16):
            s = featureExtraction(c[j][lbnds:ubnds], fs, lowcut, highcut, pcti)
            featall = np.squeeze(s[0:features])
            featureMatrix[i, j*features:(j+1)*features] = featall

    return featureMatrix


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = (lowcut / nyq)
    high = (highcut / nyq)
    if high >= 1:
        high = .99
    if low <= 0:
        low = .001
    b, a = butter(order, [low, high], btype='band')
    return b, a


def get_filepaths(folder_path):
    """
    Returns a list of complete filepaths to each data file and list of phoneme labels
    :param folder_path: string containing path to folder name for a single day of tests
    :return: List of filepaths, path to background sample, list of phoneme labels
    """
    # Initialize return values
    file_paths = []
    phoneme_labels = []
    bg_sample = ""
    for filename in os.listdir(folder_path):
        # Combine folder path with filename for file path
        file_path = os.path.join(folder_path, filename)
        if filename.endswith("_BC367.txt"):
            # Extract bg sample
            bg_sample = file_path
        else:
            # if not bg sample, add path to file_paths and phoneme label.
            file_paths.append(file_path)
            phoneme_labels.append(extract_middle_int(filename))
    return file_paths, bg_sample, phoneme_labels


def extract_middle_int(s):
    """
    Returns the middle integer based on our naming convention ABC_12_12 using regex
    """
    match = re.search(r"[A-Za-z]{3}_(\d+)_(\d+)", s)
    if match:
        return int(match.group(1))
    else:
        return None


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def welchProc(data, fs):
    # wsize=round(fs/10)
    f, P = signal.welch(data, fs)
    return f, P


def peakFinder(f, P):
    peakFLoc = np.where(P == np.amax(P))
    peakFLoc = peakFLoc[0]
    peakF = f[peakFLoc]
    vrms = np.sqrt(P.max())
    return peakF, peakFLoc, vrms


def smooth(x, window_len=11, window='hanning'):
    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


def featureExtraction(data, fs, lowcut, highcut, pcti):
    widths = np.arange(1, 31)
    data[np.isnan(data)] = 0
    data[np.isinf(data)] = 0
    intensityPcti = np.percentile(data, pcti)
    data = data-np.mean(data)
    data = smooth(data.flatten())
    data = butter_bandpass_filter(data, lowcut, highcut, fs, order=4)
    data = signal.cwt(data, signal.ricker, widths)
    f, P = welchProc(data, fs)
    peakFLoc = np.where(P == np.amax(P))
    peakFLoc = peakFLoc[0]
    peakF = f[peakFLoc]
    vrms = np.sqrt(P.max())
    Psum = np.sum(P, axis=1)
    Psum = Psum.flatten()
    # print(np.shape(Psum))
    # print(np.shape(vrms))
    # print(np.shape(peakF))
    # print(np.shape(peakFLoc))
    # print(np.shape(intensityPcti))
    featureVector = np.hstack(
        (Psum.flatten(), vrms, peakF, peakFLoc, intensityPcti))
    # print(featureVector)
    featureVector[np.isnan(featureVector)] = 0
    featureVector[np.isinf(featureVector)] = 0
    # print(featureVector)
    return featureVector


def balancedMatrix(a, totalLength):
    maxLen = np.max(totalLength)
    minLen = np.min(totalLength)
    ratioL = np.floor(maxLen/minLen)
    finalR = np.ceil(minLen*ratioL)
    aT = np.copy(a)
    features = np.shape(a)[1]
    aTT = np.zeros([1, features])
    for ii in range(0, int(ratioL)):
        aTT1 = np.copy(aT)
        aTT = np.vstack([aTT1, aTT])
        aTT = np.squeeze(aTT)

    aTT = aTT[0:(maxLen-1), :]
    aTT = np.squeeze(aTT)
    return (aTT)


def eegFeatureReducer(featureMatrixA, featureMatrixB, featureNumber):
    m0 = np.mean(featureMatrixA, axis=0)
    m1 = np.mean(featureMatrixB, axis=0)
    distancesVec = np.abs(m0-m1)
    tempR = np.argpartition(-distancesVec, featureNumber)
    resultArgs = tempR[:featureNumber]
    topFeatures = np.flip(resultArgs)

    return (topFeatures)


def featureReducer(Xf, yf, features):
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, features, step=15)
    selector = selector.fit(Xf, yf)
    topFeatures = np.where(selector.ranking_ == 1)
    Xnew = np.squeeze(Xf[:, topFeatures])
    return (Xnew, topFeatures)


def crossValClass(clf, X, y, xfold):
    scores = cross_val_score(clf, X, y, cv=xfold)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(clf, X, y, cv=xfold, scoring='f1_macro')
    print("F1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# An element is one combination of trial, phoneme, and subject


def vectorizeElement(df):
    df




def multiclass_performance(X, y, model_fitted):

    scoring = {
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro',
        'accuracy': 'accuracy'
    }
    kf = KFold(n_splits=100, shuffle=True, random_state=9)

    cv_results = cross_validate(model_fitted, X, y, cv=kf, scoring=scoring)

    # Extract test metrics
    prec = cv_results['test_precision_macro']
    rec = cv_results['test_recall_macro']
    f1 = cv_results['test_f1_macro']
    acc = cv_results['test_accuracy']

    # Take average values of the metrics
    precm_unreg = np.mean(prec)
    recm_unreg = np.mean(rec)
    f1m_unreg = np.mean(f1)
    accm_unreg = np.mean(acc)

    # Compute the standard errors
    prec_se_unreg = np.std(prec, ddof=1)/np.sqrt(10)
    rec_se_unreg = np.std(rec, ddof=1)/np.sqrt(10)
    f1_se_unreg = np.std(f1, ddof=1)/np.sqrt(10)
    acc_se_unreg = np.std(acc, ddof=1)/np.sqrt(10)

    print('Precision = {0:.4f}, SE={1:.4f}'.format(precm_unreg, prec_se_unreg))
    print('Recall =    {0:.4f}, SE={1:.4f}'.format(recm_unreg, rec_se_unreg))
    print('f1 =        {0:.4f}, SE={1:.4f}'.format(f1m_unreg, f1_se_unreg))
    print('Accuracy =  {0:.4f}, SE={1:.4f}'.format(accm_unreg, acc_se_unreg))

    yhat = cross_val_predict(model_fitted, X, y, cv=kf)
    C = confusion_matrix(y, yhat, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=C)
    disp.plot()
