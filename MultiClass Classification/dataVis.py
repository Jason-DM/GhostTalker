import supportFunctions as sp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import signal

absolute_path = os.path.dirname(__file__)
relative_path = "DLR_2_6.txt"
full_path = os.path.join(absolute_path, relative_path)
data = pd.read_csv(full_path, sep='\t')
fs = 250

f, P = signal.welch(data, fs)
plt.semilogy(f, P)
plt.show()