# %%
# Combine individual features into larger one

import numpy as np
import scipy as sp
import pandas as pd


from utilityFunctions import pairLoader

allSubNames=list(['GTP000','GTP003','GTP004','GTP007','GTP044','GTP045','GTP047','GTP104','GTP223','GTP303','GTP308','GTP455','GTP545','GTP556','GTP762','GTP765'])
subName='GTP000'

subNames=list(['GTP003','GTP004','GTP007','GTP044','GTP045','GTP047','GTP104','GTP223','GTP303','GTP308','GTP455','GTP545','GTP556','GTP762','GTP765'])
[X0,y0]=pairLoader(subName)
[xw,xh]=np.shape(X0)

#[X1,y1]=pairLoader(subName)

#X2=np.squeeze(np.concatenate((X0,X1),axis=0))

#y2=np.squeeze(np.concatenate((y0,y1),axis=0))

#print(np.shape(X2))
#print(np.shape(y2))

for subName in subNames:
	print(subName)
	[X1,y1]=pairLoader(subName)
	print(np.shape(X1))
	print(np.shape(y1))
	X1=X1[:,0:xh]
	X0=np.squeeze(np.concatenate((X0,X1),axis=0))
	y0=np.squeeze(np.concatenate((y0,y1),axis=0))
	print(np.shape(X0))
	print(np.shape(y0))
df = pd.DataFrame(X0)
outNamData='allSubData.csv'
outNamLabels='allSubLabels.csv'

df.to_csv(outNamData,sep=',')

np.savetxt(outNamLabels, y0, delimiter=",", fmt='%s')
