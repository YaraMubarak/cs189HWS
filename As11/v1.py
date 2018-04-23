#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 16:25:53 2018

@author: ymubarak
"""

import numpy as np 
import pandas as pd 
from scipy.stats import pearsonr 

folder = "/home/ymubarak/Documents/CS189/As11/"
hdi = pd.read_csv(folder + "world-values-hdi.csv")
hdi = np.array(hdi)
hdi = hdi[:,1]
hdi = hdi.astype(float)

X = pd.read_csv(folder + "world-values.csv" )
feature_headers = list(X)
feature_headers.pop(0)
X = np.array(X)
X = X[:,range(1,X.shape[1])]
X = X.astype(float)


#%% Part A 

corrs = [] 
for i in range(X.shape[1]) : 
    corrs.append(pearsonr(X[:,i], hdi)[0]  )
    
print("Most Pos Correlated feature is " + str(feature_headers[np.argmax(corrs)]))

print("Most Neg Correlated feature is " + str(feature_headers[np.argmin(corrs)]))

print("Least Correlated feature is " + str(feature_headers[np.argmin(np.absolute(corrs))]))

# %% PART B 

