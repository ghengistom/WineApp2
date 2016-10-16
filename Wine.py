# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 00:09:15 2016

@author: tom
"""

import numpy as np

from sklearn import linear_model


X = np.load("predictors.npy")

y1 = np.load("labels_color.npy")

y2 = np.load("labels_quality.npy")

y3 = np.load("labels_quality_binary.npy")

#insert scikit-learn code here

#fit(X,y1)  color  predictors  big X training set 

#fit(X,y2)  quality

#fit(X,y3)  color binary

print(X.shape)