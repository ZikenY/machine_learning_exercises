#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 14:48:00 2017

@author: zhiqianyou
"""

import numpy as np
import matplotlib.pyplot as plt

ra = np.pi / 6
rotate = np.array([[np.cos(ra), -np.sin(ra)], [np.sin(ra), np.cos(ra)]])

# covar matrix must be positive definite, or symmetric with an inverse for Gaussian distribution
covarMat = np.array([[1.0, 0.0], [0.0, 6.0]]) #2x2 array

# ------------------------  eigenvalue decomposition ------------------------
eig_val, eig_vec = np.linalg.eig(covarMat)                              # decompose covarMat to eigen-value matrix and eigen-vector matrix 
eig_vec = np.dot(rotate, eig_vec)                                       # Rotate the eigenvectors
eigval_mat = np.diag(eig_val[0:])                                       # extract eigen-value matrix
covarMat = np.dot(np.dot(eig_vec, eigval_mat), np.linalg.inv(eig_vec))  # Recomposition the covarMat
# ---------------------------------------------------------------------------

covarInv = np.linalg.inv(covarMat)

# In this way the result is the same, but the mechanism is little bit tricky.
#covarMat = np.dot(covarMat, rotate)
#covarInv = np.linalg.inv(covarMat)
#covarInv = np.dot(covarInv, rotate)
# ---------------------------------------------------------------------------

def gaussValue(x): # x is 2-element column vec
    return np.exp(-0.5 * np.dot(np.dot(x.T, covarInv), x))

# create a range from -6 to 6 in 0.2 increments. Length is 61.
pltRange = np.linspace(-3, 7, num=61, endpoint = True)
# array to hold the values for a gaussian function
# index z arrays by [i, j)
# indices are from 0 to 60
z1 = np.zeros((61, 61))

# for a 2D array, (0, 0) is top left:
# also 1st component of array index is y (vertical)

# Generate values
i = 0
for x in pltRange:
    j = 0
    for y in pltRange:
        pt1 = np.array([[x-1], [y-2]]) # column vector. 2x1 array.
        z1[j, i] = gaussValue(pt1)
        j = j + 1
    i = i + 1
    
plt.figure()
CS = plt.contour(pltRange, pltRange, z1)
plt.grid