# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:35:09 2018

@author: Simon
"""
#%%

import numpy as np
from numpy import linalg as la

def nearestPD(A):
    """Find the nearest positive-definite matrix to input"""
    d = np.shape(A)[0]
    a = 0  #accumulator for the negative eigenvalues
    k = 0 #iterator on the eigenvalues
    (eigvals,P) = la.eigh(A) #A = P*eigvals*P'
    eigvals = np.real(eigvals)
    while eigvals[k] + a/(d - k) < 0:
        a += eigvals[k]
        eigvals[k] = 0
        k += 1
    eigvals[k:] += (a/(d - k)) * np.ones((d - k,))
    A1 = np.dot(np.dot(P,np.diag(eigvals)),la.inv(P))
    return A1/np.trace(A1)

