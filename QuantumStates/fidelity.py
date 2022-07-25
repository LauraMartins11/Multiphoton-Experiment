# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 02:48:41 2018

@author: Simon
"""

#%%

from .QuantumState import *
#%%

from scipy.linalg import sqrtm
import numpy as np

def fidelity(qs1,qs2):
        """Returns mixed state qs1's fidelity to mixed state qs2 (symetric)."""
        matrix1, matrix2 = qs1.get_density_matrix(), qs2.get_density_matrix()
        sqrtmatrix1 = sqrtm(matrix1)
        return np.trace(sqrtm(sqrtmatrix1 @ matrix2 @ sqrtmatrix1))**2
    
def trace_distance(qs1,qs2): 
    """Returns the trace distance between mixed states qs1 and qs2."""
    mat1, mat2 = qs1.get_density_matrix(), qs2.get_density_matrix()
    return 0.5*np.trace(sqrtm((mat1-mat2).transpose().conj() @ (mat1-mat2)))
