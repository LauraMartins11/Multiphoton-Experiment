### Created on: 25-07-2022
### Author: Laura Martins

import numpy as np
import scipy
from scipy.stats import norm, truncnorm
import time
import functools as ft

import os
import glob

from pathlib import Path
import fnmatch

import errors
# from tomography import LRETomography
from player import Player

def general_unitary(x):
        return np.array([[np.exp(1j*x[0])*np.cos(x[2]), np.exp(1j*x[1])*np.sin(x[2])],
                        [-np.exp(-1j*x[1])*np.sin(x[2]), np.exp(-1j*x[0])*np.cos(x[2])]])

def wp_rotation(t, n):
    R= np.exp(-1j*n/2)*np.array([[np.cos(t)**2+np.exp(1j*n)*np.sin(t)**2,(1-np.exp(1j*n))*np.cos(t)*np.sin(t)],
            [(1-np.exp(1j*n))*np.cos(t)*np.sin(t),np.sin(t)**2+np.exp(1j*n)*np.cos(t)**2]])
    return(R)

def apply_unitary_to_dm(dm, U):
    return U@dm@np.transpose(np.conjugate(U))

def fid(dm, target):
    shape=np.shape(target)

    if (len(shape)>1):
        return (np.trace(scipy.linalg.sqrtm(scipy.linalg.sqrtm(target)@dm@scipy.linalg.sqrtm(target))))**2/(np.trace(target)*np.trace(dm))
    else:
        return np.transpose(np.conjugate(target))@dm@target


class DensityMatrix:
    """
    Once we have a state in density matrix form we want to calculate different things with it
    With this class we want to be able to easily calculate all these quantities regarding our state
    """
    def __init__(self, state):
        self.state = state

    @property
    def qbit_number(self):
        return np.log2(self.state.shape[0]).astype(int)

    def apply_unitary(self, U):
        return DensityMatrix(apply_unitary_to_dm(self.state, U))

    ### Returns the fidelity of our density matrix to a pure state
    def fidelity(self, target):
        shape=np.shape(target)
        if (len(shape)>1):
            return (np.trace(scipy.linalg.sqrtm(scipy.linalg.sqrtm(self.state)@target@scipy.linalg.sqrtm(self.state))))**2/(np.trace(self.state)*np.trace(target))
        else:
            return np.transpose(np.conjugate(target))@self.state@target

    def __repr__(self):
        return repr(self.state)