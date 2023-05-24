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
    
def tensor_product(state1,state2):
    return(np.kron(state1,state2))


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
    
    def fock_basis(self,qbit_number):
        fock_state = []
        fock_state = self.state
        self.state = np.zeros((((qbit_number)**4),(qbit_number**4)),dtype = complex)
        if len(fock_state) < 5 :
            for w in range(0,4):
                for i in range (0,4):
                    self.state[w*3+3,i*3+3] = fock_state[w,i]
        else :
                self.state = fock_state
class GHZ: 

    def __init__(self,state1,state2):
        self.fock_state1 = state1
        self.fock_state2 = state2

    def GHZ_before_BS(self):
        self.GHZ_state_before_BS = tensor_product(self.fock_state1,self.fock_state2)
        
    
    def fidelity(self, target):
        shape=np.shape(target)
        if (len(shape)>1):
            return (np.trace(scipy.linalg.sqrtm(scipy.linalg.sqrtm(self.GHZ_state)@target@scipy.linalg.sqrtm(self.GHZ_state))))**2/(np.trace(self.GHZ_state)*np.trace(target))
        else:
            return np.transpose(np.conjugate(target))@self.GHZ_state@target
        
    def BeamSplitter(self):
        X = np.array([[0,1],[1,0]])
        Z = np.array([[1,0],[0,-1]])
        PBS = np.bmat([[np.diag([1,1]),np.zeros((2,2))],[np.zeros((2,2)),X@Z]])
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
        PBS_swap = swap@PBS
        Beamsplitter = tensor_product(np.diag([1,1]),tensor_product(PBS_swap,np.diag([1,1])))
        self.GHZ_state = Beamsplitter@self.GHZ_state_before_BS
