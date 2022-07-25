### Created on: 25-07-2022
### Author: Laura Martins

import numpy as np
import scipy
import time

import os
import glob

from pathlib import Path
import fnmatch

class DensityMatrix():
### Once we have a state in density matrix form we want to calculate different things with it
### With this class we want to be able to easily calculate all these quantities regarding our state

    def __init__(self, qbit_number, state, working_dir):
        """
        Initialisation of the tomography.
        - 'qbit_number' : number of qubits
        - 'xp_counts_array': array of experimental counts that can be passed
        as an argument to initialize an object of class XPCounts
        - 'working_dir' : directory to save and load data
        """
        self.qbit_number = qbit_number
        self.state = state
        self.working_dir = working_dir
        os.chdir(self.working_dir)

    def general_unitary(x):
        return np.array([[np.exp(1j*x[0])*np.cos(x[2]), np.exp(1j*x[1])*np.sin(x[2])],
                        [-np.exp(-1j*x[1])*np.sin(x[2]), np.exp(-1j*x[0])*np.cos(x[2])]])

    def wp_rotation(t, n):
        R= np.exp(-1j*n/2)*np.array([[np.cos(t)**2+np.exp(1j*n)*np.sin(t)**2,(1-np.exp(1j*n))*np.cos(t)*np.sin(t)],
                [(1-np.exp(1j*n))*np.cos(t)*np.sin(t),np.sin(t)**2+np.exp(1j*n)*np.cos(t)**2]])
        return(R)

    def apply_unitary_to_dm(self, U):
        return U@self.state@np.transpose(np.conjugate(U))

    ### Returns the fidelity of our density matrix to a pure state
    def fidelity_to_pure(self, pure):
        return (pure@self.state@np.transpose(np.conjugate(pure)))

    