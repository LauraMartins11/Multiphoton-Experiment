### Created on: 25-07-2022
### Author: Laura Martins

import numpy as np
import scipy
import time

import os
import glob

from pathlib import Path
import fnmatch

import mystic
from mystic.solvers import DifferentialEvolutionSolver, diffev2
from mystic.strategy import Best1Bin
from mystic.monitors import Monitor,VerboseMonitor
from DensityMatrix import general_unitary, apply_unitary_to_dm, fidelity_to_pure

### Returns the optimized fidelity of our density matrix to a pure state
### This is to find what unitaries would need to be applied to our state to get max fidelity
### and what fidelity that would be
def function_fidelity(x, data_points, bell):
    U1=general_unitary(x[:3])
    U2=general_unitary(x[3:])
    return -np.abs(fidelity_to_pure(bell, apply_unitary_to_dm(np.kron(U1,U2), data_points)))

class Optimizer():
### Once we have a state in density matrix form we want to calculate different things with it
### With this class we want to be able to easily calculate all these quantities regarding our state

    def __init__(self, qbit_number, params, function, working_dir):
        """
        Initialisation of the tomography.
        - 'qbit_number' : number of qubits
        - 'xp_counts_array': array of experimental counts that can be passed
        as an argument to initialize an object of class XPCounts
        - 'working_dir' : directory to save and load data
        """
        self.qbit_number = qbit_number
        self.params = params
        self.working_dir = working_dir
        self.function = function
        os.chdir(self.working_dir)


    def optimize(self, x0, bounds, data_points, *args):
        self.params=diffev2(self.function, x0, args=(data_points, *args), strategy=Best1Bin, bounds=bounds, npop=100, gtol=100, disp=True, ftol=1e-20, itermon=VerboseMonitor(50), handler=False)
        #self.minimum=eval(self.function)