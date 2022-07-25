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
from DensityMatrix import general_unitary#, apply_unitary_to_dm, fidelity_to_pure

### Returns the optimized fidelity of our density matrix to a pure state
### This is to find what unitaries would need to be applied to our state to get max fidelity
### and what fidelity that would be
def function_fidelity(x, dm, bell):
    U1=general_unitary(x[:3])
    U2=general_unitary(x[3:])
    return -np.abs(dm.apply_unitary(np.kron(U1,U2)).fidelity_to_pure(bell))


class Results:
    def __init__(self, params, data_points, function, *fargs):
        self.params = params
        self.data_points = data_points
        self.function = function
        self.fargs = fargs
    
    def minimum(self):
        return self.function(self.params, self.data_points, *self.fargs)

class FidelityResults(Results):
    @property
    def u1(self):
        return general_unitary(self.params[:3])

    @property
    def u2(self):
        return general_unitary(self.params[3:])

    @property
    def optimized_state(self):
        return self.data_points.apply_unitary(np.kron(self.u1,self.u2))

class Optimizer:
### Once we have a state in density matrix form we want to calculate different things with it
### With this class we want to be able to easily calculate all these quantities regarding our state

    def __init__(self, initial_guess, function, results=Results):
        self.initial_guess = initial_guess
        self.function = function
        self.results = results

    def optimize(self, data_points, *fargs, bounds=None):
        params=diffev2(self.function, self.initial_guess, args=(data_points, *fargs), strategy=Best1Bin, bounds=bounds, npop=100, gtol=100, disp=True, ftol=1e-20, itermon=VerboseMonitor(50), handler=False)
        return self.results(params, data_points, self.function, *fargs)