
from .NearestPD import *
from .PlotMatrix import *
import matplotlib.pyplot as plt

#%%
import numpy as np

from numpy import linalg as la

class PseudoDensityOperator():
    """
    Class representing a density-like operator. All the operations are the 
    same, with the exception that the operator is a priori not non-negative.
    """

    def __init__(self, matrix):
        self.matrix = matrix

    def eig(self):
        """
        Returns the eigenvectors and eigenvalues of the density matrix. See
        `numpy.linalg.eig()` for details on the implementation.
        """
        return np.linalg.eig(self.matrix)

    def expectation(self, operator):
        """
        Returns the expectation value of the argument operator for the current
        state.
        """
        return np.trace(self.matrix @ operator)

    def make_positive(self):
        """
        Makes the density matrix non-negative.
        """
        self.matrix = nearestPD(self.matrix)

    def fidelity(self, pure_state):
        """
        Returns the state's fidelity to a pure_state.
        Argument : 'pure_state' of class PureState
        """
        return abs((pure_state.bra @ self.matrix @ pure_state.ket)[0])

    def purity(self):
        return abs(np.trace(np.linalg.matrix_power(self.matrix, 2)))

    def trace(self):
        return abs(np.trace(self.matrix))
    
    
    def plot(self):
        """
        Plots the real and imaginary parts of the density matrix,
        in two separate subplots.
        """
        real_part = np.real(self.matrix)
        imag_part = np.imag(self.matrix)
        z_max = np.max(np.concatenate((real_part,imag_part)))
        z_min = np.min(np.concatenate((real_part,imag_part)))
        fig = plt.figure(figsize=(9,5))
        real_sub = fig.add_subplot(121, projection='3d')
        plotmatrix(np.real(self.matrix), real_sub)
        real_sub.set_zlim(z_min,z_max)
        imag_sub = fig.add_subplot(122, projection='3d')
        plotmatrix(np.imag(self.matrix), imag_sub)
        imag_sub.set_zlim(z_min,z_max)
        plt.tight_layout()
        plt.show()
    
class DensityOperator(PseudoDensityOperator):
    """
    Class representing a standard density matrix and associated functions.

    Initialisation Arguments:
    - `matrix` : density matrix representation of the initial state. It is 
    systematically made positive, thanks to 'make_positive' method.
    """

    def __init__(self, matrix):
        self.matrix = matrix
        self.make_positive()
