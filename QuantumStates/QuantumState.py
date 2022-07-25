# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:33:42 2018

@author: Simon Neves, Robert Booth
"""

#%%


from .DensityOperator import *
from .TriDiagonalMatrix import *

#%%
import matplotlib.pyplot as plt
import numpy as np
import chart_studio.plotly.plotly as ly
from matplotlib import gridspec
import numpy.linalg as la

class QuantumState():
    """
    Class representing the current state of the system undergoing the
    tomography, independently of choosing a representation.
    
    3 Representations possible : 
        - the density operator : Class DensityOperator
        - the tridiagonal matrix : Class TriDiagonalMatrix
        - the vector t : Class numpy.ndarray

    the vector t representation is implicite, as it is included into the 
    TriDiagonalMatrix class.                             

    Initialisation Arguments:
    - `initial_matrix` : density matrix for the initial state of the system to 
    be represented
    """
    projH, projV, projD,projA, projL, projR = np.array(
            [[1, 0], [0, 0]]),np.array([[0, 0], [0, 1]]), 0.5*np.array(
            [[1,1], [1,1]]), 0.5*np.array([[1,-1], [-1,1]]) ,0.5*np.array(
                    [[1, -1j], [1j, 1]]),0.5*np.array([[1, 1j], [-1j, 1]])
    
    pauli_states = [[projD,projA],[projL,projR],[projH,projV]]

    ###########
    # PRIVATE #
    ###########
    def __init__(self, density_matrix):
        #Initialising the representations
        self._density_repr = DensityOperator(density_matrix)
        self.qbit_number = int(np.log2(np.shape(density_matrix))[0])
        self._tridiagonal_repr = TriDiagonalMatrix(self._density_repr.matrix)
        # To save time, we don't always update all the representations when one
        # is updated. The following attribut saves which representation was 
        # updated last, so it is the current representation.
        self._current_representation = "density"
    
    def _update_density_repr(self):
        """ 
        Updates the density operator representation when the state is in
        tridiagonal representation. 
        """
        if self._current_representation != "density":
            self._density_repr.matrix = self._tridiagonal_repr.get_density_matrix(
                    ).matrix
            self._current_representation = "density"

    def _update_tridiagonal_repr(self):
        """ 
        Updates the tridiagonal representation when the state is in density
        operator representation.
        """
        if self._current_representation != "tridiagonal":
            self._tridiagonal_repr.set_matrix(self._density_repr.matrix)
            self._current_representation = "tridiagonal"

    ###########
    # METHODS #
    ###########
    def density_representation(self):
        """
        Returns the `DensityOperator` representation for the current state.
        """
        self._update_density_repr()
        return self._density_repr

    def tridiagonal_representation(self):
        """
        Returns the `TriDiagonalMatrix` representation for the current state.
        """
        self._update_tridiagonal_repr()
        return self._tridiagonal_repr
    
    def set_vector(self,t_vect):
        """
        Sets the vector representation, and so the tridiagonal representation 
        of the state.
        """
        self._current_representation = 'tridiagonal'
        self._tridiagonal_repr.set_vector(t_vect)
    
    def set_density_matrix(self,density_matrix):
        """
        Sets the density operator representation of the state.
        """
        self._current_representation = 'density'
        self._density_repr.matrix = density_matrix
        self._density_repr.make_positive()

    def get_t_matrix(self):
        """
        Returns the matrix of the tridiagonal representation of the state.
        """
        self._update_tridiagonal_repr()
        return self._tridiagonal_repr.get_t_matrix()
    
    def get_vector(self):
        """
        Returns the vector representation of the state.
        """
        self._update_tridiagonal_repr()
        return self._tridiagonal_repr.get_vector()
    
    def get_density_matrix(self):
        """
        Returns the density matrix of the density operator of the state.
        """
        self._update_density_repr()
        return self._density_repr.matrix

    def expectation(self, operator):
        """
        Returns the expectation value for an operator in the current state.
        """
        self._update_density_repr()
        return self._density_repr.expectation(operator)

    def pauli_state_proba(self, pauli_combination,eigenvector_combination):
        """
        Returns the expectation value for the tensor product of projector.

        Arguments:
        - `projector_combination` : tuple of indices correponding to a tensor
        product of projectors, the n-th of which indicates which projector
        to measure on the n-th qbit.
        """
        
        proj_tensor = 1
        for qbit in range(self.qbit_number):
            proj_tensor = np.kron(proj_tensor, self.pauli_states[
                    pauli_combination[qbit]][eigenvector_combination[qbit]])
        if self._current_representation == "tridiagonal":
            return self._tridiagonal_repr.get_density_matrix(
                    ).expectation(proj_tensor)
        else:
            return self._density_repr.expectation(proj_tensor)

    def fidelity(self, pure_state):
        """
        Returns the fidelity relative to the `PureState` object 
        passed as argument.
        """
        self._update_density_repr()
        return self._density_repr.fidelity(pure_state)

    def purity(self):
        self._update_density_repr()
        return self._density_repr.purity()
    
    def eig(self):
        """ 
        Returns the eigen values and vectors of the current state
        """
        self._update_density_repr()
        return self._density_repr.eig()
    
    def trace(self):
        self._update_density_repr()
        return self._density_repr.trace()
    
    def plot(self):
        """
        Plots the real and imaginary parts of the density matrix of the state,
        in two separate subplots.
        """
        self._update_density_repr()
        self.density_representation().plot()
    