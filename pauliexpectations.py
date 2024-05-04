# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 20:56:03 2018

@author: Simon Neves, Robert Booth
"""

# %%

from projectorcounts import *
from nestedforloop import get_iterator

# %%
# Measurement bases indexation :
# X, Y, Z  -> 0, 1, 2

# Eigenstates indexation
# |+> , |-> -> 0,1

# Pauli operators indexation :
# I, X, Y, Z -> 0, 1, 2, 3


import os
from projectorcounts import *
import numpy as np


class ExperimentalPauliExpectations:
    """
    Derived class for generating and interacting with the experimental pauli
    operators expectation values.
    Initialisation Arguments:
    - `qbit_number` : number of qbits
    - 'xp_counts': object of class XPCounts
    This last argument is used to generate the experimental Pauli Expectations.
    """

    # Tensor of conversion between Pauli projectors and Pauli matrices
    Gamma_matrix = np.array(
        [
            [[1 / 3, 1 / 3], [1 / 3, 1 / 3], [1 / 3, 1 / 3]],
            [[+1, -1], [0, 0], [0, 0]],
            [[0, 0], [+1, -1], [0, 0]],
            [[0, 0], [0, 0], [+1, -1]],
        ]
    )  # /2

    def __init__(self, qbit_number, xp_counts, xp_counts_2_emissions=None):
        # In the comments, N is the number of qbits.
        self.qbit_number = qbit_number
        self.xp_counts = XPCounts(xp_counts, self.qbit_number, xp_counts_2_emissions)
        self.xp_probas = self.xp_counts.get_xp_probas()
        # definition of iterators. lists of arrays of indices, corresponding
        # to operators, basis and states in tensors products
        self.eigen_iterator = get_iterator(2, self.qbit_number)
        self.basis_iterator = get_iterator(3, self.qbit_number)
        self.pauli_iterator = get_iterator(4, self.qbit_number)
        self.base_4 = 4 ** np.linspace(0, qbit_number - 1, qbit_number)
        # Creation of the array containing the experimental expectations
        # of pauli operators. To get the structure, see the function
        # "get_xp_pauli_expectation_array" below.
        self.create_xp_pauli_expectation_array()

    def create_xp_pauli_expectation_array(self):
        """
        Create an (4**qbit_number,)-array containing the expectations of tensor
        products of pauli operators (including identity), out of the
        experimental probability of tensor products of eigenstates of pauli
        operators.
        index : correspond to a certain tensor product of pauli operators,
        including identity.
        value : experimental expectation value of the tensor product
        """
        self.pauli_expectations_array = np.zeros(4**self.qbit_number)
        for pauli_index in range(4**self.qbit_number):
            pauli_expect = 0
            for basis_index in range(3**self.qbit_number):
                for eigen_index in range(2**self.qbit_number):
                    Gamma_prod = 1
                    for i, j, k in zip(
                        self.pauli_iterator[pauli_index],
                        self.basis_iterator[basis_index],
                        self.eigen_iterator[eigen_index],
                    ):
                        Gamma_prod *= self.Gamma_matrix[i, j, k]
                    pauli_expect += (
                        Gamma_prod * self.xp_probas[basis_index, eigen_index]
                    )
            self.pauli_expectations_array[pauli_index] = pauli_expect

    def get_pauli_expectation(self, pauli_combination):
        """
        Returns the expactation value of a tensor product of N Pauli
        operators (including identity).

        Arguments:
        - `pauli_combination` : a line-array of indices, corresponding to
        a line from self.pauli_iterator.
        """
        pauli_index = int(np.dot(self.base_4, pauli_combination))
        return self.pauli_expectations_array[pauli_index]
