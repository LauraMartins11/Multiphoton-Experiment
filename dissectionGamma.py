# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 20:13:07 2021

@author: Verena
"""

from ProjectorCounts import *
from NestedForLoop import get_iterator

#%%
# Measurement bases indexation :
# X, Y, Z  -> 0, 1, 2

#Eigenstates indexation
# |+> , |-> -> 0,1

# Pauli operators indexation :
# I, X, Y, Z -> 0, 1, 2, 3


# Tensor of conversion between Pauli projectors and Pauli matrices
Gamma_matrix = np.array([[[1/3,1/3],[1/3,1/3],[1/3,1/3]],
                             [[+1,-1],[0,0],[0,0]],
                             [[0,0],[+1,-1],[0,0]],
                             [[0,0],[0,0],[+1,-1]]])/2

qbit_number= 4

eigen_iterator  = get_iterator(2,qbit_number) 
basis_iterator = get_iterator(3,qbit_number)
pauli_iterator = get_iterator(4,qbit_number)




pauli_expectations_array = np.zeros(4**qbit_number)
for pauli_index in range(4**qbit_number):
    pauli_expect = 0
    for basis_index in range(3**qbit_number):
        for eigen_index in range(2**qbit_number):
            Gamma_prod = 1
            for i,j,k in zip(pauli_iterator[pauli_index],
                             basis_iterator[basis_index],
                             eigen_iterator[eigen_index]):
                Gamma_prod *= Gamma_matrix[i,j,k]
                print(i,j,k,Gamma_prod)

    