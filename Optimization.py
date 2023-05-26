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
from densitymatrix import DensityMatrix, general_unitary, apply_unitary_to_dm, fid
from processmatrix import conversion, P_matrix, ChannelCJ, real_to_complex
from constants import *

class Results:
    def __init__(self, params, qubit_number,data_points, function, *fargs):
        self.params = params
        self.data_points = data_points
        self.function = function
        self.fargs = fargs
        self.qubit_number = qubit_number

    def minimum(self):
        return self.function(self.params,self.qubit_number, self.data_points, *self.fargs)


"""
Functions needed for the quantum state tomography
"""
"""
function_fidelity returns the optimized fidelity of our density matrix to a pure state
This is to find what unitary, U2, would need to be applied to our state to get max fidelity
and what fidelity that would be

class FidelityResults has the respective useful properties we might want to reconver
"""

def function_fidelity(x,qubit_number, dm, bell):
    U1=np.diag([1,1])
    U2=general_unitary(x[0:3])
    P = np.kron(U2,U1)
    if qubit_number > 2:
        U3=general_unitary(x[3:6])
        P = np.kron(U3,np.kron(U2,U1))
        if qubit_number > 3:
            U4=general_unitary(x[6:9])
            P = np.kron(U4,np.kron(U3,np.kron(U2,U1)))
    return -np.abs(dm.apply_unitary(P).fidelity(bell))

class FidelityResults(Results):
    @property
    def u1(self):
        return np.diag([1,1])
    
    @property
    def u2(self):
        return general_unitary(self.params[0:3])
    
    @property
    def u3(self):
        return general_unitary(self.params[3:6])
    
    @property
    def u4(self):
        return general_unitary(self.params[6:9])    

    @property
    def optimized_state(self):
        return self.data_points.apply_unitary(self.u)

    @property
    def u(self):
        return np.kron(self.u2,self.u2)

    def correct(self,qubit_number):
        P = np.kron(self.u2,self.u1)
        if qubit_number > 2 :
            P = np.kron(self.u3,np.kron(self.u1,self.u2))
            if qubit_number > 3 :
                P = np.kron(self.u4,np.kron(self.u3,np.kron(self.u1,self.u2)))      
        return P

    def Density(self,fock_state):
        return(DensityMatrix(fock_state))
    
    def fock_basis(self,fock_state,qbit_number):
        
        fock_state_denisty = np.zeros((((qbit_number)**4),(qbit_number**4)),dtype = complex)
        for w in range(0,4):
            for i in range (0,4):
                fock_state_denisty[w*3+3,i*3+3] = fock_state[w,i]
        return(DensityMatrix(fock_state_denisty))
    


"""
Functions needed for the quantum process tomography
"""
"""
function_process returns the minimum of a cost function that quantifies the least sqaures difference between the experimental # of counts we meansured - data_points -
and the # of counts we should have measured theorically - repetitions*soma_f(X, oput, rhoIn) (number of runs times the probability of rhoIn colapsing into oput, 
considering rhoIn goes through channel X - that is the parameter to be optimized - and is measured in oput)

class ProcessResults has the respective useful properties we might want to reconver
"""
def function_process(X, data_points, input_number, mbasis_number, oput, rhoIn, repetitions):
    f_min=0
    counts=np.reshape(data_points, (input_number, mbasis_number)) #This reshapes counts to [j][o], j is input (order: V, H, D, R, A, L) and o is measurement basis (order: D R V A L H)
    X_real, X_imag = conversion(X)
    
    for j in range(input_number):
        for o in range(mbasis_number):
            # Defining n as a probability of measurement outcome
            nab=counts[j][o]
            soma=0+0j
            for m in range(4):
                for n in range(4):
                    t=n%4+4*(m%4)
                    soma += (X_real[t]+1j*X_imag[t])*np.conjugate(oput[o])@Pauli[m]@rhoIn[j]@Pauli[n]@np.transpose(oput[o])
            soma_f = np.abs(soma[0][0])
            f_min += ((nab-soma_f*repetitions)**2)/float(nab)
    return f_min

class ProcessResults(Results):
    @property
    def chi_matrix(self):
        return real_to_complex(self.params)

    @property
    def chi_matrix_normalized(self):
        return real_to_complex(self.params)/float(np.max(np.real(np.linalg.eig(P_matrix(self.params))[0])))

    # @property
    # def trace(self):
    #     return (np.trace(P_matrix(self.chi_matrix_normalized)))

    # @property
    # def eigenvalue(self):
    #     return np.linalg.eig(P_matrix(self.chi_matrix_normalized))[0]




"""
Functions needed for the optimization to find the unitary closest to the process matrix
"""
"""
function_unitary returns the (negative) maximum fidelity we can find between rho_jE and a unitary U (to be optimized) applied to the Bell state
it is used to find the closest unitary to a given channel, according to the Choi–Jamiołkowski isomorphism.

class UnitaryResults has the respective useful properties we might want to reconver
"""
def function_unitary(x, channel, bellmatrix):
    rho_jE=DensityMatrix(ChannelCJ(channel, bellmatrix))
    U=general_unitary(x)

    return -np.abs(rho_jE.fidelity(apply_unitary_to_dm(bellmatrix, np.kron(Pauli[0],U))))


class UnitaryResults(Results):
    @property
    def unitary(self):
        return self.params



"""
class Optimizer to find the minimum of a certain function, according to the initial_guess, the bounds, penalty and data_points

we can use the functions defined above for an optimization process and recover the paramaters associated with the minimum with the recpective classes
"""
class Optimizer:
### Once we have a state in density matrix form we want to calculate different things with it
### With this class we want to be able to easily calculate all these quantities regarding our state

    def __init__(self, initial_guess, function, results=Results):
        self.initial_guess = initial_guess
        self.function = function
        self.results = results


    def optimize(self, qubit_number, data_points, *fargs,  bounds=None, penalty=None):
        params=diffev2(self.function, self.initial_guess, args=(qubit_number,data_points, *fargs),penalty=penalty, strategy=Best1Bin, bounds=bounds, npop=50, gtol=100, disp=False, ftol=1e-8, handler=False)#, itermon=VerboseMonitor(50))
        return self.results(params, qubit_number,data_points, self.function, *fargs)