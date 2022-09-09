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
from tomography import LRETomography
from player import Player
from constants import *

def complex_to_real(z):      # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))

def real_to_complex(z):      # real vector of length 2n -> complex of length n
    return z[:len(z)//2] + 1j * z[len(z)//2:]

#The function "conversion" converts the Chi vector from: 1 vector: 16 reals + 16 imag params;
#to: 2 vectors: 1 real with 10 params + 1 imag with 6 params
#This is to make sure Chi is Hermitian in it's structure, reducing the number of free params in the optimization process
def conversion(X):
    X_real=np.array([[X[0],X[1],X[2],X[3]],
                    [X[1],X[5],X[6],X[7]],
                    [X[2],X[6],X[10],X[11]],
                    [X[3],X[7],X[11],X[15]]]).flatten()
    X_imag=np.array([[0.0,X[17],X[18],X[19]],
                    [-X[17],0.0,X[22],X[23]],
                    [-X[18],-X[22],0.0,X[27]],
                    [-X[19],-X[23],-X[27],0.0]]).flatten()
    return(X_real, X_imag)

# P_matrix returns the P 2x2 matrix, that determins es the probability of a certain state being measured (not lost)
def P_matrix(X):
    X_real, X_imag = conversion(X)
    
    soma_c=np.zeros((2,2), dtype=complex)
    for m_c in range(4):
        for n_c in range(4):
            t_c=n_c%4+4*(m_c%4)
            soma_c += (X_real[t_c]+1j*X_imag[t_c])*Pauli[n_c]@Pauli[m_c]
    #if np.imag(np.trace(soma_c))>1e-17:
        #print('Imaginary part of trace is too big: ', np.imag(np.trace(soma_c)))
    return soma_c

# X_matrix returns the Chi 4x4 matrix (n are columns and m are lines)
def X_matrix(X):
    X_real, X_imag = conversion(X)
    
    a=np.append(X_real,X_imag)
    b=real_to_complex(a)
    c=np.reshape(b,(4,4))
    return(c)

"""
ChannelCJ returns the output state bell state going through a channel X. This is used to caluclate the fidelity of a channel,
according to the Choi–Jamiołkowski ismorphism
"""
def ChannelCJ(X, bellmatrix):

    X_real, X_imag = conversion(X)
    
    Final_state=np.zeros((4,4), dtype=complex)
    for m in range (4):
        for n in range(4):
            t=n%4+4*(m%4)
            Final_state += (X_real[t]+1j*X_imag[t])*np.kron(Pauli[0],Pauli[m])@bellmatrix@np.kron(Pauli[0],Pauli[n])
    return Final_state


"""
The class ProcessMatrix receives a process matrix that characterizes a channel in the "channel" argument
"""

class ProcessMatrix:

    def __init__(self, channel):
        self.channel_n = channel
        ### For optimization puporses, a complex matrix needs to be turned into a real array with the real and imaginary parameters as different entries of the array
        self.channel_2n = complex_to_real(channel)

    @property
    def qbit_number(self):
        return np.log2(self.state.shape[0]).astype(int)

    def find_closest_unitary(self, opt, bounds=None, penalty=None):
        bell=(np.array([1,0,0,0])+np.array([0,0,0,1]))/np.sqrt(2)
        bellmatrix=np.array(np.outer(bell, np.conjugate(bell)))

        result=opt.optimize(self.channel_2n, bellmatrix, bounds=bounds, penalty=penalty)
        return result

    def calculate_fidelity_error(self, tomographies, players, error_runs, opt, opt2, input_number, mbasis_number, oput, rhoIn, repetitions, bounds=None, penalty=None):
        
        fidelity_sim=np.zeros((error_runs), dtype=float)

        for i in range(error_runs):
            print("Simulating error_run: ", i)
            simulated_counts=[]
            for j in range(input_number):
                simulated_counts.append(tomographies[j].simulate_new_counts_with_uncertainties(players))
            
            result=opt.optimize(simulated_counts, input_number, mbasis_number, oput, rhoIn, repetitions, bounds=bounds, penalty=penalty)
            chi=ProcessMatrix(result.chi_matrix_normalized)
            fidel=chi.find_closest_unitary(opt2, bounds=[(-np.pi,np.pi)]*3)

            fidelity_sim[i]=fidel.minimum()
            
            ### I should make sure the fidelity is acutally real and not complex
            # fidelity_sim[i]=np.real(fid(dm_sim[i], target))

        """
        Should make sure that np.std is not assuming a normal distribution for the fidelities
        Otherwise I should fit a truncated normal distribution
        self.mu, self.std, skew, kurt = truncnorm.fit(fidelity_sim, 0 , 1)
        Also hould divide for the sqrt(samples)?
        """
        self.fidelity_mu = np.mean(fidelity_sim)
        self.fidelity_std = np.std(fidelity_sim)