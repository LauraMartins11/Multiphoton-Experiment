### Created on: 25-07-2022
### Author: Laura Martins

import numpy as np
import scipy
from scipy.stats import norm
import time

import os
import glob

from pathlib import Path
import fnmatch

import errors
from Tomography import LRETomography

def general_unitary(x):
        return np.array([[np.exp(1j*x[0])*np.cos(x[2]), np.exp(1j*x[1])*np.sin(x[2])],
                        [-np.exp(-1j*x[1])*np.sin(x[2]), np.exp(-1j*x[0])*np.cos(x[2])]])

def wp_rotation(t, n):
    R= np.exp(-1j*n/2)*np.array([[np.cos(t)**2+np.exp(1j*n)*np.sin(t)**2,(1-np.exp(1j*n))*np.cos(t)*np.sin(t)],
            [(1-np.exp(1j*n))*np.cos(t)*np.sin(t),np.sin(t)**2+np.exp(1j*n)*np.cos(t)**2]])
    return(R)

def apply_unitary_to_dm(dm, U):
    return U@dm@np.transpose(np.conjugate(U))

def fidelity(dm, pure):
        return pure@dm@np.transpose(np.conjugate(pure))


class DensityMatrix():
### Once we have a state in density matrix form we want to calculate different things with it
### With this class we want to be able to easily calculate all these quantities regarding our state

    def __init__(self, state):
        self.state = state

    @property
    def qbit_number(self):
        return np.log2(self.state.shape[0]).astype(int)

    def apply_unitary(self, U):
        return DensityMatrix(apply_unitary_to_dm(self.state, U))

    ### Returns the fidelity of our density matrix to a pure state
    def fidelity_to_pure(self, pure):
        return pure@self.state@np.transpose(np.conjugate(pure))

    def calculate_errors(self, xp_counts_arr, error_runs, bell):
        lines, columns = np.shape(errors.OBSERVABLES)
        xp_counts_err=np.zeros((3**self.qbit_number,2**self.qbit_number), dtype=int)
        dm_sim=np.zeros((error_runs, 2**self.qbit_number,2**self.qbit_number), dtype=complex)
        fidelity_sim=np.zeros((error_runs), dtype=float)

        proj=['vv', 'vh', 'hv', 'hh']
        for i in range(error_runs):
            for k in range(lines):
                N_total=np.sum(xp_counts_arr[k])
                for l in range(columns):
                    proj_basis=errors.OBSERVABLES[k][0]
                    angle_hwp_arya=np.random.normal(loc=errors.HWP_DICT[proj_basis[0]], scale=errors.SIGMA_HWP_ARYA, size=None)
                    angle_qwp_arya=np.random.normal(loc=errors.QWP_DICT[proj_basis[0]], scale=errors.SIGMA_QWP_ARYA, size=None)
                    angle_hwp_cersei=np.random.normal(loc=errors.HWP_DICT[proj_basis[1]], scale=errors.SIGMA_HWP_CERSEI, size=None)
                    angle_qwp_cersei=np.random.normal(loc=errors.QWP_DICT[proj_basis[1]], scale=errors.SIGMA_QWP_CERSEI, size=None)

                    r_arya=wp_rotation(angle_qwp_arya,np.pi/2)@wp_rotation(angle_hwp_arya,np.pi)
                    r_cersei=wp_rotation(angle_qwp_cersei,np.pi/2)@wp_rotation(angle_hwp_cersei,np.pi)

                    ### Here we need to calculate the probability for a given projector and with that calculate N_total
                    ### for the xp_counts matrix and then we apply poissonian noise
                    MB_change=np.kron(r_arya,r_cersei)

                    dm_sim_WP=MB_change@self.state@np.transpose(np.conjugate(MB_change))
                    proj_basis_array=np.kron(errors.PROJECTORS[proj[l][0]],errors.PROJECTORS[proj[l][1]])

                    p=proj_basis_array@dm_sim_WP@np.transpose(np.conjugate(proj_basis_array))

                    xp_counts_err[k][l]=np.random.poisson(lam=p*N_total)

            statetomo_err=LRETomography(int(self.qbit_number), xp_counts_err, str(Path(__file__).parent))
            statetomo_err.run() ### Runs fast maximum likelihood estimation
            statetomo_err.quantum_state.get_density_matrix()
            dm_sim[i]=statetomo_err.quantum_state.get_density_matrix()

            fidelity_sim[i]=fidelity(dm_sim[i], bell)
        self.mu, self.std = norm.fit(fidelity_sim)

    def __repr__(self):
        return repr(self.state)