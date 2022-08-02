### Created on: 25-07-2022
### Author: Laura Martins

import numpy as np
import scipy
from scipy.stats import norm, truncnorm
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

def fid(dm, target):
    shape=np.shape(target)
    if (len(shape)>1):
        return np.trace(scipy.linalg.sqrtm(scipy.linalg.sqrtm(target)@dm@scipy.linalg.sqrtm(target)))**2
    else:
        return np.transpose(np.conjugate(target))@dm@target


class DensityMatrix:
    """
    Once we have a state in density matrix form we want to calculate different things with it
    With this class we want to be able to easily calculate all these quantities regarding our state
    """
    def __init__(self, state):
        self.state = state

    @property
    def qbit_number(self):
        return np.log2(self.state.shape[0]).astype(int)

    def apply_unitary(self, U):
        return DensityMatrix(apply_unitary_to_dm(self.state, U))

    ### Returns the fidelity of our density matrix to a pure state
    def fidelity(self, target):
        shape=np.shape(target)
        if (len(shape)>1):
            return np.trace(scipy.linalg.sqrtm(scipy.linalg.sqrtm(target)@self.state@scipy.linalg.sqrtm(target)))**2
        else:
            return np.transpose(np.conjugate(target))@self.state@target            

    """
    calculate_errors is a method that associates an uncertainty self.std to a density matrix based on the xp_counts_arr
    it considers poissonian noise and waveplates uncertainties
    """
    def calculate_errors(self, xp_counts_arr, error_runs, target):
        lines, columns = 3**self.qbit_number,2**self.qbit_number
        xp_counts_err=np.zeros((3**self.qbit_number,2**self.qbit_number), dtype=int)
        dm_sim=np.zeros((error_runs, 2**self.qbit_number,2**self.qbit_number), dtype=complex)
        fidelity_sim=np.zeros((error_runs), dtype=float)

        proj=['hh','hv','vh','vv']
        for i in range(error_runs):
            """
            we sample a value within a normal distribution for each waveplate we are using,
            given a mean value that is definied in the dictionaries in errors.py
            """
            shift_hwp_arya=np.random.normal(scale=errors.SIGMA_HWP_ARYA, size=None)
            shift_qwp_arya=np.random.normal(scale=errors.SIGMA_QWP_ARYA, size=None)
            shift_hwp_cersei=np.random.normal(scale=errors.SIGMA_HWP_CERSEI, size=None)
            shift_qwp_cersei=np.random.normal(scale=errors.SIGMA_QWP_CERSEI, size=None)
            for k in range(lines):
                N_total=np.sum(xp_counts_arr[k])
                for l in range(columns):
                    proj_basis=errors.OBSERVABLES[k]
                    angle_hwp_arya=errors.HWP_DICT[proj_basis[0]] + shift_hwp_arya
                    angle_qwp_arya=errors.QWP_DICT[proj_basis[0]] + shift_qwp_arya
                    angle_hwp_cersei=errors.HWP_DICT[proj_basis[1]] + shift_hwp_cersei
                    angle_qwp_cersei=errors.QWP_DICT[proj_basis[1]] + shift_qwp_cersei
                    """
                    r_arya (r_cersei) is the rotation matrix that arya's (cersei's) qubit goes through
                    before being measured in {V,H}
                    - the angle of rotation is determined based on the uncertainty of our waveplates
                    (determined the in lines above)
                    
                    when we do the tensor product of both (MB_change) and apply it to our state's density
                    matrix (dm_sim_WP), we are performing a basis rotation before the projection in {V, H}
                    (proj_basis_array)
                    """
                    r_arya=wp_rotation(angle_qwp_arya,np.pi/2)@wp_rotation(angle_hwp_arya,np.pi)
                    r_cersei=wp_rotation(angle_qwp_cersei,np.pi/2)@wp_rotation(angle_hwp_cersei,np.pi)

                    MB_change=np.kron(r_arya,r_cersei)

                    dm_sim_WP=MB_change@self.state@np.transpose(np.conjugate(MB_change))
                    ###proj[x][y] x and y refer to the output port of the PBS and to the player, respectively
                    proj_basis_array=np.kron(errors.PROJECTORS[proj[l][0]],errors.PROJECTORS[proj[l][1]])
                    """
                    next we need to calculate the probability (p) of the state collapsing into the state represented
                    by the projector

                    with that we can generate a new xp_counts matrix (xp_counts_err), with each entry being a sample
                    of the poissonian distibution with mean value (lam=p*N_total)
                    """
                    p=proj_basis_array@dm_sim_WP@np.transpose(np.conjugate(proj_basis_array))

                    if np.imag(p)<1e-13:
                        xp_counts_err[k][l]=np.random.poisson(lam=np.real(p)*N_total)
                    else:
                        print('You are getting complex probabilities')

            statetomo_err=LRETomography(int(self.qbit_number), xp_counts_err, str(Path(__file__).parent))
            statetomo_err.run()
            statetomo_err.quantum_state.get_density_matrix()
            dm_sim[i]=statetomo_err.quantum_state.get_density_matrix()
            
            ### I should make sure the fidelity is acutally real and not complex
            fidelity_sim[i]=np.real(fid(dm_sim[i], target))

        ### Should make sure that np.std is not assuming a normal distribution for the fidelities
        ### Otherwise I should fit a truncated normal distribution
        #self.mu, self.std, skew, kurt = truncnorm.fit(fidelity_sim, 0 , 1)
        ### Also hould divide for the sqrt(samples)?
        self.mu = np.mean(fidelity_sim)
        self.std = np.std(fidelity_sim)

    def __repr__(self):
        return repr(self.state)