# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 16:57:04 2021

@author: Verena
"""

import numpy as np
import time
from copy import deepcopy

# Import Qiskit classes
import qiskit
import qiskit.quantum_info as qi
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit.providers.aer import noise
from qiskit.compiler import assemble


# Tomography functions
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.ignis.verification.tomography import process_tomography_circuits, ProcessTomographyFitter
from qiskit.ignis.verification.tomography import gateset_tomography_circuits, GatesetTomographyFitter
import qiskit.ignis.mitigation.measurement as mc

# Auxiliary methods
from qiskit.quantum_info import Choi, Kraus
from qiskit.extensions import HGate, XGate

# Process tomography of a Hadamard gate
q = QuantumRegister(1)
circ = QuantumCircuit(q)
circ.h(q[0])

# Get the ideal unitary operator
target_unitary = qi.Operator(circ)

# Generate process tomography circuits and run on qasm simulator
qpt_circs = process_tomography_circuits(circ, q)
job = qiskit.execute(qpt_circs, Aer.get_backend('qasm_simulator'), shots=4000)


###################################################################
#####################
#testing the state tomography to compare with choi
# Fit result

qst_state= state_tomography_circuits(circ, q)
job1 = qiskit.execute(qst_state, Aer.get_backend('qasm_simulator'), shots=4000)

tomo_fitter_state = StateTomographyFitter(job1.result(), qst_state)


# Perform the tomography fit
# which outputs a density matrix
rho_fit_state = tomo_fitter_state.fit(method='lstsq')
###################################################################
#######################################################################


# Extract tomography data so that counts are indexed by measurement configuration
qpt_tomo = ProcessTomographyFitter(job.result(), qpt_circs)
qpt_tomo.data


# Tomographic reconstruction
#t = time.time()
choi_fit_lstsq = np.asarray(qpt_tomo.fit(method='lstsq'))

#print('Fit time:', time.time() - t)
print('Average gate fidelity: F = {:.5f}'.format(qi.average_gate_fidelity(choi_fit_lstsq, target=target_unitary)))



hoppa=np.dot(choi_fit_lstsq,np.kron(np.transpose(rho_fit_state), np.eye(2)))
hoppa2=np.dot(choi_fit_lstsq, np.transpose(rho_fit_state))


