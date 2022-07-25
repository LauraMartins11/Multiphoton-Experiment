# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 14:01:50 2021

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

# Create the expected statevector
q2 = QuantumRegister(2)
bell = QuantumCircuit(q2)
bell.h(q2[0])
bell.cx(q2[0], q2[1])
print(bell)

target_state_bell = qi.Statevector.from_instruction(bell)
print(target_state_bell)



# Create the actual circuit
q2 = QuantumRegister(6)
bell = QuantumCircuit(q2)
bell.h(q2[3])
bell.cx(q2[3], q2[5])
print(bell)


# Generate circuits and run on simulator
t = time.time()

# Generate the state tomography circuits.
qst_bell = state_tomography_circuits(bell, [q2[3], q2[5]])

# Execute
job = qiskit.execute(qst_bell, Aer.get_backend('qasm_simulator'), shots=5000)
print('Time taken:', time.time() - t)

# Fit result
tomo_fitter_bell = StateTomographyFitter(job.result(), qst_bell)


# Perform the tomography fit
# which outputs a density matrix
rho_fit_bell = tomo_fitter_bell.fit(method='lstsq')
F_bell = qi.state_fidelity(rho_fit_bell, target_state_bell)
print('State Fidelity: F = {:.5f}'.format(F_bell))



##################################################
##################################################
#Add measurement noise
noise_model = noise.NoiseModel()
for qubit in range(6):
    read_err = noise.errors.readout_error.ReadoutError([[0.75, 0.25],[0.1,0.9]])
    noise_model.add_readout_error(read_err,[qubit])

#generate the calibration circuits
meas_calibs, state_labels = mc.complete_meas_cal(qubit_list=[3,5])

backend = Aer.get_backend('qasm_simulator')
job_cal = qiskit.execute(meas_calibs, backend=backend, shots=15000, noise_model=noise_model)
job_tomo = qiskit.execute(qst_bell, backend=backend, shots=15000, noise_model=noise_model)

meas_fitter = mc.CompleteMeasFitter(job_cal.result(),state_labels)

tomo_bell = StateTomographyFitter(job_tomo.result(), qst_bell)

#no correction
rho_bell = tomo_bell.fit(method='lstsq')
F_bell = qi.state_fidelity(rho_bell, target_state_bell)
print('State fidelity (no correction): F = {:.5f}'.format(F_bell))

#correct data
correct_tomo_results = meas_fitter.filter.apply(job_tomo.result(), method='least_squares')
tomo_bell_mit = StateTomographyFitter(correct_tomo_results, qst_bell)
rho_fit_bell_mit = tomo_bell_mit.fit(method='lstsq')
F_bell_mit = qi.state_fidelity(rho_fit_bell_mit, target_state_bell)
print('State fidelity (w/ correction): F = {:.5f}'.format(F_bell_mit))






