# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 22:37:17 2021

@author: Verena & Laura
"""

import numpy as np
import scipy
import time

import mystic
from mystic.solvers import DifferentialEvolutionSolver, diffev2
from mystic.strategy import Best1Bin
from mystic.monitors import Monitor,VerboseMonitor

from copy import deepcopy

from Tomography import *

from NestedForLoop import get_iterator
from pathlib import Path

#working_dir=r'/Users/francmartins/Documents/PhD/tomography_run_main.py/tomopy/'
working_dir = r"C:\Users\LauraMartins\Documents\PhD\Shared\tomography_run_main.py\tomopy"
print("Working Directory:", working_dir)

rhoIn=[]
rhoOUT=[]
Lambdas=[]
Rs=[]
pass_prob=[]

input_number=6
mbasis_number=6

oput=np.zeros((mbasis_number,1,2), dtype=complex)
iput=np.zeros((input_number,1,2), dtype=complex)
# The order of state: D R V A L H (To match the order we use below that is already too deep in the code)
oput[2]=V=np.array([1,0])
oput[5]=H=np.array([0,1])
oput[0]=D=np.array([1,1])/np.sqrt(2)
oput[3]=A=np.array([1,-1])/np.sqrt(2)
oput[1]=R=np.array([1,1j])/np.sqrt(2)
oput[4]=L=np.array([1,-1j])/np.sqrt(2)

counts=np.zeros((input_number,2,3))
sigma_counts=np.zeros((input_number,2,3), dtype=float)
xp_counts=np.zeros((input_number,3,2))
dirinv=np.zeros((input_number,2,2), dtype=complex)

qubit_number=1
repetitions=1e16 #This should be in a dictionary or somthing and should be updaed when we run the measurement

Pauli=np.asarray([
    [[1,0],
    [0,1]],


    [[0,1],
    [1,0]],


    [[0,-1j],
    [1j,0]],


    [[1,0],
    [0,-1]]])

for j in range(input_number):
    if j==0:
        print("Input state 0: |VxV|")
        iput[j]=V
        rhoIn.append(np.outer(V,np.conjugate(V)))
    if j==1:
        print("Input state 1: |HxH|")
        iput[j]=H
        rhoIn.append(np.outer(H,np.conjugate(H)))
    if j==2:
        print("Input state 2: |DxD|")
        iput[j]=D
        rhoIn.append(np.outer(D,np.conjugate(D)))
    if j==3:
        print("Input state 3: |RxR|")
        iput[j]=R
        rhoIn.append(np.outer(R,np.conjugate(R)))
    if j==4:
        print("Input state 4: |AxA|")
        iput[j]=A
        rhoIn.append(np.outer(A,np.conjugate(A)))
    if j==5:
        print("Input state 5: |LxL|")
        iput[j]=L
        rhoIn.append(np.outer(L,np.conjugate(L)))

    ######## Simulated counts using Simon's tomography functions ##########
    bases=np.array([
        [str(j+1)+'D',str(j+1)+'R',str(j+1)+'V'],
        [str(j+1)+'A',str(j+1)+'L',str(j+1)+'H']])

    for v in range(2):
        for w in range(3):

            with open("./Generated_data/"+bases[v][w]+".txt") as file:
                mean=0
                #Repeat for each song in the text file
                for i, line in enumerate(file):
                #Let's split the line into an array called "fields" using the " " as a separator:
                    fields = line.split()
                    mean = mean+int(np.longdouble(fields[0]))

                mean = int(mean/(i+1))

                #counts[j][v][w]=mean
                counts[j][v][w]=np.longdouble(fields[0])

    xp_counts[j][:][:]=np.array(np.transpose(counts[j][:][:])) # get the experimental count

    Iout=np.sum(counts[j][:][:])
    pass_prob.append(Iout/(3*repetitions))

    # State tomo with direct inversion
    #run=Tomography(qubit_number,xp_counts[j][:][:],working_dir)
    #dirinv[j][:][:]=run.direct_inversion()
    # State tomo with linear regression
    run=LRETomography(int(qubit_number), xp_counts[j][:][:], working_dir)
    dirinv[j][:][:]=run.LREstate()
    #print('DIRECT INVERSION: \n', dirinv[j][:][:])

    # Decomposing the output states into lambdas for the Chi inversion (we only use the first 4 input states)
    if j<4:
        a = np.array([[1, 1],[1,-1]])
        b = np.array([dirinv[j][0][0], dirinv[j][1][1]])
        sol_L = np.linalg.solve(a,b)

        e = np.array([[1, -1j],[1,+1j]])
        f = np.array([dirinv[j][0][1], dirinv[j][1][0]])
        sol_I = np.linalg.solve(e,f)

        c = np.array([[1, 1],[1,-1]])
        d = np.array([rhoIn[j][0][0], rhoIn[j][1][1]])
        sol_R = np.linalg.solve(c,d)

        # Lambdas.append([sol_L[0],np.real(rho_fit_state[1][0]),np.imag(rho_fit_state[1][0]),sol_L[1]])
        Lambdas.append([sol_L[0],sol_I[0],sol_I[1],sol_L[1]])
        Rs.append([sol_R[0],np.real(rhoIn[j][1][0]),np.imag(rhoIn[j][1][0]),sol_R[1]])

        # Considering the losses, we multiply the density matrix by the probability of measuring a photon
        Lambdas[j][:]=np.array(Lambdas[j][:])*pass_prob[j]

#print('LAMBDA: ', Lambdas)
#print('RS: ', Rs)

iterator = get_iterator(4,3)
B=np.zeros((4,4,4,4), dtype=complex)

for m, n, j in iterator:
    temp = np.zeros((2,2), dtype=complex)
    for i in range(4):
        temp += (Rs[j][i]*Pauli[m]@Pauli[i]@Pauli[n])

    a = np.array([[1, 1],[1,-1]])
    b = np.array([temp[0][0],temp[1][1]])
    sol_diagonal = np.linalg.solve(a,b)
    c = np.array([[1, -1j],[1,1j]])
    d = np.array([temp[0][1],temp[1][0]])
    sol_anti_diagonal = np.linalg.solve(c,d)
    B[m][n][j][0]=sol_diagonal[0]
    B[m][n][j][1]=sol_anti_diagonal[0]
    B[m][n][j][2]=sol_anti_diagonal[1]
    B[m][n][j][3]=sol_diagonal[1]

    # Verification step
    ver=B[m][n][j][0]*Pauli[0]+B[m][n][j][1]*Pauli[1]+B[m][n][j][2]*Pauli[2]+B[m][n][j][3]*Pauli[3]==temp
    if False in ver:
        print('B matrix is not verifying right condidion')

Bs_new = np.transpose(np.reshape(B,(16,16)))

Kabba=np.linalg.inv(Bs_new)

lambdas_vect=np.reshape(Lambdas,[16,1])

Chi_vector=Kabba@lambdas_vect

############### Now we need to define the function that will be optimized in order to find the Chi ###############
############### I don't understand why I have to pass counts as an argumen but not oput, iput, etc ###############

# f is the function w want to optimize
def f(X, *args):
    counts=args
    f_min=0
    counts=np.reshape(counts, (input_number, mbasis_number)) #This reshapes counts to [j][o], j is input (order: V, H, D, R, A, L) and o is measurement basis (order: D R V A L H)
    for j in range(input_number):
        for o in range(mbasis_number):
            # Defining n as a probability of measurement outcome
            nab=counts[j][o]/float((counts[j][o]+counts[j][o+(-1)**int(o/3)*3]))
            soma=0+0j
            for m in range(4):
                for n in range(4):
                    t=n%4+4*(m%4)
                    soma += (X[t]+1j*X[t+16])*np.conjugate(oput[o])@Pauli[m]@rhoIn[j]@Pauli[n]@np.transpose(oput[o])
            soma_f = np.abs(soma[0][0])
            #f_min += ((nab-soma_f)**2)/float(nab)
            f_min += ((nab-soma_f)**2) # We need to be sure about which function we want to minimize (if it makes sense to use the number of counts or the probability)
    return f_min
# Now the problem is that some is complex and in order to find a minimum we have to take the real part or the abs value

# Defining the constraints of the minimization function
def P_matrix(X):
    soma_c=np.zeros((2,2), dtype=complex)
    for m_c in range(4):
        for n_c in range(4):
            t_c=n_c%4+4*(m_c%4)
            soma_c += (X[t_c]+1j*X[t_c+16])*Pauli[n_c]@Pauli[m_c]
    #if np.imag(np.trace(soma_c))>1e-17:
        #print('Imaginary part of trace is too big: ', np.imag(np.trace(soma_c)))
    return soma_c

# Defining restrictions for the optimization function
def neg_part(Y):
    return np.sum(np.clip(Y, -np.inf, 0))

from mystic.penalty import quadratic_inequality,quadratic_equality
def penalty1 (X):
    return np.real(np.trace(P_matrix(X)))-2
def penalty2 (X):
    return -neg_part(np.real(np.linalg.eig(P_matrix(X))[0]))
def penalty3(X):
    return neg_part(np.real(np.linalg.eig(P_matrix(X))[0]))-1

def penalty4 (X):
    return np.real(P_matrix(X)[0][1]-np.conjugate(P_matrix(X)[1][0]))
def penalty5 (X):
    return np.imag(P_matrix(X)[0][1]-np.conjugate(P_matrix(X)[1][0]))
def penalty6(X):
    return np.imag(P_matrix(X)[0][0])
def penalty7(X):
    return np.imag(P_matrix(X)[1][1])
def penalty8(X):
    return neg_part(np.imag(np.linalg.eig(P_matrix(X))[0]))


@quadratic_inequality(penalty1, k=1e12)
@quadratic_inequality(penalty2, k=1e12)
@quadratic_inequality(penalty3, k=1e12)

@quadratic_equality(penalty4, k=1e12)
@quadratic_equality(penalty5, k=1e12)
@quadratic_equality(penalty6, k=1e12)
@quadratic_equality(penalty7, k=1e12)
@quadratic_equality(penalty8, k=1e12)

def penalty(x):
    return 0.0

# Function f doesn't optimize on the complex space, so we need functions to double the params:
def complex_to_real(z):      # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))

def real_to_complex(z):      # real vector of length 2n -> complex of length n
    return z[:len(z)//2] + 1j * z[len(z)//2:]

Chi_initial=complex_to_real(Chi_vector).flatten()
print('FUNCTION: ', f(Chi_initial, counts))
print('PENALTY: ', penalty(Chi_initial))

# Running the optimization
monitor = VerboseMonitor(50)
npop = 50
result_y=diffev2(f, x0=Chi_initial, args=counts, strategy=Best1Bin, bounds=[(-1,1)]*32, penalty=penalty, npop=npop, gtol=100, disp=True, ftol=1e-10, itermon=monitor, handler=False)

# Normalizing the Chi matrix to the maximum eigenvalue
def Eigenvalue(X):
    return np.linalg.eig(P_matrix(X))[0]
Final_Chi_vector=real_to_complex(result_y/float(np.max(Eigenvalue(result_y))))


# Prints to verify the optimization ran properly
def Trace_cond(X):
    return 2.0-np.real(np.trace(P_matrix(X)))

def Hermitian(X):
    return P_matrix(X)[0][1]-np.conjugate(P_matrix(X)[1][0])

def Trace_real(X):
    return np.imag(P_matrix(X)[0][0]+P_matrix(X)[1][1])

def Eigenvalue_real_cond(X):
    return np.imag(np.linalg.eig(P_matrix(X))[0])

print('CONDITIONS TO VERIFY: \n')

print('Trace of P normalized (should be <= 2): \n i: ', np.trace(P_matrix(Chi_initial))/float(np.max(Eigenvalue(Chi_initial))), '\n f: ', np.trace(P_matrix(result_y/float(np.max(Eigenvalue(result_y))))), '\n')
print('Eigenvalues of P matrix (should be positive and <= 1): \n i: ',np.linalg.eig(P_matrix(Chi_initial))[0]/float(np.max(Eigenvalue(Chi_initial))) ,'\n f: ', np.linalg.eig(P_matrix(result_y))[0], '\n')
print('Function value (should be minimized): \n i: ', f(Chi_initial, counts), '\n f: ', f(result_y, counts))
print('Max P matrix (should be 1): ', np.max(Eigenvalue(result_y)))

print('\n')

print('COND 1: Trace >= 0: ', Trace_cond(result_y))
print('COND 2: Hermitian = 0: ', Hermitian(result_y))
print('COND 3: Trace_real = 0: ', Trace_real(result_y))
print('COND 5: Eigenvalue_real_cond = 0: ', Eigenvalue_real_cond(result_y))
print('COND 6: Eigenvalue_pos_cond >= 0 && <=1: ', Eigenvalue(result_y))

print('\n')

print('COND 1: Trace >= 0: ', Trace_cond(Chi_initial))
print('COND 2: Hermitian = 0: ', Hermitian(Chi_initial))
print('COND 3: Trace_real = 0: ', Trace_real(Chi_initial))
print('COND 5: Eigenvalue_real_cond = 0: ', Eigenvalue_real_cond(Chi_initial))
print('COND 6: Eigenvalue_pos_cond >= 0 && <=1: ', Eigenvalue(Chi_initial))

# Verification of the Chi matrix for the input states
# For an identity channel, the final state should be the same as the initial state but with noise
def verification(X, Y):
    for initial in range(input_number):
        verification=rhoIn[initial]
        Finali_state=np.zeros((2,2), dtype=complex)
        Finalf_state=np.zeros((2,2), dtype=complex)
        for m in range (4):
            for n in range(4):
                t=n%4+4*(m%4)
                Finali_state += Y[t]*Pauli[m]@verification@Pauli[n]
                Finalf_state += X[t]*Pauli[m]@verification@Pauli[n]
        print('INITIAL STATE: \n', verification)
        print('OUTPUT STATE: \n', dirinv[initial][:][:])
        print('FINAL STATE (Chi direct inversion): \n', Finali_state)
        print('FINAL STATE (Chi optmized): \n', Finalf_state)
        print('\n \n')
    pass

verification(Final_Chi_vector, real_to_complex(Chi_initial/float(np.max(Eigenvalue(Chi_initial)))))
