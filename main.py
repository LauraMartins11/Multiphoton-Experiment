# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:07:43 2019

@author: Verena Maged
"""
import numpy as np 
#from PauliExpectations import *
from ProjectorCounts import *
from QuantumStates import QuantumState
from QuantumStates import NearestPD
from Tomography import *
import heapq
import os
from itertools import combinations
import time
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import scipy.linalg as lin
from tqdm import tqdm
import math as m
import random 
import glob
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

################ GHZ state generator ####################################
def generate_GHZ (qubit_number):
    Jones=np.asarray([[complex(0,0),complex(1,0)],[complex(1,0),complex(0,0)]])
    sing=1
    for i in range(0,qubit_number):
        sing=np.kron(sing,Jones)
    mix=[]
    for i in range (0, len(sing)):
        mix.append (sing[i,:]+np.delete(sing,i,0))
        mix.append (sing[i,:]-np.delete(sing,i,0))
    mix= np.concatenate(np.asarray(mix))
    
    mixstates=[]
    for i in range(0,len(mix)):
        mixstates.append((1/m.sqrt(qubit_number))* np.outer(mix[i],mix[i]))
        
    mixstates=np.asarray(mixstates)
    return mixstates
#########################################################################
    

##################### Generate projectors ###############################
def generate_projectors(qubit_number):
    Pauli= np.asarray([[[0,complex(0.5,0)],[complex(0.5,0),0]],[[0,complex(0,-0.5)],[complex(0,0.5),0]],[[complex(0.5,0),0],[0,complex(-0.5,0)]]])    
    proj=[];
    preproj=1;
    for i in range (0,qubit_number):
        preproj=np.kron(preproj,Pauli)
    for i in range(0,len(preproj)):   #the appension order of the vectors depends on ++/+-/-+/-- considered in tomography 
         [ep,evp]=np.linalg.eigh(preproj[i])
                 
         proj.append(np.outer(evp[:,3],evp[:,3]))
         
         proj.append(np.outer(evp[:,1],evp[:,1]))
         
         proj.append(np.outer(evp[:,0],evp[:,0]))
         proj.append(np.outer(evp[:,2],evp[:,2]))
          
        
         
         
         
         
    proj= np.asarray(proj) 
    return proj #, preproj
###########################################################################    
    
def hist2d(data_array,color):
    # Create a figure for plotting the data as a 3D histogram.
    #
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #
    # Create an X-Y mesh of the same dimension as the 2D data. You can
    # think of this as the floor of the plot.
    #
    x_data, y_data = np.meshgrid( np.arange(data_array.shape[1]),
                                  np.arange(data_array.shape[0]) )
    #
    # Flatten out the arrays so that they may be passed to "ax.bar3d".
    # Basically, ax.bar3d expects three one-dimensional arrays:
    # x_data, y_data, z_data. The following call boils down to picking
    # one entry from each array and plotting a bar to from
    # (x_data[i], y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
    #
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = data_array.flatten()
    ax.bar3d( x_data,
              y_data,
              np.zeros(len(z_data)),
              1, 1, z_data,color=color)
    
# rotate the axes and update
   
    ax.view_init(20, 120)
        
    plt.show()   
    return x_data, y_data, z_data
 ########################################################################################   
 
def entanglement(rho):
    if min(np.shape(rho))==1:
        rho = np.dot(rho,rho.conj().transpose())
        #psi??, I think there something wrong here in m files.
    
    Z = np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])
    R = np.dot(rho,np.dot(Z,np.dot(rho.conj(),Z)))
    [r,right] = np.linalg.eig(R)
    #left = np.linalg.inv(right)
    r = np.real(r)
    
    tmp = np.sort(np.sqrt(r+0j))
    C = np.real(tmp[3]-tmp[2]-tmp[1]-tmp[0])
    C = np.max([C,0])
    
    t = C**2
    x = (1 + np.sqrt(1-t))/2
    if x == 0:
        E = 0
    elif x == 1:
        E = 1
    else:
        E = -x*np.log2(x) - (1-x)*np.log2(1-x)
    
    return E



def entropy(rho):
    [D,ignore] = np.linalg.eig(rho)
    E = np.real(D)
    S = 0
    for a in range(len(E)):
        if E[a] > 0:
            S = S - E[a]*np.log2(E[a])
            
    SS= np.trace(np.dot(rho,np.log2(rho)))       
    
    return SS

def linear_entropy(rho):
    if min(np.shape(rho)) == 1:
        lin_e = 0
    else:
        d = len(rho)
        lin_e = d * np.real(1-np.trace(np.dot(rho,rho)))/(d-1)
    
    return lin_e    

def tracedist (rho,init):
    D= 0.5* np.trace(np.sqrt((init-rho)**2))    
    return D 


def fidelityy(rho,init):
   f=  (np.trace(np.sqrt((np.dot(np.dot(np.sqrt(init),rho),np.sqrt(init))))))**2
   return f
########################################################################################
class LRETomography():
    """
    'LRETomography' is the class including all function needed to perform a
    tomography of a multi-qubits state, via Linear Regression Estimation. 
    It provides the interface for running a tomography on experimental data. 
    This class uses the Linear Regression Estimation method and fast Maximum
    Likelihood estimation.
    """
    
    def __init__(self, qbit_number, xp_counts, working_dir):
        """
        Initialisation of the tomography.
        - 'qbit_number' : number of qubits
        - 'xp_counts_array': array of experimental counts that can be passed
        as an argument to initialize an object of class XPCounts
        - 'working_dir' : directory to save and load data
        """
        self.qbit_number = qbit_number
        self.xp_counts = XPCounts(xp_counts, self.qbit_number)
        self.working_dir = working_dir
        self.quantum_state = QuantumState(
            np.eye(2**self.qbit_number) / 2**self.qbit_number)
        os.chdir(self.working_dir)        
        
    def get_theta_LS(self):
        """Function to get the vector of coordinates of the density matrix
        in the basis of Pauli operators."""
        X = np.load(self.working_dir+'/SavedVariables/X_matrix'+str(
                self.qbit_number)+".npy")
        invXtX_matrix = np.load(
                self.working_dir+'/SavedVariables/invXtX_matrix'+str(
                        self.qbit_number)+".npy")
        Y = self.xp_counts.get_xp_probas() - 1/(2**self.qbit_number)
        
        XtY = np.sum(X*np.resize(Y,(4**self.qbit_number - 1,
                                    3**self.qbit_number,2**self.qbit_number
                                    )).transpose((1,2,0)),axis = (0,1))
        return np.dot(invXtX_matrix,XtY)
        
    def LREstate(self):
        """Function that calculates an approximation of the density matrix,
        using the linear regression estimation method"""
        pauli_operators_array = np.load(
                self.working_dir+'/SavedVariables/Pauli'+str(self.qbit_number
                                                             )+'.npy')
        theta_LS = self.get_theta_LS()
        
        return np.eye(2**self.qbit_number)/2**self.qbit_number + np.sum(
                np.resize(theta_LS,(2**self.qbit_number,2**self.qbit_number,
                        4**self.qbit_number-1)).transpose((
                                2,0,1))*pauli_operators_array[1:,:,:],axis = 0)
                
    def run_pseudo_tomo(self):
        """
        Runs the pseudo tomography to get generate a pseudo_state out of the
        experimental data, using the LRE method.

        WARNING: The resulting density matrix might not be physical, meaning 
        that it is not necessarily positive semi-definite.
        """
        #Defining the "pseudo" density matrix calculated with LRE
        #method, from experimental data.
        self.pseudo_state = self.LREstate()
    
    def run(self):
        """
        Runs the pseudo tomography to get generate a state out of the
        experimental data, using the LRE method and fast maximum likelyhood
        estimation."""
        self.run_pseudo_tomo()  
        self.quantum_state.set_density_matrix(self.pseudo_state)
##############################################################################
qubit_number= 2            
x=m.radians(47)
qwp= np.asarray([[(complex(0,1)-m.cos(2*x)),m.sin(2*x)],[m.sin(2*x),(complex(0,1)+m.cos(2*x))]])
qwp2=np.kron(qwp,qwp)

states= generate_GHZ (qubit_number)       
projectors= generate_projectors(qubit_number)
working_dir= r'/Users/francmartins/Documents/PhD/tomography_run_main.py/tomopy'



m=np.empty((3**qubit_number,2**qubit_number))

iterations=1000
statehobba=np.asarray([-1,0,0,1])*(1/np.sqrt(2))
'''
while i< len(projectors):
    if i/4 ==1 or  i/4 ==3 or  i/4 ==4 or  i/4 ==5 or  i/4 ==7:
          for j in range(0,4):
              m[(i)//4,j]= np.random.poisson(Numofexp*abs(np.trace(np.dot(projectors[i+j]+qwp2,(states[2])))),1)
          
    else: 
          for j in range(0,4):
              m[(i)//4,j]= np.random.poisson(Numofexp*abs(np.trace(np.dot(projectors[i+j],states[2]))),1)
    i=i+4     
   
'''  
ERRFid=[]
ERRPur=[]
ERRLE=[]
ERRE=[]
ERREnt=[]
Numofexp=10000
#Numofexp=np.asarray([1000,1500,1700,1900,2100,2300,2500,2700,2900,3200,3500])
for k in range (0,iterations):

    i=0 
    while i< len(projectors):
           for j in range(0,4):
                  m[(i)//4,j]= 0.9*np.random.poisson(Numofexp *abs(np.trace(np.dot(projectors[i+j],states[21]))),1)+random.uniform(28,30)
           i=i+4  
           
     
    run3=LRETomography(qubit_number,m,working_dir)
    #hist2d(states[21].real,'red')
    
    w3=run3.LREstate()
    w33=NearestPD.nearestPD(w3)
    w3r=w3.real
    ERRFid.append( np.dot(statehobba, np.dot(w33,np.transpose(statehobba))))
    ERRPur.append( np.trace(np.dot(w33,w33)))
    ERRLE.append(linear_entropy(w33))
    ERRE.append(entropy(w33))
    ERREnt.append(entanglement(w33))
     #hist2d(w3r,'cyan')
    
ERRFid=np.asarray(ERRFid)
ERRPur=np.asarray(ERRPur)
ERRLE= np.asarray(ERRLE)
ERRE=np.asarray(ERRE)
ERREnt= np.asarray(ERREnt)
print(ERRE)
stderrfid=np.sqrt(np.std(ERRFid))
stderrPur=np.sqrt(np.std(ERRPur))
stderrLE=np.sqrt(np.std(ERRLE))
stderrE=np.sqrt(np.std(ERRE))
stderrEnt=np.sqrt(np.std(ERREnt))
##########################################################################


precounts=[]
bases=[['DD','DL','DH','LD','LL','LH', 'HD','HL', 'HH'],
       
       ['DA','DR','DV','LA','LR','LV', 'HA', 'HR', 'HV'],
       ['AD','AL','AH','RD','RL','RH', 'VD','VL','VH'],
       ['AA','AR','AV','RA','RR','RV','VA','VR','VV']]

counts=np.empty((4,9))

for v in range(0,4):
    
    for j in range(0,9):
                                       
        file = open("./Choptom/"+bases[v][j]+".txt")
        mean=0
            #Repeat for each song in the text file
        for i, line in enumerate(file):
              
              #Let's split the line into an array called "fields" using the " " as a separator:
              fields = line.split()
              mean= mean+int(fields[0])
          
        mean = int(mean/(i+1))
        
        counts[v][j]= mean


xp_counts= np.transpose(counts); # get the experimental count


###########################################################################################

#working_dir= r'/Users/francmartins/Documents/PhD/tomography_run_main.py/tomopy'

run=Tomography(qubit_number,xp_counts,working_dir)
dirinv=run.direct_inversion()

run2=LRETomography(qubit_number,xp_counts,working_dir)


w=run2.LREstate() #get the state
wND=NearestPD.nearestPD(w)
wr=w.real #real part of the state
wi=w.imag

#[xr,yr,zr]=hist2d(wND.real,'blue')
#[xi,yi,zi]=hist2d(wND.imag,'blue')






print("fidelity :",np.dot(statehobba, np.dot(wND,np.transpose(statehobba))))
print("purity:",np.trace(np.dot(wND,wND)))
print ("entanglement :", entanglement(wND))
print("Linear_entropy of pure :", linear_entropy(wND))
print("von neuman:",entropy(wND))



####################################################################


######## Simulated state ##########

counts=np.empty((4,2,3))
xp_counts=np.empty((4,3,2))
dirinv=np.empty((4,2,2))
qubit_number=1

for j in range(4):
    bases=[
        [str(j)+'D',str(j)+'R',str(j)+'V'], 
        [str(j)+'A',str(j)+'L',str(j)+'H']]

    for v in range(2):
        for w in range(3):
                                           
            file = open("./Random_data/"+bases[v][w]+".txt")
            mean=0
            #Repeat for each song in the text file
            for i, line in enumerate(file):
                  
            #Let's split the line into an array called "fields" using the " " as a separator:
                fields = line.split()
                #print('I, fields: ', i, fields)
                mean= mean+int(fields[0])
                #print('I, MEAN: ', i, mean)
              
            mean = int(mean/(i+1))
            #print('MEAN: ', mean)    
            
            counts[j][v][w]= mean

    xp_counts[j][:][:]=np.array(np.transpose(counts[j][:][:])) # get the experimental count
    print('j, xp_counts: \n', j, xp_counts[j][:][:])

    run=Tomography(qubit_number,xp_counts[j][:][:],working_dir)
    dirinv[j][:][:]=run.direct_inversion()
    print('DIRECT INVERSION: \n', dirinv)

    ######################################################



