# -*- coding: utf-8 -*-
"""
Created on Wed May 17 01:42:38 2023

@author: Verena, Nicolas Laurent-Puig, Laura
"""

import numpy as np
import scipy
import time

import os
import glob

from pathlib import Path
import fnmatch


def finding_file(general, containing, filenames):
    for file in filenames:
        if fnmatch.fnmatch(file, general+'[!a-z]_'+containing+'*'):
            return file
        elif fnmatch.fnmatch(file, general+'[!a-z][!a-z]_'+containing+'*'):
            return file
    print('No file containing: ', general+containing, '...')
    pass

def get_iterator(K,n):
    """
    Function to generate the indices used during a for loop of n indices
    starting from 0 to K-1  (n nested for loops of K iterations).
    Returns a list of K^n tuples of n elements.
    the r-th tuple contain the number "r" written in base K.
    """
    iterator = []
    for r in range(K**n):
        r_in_base_K = np.base_repr(r, K)
        list_r_in_base_K = [0]*n
        for c in range(len(r_in_base_K)):
            list_r_in_base_K[n - len(r_in_base_K) + c] = np.int(r_in_base_K[c])
        iterator.append(np.array(list_r_in_base_K))
    return iterator

def get_char_iterator(N_qubit):
    d={"x": "DA", "y":"RL", "z":"HV" }
    basis_order= get_iterator(3,N_qubit)
    basis=["x","y","z"]
    projectors_order=get_iterator(2,N_qubit)
    columns=[]
    
    
    character_Matrix=[]
    for i in basis_order:
        string=""
        for j in i:
            string=string+basis[j]
        lines=[]    
        for k in projectors_order:
            o=0
            strr=""
            for n in k:
                strr=strr+d[string[o]][n]
                o=o+1
            lines.append(strr)    
        character_Matrix.append(lines)         
        columns.append(string)   
        
    return character_Matrix 


def get_proj_iterator(N_qubit):
    d={"x": [[1/np.sqrt(2),1/np.sqrt(2)],[1/np.sqrt(2),-1/np.sqrt(2)]], "y":[[1/np.sqrt(2),complex(0,1/np.sqrt(2))],[1/np.sqrt(2),complex(0,-1/np.sqrt(2))]], "z":[[1,0],[0,1]] }
    basis_order= get_iterator(3,N_qubit)
    basis=["x","y","z"]
    projectors_order=get_iterator(2,N_qubit)
    columns=[]
    
    
    character_Matrix=[]
    for i in basis_order:
        string=""
        for j in i:
            string=string+basis[j]
        lines=[]    
        for k in projectors_order:
            o=0
            vect=[]
            for n in k:
                vect=vect+d[string[o]][n]
                o=o+1
            lines.append(vect)    
        character_Matrix.append(lines)         
        columns.append(string)   
        
    return character_Matrix 

### set_raw_counts() returns an array with the counts recorded in the columns 4:8 of the datafile (where the coincidence counts are written)
def set_raw_counts(datafiles, qubit_number, column_start, column_stop, directory):
    os.chdir(directory)
    counts_aux=np.zeros((2**qubit_number,3**qubit_number), dtype=float)
    
    
    letters = ['x', 'y', 'z']
    bases = []

    for i in letters:
        if qubit_number == 1:
            base = i 
            bases.append(base)
        for j in letters:
            if qubit_number == 2:
                base = i + j
                bases.append(base)
            for k in letters:
                if qubit_number == 3:
                    base = i + j + k 
                    bases.append(base)
                for l in letters:
                    if qubit_number == 4:
                        base = i + j + k + l
                        bases.append(base)
    print(bases)

    counts=select_lines_in_file(column_start, column_stop, datafiles, np.shape(counts_aux), bases, os.getcwd())

    """
    Just ordering the array such that it matches with the covention counts_aux[x] (this changes depending on how we save data):
    - x=0: HH
    - x=1: HV
    - x=2: VH
    - x=3: VV
    If we change this order we need to do the same in correct_counts_with_channels_eff in projectorcounts.py
    """
    if qubit_number == 1:
        counts_aux=counts[[0,1]]
        return(counts_aux)
    if qubit_number == 2:
        counts_aux=counts[[0,1,2,3]]
        return(counts_aux)
    if qubit_number == 3:
        counts_aux=counts[[0,1,2,3,4,5,6,7]]
        return(counts_aux)
    if qubit_number == 4:
        counts_aux=counts[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
        return(counts_aux)


def select_lines_in_file(start_line, finish_line, datafiles, shape, bases, directory):
    array=np.zeros(shape, dtype='float')
    os.chdir(directory)

    for w in range(shape[1]):
        file=finding_file('Bigiteration_', bases[w], datafiles)
        with open(file) as file:
            for line in file:
                fields = line.split()
                for iter, field in enumerate(fields[start_line:finish_line]):
                    array[iter][w]=float(field)
                    
    return(array)

"""
get_channels_eff() returns the normalized efficiency of each channel
Not all channels have the same efficiency for various reasons
We need to be able to characterize each efficiency to be able to correct the number of counts_array
"""
def get_channels_eff(datafiles,qubit_number, column_start, column_stop, directory):
    os.chdir(directory)
    letters = ['a', 'z']
    eff = []

    for i in letters:
        if qubit_number == 1:
            base = i 
            eff.append(base)
        for j in letters:
            if qubit_number == 2:
                base = i + j
                eff.append(base)
            for k in letters:
                if qubit_number == 3:
                    base = i + j + k 
                    eff.append(base)
                for l in letters:
                    if qubit_number == 4:
                        base = i + j + k + l
                        eff.append(base)

    efficiencies_aux=np.zeros((len(eff), column_stop-column_start), dtype=float)
    
    ### !Don't forget that we transpose here. That's why we sum over the 0 axis and not 1 in the next line!
    efficiencies_aux=select_lines_in_file(column_start, column_stop, datafiles, np.shape(efficiencies_aux), eff, os.getcwd()).transpose()

    """
    Each column (channel) will be turned into the sum over the lines (that correspond to different measurement basis)
    When normalized we end up with a an array with the relative channel efficiencies in the order we saved the data
    The re-ordiring of the basis to match each channel to a measurement basis happens in
    correct_counts_with_channels_eff in ProjectorCounts.py
    """
    efficiencies=np.sum(efficiencies_aux, 0)/np.max(np.sum(efficiencies_aux, 0))

    return(efficiencies)