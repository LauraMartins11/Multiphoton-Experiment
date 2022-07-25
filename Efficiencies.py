### Created on: 18-07-2022
### Author: Laura Martins

import numpy as np
import scipy
import time

import os
import glob

from pathlib import Path
import fnmatch

### This function finds files in the form 'general'+'containing'
### This can be used if we want to find files that have 'general' in common but that can be distinguished by 'containing'
def finding_file(general, containing, filenames):
    for file in filenames:
         if fnmatch.fnmatch(file, general+containing+'*'):
            return file
    print('No file containing: ', general+containing, '...')
    pass


### select_lines_in_file() writes the data of a file in an array - each column is a new file
### - each column corresponds to a new datafile
###- each line corresponds to a different column of the datafile
def select_lines_in_file(start_line, finish_line, datafiles, shape, bases, directory):
    array=np.zeros(shape, dtype='float')
    os.chdir(directory)

    for w in range(shape[1]):
        file=finding_file('Bigiteration_0_', bases[w], datafiles)
        with open(file) as file:
            for line in file:
                fields = line.split()
                for iter, field in enumerate(fields[start_line:finish_line]):
                    array[iter][w]=float(field)
    return(array)

### get_channels_eff() returns the normalized efficiency of each channel {zz, z-z, -zz, -z-z} where '-z' is labeled as 'a'
### Not all channels have the same efficiency for various reasons
### We need to be able to characterize each efficiency to be able to correct the number of counts_array
def get_channels_eff(datafiles, directory):
    os.chdir(directory)
    eff=np.array(['zz', 'za', 'az', 'aa'])
    efficiencies_aux=np.zeros((len(eff), 4), dtype=float)

    efficiencies_aux=select_lines_in_file(4, 8, datafiles, np.shape(efficiencies_aux), eff, os.getcwd()).transpose()
    efficiencies=np.sum(efficiencies_aux, 0)/np.max(np.sum(efficiencies_aux, 0))


    return(efficiencies)

### set_raw_counts() returns an array with the counts recorded in the columns 4:8 of the datafile (where the coincidence counts are written)
def set_raw_counts(datafiles, qubit_number, directory):
    os.chdir(directory)

    bases=np.array(['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz'])#, ## order: D, L, H

    counts=np.zeros((4,3**qubit_number), dtype=int)
    counts_aux=np.zeros((2**qubit_number,3**qubit_number), dtype=float)

    counts=select_lines_in_file(4, 8, datafiles, np.shape(counts_aux), bases, os.getcwd())

    ### Just ordering the array such that it matches with the covention: 0: VV; 1: VH; 2: HV; 3: HH (this changes depending on how we save data)
    counts_aux[0]=counts[2]
    counts_aux[1]=counts[3]
    counts_aux[2]=counts[0]
    counts_aux[3]=counts[1]

    #counts.transpose()

    return(counts_aux)
