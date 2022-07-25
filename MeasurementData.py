# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 23:24:38 2018

@author: Simon Neves, Robert Booth
"""

#%%

#importdir = "C:/Users/Simon/Documents/Travail/LIP6/projet_info/tomo-dense/src/"
import os
#os.chdir(importdir)
import numpy as np

from NestedForLoop import get_iterator

# Measurement bases indexation :
# Comp, Diag, Circ -> 0, 1, 2

# Projectors indexation :
# V, D, L, H -> 0, 1, 2, 3

# Pauli operators indexation :
# I, X, Y, Z -> 0, 1, 2, 3


class MeasurementData():
    """
    Class for interactions with experimental data.
    Experimental data typically consist of the numbers of events (photon
    detection or detection coincidences) recorded during a certain period, when
    measuring simultaneously the different qubits in certain combinations of 
    bases. A data file has a name corresponding to the combination of 
    measurement bases.
    
    This class is used to extract the data from text files that are built from
    the experiment, and to adapt the data to our classes "XPProjectorCounts" 
    and "TotalProjectorCounts". It is pretty specific to the measurement setup
    used to acquire the data.

    Initialisation Arguments:
    - 'qbit_number' : number of qubits
    - 'working_dir' : working directory
    - 'data_dir' : directory for the experimental data.
    """

    # Dictionaries to save and access the experimental data.
    experimental_data = {}

    ##################
    # INITIALISATION #
    ##################
    
    def __init__(self, qbit_number, working_dir, data_dir='C:/Users/Verena Maged/.spyder-py3/Tomography/tomopy/data/'):
        self.qbit_number = qbit_number
        self.data_dir = data_dir
        os.chdir(working_dir)

        # Definition of iterators
        # These are used to iterate on all tensor products of projectors,
        # or all combination of bases used to measure the different qubits,...
        self.projector_iterator = get_iterator(4, self.qbit_number)
        self.basis_iterator = get_iterator(3, self.qbit_number)
        binary_iterator = get_iterator(2, self.qbit_number)
        self.detector_iterator = []
        for iterator in binary_iterator:
            self.detector_iterator.append(
                tuple(iterator[k] + 2 * k for k in range(self.qbit_number)))
            
        self.fill_data_dictionary()

    ###########
    # METHODS #
    ###########
    def convert_bases(self, basis_combination):
        """
        Converts a tuple of bases to a data file name.
        Argument : 'basis_combination' is a tuple of indices corresponding to 
        a basis combination.
        """
        return self.data_dir + ''.join([str(e)
                                        for e in basis_combination]) + '.txt'

    def load_data(self, basis_combination):
        """
        Loads the data file corresponding to a simultaneous measurement of 
        photons in a particular combination of bases. It loads it into a numpy 
        array.
        Argument : 'basis_combination' is a tuple of indices corresponding to 
        the basis combination.
        """
        filename = self.convert_bases(basis_combination)
        return np.loadtxt(filename)

    def fill_data_dictionary(self):
        """
        Filling the dictionary of experimental data. In the end,
        self.experimental_data has the following structure :
        -key : tuple containing a combination of indices corresponding
        to a combination of measurement bases.
        -value : line array containing the numbers of events recorded 
        during the experiment.
        """
        for basis_combination in self.basis_iterator:
            self.experimental_data[tuple(basis_combination)] = self.load_data(basis_combination)

    def get_count(self, basis_combination, detector_combination,
                  amplification_factor = 1):
        """
        Function to access the number of event, for
        measurement in a particular combinations of bases, and in specific
        detectors (or coincidences between detectors).
        Arguments:  - `basis_combination` is a tuple of measurement bases
                    configurations.
                   - `detector_combination` is a tuple of the detectors whose
                   coincidences to access.
        """
        #the formula for his index is specific to our experimental setup
        total_index = sum([2**detector
                           for detector in detector_combination]) - 1
        return amplification_factor*self.experimental_data[basis_combination][
                total_index]

    def get_total_counts(self, projector_combination):
        """
        Function to get the total number of coincidences of N-photon
        detection events, in the combination of measurement bases used to 
        perform a measurement of the tensor product of projectors correponding
        to 'projector_combination'
        Arguments:  - `projector_combination` is a tuple of indices 
        corresponding to the projectors of the tensor product.
        """
        basis_combination = tuple(0 if proj == 3 else proj 
                                  for proj in projector_combination)
        total_counts = 0
        for detectors in self.detector_iterator:
            total_counts += self.get_count(basis_combination, detectors)
        return total_counts

    def get_total_counts_dict(self):
        """
        Function that returns a dictionary, containing the total number of 
        coincidences of N-photon detection events, in the combination of
        measurement bases used to perform measurements of each tensor product 
        of projectors.

        key : tuple containing a combination of indices corresponding
            to projectors
        value : total number of coincidences of N-photon detection events,
            detected in the bases used to measure the tensor product of 
            projectors corresponding to the key
        """
        total_counts_dict = {}
        for projector_combination in self.projector_iterator:
            total_counts_dict[
                projector_combination] = self.get_total_counts(
                    projector_combination)
        return total_counts_dict

    def get_xp_projector_counts(self, projector_combination):
        """
        Function that returns the number of coincidence events detected when 
        the  photons are measured with a certain tensor product of projectors, 
        out of the experimental data contained in dictionary experimental_data.
        
        Arguments : - 'projector_combination' is the tuple of indices
        corresponding to the projectors in the tensor product.
        """
        #defining which measurement base to use so as to get the useful data
        #for the given projector combination
        basis_combination = tuple(
            0 if projector_index == 3 else projector_index
            for projector_index in projector_combination)
        #defining which detectors to use so as to get the useful data
        #for the given projector combination
        detector_combination = tuple(
            1 + 2 * k if projector_combination[k] == 0 else 2 * k
            for k in range(self.qbit_number))
        return self.get_count(basis_combination, detector_combination) 

    def get_xp_projector_counts_dict(self):
        """
        Fills the dictionary self.xp_projectors_counts_dict.
        key : tuple of  indices corresponding to a tensor product of
        projectors.
        value : number of coincidence events detected when the photons are 
        measured with the tensor product of projectors corresponding to the 
        key.
        """
        xp_projector_counts_dict = {}
        for projector_combination in self.projector_iterator:
            xp_projector_counts_dict[
                projector_combination] = self.get_xp_projector_counts(
                    projector_combination)
        return xp_projector_counts_dict
    

    
