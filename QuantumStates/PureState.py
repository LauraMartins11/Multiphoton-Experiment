import numpy as np

from .DensityOperator import DensityOperator
from .QuantumState import QuantumState


class PureState():
    """
    Class to define and interact with a pure state. 
    Initialization with "ket", a line array corresponding to the ket vector of
    the state.
    """
    def __init__(self, ket):
        self.dim = np.shape(ket)[0]
        self.qbit_number = int(np.log2(self.dim))
        #ket becomes a column array -> vector
        self.ket = np.reshape(ket, (self.dim, 1))
        #bra is a line array
        self.bra = np.reshape(np.conj(self.ket), (1, self.dim))
        #associated projector
        self.proj = self.ket @ self.bra

    def expectation(self, operator):
        """
        Returns the expectation value of an operator measured on pure state.
        Argument : "operator" is a square array of shape (self.dim,self.dim)
        """
        return self.bra @ operator @ self.ket

    def get_quantum_state(self):
        """
        Returns a QuantumState object associated to the same state than 
        PureState.
        """
        return QuantumState(self.proj)

    def plot(self):
        """
        Plot the density matrix associated to pure state.
        """
        self.get_quantum_state().plot()
