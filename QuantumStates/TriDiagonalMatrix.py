import numpy as np

from .DensityOperator import DensityOperator

#%%
from numpy import linalg
import numpy as np

class TriDiagonalMatrix():
    """
    Represents a tri-diagonal matrix with real diagonal elements, used to
    parametrize a physical density matrix for the optimization procedure.
    This triangular matrix results from the Cholesky factorisation of the
    argument matrix, into the product of a lower tri-diagonal matrix and it's
    hermitian conjugate.

    Initialisation Arguments:
    - `initial_matrix` : density matrix of the initial state to be represented by
                         the factored matrix
    """

    # From an implementation perspective, since the diagonal elements are real and
    # only the sub-diagonal elements have complex parts, we use a real square
    # matrix: the sub-diagonal elements represent the real part of the complex
    # elements of the matrix, and the sup-diagonal elements contain the complex
    # part.
    def __init__(self, matrix):
        temp = linalg.cholesky(matrix)
        self.array = temp.real + temp.imag.transpose()

    # This method uses reshapes to "create" the vector. As a result it is FAST.
    def get_vector(self):
        """
        Returns a flattened version of the matrix (a vector), to be passed as
        argument to the optimization procedure. This vector is mutable so can
        be directly acted upon by the optimization.
        """
        return self.array.reshape(-1)

    def set_vector(self, new_vector):
        """
        Sets the array representing the tri-diagonal matrix, given a vector
        parametrisation. In effect, this is the inverse operation of
        `get_vector`.
        """
        self.array = new_vector.reshape(self.array.shape)

    def set_matrix(self, matrix):
        temp_matrix = linalg.cholesky(matrix)
        self.array = temp_matrix.real + temp_matrix.imag.transpose()

    def get_t_matrix(self):
        """
        Returns a copy of the tri-diagonal matrix.
        """
        return np.tril(self.array) + np.triu(self.array, k=1).transpose() * 1j

    # NOTE: This method can potentially be sped up by overloading the matrix
    # multiplication operator `__matmul__` to multiply the condensed array
    # `self.array` directly, so as to bypass the call to `get_t_matrix()` (which
    # is quite slow).
    def get_density_matrix(self):
        """
        Constructs and returns the density matrix resulting from the
        parametrization.
        """
        triangular = self.get_t_matrix()
        density = triangular @ triangular.transpose().conj()
        return DensityOperator(density / np.trace(density))
