from tomography.QuantumStates.TriDiagonalMatrix import TriDiagonalMatrix
import numpy as np


def test_Tri_Diagonal():
    initial_matrix = np.array([[2, 1j], [-1j, 2]])
    test_Tri_Diagonal = TriDiagonalMatrix(initial_matrix)

    # Makes sure the matrix return is the original one factorised.
    assert np.all(test_Tri_Diagonal.get_density_matrix().matrix -
                  initial_matrix / np.trace(initial_matrix) < 1E-7)

    # Makes sure that the symmetrisation of the matrix removes all complex values.
    assert np.all(test_Tri_Diagonal.array.imag == np.zeros(
        test_Tri_Diagonal.array.shape))
