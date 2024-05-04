### Created on: 25-07-2022
### Author: Laura Martins

import numpy as np
import scipy
from scipy.stats import norm, truncnorm
import time
import functools as ft

import os
import glob

from pathlib import Path
import fnmatch

import errors

# from tomography import LRETomography
from player import Player

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from nestedforloop import get_iterator


def general_unitary(x):
    return np.array(
        [
            [np.exp(1j * x[0]) * np.cos(x[2]), np.exp(1j * x[1]) * np.sin(x[2])],
            [-np.exp(-1j * x[1]) * np.sin(x[2]), np.exp(-1j * x[0]) * np.cos(x[2])],
        ]
    )


def z_rotation(x):
    return np.array([[np.exp(-1j * x[0] / 2), 0], [0, np.exp(1j * x[0] / 2)]])


def wp_rotation(t, n):
    R = np.exp(-1j * n / 2) * np.array(
        [
            [
                np.cos(t) ** 2 + np.exp(1j * n) * np.sin(t) ** 2,
                (1 - np.exp(1j * n)) * np.cos(t) * np.sin(t),
            ],
            [
                (1 - np.exp(1j * n)) * np.cos(t) * np.sin(t),
                np.sin(t) ** 2 + np.exp(1j * n) * np.cos(t) ** 2,
            ],
        ]
    )
    return R


def apply_unitary_to_dm(dm, U):
    return U @ dm @ np.transpose(np.conjugate(U))


def fid(dm, target):
    shape = np.shape(target)

    if len(shape) > 1:
        return (
            np.trace(
                scipy.linalg.sqrtm(
                    scipy.linalg.sqrtm(target) @ dm @ scipy.linalg.sqrtm(target)
                )
            )
        ) ** 2 / (np.trace(target) * np.trace(dm))
    else:
        return np.transpose(np.conjugate(target)) @ dm @ target


def tensor_product(state1, state2):
    return np.kron(state1, state2)


class DensityMatrix:
    """
    Once we have a state in density matrix form we want to calculate different things with it
    With this class we want to be able to easily calculate all these quantities regarding our state
    """

    def __init__(self, state):
        self.state = state

    @property
    def qbit_number(self):
        return np.log2(self.state.shape[0]).astype(int)

    def apply_unitary(self, U):
        return DensityMatrix(apply_unitary_to_dm(self.state, U))

    ### Returns the fidelity of our density matrix to a pure state
    def fidelity(self, target):
        shape = np.shape(target)
        if len(shape) > 1:
            return (
                np.trace(
                    scipy.linalg.sqrtm(
                        scipy.linalg.sqrtm(self.state)
                        @ target
                        @ scipy.linalg.sqrtm(self.state)
                    )
                )
            ) ** 2 / (np.trace(self.state) * np.trace(target))
        else:
            return np.transpose(np.conjugate(target)) @ self.state @ target

    def __repr__(self):
        return repr(self.state)

    def fock_basis(self, qbit_number):
        fock_state = []
        fock_state = self.state
        self.state = np.zeros((((qbit_number) ** 4), (qbit_number**4)), dtype=complex)
        if len(fock_state) < 5:
            for w in range(0, 4):
                for i in range(0, 4):
                    self.state[w * 3 + 3, i * 3 + 3] = fock_state[w, i]
        else:
            self.state = fock_state

    def plot_dm(self):
        density_matrix_plot = self.state
        real_density_matrix = density_matrix_plot.real
        imag_density_matrix = density_matrix_plot.imag
        HV_label = {0: "H", 1: "V"}

        HV_iterator = get_iterator(2, self.qbit_number)
        axes_labels = []
        for k in range(np.shape(density_matrix_plot)[0]):
            axes_labels.append("".join(tuple(map(HV_label.get, HV_iterator[k]))))

        nrows, ncols = density_matrix_plot.shape

        # Create a meshgrid for the x and y values
        x = np.arange(ncols)
        y = np.arange(nrows)
        X, Y = np.meshgrid(x, y)

        # Flatten the density matrix values
        Z = real_density_matrix.flatten()
        W = imag_density_matrix.flatten()

        # Set the column heights as Z values
        column_heights = abs(Z)
        column_heights2 = abs(W)

        # Set the column widths and depths
        column_width = column_depth = 0.7

        # Create the first figure
        fig1 = plt.figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111, projection="3d")

        # Plot the density matrix as 3D columns using a colormap
        cmap = cm.get_cmap("plasma")
        # cmap = cm.get_cmap('set3')
        ax1.bar3d(
            X.ravel(),
            Y.ravel(),
            np.zeros_like(Z),
            column_width,
            column_depth,
            column_heights,
            shade=True,
            color=cmap(Z),
            alpha=0.5,
        )

        ax1.set_xticks(np.arange(ncols) + 0.5)
        ax1.set_yticks(np.arange(nrows) + 0.5)
        ax1.set_xticklabels(
            axes_labels, rotation=45, ha="right", fontsize=8, fontweight="light"
        )
        ax1.set_yticklabels(axes_labels, rotation=-60, fontsize=8, fontweight="light")

        # Set the title
        ax1.set_title("Real Density Matrix", fontsize=14, fontweight="bold")

        # Add a colorbar
        cbar = fig1.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax1)
        cbar.set_label("Real Density", rotation=270, labelpad=15)

        # Adjust plot limits to avoid cutoff of tick labels
        ax1.set_xlim(0, ncols)
        ax1.set_ylim(0, nrows)
        ax1.view_init(elev=15, azim=-45)
        # Create the second figure
        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111, projection="3d")

        # Plot the density matrix as 3D columns using a colormap
        ax2.bar3d(
            X.ravel(),
            Y.ravel(),
            np.zeros_like(W),
            column_width,
            column_depth,
            column_heights2,
            shade=True,
            color=cmap(W),
            zsort="max",
        )

        # Set the x and y axis labels
        ax2.set_xticks(np.arange(ncols) + 0.5)
        ax2.set_yticks(np.arange(nrows) + 0.5)
        ax2.set_xticklabels(
            axes_labels, rotation=45, ha="right", fontsize=10, fontweight="bold"
        )
        ax2.set_yticklabels(axes_labels, rotation=-60, fontsize=10, fontweight="bold")
        ax2.set_zlim(0, np.max([column_heights, column_heights2]))  # Set z-axis limits

        # Set the title
        ax2.set_title("Imaginary Density Matrix", fontsize=14, fontweight="bold")

        # Add a colorbar
        cbar = fig2.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax2)
        cbar.set_label("Imag Density", rotation=270, labelpad=15)

        # Adjust plot limits to avoid cutoff of tick labels
        ax2.set_xlim(0, ncols)
        ax2.set_ylim(0, nrows)
        ax2.view_init(elev=15, azim=-45)
        # Display the figures
        plt.show()


class GHZ:

    def __init__(self, state1, state2):
        self.fock_state1 = state1
        self.fock_state2 = state2

    def GHZ_before_BS(self):
        self.GHZ_state_before_BS = tensor_product(self.fock_state1, self.fock_state2)

    def fidelity(self, target):
        shape = np.shape(target)
        if len(shape) > 1:
            return (
                np.trace(
                    scipy.linalg.sqrtm(
                        scipy.linalg.sqrtm(self.GHZ_state)
                        @ target
                        @ scipy.linalg.sqrtm(self.GHZ_state)
                    )
                )
            ) ** 2 / (np.trace(self.GHZ_state) * np.trace(target))
        else:
            return np.transpose(np.conjugate(target)) @ self.GHZ_state @ target

    def BeamSplitter(self):
        X = np.array([[0, 1], [1, 0]])
        Z = np.array([[1, 0], [0, -1]])
        PBS = np.bmat([[np.diag([1, 1]), np.zeros((2, 2))], [np.zeros((2, 2)), X @ Z]])
        swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        PBS_swap = swap @ PBS
        Beamsplitter = tensor_product(
            np.diag([1, 1]), tensor_product(PBS_swap, np.diag([1, 1]))
        )
        self.GHZ_state = Beamsplitter @ self.GHZ_state_before_BS
