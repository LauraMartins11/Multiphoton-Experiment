# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 01:28:26 2018

@author: Simon
"""

#%%
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

from NestedForLoop import get_iterator


def plotmatrix(matrix, fig):
    """
    Function to plot a real matrix on a particular figure. You get a 3D plot.
    X and Y coordinates are the indices of the matrix. At each (x=i,y=j) point, 
    you get a bar, the height of which corresponds to the (i,j) coefficient of
    the matrix. 
    The function is more adapted to a multi-qubit space, as we expect the 
    matrix to be of shape (2**n,2**n), and the XY-axes labels are of the form
    "HHH,HHV,HVH,..."
    """
    dim = np.shape(matrix)[0]
    qbit_number = int(np.log2(dim))
    HV_label = {0: "H", 1: "V"}
    #Creating the coordinates for plot
    x = np.array(range(dim))
    y = x.copy()
    xx, yy = np.meshgrid(x, y)
    x, y = xx.ravel(), yy.ravel()
    z = np.zeros(dim**2)
    dx = 0.5 * np.ones(dim**2)
    dy = dx.copy()
    dz = matrix.ravel()
    #ploting the bar figure.
    fig.bar3d(x - dx / 2, y - dy / 2, z, dx, dy, dz)
    
    axes = plt.gca()
    #forcing the ticks to be integers
    plt.xticks(np.array(range(dim)))
    plt.yticks(np.array(range(dim)))
    
    #Setting the axes ticks labels with qbits states names.
    HV_iterator = get_iterator(2, qbit_number)
    axes_labels = []
    for k in range(dim):
        axes_labels.append("".join(tuple(map(HV_label.get, HV_iterator[k]))))
    axes.xaxis.set_ticklabels(axes_labels, fontsize=20)
    axes.yaxis.set_ticklabels(axes_labels, fontsize=20)
    for tick in axes.zaxis.get_ticklabels():
        tick.set_fontsize(20)
