### Created on: 25-07-2022
### Author: Laura Martins

import numpy as np
""""
ERROR DUE TO WAVEPLATES UNCERTAINTY ###
In our xp_counts matrix, every entry corresponds to a different projection basis, which is associated to associated to
a different set of {HWP,QWP}. We need to simulate new number of counts for each entry, given the angle and the
the uncertainty of the WP's we are using
"""
HWP_DICT={"x": np.pi/8,
          "y": 0,
          "z": 0}
        #   "a": -np.pi/8,
        #   "r": 0,
        #   "v": np.pi/4}

QWP_DICT={"x": np.pi/2,
          "y": 3*np.pi/4,
          "z": np.pi/2}
        #   "a": np.pi/2,
        #   "r": np.pi/4,
        #   "v": np.pi/2}

PROJECTORS={"d": np.array([1,1])/np.sqrt(2),
            "l": np.array([1,1j])/np.sqrt(2),
            "h": np.array([1,0]),
            "a": np.array([1,-1])/np.sqrt(2),
            "r": np.array([1,-1j])/np.sqrt(2),
            "v": np.array([0,1])}

### Uncertainty on the WP
SIGMA_HWP_ARYA=0.026*np.pi/180
SIGMA_QWP_ARYA=0.356*np.pi/180
SIGMA_HWP_BRAN=0.018*np.pi/180
SIGMA_QWP_BRAN=0.046*np.pi/180
SIGMA_HWP_CERSEI=0.03*np.pi/180
SIGMA_QWP_CERSEI=0.18*np.pi/180
SIGMA_HWP_DANY=0.09*np.pi/180
SIGMA_QWP_DANY=0.04*np.pi/180