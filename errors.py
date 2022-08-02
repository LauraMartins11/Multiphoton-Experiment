### Created on: 25-07-2022
### Author: Laura Martins

import numpy as np

### ERROR DUE TO WAVEPLATES UNCERTAINTY ###
## In our xp_counts matrix, every entry corresponds to a different projection basis, which is associated to associated to
## a different set of {HWP,QWP}. We need to simulate new number of counts for each entry, given the angle and the
## the uncertainty of the WP's we are using
HWP_DICT={"d": np.pi/8,
          "l": 0,
          "h": 0,
          "a": -np.pi/8,
          "r": 0,
          "v": np.pi/4}

QWP_DICT={"d": np.pi/2,
          "l": 3*np.pi/4,
          "h": np.pi/2,
          "a": np.pi/2,
          "r": np.pi/4,
          "v": np.pi/2}

PROJECTORS={"d": np.array([1,1])/np.sqrt(2),
            "l": np.array([1,1j])/np.sqrt(2),
            "h": np.array([1,0]),
            "a": np.array([1,-1])/np.sqrt(2),
            "r": np.array([1,-1j])/np.sqrt(2),
            "v": np.array([0,1])}


OBSERVABLES=np.transpose(np.array([
    ['dd', 'dl', 'dh', 'ld', 'll', 'lh', 'hd', 'hl', 'hh'],
    ['da', 'dr', 'dv', 'la', 'lr', 'lv', 'ha', 'hr', 'hv'],
    ['ad', 'al', 'ah', 'rd', 'rl', 'rh', 'vd', 'vl', 'vh'],
    ['aa', 'ar', 'av', 'ra', 'rr', 'rv', 'va', 'vr', 'vv']])
    )

# OBSERVABLES=['dd','da','ad','aa']

### Uncertainty on the WP
SIGMA_HWP_ARYA=0.04*np.pi/180
SIGMA_QWP_ARYA=0.1*np.pi/180
SIGMA_HWP_CERSEI=0.01*np.pi/180
SIGMA_QWP_CERSEI=0.11*np.pi/180