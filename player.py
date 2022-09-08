### Created on: 29-07-2022
### Author: Laura Martins

import numpy as np
import errors

class Player:
    def __init__(self, player):
        self.name=player
        if self.name==("Arya" or "arya" or "ARYA"):
            self.sigma_wp=np.array([errors.SIGMA_HWP_ARYA, errors.SIGMA_HWP_ARYA])
        elif self.name==("Bran" or "bran" or "BRAN"):
            self.sigma_wp=np.array([errors.SIGMA_HWP_BRAN, errors.SIGMA_HWP_BRAN])
        elif self.name==("Cersei" or "cersei" or "CERSEI"):
            self.sigma_wp=np.array([errors.SIGMA_HWP_CERSEI, errors.SIGMA_HWP_CERSEI])