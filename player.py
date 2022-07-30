### Created on: 29-07-2022
### Author: Laura Martins

import numpy as np
import errors

class Player:
    def __init__(self, player):
        self.player=player
        self.sigma_wp=np.array([errors.SIGMA_HWP_ARYA, errors.SIGMA_HWP_ARYA])