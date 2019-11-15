# -*- coding: utf-8 -*-

import jax.numpy as np
from jax import grad


class Optimizer(object):
    """Base class for optimizers."""

    def __init__(self, inp, dvs):
        self.inp = inp
        self.dvs = dvs

        for var in self.dvs:
            setattr(self, var, 0)

    def add_nd(self, idx):
        """Updates optimizer state to reflect expanded network."""
        col = idx - self.inp

        for var in self.dvs:
            arr = getattr(self, var)
            arr = np.insert(np.insert(arr, idx, 0, 0), col, 0, 1)
            setattr(self, var, arr)

    def del_nd(self, idx):
        """Updates optimizer state to reflect reduced network."""
        col = idx - self.inp

        for var in self.dvs:
            arr = getattr(self, var)
            arr = np.delete(np.delete(arr, idx, 0), col, 1)
            setattr(self, var, arr)
