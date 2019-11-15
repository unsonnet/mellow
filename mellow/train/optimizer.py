# -*- coding: utf-8 -*-

import jax.numpy as np
from jax import grad


class Optimizer(object):
    """Base class for optimizers."""

    def __init__(self, net, dvs):
        self.inp = net.shape[0]
        self.dvs = dvs

        for var in self.dvs:
            setattr(self, var, 0)

    def insert(self, idx):
        col = idx - self.inp

        for var in self.dvs:
            arr = getattr(self, var)
            arr = np.insert(np.insert(arr, idx, 0, 0), col, 0, 1)
            setattr(self, var, arr)

    def delete(self, idx):
        col = idx - self.inp

        for var in self.dvs:
            arr = getattr(self, var)
            arr = np.delete(np.delete(arr, idx, 0), col, 1)
            setattr(self, var, arr)
