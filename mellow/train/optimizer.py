# -*- coding: utf-8 -*-

import jax.numpy as np


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


class Momentum(Optimizer):
    """Optimizer that implements Momentum."""

    def __init__(self, inp, lr=0.01, decay=0.9):
        super().__init__(inp, ["m"])
        self.λ = init_schedule(lr)
        self.μ = decay

    def __call__(self, t, g):
        self.m = self.μ * self.m + g

        return -self.λ(t) * self.m


class Nesterov(Optimizer):
    """Optimizer that implements Nesterov Momentum."""

    def __init__(self, inp, lr=0.01, decay=0.9):
        super().__init__(inp, ["m"])
        self.λ = init_schedule(lr)
        self.μ = decay

    def __call__(self, t, g):
        self.m = self.μ * self.m + g

        return -self.λ(t) * (self.μ * self.m + g)
