# -*- coding: utf-8 -*-

import jax.numpy as np

from mellow.train import schedule


class Optimizer(object):
    """Base class for optimizers."""

    def __call__(self, i, g, s):
        raise NotImplementedError


def init_schedule(obj):
    """Declares learning rate schedule."""
    return obj if callable(obj) else schedule.Constant(obj)


# ------------------ optimizers ------------------


class Momentum(Optimizer):
    """Optimizer that implements Momentum."""

    def __init__(self, lr=0.01, decay=0.9):
        self.λ = init_schedule(lr)
        self.μ = decay

    def __call__(self, i, g, s):
        s.m = self.μ * s.m + g

        return -self.λ(i) * s.m


class Nesterov(Optimizer):
    """Optimizer that implements Nesterov Momentum."""

    def __init__(self, lr=0.01, decay=0.9):
        self.λ = init_schedule(lr)
        self.μ = decay

    def __call__(self, i, g, s):
        s.m = self.μ * s.m + g

        return -self.λ(i) * (self.μ * s.m + g)
