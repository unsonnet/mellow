# -*- coding: utf-8 -*-

import jax.numpy as np


class Schedule(object):
    """Base class for hyperparameter schedules."""

    def __call__(self, i):
        raise NotImplementedError


# ----------------- lr schedules -----------------


class Constant(Schedule):
    """Learning rate schedule that remains constant."""

    def __init__(self, lr):
        self.λi = lr

    def __call__(self, i):
        return self.λi


class PiecewiseConstant(Schedule):
    """Learning rate schedule that implements Piecewise Constant."""

    def __init__(self, bounds, lrs):
        self.intervals = np.array(bounds)
        self.λs = np.array(lrs)

    def __call__(self, i):
        return self.λs[np.sum(i > self.intervals)]


class InverseTime(Schedule):
    """Learning rate schedule that implements Inverse Time Decay."""

    def __init__(self, lr, decay):
        self.λi = lr
        self.μ = decay

    def __call__(self, i):
        return self.λi / (1 + self.μ * i)


class Polynomial(Schedule):
    """Learning rate schedule that implements Polynomial Decay."""

    def __init__(self, lr, iter_max, lr_min, power=1.0):
        self.λi = lr
        self.im = iter_max
        self.λf = lr_min
        self.p = power

    def __call__(self, i):
        μ = (1 - min(i, self.im) / self.im) ** self.p

        return μ * (self.λi - self.λf) + self.λf


class Exponential(Schedule):
    """Learning rate schedule that implements Exponential Decay."""

    def __init__(self, lr, decay):
        self.λi = lr
        self.μ = decay

    def __call__(self, i):
        return self.λi * self.μ ** i
