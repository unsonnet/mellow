# -*- coding: utf-8 -*-

import jax.numpy as np


def mae(y, yh):
    """Computes mean absolute error."""
    return np.mean(np.abs(y - yh))


def mse(y, yh):
    """Computes mean squared error."""
    return np.mean((y - yh) ** 2)
