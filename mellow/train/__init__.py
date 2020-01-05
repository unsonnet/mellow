# -*- coding: utf-8 -*-

import jax.numpy as np

from mellow.train.core import SGD


# ---------------- cost functions ----------------


def mae(y, yh):
    """Computes mean absolute error."""
    assert y.shape == yh.shape
    return np.mean(np.abs(y - yh))


def mse(y, yh):
    """Computes mean squared error."""
    assert y.shape == yh.shape
    return np.mean((y - yh) ** 2)
