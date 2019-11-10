# -*- coding: utf-8 -*-

import jax.numpy as np


def mae(y, yh):
    """Computes mean absolute error."""
    return np.sum(np.abs(y - yh)) / y.size


def mse(y, yh):
    """Computes mean squared error."""
    return np.sum((y - yh) ** 2) / y.size


def log_loss(y, yh):
    """Computes logistic loss."""
    return -np.sum(y * np.log(yh)) / y.size


def hinge(y, yh):
    """Computes hinge loss."""
    return np.sum(np.maximum(0, 1.0 - y * yh)) / y.size


def hubert(y, yh, δ=1.0):
    """Computes hubert loss."""
    diff = np.abs(y - yh)
    return np.sum(np.where(diff < δ, 0.5 * (diff) ** 2, δ * (diff - δ / 2)))
