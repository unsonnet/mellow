# -*- coding: utf-8 -*-

import jax.numpy as np

from mellow.core import Network


# ------------- activation functions -------------


def elu(z):
    """Computes exponential linear unit."""
    return np.where(z > 0, z, np.exp(z) - 1.0)


def relu(z, a=0.0):
    """Computes rectified linear unit."""
    return np.where(z > 0, z, a * z)


def sigmoid(z):
    """Computes logistic sigmoid."""
    return np.where(z > 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def tanh(z):
    """Computes hyperbolic tangent."""
    return np.tanh(z)
