# -*- coding: utf-8 -*-

import jax.numpy as np


def elu(z):
    """Computes exponential linear unit."""
    return z if z > 0 else np.exp(z) - 1.0


def relu(z):
    """Computes rectified linear unit."""
    return z if z > 0 else 0.0


def leaky_relu(z, α=0.2):
    """Computes leaky rectified linear unit."""
    return z if z > 0 else α * z


def sigmoid(z):
    """Computes logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-z)) if z > 0 else np.exp(z) / (1.0 + np.exp(z))


def tanh(z):
    """Computes hyperbolic tangent."""
    return np.tanh(z)
