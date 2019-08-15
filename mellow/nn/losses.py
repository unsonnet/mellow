# -*- coding: utf-8 -*-

import autograd.numpy as np


def mae(net, x, θ, y):
    """Computes mean absolute error."""
    return np.sum(np.abs(y - net.evaluate(x, θ))) / y.size


def mse(net, x, θ, y):
    """Computes mean squared error."""
    return np.sum((y - net.evaluate(x, θ)) ** 2) / y.size


def log_loss(net, x, θ, y):
    """Computes logistic loss."""
    return -np.sum(y * np.log(net.evaluate(x, θ))) / y.size


def hinge(net, x, θ, y):
    """Computes hinge loss."""
    return np.sum(np.maximum(0, 1.0 - y * net.evaluate(x, θ))) / y.size


def hubert(net, x, θ, y, δ=1.0):
    """Computes hubert loss."""
    diff = np.abs(y - net.evaluate(x, θ))
    return np.sum(np.where(diff < δ, 0.5 * (diff) ** 2, δ * (diff - δ / 2)))
