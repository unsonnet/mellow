# -*- coding: utf-8 -*-

import autograd.numpy as np


def mse(net, x, θ, y):
    """Computes mean squared error."""
    return np.sum((y - net.evaluate(x, θ)) ** 2) / y.size
