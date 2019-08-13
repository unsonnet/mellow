# -*- coding: utf-8 -*-

import autograd.numpy as np


def elu(z):
    """Computes exponential linear unit."""
    return z if z > 0 else np.exp(z) - 1.0
