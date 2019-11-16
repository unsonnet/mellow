# -*- coding: utf-8 -*-

import jax.numpy as np
import jax.ops as jo


# ------------- network constructors -------------


def depth(inp, out, params):
    """Deduces depth of hidden layer."""
    dis = 1 + 4 * inp * (inp - 1) + 4 * out * (out - 1) + 8 * len(params)
    hid = (1 - 2 * (inp + out) + np.sqrt(dis)) / 2

    if not float(hid).is_integer():
        raise ValueError("Insufficient number of parameters.")

    return hid


def adj_arr(shape, params):
    """Constructs a sliced weighted acyclic-digraph adjacency matrix."""
    inp, _, out = shape
    Σ = np.sum(shape)

    outbound = np.array([Σ - max(inp, n + 1) for n in range(Σ - out)])
    arr = np.flip(np.arange(Σ - inp), 0) < outbound[:, None]

    return jo.index_update(arr.astype(float), arr, params)


def nd_vect(shape):
    """Constructs a node vector."""
    Σ = np.sum(shape[0:2])
    v = np.zeros(int(Σ), dtype=float)

    return jo.index_update(v, 0, 1.0)


# ------------- activation functions -------------


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
