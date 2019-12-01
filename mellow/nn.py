# -*- coding: utf-8 -*-

import jax.numpy as np
import jax.ops as jo

from mellow.typing import Shape, Array, Tensor


# ------------- network constructors -------------


def depth(inb: int, out: int, params: Array) -> int:
    """Deduces depth of hidden layer.
    
    Args:
        inb: Number of input nodes not excluding bias unit.
        out: Number of output nodes.
        params: One-dimensional array of weights.

    Returns:
        Number of hidden nodes that, when taken with respect to `inb`
        and `out`, represent a quasi-complete acyclic digraph with the
        same number of arcs as weights in `params`.

    Raises:
        ValueError: If a quasi-complete acyclic digraph cannot be found
            with the same number of arcs as weights in `params`.
    """
    dis = 1 + 4 * inb * (inb - 1) + 4 * out * (out - 1) + 8 * len(params)
    hid = (1 - 2 * (inb + out) + np.sqrt(dis)) / 2

    if not float(hid).is_integer():
        msg = "{} weights insufficient for deducing depth of a [{}, :, {}] network."
        raise ValueError(msg.format(len(params), inb, out))

    return hid


def adj_arr(shape: Shape, params: Array) -> Tensor:
    """Constructs a sliced weighted acyclic-digraph adjacency matrix.

    Assigns weights from `params` to an acylic-digraph adjacency matrix
    in row-major order. Nodes are assumed to be topologically sorted.

    Args:
        shape: Tuple containing number of nodes per type.
        params: One-dimensional array of weights.

    Returns:
        Two-dimensional array of weights where the rows and columns
        represent source and target nodes respectively. Only arcs with
        non-output source and non-input target are represented.

    Raises:
        AttributeError: If weighted acyclic-digraph adjacency matrix
            cannot be completely initialized.
    """
    inb, _, out = shape
    Σ = np.sum(shape)

    outbound = np.array([Σ - max(inb, n + 1) for n in range(Σ - out)])
    arr = np.flip(np.arange(Σ - inb), 0) < outbound[:, None]

    if np.count_nonzero(arr) != len(params):
        msg = "{} weights required to initialize adj matrix of a {} network, got {}."
        raise AttributeError(msg.format(np.count_nonzero(arr), shape, len(params)))

    return jo.index_update(arr.astype(float), arr, params)


def nd_vect(shape: Shape) -> Tensor:
    """Constructs a node vector.

    Assigns 0 to all non-output nodes except the bias unit which
    defaults to 1. Nodes are assumed to be topologically sorted.

    Args:
        shape: Tuple containing number of nodes per type.

    Returns:
        One-dimensional array of values where each element represents
        the stored value of a non-output node.
    """
    Σ = np.sum(shape[0:2])
    v = np.zeros(int(Σ), dtype=float)

    return jo.index_update(v, 0, 1.0)


# ------------- activation functions -------------


def elu(z: Tensor) -> Tensor:
    """Computes exponential linear unit."""
    return np.where(z > 0, z, np.exp(z) - 1.0)


def relu(z: Tensor, a: float = 0.0) -> Tensor:
    """Computes rectified linear unit."""
    return np.where(z > 0, z, a * z)


def sigmoid(z: Tensor) -> Tensor:
    """Computes logistic sigmoid."""
    return np.where(z > 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def tanh(z: Tensor) -> Tensor:
    """Computes hyperbolic tangent."""
    return np.tanh(z)
