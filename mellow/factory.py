# -*- coding: utf-8 -*-

import jax.numpy as np
import jax.ops as jo
import jax.random as random


def depth(inb, out, params):
    """Deduces depth of hidden layer.

    Args:
        inb: Number of input nodes not excluding bias unit.
        out: Number of output nodes.
        params: Sequence of weights.

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


def adj_arr(shape, params):
    """Constructs a sliced weighted acyclic-digraph adjacency matrix.

    Assigns weights from `params` to an acylic-digraph adjacency matrix
    in row-major order. Nodes are assumed to be topologically sorted.

    Args:
        shape: Tuple containing number of nodes per type.
        params: Sequence of weights.

    Returns:
        Weighted adjacency matrix where the rows and columns represent
        source and target nodes respectively. Only arcs with non-output
        source and non-input target are represented.

    Raises:
        AttributeError: If weighted adjacency matrix cannot be
            completely initialized.
    """
    inb, _, out = shape
    Σ = np.sum(shape)

    outbound = np.array([Σ - max(inb, n + 1) for n in range(Σ - out)])
    arr = np.flip(np.arange(Σ - inb), 0) < outbound[:, None]

    if np.count_nonzero(arr) != len(params):
        msg = "{} weights required to initialize adj matrix of a {} network, got {}."
        raise AttributeError(msg.format(np.count_nonzero(arr), shape, len(params)))

    return jo.index_update(arr.astype(float), arr, params)


def nd_vect(shape):
    """Constructs a node vector.

    Assigns 0 to all non-output nodes except the bias unit which
    defaults to 1. Nodes are assumed to be topologically sorted.

    Args:
        shape: Tuple containing number of nodes per type.

    Returns:
        Node vector where each element represents the stored value of a
        non-output node.
    """
    Σ = np.sum(shape[0:2])
    v = np.zeros(int(Σ), dtype=float)

    return jo.index_update(v, 0, 1.0)


def drop_mask(key, shape, p):
    """Constructs a dropout mask.

    Assigns a dropout coefficient to hidden nodes with probabilty |`p`|.
    Outgoing arcs of affected nodes are scaled by the inverse of the
    activation probabilty to accommodate for the expected reduced
    network capacity during training.
    
    Args:
        key: Pseudo-random generator state.
        shape: Tuple containing number of nodes per type.
        p: Dropout probabilty.

    Returns:
        Non-negative dropout mask of equal shape to network's weighted
        adjacency matrix.
    """
    inb, hid, out = shape
    q = 1 - p

    D = (random.uniform(key, (hid, 1)) < q) / q
    D = np.pad(D, ((inb, 0), (0, 0)), constant_values=1)

    return np.repeat(D, hid + out, axis=1)
