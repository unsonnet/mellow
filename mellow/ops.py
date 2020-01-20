# -*- coding: utf-8 -*-

import jax.numpy as np
import jax.ops as jo
import jax.random as random
import prox_tv as ptv

from mellow.typing import Dataset, List, Shape, Tensor


# ------------- network constructors -------------


def depth(inb: int, out: int, params: Tensor) -> int:
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


def adj_arr(shape: Shape, params: Tensor) -> Tensor:
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


def nd_vect(shape: Shape) -> Tensor:
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


# ------------- statistics operators -------------


def shuffle(key, tensors: Dataset, axis: int = 0) -> List[Tensor]:
    """Shuffles the contents of tensors in unison.

    Args:
        key: Pseudo-random generator state.
        tensors: Iterator of tensors.
        axis: Optional, axis along which to shuffle (default 0).

    Returns:
        List of shuffled tensors.

    Raises:
        ValueError: If shape of tensors do not match along `axis`.
    """
    size(tensors, axis=axis)

    return [random.shuffle(key, tsr, axis=axis) for tsr in tensors]


def batch(tensors: Dataset, step: int = 1) -> List[Tensor]:
    """Generates uniform batches from tensors in unison.

    Args:
        tensors: Iterator of tensors.
        step: Optional, batch size (default 1).

    Yields:
        List of evenly-sliced tensors along first axis.

    Raises:
        ValueError: If shape of tensors do not match along first axis.
    """
    length = size(tensors, axis=0)

    for idx in range(0, length, step):
        end = min(idx + step, length)

        yield [tsr[idx:end] for tsr in tensors]


def update_mean(t: int, val: float, mean: float) -> float:
    """Computes triangular-weighted mean.

    Args:
        t: Time step.
        val: New datapoint.
        mean: Current mean.

    Returns:
        Updated mean with the new value having a greater weight than
        previous terms in the sequence.
    """
    return mean + 2 * (val - mean) / (t + 2)


def welford(t: int, val: float, aggregates):
    """Computes triangular-weighted standard deviation.

    Approximates sample standard deviation of time-series data using a
    weighted-form of Welford's method for numerical stability.

    Args:
        t: Time step.
        val: New datapoint.
        aggregates: Current mean and sum of squares.

    Returns:
        Updated aggregates and its respective sample standard deviation
        with the new value having a greater weight than previous terms.
    """
    if t == 0:
        return (val, 0), 1

    mean, suma = aggregates
    mean = update_mean(t, val, mean)
    suma += (t + 1) * (t + 2) * (val - mean) ** 2 / t

    return (mean, suma), np.sqrt(2 * suma / t / (t + 2))


def tv_diff(data: Tensor, lambd: float) -> float:
    """Computes time derivative at endpoint.

    Approximates time derivative of `data` with a second-order backward
    finite difference formula. Assuming the time series contains noise,
    datapoints are filtered using Total Variation Regularization.

    Args:
        data: Uniformly-spaced time series.
        lambd: Non-negative regularization parameter.

    Returns:
        Time derivative at endpoint of filtered `data`.
    """
    u = ptv.tv1_1d(data, lambd)

    return (3 * u[-1] - 4 * u[-2] + u[-3]) / 2


# --------------- helper functions ---------------


def size(tensors: Dataset, axis: int = 0) -> int:
    """Measures the size of tensors along an axis.

    Args:
        tensors: Iterator of tensors.
        axis: Optional, axis along which to measure (default 0).

    Returns:
        Size of tensors along `axis`.

    Raises:
        ValueError: If shape of tensors do not match along `axis`.
    """
    sizes = set([tsr.shape[axis] for tsr in tensors])

    if len(sizes) not in (0, 1):
        msg = "tensors of uniform size along {} axis required, got shapes {}."
        raise ValueError(msg.format(axis, [tsr.shape for tsr in tensors]))

    return sizes.pop()


def shift(tsr: Tensor, fill=np.nan) -> Tensor:
    """Rolls tensor backwards by one.
    
    Shifts one-dimensional tensor to the left, discarding the first
    element and filling the empty slot at the end with a new value.

    Args:
        tsr: One-dimensional tensor.
        fill: Value to add to tensor.

    Returns:
        Tensor with same shape as `tsr` but whose elements are shifted
        to the left by one with a new element at the end.
    """
    out = jo.index_update(np.empty_like(tsr), -1, fill)

    return jo.index_update(out, jo.index[:-1], tsr[1:])
