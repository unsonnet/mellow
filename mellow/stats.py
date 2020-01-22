# -*- coding: utf-8 -*-

import jax.numpy as np
import jax.random as random
import prox_tv as ptv

import mellow.ops as mo
from mellow.typing import Dataset, Tensor


def shuffle(key, tensors: Dataset, axis: int = 0) -> Dataset:
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
    mo.size(tensors, axis=axis)

    return [random.shuffle(key, tsr, axis=axis) for tsr in tensors]


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
