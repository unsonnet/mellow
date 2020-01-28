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


def update_mean(t: int, z: float, mean: float) -> float:
    """Computes triangular-weighted mean.

    Args:
        t: Time step.
        z: New datapoint.
        mean: Current mean.

    Returns:
        Updated mean with the new value having a greater weight than
        previous terms in the sequence.
    """
    return mean + 2 * (z - mean) / (t + 2)


def update_variance(t: int, z: float, sv: float) -> float:
    """Computes triangular-weighted sample variance.

    Args:
        t: Time step.
        z: New datapoint.
        sv: Current sample variance.

    Returns:
        Updated sample variance with the new value having a greater
        weight than previous terms in the sequence.
    """
    if t == 0:
        suma = z ** 2
    elif t == 1:
        suma = (4 * z ** 2 - sv) / 3
    else:
        suma = (2 * (t + 1) * z ** 2 - (2 * t + 1) * sv) / t / (t + 2)

    return sv + suma


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
