# -*- coding: utf-8 -*-

import jax.numpy as np
import jax.random as random
import prox_tv as ptv

import mellow.ops as mo


def shuffle(key, tensors, axis=0):
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
    a = mo.size(tensors, axis=axis)
    p = random.shuffle(key, np.arange(a))

    return [np.take(tsr, p, axis=axis) for tsr in tensors]


def update_mean(t, z, mean):
    """Computes arithmetic-weighted mean.

    Args:
        t: Time step.
        z: New datapoint.
        mean: Current mean.

    Returns:
        Updated mean with `z` having equal weight to previous terms in
        the sequence.
    """
    return mean + (z - mean) / (t + 1)


def init_prob(u):
    u = np.abs(u)

    return u / (1 + u + u ** 2 / (1 + u))


def update_prob(u0, u, p):
    """Computes fuzzy probabilty measure.

    Maps `u` to a mellow probability value influenced by past values.
    Numbers are scaled such that relatively large updates in `u` incur
    higher probabilities than smaller perturbations.

    Args:
        u0: Current datapoint associated with `p`.
        u: New datapoint.
        p: Current probability.

    Returns:
        Updated probability given `u`.
    """
    u = np.abs(u)
    a = u / (1 + u) + 1

    return 0 if u == 0 else p / (u0 / u * (1 - p) + a * p)


def tv_diff(data, lambd):
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
