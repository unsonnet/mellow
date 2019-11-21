# -*- coding: utf-8 -*-

import numpy as onp


def permute(*arrs):
    """Shuffles the contents of arrays in unison."""
    rng = onp.random.get_state()

    for arr in arrs:
        onp.random.set_state(rng)
        yield onp.random.permutation(arr)


def batch(*arrs, n=1):
    """Iterates over evenly sliced arrays in unison."""
    length = min(map(len, arrs))
    arrs = [onp.copy(arr) for arr in arrs]

    for idx in range(0, length, n):
        end = min(idx + n, length)
        yield [arr[idx:end] for arr in arrs]
