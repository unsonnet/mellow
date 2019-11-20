# -*- coding: utf-8 -*-

import numpy as onp


def shuffle(*arrs):
    """Shuffles the contents of a sequence of arrays in unison."""
    rng = onp.random.get_state()

    for arr in arrs:
        onp.random.set_state(rng)
        onp.random.shuffle(arr)


def batch(*arrs, n=1):
    """Iterates over a sequence of evenly sliced arrays in unison."""
    length = min(map(len, arrs))

    for idx in range(0, length, n):
        end = min(idx + n, length)
        yield [arr[idx:end] for arr in arrs]
