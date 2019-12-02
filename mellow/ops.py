# -*- coding: utf-8 -*-

import jax.random as random


def shuffle(key, tensors, axis=0):
    """Shuffles the contents of tensors in unison."""
    size(tensors, axis=axis)

    return [random.shuffle(key, tsr, axis=axis) for tsr in tensors]


def batch(tensors, step=1):
    """Generates uniform batches from tensors in unison."""
    length = size(tensors, axis=0)

    for idx in range(0, length, step):
        end = min(idx + step, length)

        yield [tsr[idx:end] for tsr in tensors]


def size(tensors, axis=0):
    """Measures the size of tensors along an axis."""
    sizes = set([tsr.shape[axis] for tsr in tensors])

    if len(sizes) not in (0, 1):
        msg = "tensors of uniform size along {} axis required, got shapes {}."
        raise ValueError(msg.format(axis, [tsr.shape for tsr in tensors]))

    return sizes.pop()
