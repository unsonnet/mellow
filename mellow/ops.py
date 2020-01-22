# -*- coding: utf-8 -*-

import jax.numpy as np
import jax.ops as jo
import numpy as onp


def size(tensors, axis=0):
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


def batch(tensors, step=1):
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


def shift(tsr, fill=np.nan):
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


def insert(tsr, idx, val, axis=0):
    """Wrapper of numpy.insert for jax.

    Args:
        tsr: Input tensor.
        idx: Index before which `val` is inserted.
        val: Values to insert into `tsr`.
        axis: Optional, axis along which to insert (default 0).

    Returns:
        Copy of `tsr` with `val` inserted.
    """
    tsr = onp.insert(tsr, idx, val, axis=axis)

    return np.array(tsr)
