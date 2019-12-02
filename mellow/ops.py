# -*- coding: utf-8 -*-

import jax.random as random

from mellow.typing import Dataset, List, Tensor


# -------------- data manipulators ---------------


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


# ------------ statistical functions -------------


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
