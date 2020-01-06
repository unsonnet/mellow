# -*- coding: utf-8 -*-

import jax.numpy as np

from mellow.core import Network
from mellow.typing import Tensor


# ------------- activation functions -------------


def elu(z: Tensor) -> Tensor:
    """Computes exponential linear unit."""
    return np.where(z > 0, z, np.exp(z) - 1.0)


def relu(z: Tensor, a: float = 0.0) -> Tensor:
    """Computes rectified linear unit."""
    return np.where(z > 0, z, a * z)


def sigmoid(z: Tensor) -> Tensor:
    """Computes logistic sigmoid."""
    return np.where(z > 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def tanh(z: Tensor) -> Tensor:
    """Computes hyperbolic tangent."""
    return np.tanh(z)
