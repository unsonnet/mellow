# -*- coding: utf-8 -*-

import warnings

import jax.numpy as np
import jax.ops as jo

from mellow.typing import Array, Tensor, UFunc
from mellow import nn


class Network(object):
    """Homogenous feedforward neural network."""

    def __init__(self, inp: int, out: int, params: Array, act: UFunc) -> None:
        """Inits Network.

        Deduces the structure of a network represented by `inp`, `out`,
        and `params`. A weighted adjacency matrix and node vector of
        appropriate shape are constructed as well.

        Args:
            inp: Number of input nodes excluding bias unit.
            out: Number of output nodes.
            params: One-dimensional array of weights.
            act: Activation function.

        Raises:
            ValueError: If a network structure cannot be found with the
                same number of arcs as weights in `params`.
        """
        inb = inp + 1  # Accounts for bias unit.
        hid = nn.depth(inb, out, params)

        with warnings.catch_warnings():  # Filters precision warnings.
            warnings.filterwarnings("ignore", message="Explicitly requested dtype.*")

            self.shape = np.array([inb, hid, out], dtype=int)
            self.θ = nn.adj_arr(self.shape, params)
            self.v = nn.nd_vect(self.shape)

        self.A = act

    def predict(self, z: Tensor) -> Tensor:
        """Produces a hypothesis from `z`.
        
        Args:
            z: Single or vertically-stacked array of input.

        Returns:
            Vertically-stacked output layers.

        Raises:
            AttributeError: If insufficient input data is given.
        """
        return self.eval(self.θ, z)

    def eval(self, θ: Tensor, z: Tensor) -> Tensor:
        """Evaluates network on input.

        Forward propagates data from `z` through network with arcs
        parameterized by `θ`.
        
        Args:
            θ: Topologically-sorted weighted adjacency matrix.
            z: Single or vertically-stacked array of input.

        Returns:
            Vertically-stacked output layers.

        Raises:
            AttributeError: If insufficient input data is given.
            ValueError: If network cannot be parameterized by `θ`.
        """
        inb, _, out = self.shape
        v = self.reshape(z)

        if self.θ.shape != θ.shape:
            msg = (
                "Weighted adjacency matrix of shape {} required to parameterize a {} "
                "network, got {}."
            )
            raise ValueError(msg.format(self.θ.shape, self.shape, θ.shape))

        for idx in range(inb, v.shape[-1]):
            Σ = np.dot(v[..., :idx], θ[:idx, idx - inb])
            v = jo.index_update(v, jo.index[..., idx], self.A(Σ))

        return np.dot(v, θ[:, -out:])

    def reshape(self, z: Tensor) -> Tensor:
        """Formats data for network evaluation.

        Assigns data from `z` to input layer. Vectors are stacked
        vertically if `z` represents multiple data samples.

        Args:
            z: Single or vertically-stacked array of input.

        Returns:
            Vertically-stacked node vectors.

        Raises:
            AttributeError: If input layers cannot be completely
                initialized.
        """
        inb, _, _ = self.shape

        if inb - 1 != np.transpose(z).shape[0]:
            msg = (
                "{} values required per input slice in order for network to evaluate "
                "data, got {}."
            )
            raise AttributeError(msg.format(inb - 1, np.transpose(z).shape[0]))

        rows = np.transpose(z)[..., None].shape[1]  # Counts number of data samples.
        v = np.tile(self.v, (rows, 1))

        return jo.index_update(v, jo.index[..., 1:inb], z)
