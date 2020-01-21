# -*- coding: utf-8 -*-

import warnings

import jax.numpy as np
import jax.ops as jo

import mellow.factory as factory
import mellow.ops as mo
from mellow.typing import Tensor, UFunc


class Network(object):
    """Homogenous feedforward neural network."""

    def __init__(self, inp: int, out: int, params: Tensor, act: UFunc) -> None:
        """Inits Network.

        Deduces the structure of a network represented by `inp`, `out`,
        and `params`. A weighted adjacency matrix and node vector of
        appropriate shape are constructed as well.

        Args:
            inp: Number of input nodes excluding bias unit.
            out: Number of output nodes.
            params: Sequence of weights.
            act: Activation function.

        Raises:
            ValueError: If a network structure cannot be found with the
                same number of arcs as weights in `params`.
        """
        inb = inp + 1  # Accounts for bias unit.
        hid = factory.depth(inb, out, params)

        with warnings.catch_warnings():  # Filters precision warnings.
            warnings.filterwarnings("ignore", message="Explicitly requested dtype.*")

            self.shape = np.array([inb, hid, out], dtype=int)
            self.θ = factory.adj_arr(self.shape, params)
            self.v = factory.nd_vect(self.shape)

        self.A = act

    def add_nd(self, params):
        """Adds a new hidden node to network.
        
        Inserts a node represented by `params` at the beginning of the
        hidden layer. Regarding the weighted adjacency matrix, arcs are
        assigned in row-major order such that it preserves topological
        ordering.

        Args:
            params: Sequence of weights.

        Raises:
            AttributeError: If insufficient number of weights is given.
        """
        inb, hid, _ = self.shape
        Σ = np.sum(self.shape)

        if Σ != len(params):
            msg = "{} weights required to add a node to a {} network, got {}."
            raise AttributeError(msg.format(Σ, self.shape, len(params)))

        self.shape = jo.index_add(self.shape, 1, 1)
        self.v = np.append(self.v, 0)

        col = np.pad(params[:inb], (0, hid), constant_values=0)
        row = np.pad(params[inb:], (1, 0), constant_values=0)
        self.θ = mo.insert(self.θ, 0, col, axis=1)
        self.θ = mo.insert(self.θ, inb, row, axis=0)

    def predict(self, z: Tensor) -> Tensor:
        """Produces a hypothesis from `z`.

        Args:
            z: Stacked input samples.

        Returns:
            Stacked output layers.

        Raises:
            AttributeError: If insufficient input is given.
        """
        return self.eval(self.θ, z)

    def eval(self, θ: Tensor, z: Tensor) -> Tensor:
        """Evaluates network on input.

        Forward propagates data from `z` through network with arcs
        parameterized by `θ`.

        Args:
            θ: Topologically-sorted weighted adjacency matrix.
            z: Stacked input samples.

        Returns:
            Stacked output layers.

        Raises:
            AttributeError: If insufficient input is given.
            ValueError: If network cannot be parameterized by `θ`.
        """
        inb, _, out = self.shape
        v = self.reshape(z)

        if self.θ.shape != θ.shape:
            msg = "{} adj matrix required to parameterize a {} network, got {}."
            raise ValueError(msg.format(self.θ.shape, self.shape, θ.shape))

        for idx in range(inb, v.shape[-1]):
            Σ = np.dot(v[..., :idx], θ[:idx, idx - inb])
            v = jo.index_update(v, jo.index[..., idx], self.A(Σ))

        return np.dot(v, θ[:, -out:])

    def reshape(self, z: Tensor) -> Tensor:
        """Formats input for network evaluation.

        Assigns data from `z` to input layer. Multiple node vectors are
        constructed if `z` represents a sequence of input samples.

        Args:
            z: Stacked input samples.

        Returns:
            Stacked node vectors.

        Raises:
            AttributeError: If input layers cannot be completely
                initialized.
        """
        inb, _, _ = self.shape

        if inb - 1 != np.transpose(z).shape[0]:
            msg = "{} values required per input sample, got {}."
            raise AttributeError(msg.format(inb - 1, np.transpose(z).shape[0]))

        rows = np.transpose(z)[..., None].shape[1]  # Counts number of input samples.
        v = np.tile(self.v, (rows, 1))

        return jo.index_update(v, jo.index[..., 1:inb], z)
