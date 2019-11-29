# -*- coding: utf-8 -*-

import warnings

import jax.numpy as np
import jax.ops as jo

from mellow import nn


class Network(object):
    """Homogenous feedforward neural network."""

    def __init__(self, inp, out, params, act):
        inb = inp + 1  # Accounts for bias unit.
        hid = nn.depth(inb, out, params)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Explicitly requested dtype.*")

            self.shape = np.array([inb, hid, out], dtype=int)
            self.θ = nn.adj_arr(self.shape, params)
            self.v = nn.nd_vect(self.shape)

        self.A = act

    def predict(self, z):
        """Produces a hypothesis."""
        return self.eval(self.θ, z)

    def eval(self, θ, z):
        """Evaluates network on input."""
        inb, _, out = self.shape
        v = self.reshape(z)

        for idx in range(inb, v.shape[-1]):
            Σ = np.dot(v[..., :idx], θ[:idx, idx - inb])
            v = jo.index_update(v, jo.index[..., idx], self.A(Σ))

        return np.dot(v, θ[:, -out:])

    def reshape(self, z):
        """Formats data for network evaluation."""
        inb, _, _ = self.shape
        assert z.transpose().shape[0] == inb - 1

        rows = z.transpose()[..., None].shape[1]
        v = np.tile(self.v, (rows, 1))

        return jo.index_update(v, jo.index[..., 1:inb], z)
