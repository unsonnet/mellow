# -*- coding: utf-8 -*-

import jax.numpy as np
import jax.ops as jo

from mellow import nn


class Network(object):
    """Homogenous feedforward neural network."""

    def __init__(self, inp, out, params, act):
        inb = inp + 1  # Accounts for bias unit.
        hid = nn.depth(inb, out, params)
        self.shape = np.array([inb, hid, out], dtype=int)

        self.θ = nn.adj_arr(self.shape, params)
        self.v = nn.nd_vect(self.shape)
        self.A = act

    def __call__(self, z):
        inb = self.shape[0]
        v = jo.index_update(self.v, jo.index[1:inb], z)

        return self._eval(self.θ, v)

    def _eval(self, θ, v):
        """Produces a hypothesis."""
        inb, _, out = self.shape

        for idx in range(inb, v.size):
            Σ = np.dot(v[:idx], θ[:idx, idx - inb])
            v = jo.index_update(v, idx, self.A(Σ))

        return np.dot(v, θ[:, -out:])

    def model(self, data, labels, optimizer):
        """Fits network to labeled training set."""
        pass
