# -*- coding: utf-8 -*-

import jax.numpy as np
import jax.ops as jo

import nn


class Network(object):
    def __init__(self, inp, out, params, act):
        hid = nn.depth(inp, out, params)
        self.shape = np.array([inp, hid, out], dtype=int)

        self.θ = nn.adj_arr(self.shape, params)
        self.v = nn.nd_vect(self.shape)
        self.A = act

    def __call__(self, z):
        inp = self.shape[0]
        v = jo.index_update(self.v, jo.index[1:inp], z)

        return self._eval(self.θ, v)

    def _eval(self, θ, v):
        """Produces hypothesis via forward pass."""
        inp, _, out = self.shape

        for idx in range(inp, v.size):
            Σ = np.dot(v[:idx], θ[:idx, idx - inp])
            v = jo.index_update(v, idx, self.A(Σ))

        return np.dot(v, θ[:, -out:])

    def model(self, data, optimizer):
        """Trains network via stochastic gradient descent."""
        pass
