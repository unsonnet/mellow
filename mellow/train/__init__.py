# -*- coding: utf-8 -*-

import jax.numpy as np
from jax import value_and_grad

import mellow.ops as mo


class SGD(object):
    """Mini-batch stochastic gradient descent."""

    def __init__(self, net, cost, optimizer):
        self.net = net
        self.i = 0

        J = lambda θ, z, y: cost(y, self.net.eval(θ, z))
        self.grad_J = value_and_grad(J)
        self.opt = optimizer

    def model(self, data, target, batch_size, epochs):
        """Fits network to labeled training set."""
        assert data.shape[0] == target.shape[0]

        for t in range(epochs):
            mo.shuffle(data, target)

            for z, y in mo.batch(data, target, n=batch_size):
                _, g = self.grad_J(self.net.θ, z, y)
                self.net.θ += self.opt(self.i, g)
                self.i += 1

        return self.net
