# -*- coding: utf-8 -*-

import jax.random as random
from jax import value_and_grad

from mellow import Network
import mellow.ops as mo
from mellow.train.optimizer import Optimizer
from mellow.typing import Dataset, Loss


class SGD(object):
    """Mini-batch stochastic gradient descent."""

    def __init__(self, net: Network, cost: Loss, optimizer: Optimizer) -> None:
        self.net = net
        self.i = 0

        J = lambda θ, z, y: cost(y, self.net.eval(θ, z))
        self.grad_J = value_and_grad(J)
        self.opt = optimizer

    def model(self, key, examples: Dataset, batch_size: int, epochs: int) -> Network:
        """Fits network to labeled training set."""
        for t in range(epochs):
            key, subkey = random.split(key)
            examples = mo.shuffle(subkey, examples)

            for z, y in mo.batch(examples, step=batch_size):
                _, g = self.grad_J(self.net.θ, z, y)
                self.net.θ += self.opt(self.i, g)
                self.i += 1

        return self.net
