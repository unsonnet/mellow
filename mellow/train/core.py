# -*- coding: utf-8 -*-

import jax.random as random
from jax import value_and_grad

from mellow import Network
import mellow.ops as mo
from mellow.train.optimizer import Optimizer
from mellow.typing import Dataset, Loss


class MetaData(object):
    def __getattr__(self, name):
        return 0


class SGD(object):
    """Mini-batch stochastic gradient descent."""

    def __init__(self, net: Network, cost: Loss, optimizer: Optimizer) -> None:
        """Inits trainer.

        Constructs a function that computes the gradient of `cost` with
        respect to the weight parameters of `net`.

        Args:
            net: Network instance.
            cost: Cost function.
            optimizer: Optimization algorithm instance.
        """
        self.net = net
        self.key = random.PRNGKey(0)

        J = lambda θ, z, y: cost(y, self.net.eval(θ, z))
        self.grad_J = value_and_grad(J)
        self.opt = optimizer

    def model(self, examples: Dataset, batch_size: int, epochs: int) -> Network:
        """Fits network to labeled training set.

        Updates network weight parameters by following the gradient of 
        the cost function, minimizing loss. Network is evaluated with 
        training data from `examples`, which is shuffled per epoch.
        Parameters are updated per completed batch.

        Args:
            examples: Labeled training data.
            batch_size: Number of iterations per update.
            epochs: Number of training cycles.

        Returns:
            Network with updated parameters.
        """
        s = MetaData()
        i = 0

        for t in range(epochs):
            self.key, subkey = random.split(self.key)
            examples = mo.shuffle(subkey, examples)

            for z, y in mo.batch(examples, step=batch_size):
                _, g = self.grad_J(self.net.θ, z, y)
                self.net.θ += self.opt(i, g, s)
                i += 1

        return self.net
