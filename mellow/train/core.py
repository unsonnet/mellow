# -*- coding: utf-8 -*-

import jax.numpy as np
import jax.random as random
from jax import value_and_grad

import mellow.factory as factory
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

        J = lambda θ, D, z, y: cost(y, self.net.eval(θ * D, z))
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
        j = np.zeros(50)
        i = p = m = Σ = 0

        for t in range(epochs):
            self.key, subkey = random.split(self.key)
            examples = mo.shuffle(subkey, examples)

            batches = mo.batch(examples, step=batch_size)
            i, avg = self.descent(subkey, i, batches, s, p)
            j = mo.shift(j, avg)

            u = mo.tv_diff(j, 100)
            (m, Σ), sd = mo.welford(t, u, (m, Σ))
            p = (u - m) / 2 / sd

        return self.net

    def descent(self, key, i: int, batches, state: MetaData, p: float):
        """Implements mini-batch gradient descent.

        Updates network weight parameters by following the gradient of
        the cost function, reducing loss. Nodes are dropped temporarily
        per batch with probabilty `p`. Update step is determined by
        an optimization algorithm provided during initialization.

        Args:
            key: Pseudo-random generator state.
            i: Iteration step.
            batches: Sample batch generator.
            state: Optimizer state tree.
            p: Dropout probabilty.

        Returns:
            Tuple containing the current iteration step and the
            triangular-weighted mean of losses computed during mini-
            batch gradient descent.
        """
        avg = 0

        for n, (z, y) in enumerate(batches):
            D = factory.drop_mask(key, self.net.shape, p)
            j, g = self.grad_J(self.net.θ, D, z, y)
            avg = mo.update_mean(n, j, avg)
            self.net.θ += self.opt(i + n, g, state)

        return i + n, avg
