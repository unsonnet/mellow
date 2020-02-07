# -*- coding: utf-8 -*-

import jax.numpy as np
import jax.random as random
from jax import value_and_grad

import mellow.factory as factory
import mellow.ops as mo
import mellow.stats as stats


class MetaData(object):
    def __init__(self):
        self._k = []

    def __setattr__(self, name, val):
        super().__setattr__(name, val)
        self._k.append(name)

    def __getattr__(self, name):
        self._k.append(name)
        return 0

    def __dir__(self):
        return self._k


class SGD(object):
    """Mini-batch stochastic gradient descent."""

    def __init__(self, net, max_depth, cost, optimizer):
        """Inits trainer.

        Constructs a function that computes the gradient of `cost` with
        respect to the weight parameters of `net`.

        Args:
            net: Network instance.
            max_depth: Maximum number of hidden nodes.
            cost: Cost function.
            optimizer: Weight optimization algorithm.
        """
        self.net = net
        self.max_depth = max_depth
        self.key = random.PRNGKey(0)

        J = lambda θ, D, z, y: cost(y, self.net.eval(θ * D, z))
        self.grad_J = value_and_grad(J)
        self.opt = optimizer

    def model(self, examples, batch_size, epochs):
        """Fits network to labeled training set.

        Updates network to model the relationship between variables via
        supervised learning. Regression is performed on training data
        from `examples` shuffled and grouped into batches per epoch.
        Weights are updated per batch while network topology may be
        altered per epoch.

        Args:
            examples: Labeled training data.
            batch_size: Number of iterations per update.
            epochs: Number of training cycles.

        Returns:
            Network with updated parameters.
        """
        s = MetaData()
        j = np.zeros(10)
        i = u0 = 0
        p0 = 0.5

        for t in range(epochs):
            u = stats.tv_diff(j, 100)
            p = stats.update_prob(u0, u, p0)
            p = stats.init_prob(u) if t < 2 else stats.update_prob(u0, u, p0)

            if p > 0:
                u0, p0 = (np.abs(u), p)

            self.key, subkey = random.split(self.key)
            self.genesis(subkey, s, (u > 0) * p)

            self.key, subkey = random.split(self.key)
            examples = stats.shuffle(subkey, examples)
            batches = mo.batch(examples, step=batch_size)

            self.key, subkey = random.split(self.key)
            i, avg = self.descent(subkey, i, batches, s, p)
            j = mo.shift(j, avg)

        return self.net

    def descent(self, key, i, batches, state, p):
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
            key, subkey = random.split(key)
            D = factory.drop_mask(subkey, self.net.shape, p)

            j, g = self.grad_J(self.net.θ, D, z, y)
            avg = stats.update_mean(n, j, avg)
            self.net.θ += self.opt(i + n, g, state)

        return i + n, avg

    def genesis(self, key, state, p):
        """Implements topology optimization.
        
        Expands network topology by inserting a node at the beginning of
        hidden layer with probabilty `p`. weights are initialized using
        He initialization. All metadata that depend on network topology
        are updated to reflect the change. Process halts when maximum
        depth is reached.

        Args:
            key: Pseudo-random generator state.
            state: Optimizer state tree.
            p: Genesis probabilty.

        Returns:
            True if network topology successfully expanded, else false.
        """
        inb, hid, _ = self.net.shape

        if hid >= self.max_depth or random.uniform(key) < (1 - p):
            return False

        for name in dir(state):
            var = getattr(state, name)

            if np.shape(var) == self.net.θ.shape:
                var = mo.insert(var, 0, 0, axis=1)
                var = mo.insert(var, inb, 0, axis=0)
                setattr(state, name, var)

        Σ = np.sum(self.net.shape)
        weights = random.normal(key, (Σ,)) * np.sqrt(2 / inb)
        self.net.add_nd(weights)

        return True
