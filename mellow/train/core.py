# -*- coding: utf-8 -*-

import jax.numpy as np
import jax.random as random
from jax import value_and_grad

import mellow.factory as factory
import mellow.ops as mo
import mellow.stats as stats


class MetaData(object):
    def __getattr__(self, name):
        return 0


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
            examples = stats.shuffle(subkey, examples)

            batches = mo.batch(examples, step=batch_size)
            i, avg = self.descent(subkey, i, batches, s, p)
            j = mo.shift(j, avg)

            u = stats.tv_diff(j, 100)
            (m, Σ), sd = stats.welford(t, u, (m, Σ))
            p = (u - m) / 2 / sd

            self.key, subkey = random.split(self.key)
            self.genesis(subkey, s, max(0, p))

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
