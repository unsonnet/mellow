# -*- coding: utf-8 -*-

from autograd.extend import primitive, defvjp
import autograd.numpy as np
from autograd import grad

from .nn.activations import elu
from .nn.losses import mse
from .nn.optimizers import Momentum


# -------------------- factory methods --------------------


def hidden_layer(inp, out, θ):
    """Determines shape of hidden layer."""
    if θ is None:  # Induces number of hidden nodes.
        hid = np.ceil(np.sqrt(inp * out))

    else:  # Deduces number of hidden nodes.
        det = 1 + 4 * inp * (inp - 1) + 4 * out * (out - 1) + 8 * len(θ)
        hid = (1 - 2 * (inp + out) + np.sqrt(det)) / 2
        if not float(hid).is_integer():
            raise ValueError("Insufficient number of parameters.")

    return hid


def adjacency_mat(struct, θ):
    """Constructs a weighted acyclic digraph adjacency matrix."""
    _sum = np.sum(struct, dtype=int)
    inp, _, out = struct

    def arcs_in_row(n):
        nonlocal _sum, inp, out
        return (_sum - n - 1) * (n < _sum - out) - (inp - n - 1) * (n < inp)

    num_arcs = np.array([arcs_in_row(n) for n in range(_sum)])
    mat = (num_arcs[:, np.newaxis] > np.flip(np.arange(_sum))).astype(float)
    θ = np.random.standard_normal(np.sum(num_arcs)) if θ is None else θ
    mat[mat > 0] = θ  # Assigns weights to valid arcs.

    return mat


def node_vect(struct):
    """Constructs a node vector."""
    vect = np.zeros(np.sum(struct, dtype=int), dtype=float)
    vect[0] = 1  # Assigns 1 to bias node.

    return vect


# -------------------- network class --------------------


class Network(object):
    def __init__(self, inp, out, θ=None, func=elu):
        inp += 1  # Accounts for bias node.
        hid = hidden_layer(inp, out, θ)
        self.struct = np.array([inp, hid, out], dtype=int)

        self.σ = func
        self.θ = adjacency_mat(self.struct, θ)
        self.v = node_vect(self.struct)

        self.grad_σ = grad(func)
        self.J = np.zeros([self.v.size] * 3, dtype=float)

    @primitive
    def evaluate(self, x, θ=None):
        """Computes output vector via forward propagation."""
        θ = self.θ if θ is None else θ
        inp, _, out = self.struct
        self.v[1:inp] = x

        for n in range(inp, self.v.size):
            self.v[n] = self.σ(θ[:n, n] @ self.v[:n])

        return self.v[-out:]

    def backprop(ans, self, x, θ):
        """Computes jacobian w.r.t. network parameters."""
        inp, _, out = self.struct

        for n in range(inp, self.v.size):
            self.J[n, :n, inp:] = np.tensordot(θ[:n, n].T, self.J[:n, :n, inp:], 1)
            self.J[n, :n, n] += self.v[:n]
            self.J[n, ...] *= self.grad_σ(θ[:n, n] @ self.v[:n])

        # Constructs vector-jacobian product operator.
        return lambda g: np.tensordot(g, self.J[-out:], 1)

    def model(self, data, batch_size=1, loss=mse, optimizer=Momentum()):
        """Trains network via stochastic gradient descent."""
        data = np.random.permutation(data)

        for batch in np.array_split(data, np.ceil(len(data) / batch_size)):
            Δθ = 0

            for x, y in batch:
                cost = lambda θ: loss(y, self.evaluate(x, θ))
                Δθ += optimizer(cost, self.θ) / len(batch)

            self.θ += Δθ

        return self


defvjp(Network.evaluate, Network.backprop, argnums=[2])
