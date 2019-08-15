# -*- coding: utf-8 -*-

from autograd.extend import primitive, defvjp
import autograd.numpy as np
from autograd import grad

from nn.activations import elu
from nn.losses import mse
from nn.optimizers import Momentum

# -------------------- computation methods --------------------


@primitive
def forward_prop(net, x, θ):
    """Computes output vector via forward propagation."""
    inp, _, out = net.struct
    net.V[1:inp] = x

    for n in range(inp, net.V.size):
        net.V[n] = net.σ(θ[:n, n] @ net.V[:n])

    return net.V[-out:]


def backprop_x(ans, net, x, θ):
    """Computes jacobian w.r.t. input vector."""
    inp, _, out = net.struct

    for n in range(inp, net.V.size):
        _sum = θ[:n, n] @ net.V[:n]
        net.Jx[n] = net.grad_σ(_sum) * (θ[:n, n] @ net.Jx[:n, :])

    # Constructs vector-jacobian product operator.
    return lambda g: np.tensordot(g, net.Jx[-out:], 1)


def backprop_θ(ans, net, x, θ):
    """Computes jacobian w.r.t. network parameters."""
    inp, _, out = net.struct

    for n in range(inp, net.V.size):
        net.Jθ[n, :n, inp:] = np.tensordot(θ[:n, n].T, net.Jθ[:n, :n, inp:], 1)
        net.Jθ[n, :n, n] += net.V[:n]
        net.Jθ[n, ...] *= net.grad_σ(θ[:n, n] @ net.V[:n])

    # Constructs vector-jacobian product operator.
    return lambda g: np.tensordot(g, net.Jθ[-out:], 1)


defvjp(forward_prop, backprop_x, backprop_θ, argnums=[1, 2])


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
        self.V = node_vect(self.struct)

        self.grad_σ = grad(func)
        self.Jx = np.eye(self.V.size, dtype=float)[:, 1:inp]
        self.Jθ = np.zeros([self.V.size] * 3, dtype=float)

    def evaluate(self, x, θ=None):
        """Computes output vector via forward propagation."""
        return forward_prop(self, x, self.θ if θ is None else θ)

    def model(self, data, loss=mse, optimizer=Momentum()):
        """Trains network via stochastic gradient descent."""
        for x, y in np.random.permutation(data):

            def cost(θ):
                nonlocal self, x, y
                return loss(self, x, θ, y)

            self.θ -= optimizer(cost, self.θ)

        return self
