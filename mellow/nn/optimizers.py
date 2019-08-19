# -*- coding: utf-8 -*-

import autograd.numpy as np
from autograd import grad


class Momentum(object):
    def __init__(self, learning_rate=0.01, decay=0.9):
        """Constructs momentum optimizer."""
        self.η = learning_rate
        self.α = decay
        self.m = 0

    def __call__(self, cost, θ):
        self.m = self.α * self.m + self.η * grad(cost)(θ)

        return -self.m


class NesterovMomentum(object):
    def __init__(self, learning_rate=0.01, decay=0.9):
        """Constructs nesterov-accelerated gradient optimizer."""
        self.η = learning_rate
        self.α = decay
        self.m = 0

    def __call__(self, cost, θ):
        θ = θ - self.α * self.m
        self.m = self.α * self.m + self.η * grad(cost)(θ)

        return -self.m


class Adagrad(object):
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        """Constructs adagrad optimizer."""
        self.η = learning_rate
        self.ϵ = epsilon
        self.G = 0

    def __call__(self, cost, θ):
        g = grad(cost)(θ)
        self.G += g ** 2

        return -self.η * g / np.sqrt(self.G + self.ϵ)


class Adadelta(object):
    def __init__(self, decay=0.9, epsilon=1e-8):
        """Constructs adadelta optimizer."""
        self.α = decay
        self.ϵ = epsilon
        self.v = 0
        self.EΔ = 0

    def rmse(self, E):
        return np.sqrt(E + self.ϵ)

    def __call__(self, cost, θ):
        g = grad(cost)(θ)
        self.v = self.α * self.v + (1 - self.α) * (g ** 2)
        Δθ = -self.rmse(self.EΔ) * g / self.rmse(self.v)
        self.EΔ = self.α * self.EΔ + (1 - self.α) * (Δθ ** 2)

        return Δθ


class RMSprop(object):
    def __init__(self, learning_rate=0.01, decay=0.9, epsilon=1e-8):
        """Constructs rmsprop optimizer."""
        self.η = learning_rate
        self.α = decay
        self.ϵ = epsilon
        self.v = 0

    def rmse(self, E):
        return np.sqrt(E + self.ϵ)

    def __call__(self, cost, θ):
        g = grad(cost)(θ)
        self.v = self.α * self.v + (1 - self.α) * (g ** 2)

        return -self.η * g / self.rmse(self.v)


class Adam(object):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Constructs adaptive moment estimation (adam) optimizer."""
        self.η = learning_rate
        self.β1 = beta1
        self.β2 = beta2
        self.ϵ = epsilon
        self.m = 0
        self.v = 0
        self.t = 0

    def __call__(self, cost, θ):
        self.t += 1

        g = grad(cost)(θ)
        self.η *= np.sqrt(1 - (self.β2 ** self.t)) / (1 - (self.β1 ** self.t))
        self.m = self.β1 * self.m + (1 - self.β1) * g
        self.v = self.β2 * self.v + (1 - self.β2) * (g ** 2)

        return -self.η * self.m / (np.sqrt(self.v) + self.ϵ)


class AdaMax(object):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Constructs adamax optimizer."""
        self.η = learning_rate
        self.β1 = beta1
        self.β2 = beta2
        self.ϵ = epsilon
        self.m = 0
        self.u = 0
        self.t = 0

    def __call__(self, cost, θ):
        self.t += 1

        g = grad(cost)(θ)
        self.η /= 1 - (self.β1 ** self.t)
        self.m = self.β1 * self.m + (1 - self.β1) * g
        self.u = np.maximum(self.β2 * self.u, np.abs(g))

        return -self.η * self.m / (self.u + self.ϵ)


class Nadam(object):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Constructs nesterov-accelerated adam optimizer."""
        self.η = learning_rate
        self.β1 = beta1
        self.β2 = beta2
        self.ϵ = epsilon
        self.m = 0
        self.v = 0
        self.t = 0

    def __call__(self, cost, θ):
        self.t += 1

        g = grad(cost)(θ)
        self.η *= np.sqrt(1 - (self.β2 ** self.t)) / (1 - (self.β1 ** self.t))
        self.m = self.β1 * self.m + (1 - self.β1) * g
        self.v = self.β2 * self.v + (1 - self.β2) * (g ** 2)

        return -(
            self.η * (self.β1 * self.m + (1 - self.β1) * g) / (np.sqrt(self.v) + self.ϵ)
        )


class AMSGrad(object):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Constructs amsgrad optimizer."""
        self.η = learning_rate
        self.β1 = beta1
        self.β2 = beta2
        self.ϵ = epsilon
        self.m = 0
        self.v = 0
        self.vh = 0
        self.t = 0

    def __call__(self, cost, θ):
        self.t += 1

        g = grad(cost)(θ)
        self.m = self.β1 * self.m + (1 - self.β1) * g
        self.v = self.β2 * self.v + (1 - self.β2) * (g ** 2)
        self.vh = np.maximum(self.vh, self.v)

        return -self.η * self.m / (np.sqrt(self.vh) + self.ϵ)
