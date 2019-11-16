# -*- coding: utf-8 -*-

import jax.numpy as np


class Schedule(object):
    """Base class for hyperparameter schedules."""

    def __call__(self, i):
        raise NotImplementedError
