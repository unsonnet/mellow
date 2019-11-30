# -*- coding: utf-8 -*-

from typing import Sequence, Union, Callable

import jax.numpy as np


Shape = Sequence[int]
Array = Sequence[float]
Tensor = Union[float, np.ndarray]
UFunc = Callable[[Tensor], Tensor]
