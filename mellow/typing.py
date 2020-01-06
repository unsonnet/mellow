# -*- coding: utf-8 -*-

from typing import Callable, Iterable, List, Sequence, Union

import jax.numpy as np


Shape = Sequence[int]
Tensor = Union[float, np.ndarray]

Dataset = Iterable[Tensor]
Loss = Callable[[Tensor, Tensor], Tensor]
UFunc = Callable[[Tensor], Tensor]
