# must place at top of each file that imports this
#   from __future__ import annotations

from typing import Optional, Sequence, Any

# Self not until 3.11
# T = TypeVar("T", bound=np.generic, covariant=True)
import numpy as np

Array1d = np.ndarray[tuple[int], np.dtype]
Array2d = np.ndarray[tuple[int, int], np.dtype]
Array3d = np.ndarray[tuple[int, int, int], np.dtype]

# DType = str | np.dtype # problematic for 3.9
Dtype = Any
