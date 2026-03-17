from typing import Any
import numpy as np

Array1d = np.ndarray[tuple[int], np.dtype]
Array2d = np.ndarray[tuple[int, int], np.dtype]
Array3d = np.ndarray[tuple[int, int, int], np.dtype]

# DType = str | np.dtype # problematic for 3.9
Dtype = Any

# anything that can be converted into a 2-element array.  The scalar case is the special case
# where both elements are the same.
ArrayLike2 = (
    float | int | list[float | int] | tuple[float | int, float | int] | np.ndarray
)

# anything that can be converted to a 1D array of floats or ints by spatialmath.base.getvector()
ArrayLike = float | int | list[float | int] | tuple[float | int, ...] | np.ndarray
