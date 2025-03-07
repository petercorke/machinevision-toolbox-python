import numpy as np


# decorators
def scalar_result(func):
    """
    Decorator @scalar_result

    This decorator converts a function that returns an iterable of scalar values
    into a function that returns a single scalar value or an ndarray.

    =============== ======================
    Function output  Decorator output
    =============== ======================
    [1,2,3]         ndarray(1,2,3)
    [1]             1
    =============== ======================

    """

    def innerfunc(*args, **kwargs):
        out = func(*args, **kwargs)
        if len(out) == 1:
            return out[0]
        else:
            return np.array(out)

    inner = innerfunc
    inner.__doc__ = func.__doc__  # pass through the doc string
    return inner


def array_result(func):
    """
    Decorator @array_result

    This decorator converts a function that returns an iterable of object values
    into a function that returns a single object value or the original iterable.

    =============================  =============================
    Function output                Decorator output
    =============================  =============================
    [1,2,3]                        [1,2,3]
    [1]                            1
    [ndarray(1,2), ndarray(3,4)]   [ndarray(1,2), ndarray(3,4)]
    [ndarray(1,2)]                 ndarray(1,2)
    =============================  =============================

    """

    def innerfunc(*args):
        out = func(*args)
        if len(out) == 1:
            return out[0]
        else:
            return out

    inner = innerfunc
    inner.__doc__ = func.__doc__  # pass through the doc string
    return inner


def array_result2(func):
    """
    Decorator @array_result2

    This decorator converts a function that returns an iterable of ndarray values
    into a function that returns a single object value or the original iterable.
    A single value is flatened to a 1D array, multiple values are stacked as columns.

    """

    def innerfunc(*args):
        out = func(*args)
        if len(out) == 1:
            return out[0].flatten()
        else:
            return np.squeeze(np.array(out)).T

    inner = innerfunc
    inner.__doc__ = func.__doc__  # pass through the doc string
    return inner
