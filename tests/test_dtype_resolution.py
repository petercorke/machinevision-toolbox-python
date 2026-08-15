"""
Consolidated dtype-resolution consistency tests.

Multiple entry points resolve a user-supplied dtype string into an actual
NumPy dtype: the Image constructor (via _infer_dtype), convert(), and (once
fixed on their own branches) Image.to()/.array_as()/.astype() and the
ImageConstantsMixin factory methods (Zeros, Constant, Random, ...). All of
them are supposed to honour the same short-name aliases (DTYPE_ALIASES:
'int'->uint8, 'float'->float32, 'double'->float64, 'half'->float16) plus
pass explicit NumPy dtype strings through unchanged.

Three independent, inconsistent implementations of this resolution existed
at once (found 2026-08 while working through RVC3-python's chap11.ipynb --
VideoFile(..., mono=True, dtype='float') produced float64 frames instead of
float32, because Image.__init__'s own dtype resolution didn't share
convert()'s alias table). This module pins every entry point against the
same shared matrix of cases so that kind of drift can't happen silently
again -- add a test method to the relevant class below rather than a new
one-off test elsewhere when another entry point is fixed.
"""

import numpy as np
import pytest
from spatialmath import Polygon2

from machinevisiontoolbox import Image
from machinevisiontoolbox.base.imageio import convert

# (dtype spec passed in, expected resolved np.dtype)
DTYPE_CASES = [
    # short-name aliases (DTYPE_ALIASES) -- NumPy's own np.dtype(...) would
    # resolve 'float' to float64 and 'int' to platform int, not these.
    ("int", np.dtype("uint8")),
    ("float", np.dtype("float32")),
    ("double", np.dtype("float64")),
    ("half", np.dtype("float16")),
    # explicit NumPy dtype strings, not in DTYPE_ALIASES -- must pass
    # through unchanged, not be (mis)matched against an alias.
    ("uint8", np.dtype("uint8")),
    ("int16", np.dtype("int16")),
    ("float32", np.dtype("float32")),
    ("float64", np.dtype("float64")),
]
DTYPE_CASE_IDS = [c[0] for c in DTYPE_CASES]


@pytest.mark.parametrize("dtype_in,expected", DTYPE_CASES, ids=DTYPE_CASE_IDS)
class TestDtypeResolutionConsistency:
    """Every entry point below must resolve the same dtype_in the same way."""

    def test_image_constructor(self, dtype_in, expected):
        im = Image(np.ones((2, 3), dtype=np.uint8), dtype=dtype_in)
        assert im.dtype == expected

    def test_convert(self, dtype_in, expected):
        arr = convert(np.ones((2, 3), dtype=np.uint8), dtype=dtype_in)
        assert arr.dtype == expected


# ImageConstantsMixin factory methods: a *different* bug from the above --
# they build their raw pixel array with the (now alias-resolved)
# dtype, but historically never forwarded dtype= to the Image
# constructor call, so the constructor's own "no dtype given" auto-detect
# (any float input becomes float32) silently downcast even an explicit,
# already-correct dtype='float64' request. Fixed by passing dtype=True
# ("trust the array I already built") through _pattern_image() and each
# factory's own final constructor call.
_SQUARE = Polygon2([(2, 2), (8, 2), (8, 8), (2, 8)])
FACTORY_CASES = [
    ("Zeros", lambda dtype: Image.Zeros(size=8, dtype=dtype)),
    ("Constant_scalar", lambda dtype: Image.Constant(1.0, size=8, dtype=dtype)),
    (
        "Constant_iterable",
        lambda dtype: Image.Constant(
            [1.0, 0.5, 0.2], size=8, colororder="RGB", dtype=dtype
        ),
    ),
    ("Random", lambda dtype: Image.Random(size=8, dtype=dtype)),
    ("Squares", lambda dtype: Image.Squares(1, size=20, dtype=dtype)),
    ("Circles", lambda dtype: Image.Circles(1, size=20, dtype=dtype)),
    ("Ramp", lambda dtype: Image.Ramp(size=20, dtype=dtype)),
    ("Sin", lambda dtype: Image.Sin(size=20, dtype=dtype)),
    ("Chequerboard", lambda dtype: Image.Chequerboard(size=20, dtype=dtype)),
    ("Polygons", lambda dtype: Image.Polygons(_SQUARE, size=10, dtype=dtype)),
]
FACTORY_CASE_IDS = [c[0] for c in FACTORY_CASES]


@pytest.mark.parametrize("dtype_in,expected", DTYPE_CASES, ids=DTYPE_CASE_IDS)
@pytest.mark.parametrize("factory_name,factory_fn", FACTORY_CASES, ids=FACTORY_CASE_IDS)
def test_image_constants_factory_respects_explicit_dtype(
    factory_name, factory_fn, dtype_in, expected
):
    im = factory_fn(dtype_in)
    assert im.dtype == expected
