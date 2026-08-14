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
