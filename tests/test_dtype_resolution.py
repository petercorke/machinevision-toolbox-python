"""
Consolidated dtype-resolution consistency tests.

Multiple entry points resolve a user-supplied dtype string into an actual
NumPy dtype: the Image constructor (via _infer_dtype), convert(), Image.to(),
Image.array_as(), Image.astype(), and (once fixed on its own branch) the
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
from machinevisiontoolbox.base.types import int_image

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

    def test_image_to(self, dtype_in, expected):
        im = Image(np.ones((2, 3), dtype=np.uint8))
        assert im.to(dtype_in).dtype == expected

    def test_image_astype(self, dtype_in, expected):
        im = Image(np.ones((2, 3), dtype=np.uint8))
        assert im.astype(dtype_in).dtype == expected

    def test_image_array_as(self, dtype_in, expected):
        im = Image(np.ones((2, 3), dtype=np.uint8))
        assert im.array_as(dtype_in).dtype == expected


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


class TestImageConstructorMaxintval:
    """Image(..., dtype=..., maxintval=...) must SCALE, not truncate.

    Regression for: Image.__init__ applied an explicit dtype= via a raw
    .astype() cast before convert()-only kwargs like maxintval ever saw the
    data, so maxintval was silently ignored and an out-of-range integer
    downcast (eg. uint16 0..4095 -> uint8) kept only the low byte instead of
    being rescaled into 0..255. Found 2026-08 via RVC3-python's visodom.py,
    which reads 12-bit .pgm frames as
    ``Image(..., dtype='uint8', maxintval=4095)`` -- the resulting images
    looked corrupted (an artifact indistinguishable from bottom-byte-only
    display) even though the actual display path (idisp/cv2.imshow) was
    independently verified correct; the data itself was already wrong by
    the time it reached display.
    """

    def test_downscale_with_maxintval_matches_manual_scaling(self):
        # 12-bit source values (0..4095) stored in a uint16 array, exactly
        # the enpeda/bridge .pgm scenario.
        src = np.linspace(0, 4095, 512, dtype=np.uint16)
        im = Image(np.tile(src, (2, 1)), dtype="uint8", maxintval=4095)
        assert im.dtype == np.uint8
        expected = np.rint(src.astype(np.float64) * 255 / 4095).astype(np.uint8)
        # int_image() truncates rather than rounds (separate, minor,
        # pre-existing quirk) -- allow the resulting off-by-one.
        assert np.max(np.abs(im._A[0].astype(int) - expected.astype(int))) <= 1

    def test_downscale_with_maxintval_is_not_byte_truncation(self):
        # the actual failure mode: a naive .astype() keeps only the bottom
        # byte, which is NOT monotonic (wraps every 256 counts). A properly
        # scaled monotonic ramp must stay monotonic.
        src = np.linspace(0, 4095, 512, dtype=np.uint16)
        im = Image(np.tile(src, (2, 1)), dtype="uint8", maxintval=4095)
        row = im._A[0].astype(int)
        assert np.all(np.diff(row) >= 0)  # monotonic non-decreasing
        assert not np.array_equal(row, src.astype(np.uint8))  # not truncated

    def test_maxintval_default_uses_source_dtype_max(self):
        # maxintval=None (default) should behave as before: scale using the
        # full range of the SOURCE dtype, eg. uint16's 65535, not 4095.
        # (mono=True is a no-op on this already-2D image; it's here purely
        # to trigger the convert()-kwargs code path alongside dtype=, since
        # dtype= on its own is a plain unscaled cast -- see
        # test_dtype_alone_still_unscaled_cast.)
        src = np.array([0, 4095, 65535], dtype=np.uint16)
        im = Image(src.reshape(1, 3), dtype="uint8", mono=True)
        expected = int_image(src, intclass="uint8")  # canonical reference
        assert np.array_equal(im._A[0], expected)

    def test_maxintval_combined_with_other_convert_kwargs(self):
        # the exact real-world combination from visodom.py: dtype +
        # maxintval alongside another convert()-only kwarg (mono) in one
        # call. R=G=B=src so ITU601 grey conversion (weights sum to 1)
        # reproduces src exactly before the dtype/maxintval scaling.
        src = np.tile(np.linspace(0, 4095, 100, dtype=np.uint16), (50, 1))
        color = np.stack([src] * 3, axis=-1)
        im = Image(color, mono=True, dtype="uint8", maxintval=4095)
        assert im.dtype == np.uint8
        assert im.ndim == 2
        assert im._A.max() >= 250  # properly scaled up to ~255, not stuck near 16

    def test_dtype_alone_still_unscaled_cast(self):
        # regression guard: dtype= with NO other kwargs must remain a plain
        # cast (no implicit scaling) -- the common/simple path, untouched
        # by the maxintval fix.
        src = np.array([0, 4095, 65535], dtype=np.uint16)
        im = Image(src.reshape(1, 3), dtype="float32")
        assert np.array_equal(im._A[0], src.astype(np.float32))

    def test_maxintval_without_explicit_dtype_stays_inert(self):
        # maxintval only makes sense paired with an explicit dtype (it's
        # the assumed max of the *source* data, used to compute the scale
        # factor to the target dtype). Without dtype=, behaviour is
        # unchanged from before this fix: maxintval is not applied.
        src = np.array([0, 4095, 65535], dtype=np.uint16)
        im = Image(src.reshape(1, 3), maxintval=4095)
        assert im.dtype == np.uint16
        assert np.array_equal(im._A[0], src)
