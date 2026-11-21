#!/usr/bin/env python
"""Input shape and data type handling of gamma_encode, gamma_decode and colorspace_convert.

Integer data (uint8, uint16) spans 0 to the maximum of its type, and float data spans
0 to 1.  The output type is the input type unless ``dtype`` says otherwise.  HSV, HLS,
XYZ, L*a*b* and L*u*v* are always floating point, in their natural units.
"""

import numpy as np
import numpy.testing as nt
import pytest

from machinevisiontoolbox import Image
from machinevisiontoolbox.base.color import (
    colorspace_convert,
    gamma_decode,
    gamma_encode,
)

DTYPES = [np.uint8, np.uint16, np.float32, np.float64]
SHAPES = [(), (5,), (4, 5), (4, 5, 3)]
NATURAL = ["hsv", "hls", "xyz", "lab", "luv"]


def make(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        return rng.integers(0, np.iinfo(dtype).max + 1, shape).astype(dtype)
    return rng.random(shape).astype(dtype)


# ---------------------------------------------------------------------------
# gamma
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("func", [gamma_encode, gamma_decode])
@pytest.mark.parametrize("gamma", ["sRGB", 2.2])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gamma_keeps_shape_and_type(func, gamma, shape, dtype):
    image = make(shape, dtype)
    out = func(image, gamma)
    assert out.shape == image.shape
    assert out.dtype == np.dtype(dtype)
    if np.issubdtype(dtype, np.floating):
        assert np.all((out >= 0) & (out <= 1))


@pytest.mark.parametrize("func", [gamma_encode, gamma_decode])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gamma_1d_is_the_same_as_2d(func, dtype):
    x = make((12,), dtype)
    nt.assert_allclose(func(x), func(x.reshape(3, 4)).reshape(12), rtol=1e-6)


def test_gamma_srgb_values():
    # the linear segment, the curve and the end points
    nt.assert_allclose(gamma_encode(np.array([0.0, 0.001, 0.5, 1.0])), [0, 0.01292, 0.735357, 1], atol=1e-5)
    nt.assert_allclose(gamma_decode(np.array([0.0, 0.01, 0.5, 1.0])), [0, 0.000774, 0.214041, 1], atol=1e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gamma_round_trip_float(dtype):
    x = make((200,), dtype)
    tol = 2e-3 if dtype == np.float32 else 1e-9
    nt.assert_allclose(gamma_decode(gamma_encode(x)), x, atol=tol)
    # a numeric gamma is the exponent, so decoding undoes encoding with its inverse
    nt.assert_allclose(gamma_decode(gamma_encode(x, 1 / 2.2), 2.2), x, atol=tol)


def test_gamma_uint16_keeps_its_precision():
    x = np.array([1000, 1001, 30000, 65535], dtype=np.uint16)
    out = gamma_decode(x, 2.2)
    assert out.dtype == np.uint16
    assert out[-1] == 65535
    # converted via uint8 these would be squashed to 256 levels
    assert len(set(gamma_encode(np.arange(1000, 1100, dtype=np.uint16), 0.45))) > 50


def test_gamma_dtype_argument_converts_and_scales():
    x = np.array([0, 128, 255], dtype=np.uint8)
    out = gamma_decode(x, 2.0, dtype="float32")
    assert out.dtype == np.float32
    nt.assert_allclose(out, (np.array([0, 128, 255]) / 255) ** 2, atol=1e-6)
    out = gamma_decode(np.array([0.0, 0.5, 1.0]), 2.0, dtype=np.uint8)
    assert out.dtype == np.uint8
    nt.assert_array_equal(out, [0, 64, 255])
    assert gamma_decode(np.array([0.5], dtype=np.float32), 2.0, dtype="double").dtype == np.float64


def test_gamma_float_out_of_range_is_clipped_for_integer_output():
    out = gamma_encode(np.array([1.5, -0.5, 0.5]), 0.5, dtype=np.uint8)
    assert out.dtype == np.uint8
    assert out[0] == 255  # not wrapped round


@pytest.mark.parametrize("bad", [np.int32, np.int64, "bogus"])
def test_gamma_bad_dtype(bad):
    with pytest.raises((ValueError, TypeError)):
        gamma_encode(np.array([0.5]), 2.0, dtype=bad)


def test_gamma_bad_integer_input():
    with pytest.raises(ValueError):
        gamma_encode(np.array([1, 2, 3], dtype=np.int32))


def test_gamma_bad_gamma():
    with pytest.raises(ValueError):
        gamma_encode(np.array([0.5]), "bogus")
    with pytest.raises(ValueError):
        gamma_decode(np.array([0.5]), [1, 2])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# colorspace_convert
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dst", ["bgr", "ycrcb"])
@pytest.mark.parametrize("shape", [(3,), (4, 3), (4, 5, 3)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_convert_image_like_keeps_shape_and_type(dst, shape, dtype):
    image = make(shape, dtype)
    out = colorspace_convert(image, "rgb", dst)
    assert out.shape == image.shape
    assert out.dtype == np.dtype(dtype)


@pytest.mark.parametrize("dst", NATURAL)
@pytest.mark.parametrize("shape", [(3,), (4, 3), (4, 5, 3)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_convert_natural_units_is_float(dst, shape, dtype):
    image = make(shape, dtype)
    out = colorspace_convert(image, "rgb", dst)
    assert out.shape == image.shape
    expect = np.float64 if dtype == np.float64 else np.float32
    assert out.dtype == np.dtype(expect)


@pytest.mark.parametrize("shape", [(3,), (4, 3), (4, 5, 3)])
def test_convert_to_grey(shape):
    out = colorspace_convert(make(shape, np.uint8), "rgb", "grey")
    assert out.dtype == np.uint8
    assert out.shape == shape[:-1] or out.shape == (1,)


@pytest.mark.parametrize("shape", [(3,), (4, 3), (4, 5, 3)])
def test_convert_integers_are_scaled_like_floats(shape):
    u8 = make(shape, np.uint8)
    u16 = (u8.astype(np.uint16) * 257).astype(np.uint16)  # the same values in 16 bits
    f = u8.astype(np.float64) / 255
    for dst in NATURAL:
        expect = colorspace_convert(f, "rgb", dst)
        nt.assert_allclose(colorspace_convert(u8, "rgb", dst), expect, rtol=1e-3, atol=1e-3)
        nt.assert_allclose(colorspace_convert(u16, "rgb", dst), expect, rtol=1e-3, atol=1e-3)


def test_convert_vector_of_uint8_is_not_treated_as_float():
    # this used to give garbage: 255 was taken as a float
    out = colorspace_convert(np.array([255, 0, 0], dtype=np.uint8), "rgb", "hsv")
    nt.assert_allclose(out, [0, 1, 1], atol=1e-6)
    nt.assert_allclose(out, colorspace_convert([1, 0, 0], "rgb", "hsv"), atol=1e-6)


def test_convert_list_is_float():
    out = colorspace_convert([0, 0, 1], "rgb", "hsv")
    assert out.dtype == np.float64
    nt.assert_allclose(out, [240, 1, 1], atol=1e-5)


@pytest.mark.parametrize("space", NATURAL)
def test_convert_natural_units_need_float(space):
    with pytest.raises(ValueError, match="floating point"):
        colorspace_convert(make((4, 3), np.uint8), space, "rgb")
    with pytest.raises(ValueError, match="floating point"):
        colorspace_convert(make((4, 3), np.uint8), "rgb", space, dtype=np.uint8)


def test_convert_other_integer_types_are_not_images():
    with pytest.raises(ValueError):
        colorspace_convert(np.array([[1, 2, 3]], dtype=np.int32), "rgb", "bgr")
    with pytest.raises(ValueError):
        colorspace_convert(np.array([[0.1, 0.2, 0.3]]), "rgb", "bgr", dtype=np.int32)


@pytest.mark.parametrize("shape", [(4, 3), (4, 5, 3)])
@pytest.mark.parametrize("space", ["hsv", "hls", "xyz", "lab", "luv", "ycrcb"])
def test_convert_round_trip_uint8(shape, space):
    rng = np.random.default_rng(1)
    rgb = rng.integers(20, 236, shape, dtype=np.uint8)
    there = colorspace_convert(rgb, "rgb", space)
    back = colorspace_convert(there, space, "rgb", dtype=np.uint8)
    assert back.dtype == np.uint8
    nt.assert_allclose(back.astype(int), rgb.astype(int), atol=2)


def test_convert_dtype_argument():
    rgb = np.array([[0.2, 0.4, 0.6]])
    out = colorspace_convert(rgb, "rgb", "bgr", dtype=np.uint8)
    assert out.dtype == np.uint8
    nt.assert_array_equal(out, [[153, 102, 51]])
    out = colorspace_convert(make((4, 3), np.uint8), "rgb", "bgr", dtype="float")
    assert out.dtype == np.float32
    assert out.min() >= 0 and out.max() <= 1


def test_convert_float64_and_uint16_images_work_for_every_space():
    # OpenCV cannot do these directly
    for dtype in (np.float64, np.uint16):
        image = make((4, 5, 3), dtype)
        for dst in NATURAL + ["ycrcb", "bgr"]:
            colorspace_convert(image, "rgb", dst)


def test_convert_bad_shape():
    with pytest.raises(ValueError):
        colorspace_convert(np.zeros((4, 4)), "rgb", "hsv")


# ---------------------------------------------------------------------------
# Image.colorspace
# ---------------------------------------------------------------------------


def test_image_colorspace_uint8_to_natural_units():
    im = Image(make((6, 8, 3), np.uint8), colororder="RGB", dtype=True)
    hsv = im.colorspace("hsv")
    assert hsv.dtype == np.float32
    assert 0 <= hsv.array[..., 0].max() <= 360 and hsv.array[..., 0].max() > 180
    assert 0 <= hsv.array[..., 1].max() <= 1
    lab = im.colorspace("lab")
    assert lab.array[..., 1].min() < 0  # a* is signed


def test_image_colorspace_keeps_type_for_image_like_spaces():
    for dtype in DTYPES:
        im = Image(make((6, 8, 3), dtype), colororder="RGB", dtype=True)
        assert im.colorspace("ycrcb").dtype == np.dtype(dtype)


def test_image_colorspace_dtype_argument():
    im = Image(make((6, 8, 3), np.uint8), colororder="RGB", dtype=True)
    assert im.colorspace("bgr", dtype="float").dtype == np.float32
    assert im.colorspace("hsv", dtype="double").dtype == np.float64
