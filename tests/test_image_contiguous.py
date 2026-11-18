"""Images read from file are C-contiguous, and draw_* work on them (OpenCV >= 5.0).

OpenCV 5.0 rejects non-C-contiguous arrays in its drawing functions.  NumPy fancy
indexing, which reordered the color planes, returns a planar layout and so broke
``Image.draw_line()`` etc. on every image from ``Image.Read``.
"""

from typing import Any, Callable

import numpy as np
import pytest

from machinevisiontoolbox import Image, iread
from machinevisiontoolbox.base.imageio import convert


def _bgra() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(12, 16, 4), dtype=np.uint8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # BGR -> RGB
        {"rgb": False},  # BGRA -> BGR (drops alpha, was a strided view)
        {"alpha": True},  # BGRA -> RGBA
        {"alpha": True, "rgb": False},  # BGRA kept
        {"mono": True},
        {"reduce": 2},  # strided view
        {"roi": [2, 10, 3, 9]},  # strided view
        {"dtype": "float32"},
        {"gamma": "sRGB"},
    ],
)
def test_convert_returns_contiguous(kwargs: dict[str, Any]) -> None:
    out = convert(_bgra(), **kwargs)
    assert out.flags["C_CONTIGUOUS"]


def test_convert_greyscale_input_roi_contiguous() -> None:
    out = convert(np.arange(100, dtype=np.uint8).reshape(10, 10), roi=[1, 5, 2, 6])
    assert out.flags["C_CONTIGUOUS"]


def test_convert_keeps_pixel_values() -> None:
    bgra = _bgra()
    out = convert(bgra)  # RGB
    np.testing.assert_array_equal(out, bgra[:, :, [2, 1, 0]])
    out = convert(bgra, alpha=True)  # RGBA
    np.testing.assert_array_equal(out, bgra[:, :, [2, 1, 0, 3]])
    out = convert(bgra, rgb=False)  # BGR
    np.testing.assert_array_equal(out, bgra[:, :, :3])


def test_iread_and_read_contiguous() -> None:
    im, _ = iread("monalisa.png")
    assert im.flags["C_CONTIGUOUS"]
    assert Image.Read("monalisa.png")._A.flags["C_CONTIGUOUS"]


DRAWERS: dict[str, Callable[[Image], None]] = {
    "draw_line": lambda im: im.draw_line((10, 10), (100, 100), color="yellow"),
    "draw_circle": lambda im: im.draw_circle((100, 100), 30, color="yellow"),
    "draw_text": lambda im: im.draw_text((50, 50), "hi", color="yellow"),
    "draw_point": lambda im: im.draw_point((50, 50), "+", "eye", color="yellow"),
}


@pytest.mark.parametrize("name", list(DRAWERS))
def test_draw_on_read_image(name: str) -> None:
    im = Image.Read("monalisa.png")
    before = im.array.copy()
    DRAWERS[name](im)
    assert not np.array_equal(before, im.array)  # something was drawn


def _noncontiguous_image() -> tuple[Image, np.ndarray]:
    """An image whose array is a transposed view of another array."""
    base = np.zeros((300, 200, 3), dtype=np.uint8)
    view = base.transpose(1, 0, 2)  # shape (200, 300, 3), not C-contiguous
    assert not view.flags["C_CONTIGUOUS"]
    return Image(view), base


@pytest.mark.parametrize("name", list(DRAWERS))
def test_draw_on_noncontiguous_raises_helpful_error(name: str) -> None:
    im, base = _noncontiguous_image()

    with pytest.raises(ValueError, match="C-contiguous") as error:
        DRAWERS[name](im)

    assert "copy=True" in str(error.value)  # says how to fix it
    assert not base.any()  # nothing was drawn, the image is not left half drawn


def test_draw_labelbox_on_noncontiguous_raises() -> None:
    im, base = _noncontiguous_image()
    with pytest.raises(ValueError, match="C-contiguous"):
        im.draw_labelbox("hi", lb=(10, 100), rt=(80, 20), color="yellow")
    assert not base.any()


def test_draw_box_on_noncontiguous_raises() -> None:
    im, base = _noncontiguous_image()
    with pytest.raises(ValueError, match="C-contiguous"):
        im.draw_box(lt=(10, 10), rb=(80, 100), color="yellow")
    assert not base.any()


@pytest.mark.parametrize("name", list(DRAWERS))
def test_copy_true_makes_a_drawable_image(name: str) -> None:
    im, base = _noncontiguous_image()
    fixed = Image(im, copy=True)  # from an Image
    assert fixed._A.flags["C_CONTIGUOUS"]
    DRAWERS[name](fixed)
    assert fixed.array.any()

    fixed2 = Image(base.transpose(1, 0, 2), copy=True)  # from a plain array
    assert fixed2._A.flags["C_CONTIGUOUS"]
    DRAWERS[name](fixed2)
    assert fixed2.array.any()


def test_image_still_references_the_array_it_is_given() -> None:
    # not copying (and not forcing contiguity) is the documented default behavior
    a = np.zeros((4, 6), dtype=np.uint8)
    assert np.shares_memory(a, Image(a).array)

    view = np.zeros((6, 4), dtype=np.uint8).T
    im = Image(view)
    assert np.shares_memory(view, im.array)
    assert not im._A.flags["C_CONTIGUOUS"]  # kept as given


def test_feature_draw_on_noncontiguous_raises() -> None:
    feats = Image.Read("monalisa.png", mono=True).SIFT()
    assert len(feats) > 0
    for method in (feats.draw, feats.draw2):
        im, base = _noncontiguous_image()
        with pytest.raises(ValueError, match="C-contiguous"):
            method(im)
        assert not base.any()


def test_feature_draw2_on_read_image_works() -> None:
    img = Image.Read("monalisa.png")
    feats = img.SIFT()
    before = img.array.copy()
    feats.draw2(img)
    assert not np.array_equal(before, img.array)
