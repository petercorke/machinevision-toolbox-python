"""Image.showwindow: the axes default to the current axes, as documented."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from machinevisiontoolbox import Image


@pytest.fixture
def img() -> Image:
    return Image(np.arange(100, dtype=np.uint8).reshape(10, 10))


def test_showwindow_defaults_to_current_axes(img: Image) -> None:
    fig, ax = plt.subplots()
    try:
        window = img.showwindow(h=1)
        assert window.patch in ax.patches
    finally:
        plt.close(fig)


def test_showwindow_explicit_axes(img: Image) -> None:
    fig, ax = plt.subplots()
    try:
        window = img.showwindow(h=1, ax=ax)
        assert window.patch in ax.patches
    finally:
        plt.close(fig)


def test_window_move_returns_pixels(img: Image) -> None:
    fig, _ = plt.subplots()
    try:
        window = img.showwindow(h=1)
        pixels = window.move(2, 3)  # centred on u=2, v=3
        assert pixels.shape == (3, 3)
        np.testing.assert_array_equal(pixels, img.array[2:5, 1:4])
    finally:
        plt.close(fig)
