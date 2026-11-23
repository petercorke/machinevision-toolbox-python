"""draw_point() says what is wrong with its marker string."""

import numpy as np
import pytest

from machinevisiontoolbox.base.graphics import draw_point


@pytest.fixture
def img():
    return np.zeros((60, 60, 3), dtype="uint8")


@pytest.mark.parametrize("marker", ["rg+", "rr+", "kw*", "+bk"])
def test_two_color_letters(img, marker):
    with pytest.raises(ValueError, match="more than one color letter"):
        draw_point(img, (30, 30), marker, fontsize=1)


@pytest.mark.parametrize("marker", ["", "r", "g"])
def test_no_marker_character(img, marker):
    with pytest.raises(ValueError, match="no marker character"):
        draw_point(img, (30, 30), marker, color="red", fontsize=1)


@pytest.mark.parametrize("marker", ["+", "*", "x"])
def test_no_color_at_all(img, marker):
    with pytest.raises(ValueError, match="no color"):
        draw_point(img, (30, 30), marker, fontsize=1)


@pytest.mark.parametrize(
    "marker, color",
    [("r+", None), ("+", "red"), ("b*", None), ("*", (0, 255, 0)), ("x", 200)],
)
def test_valid_calls_still_draw(img, marker, color):
    draw_point(img, (30, 30), marker, color=color, fontsize=1)
    assert img.any()


def test_color_argument_wins_over_the_letter(img):
    draw_point(img, (30, 30), "r+", color="blue", fontsize=1)
    assert img[..., 2].max() > 0 and img[..., 0].max() == 0
