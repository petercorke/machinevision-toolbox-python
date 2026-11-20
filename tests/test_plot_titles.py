"""Plotting functions that create a window give it a sensible title."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from machinevisiontoolbox import Image


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def window_title(fig=None):
    fig = fig or plt.gcf()
    return fig.canvas.manager.get_window_title()


def test_histogram_default_title():
    Image.Random(size=(8, 6)).hist().plot()
    assert window_title() == "Image Histogram"


def test_histogram_given_title():
    Image.Random(size=(8, 6)).hist().plot(title="mine")
    assert window_title() == "mine"


def test_threshold_interactive_default_title():
    Image.Random(size=(8, 6), dtype="uint8").threshold_interactive(block=False)
    assert window_title() == "Interactive Image Threshold"


def test_threshold_interactive_given_title():
    Image.Random(size=(8, 6), dtype="uint8").threshold_interactive(
        title="mine", block=False
    )
    assert window_title() == "mine"
