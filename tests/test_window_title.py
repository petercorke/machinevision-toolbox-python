"""Images are displayed in a window with the title that was asked for."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from machinevisiontoolbox import Image


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_disp_sets_the_window_title():
    Image.Random(size=(8, 6)).disp(title="my title", block=False)
    assert plt.gcf().canvas.manager.get_window_title() == "my title"


def test_redisplay_in_the_same_axes_updates_the_window_title():
    fig, ax = plt.subplots()
    image = Image.Random(size=(8, 6))
    image.disp(ax=ax, title="first", block=False)
    image.disp(ax=ax, title="second", block=False)
    assert fig.canvas.manager.get_window_title() == "second"


def test_the_removed_local_helper_is_not_exported():
    import machinevisiontoolbox
    from machinevisiontoolbox import base

    assert not hasattr(machinevisiontoolbox, "set_window_title")
    assert "set_window_title" not in base.__all__
