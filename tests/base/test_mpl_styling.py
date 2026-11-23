"""Compact Matplotlib markers must be consumed before checking leftovers."""

import matplotlib.pyplot as plt
import pytest
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

from machinevisiontoolbox.base import mpl_styling
from machinevisiontoolbox.base.graphics import parse_fmt


@pytest.mark.parametrize(
    "fmt, expected",
    [
        ("o", {"marker": "o", "linestyle": "None"}),
        ("ro--", {"color": "r", "marker": "o", "linestyle": "--"}),
        ("g^:", {"color": "g", "marker": "^", "linestyle": ":"}),
        ("-.b*", {"color": "b", "marker": "*", "linestyle": "-."}),
        (("ro--",), {"color": "r", "marker": "o", "linestyle": "--"}),
    ],
)
def test_marker_formats(fmt, expected):
    options = mpl_styling(fmt, {}, {})
    assert options == expected
    fig, ax = plt.subplots()
    try:
        (line,) = ax.plot([0, 1], [0, 1], **options)
        assert line.get_marker() == expected["marker"]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("fmt", ["roo", "r?", "ro--?"])
def test_unrecognized_components_still_raise(fmt):
    with pytest.raises(ValueError, match="not a valid format string"):
        mpl_styling(fmt, {}, {})


def test_keyword_options_override_compact_marker():
    assert mpl_styling("ro--", {"linewidth": 2}, {"marker": "x"}) == {
        "color": "r",
        "marker": "x",
        "linestyle": "--",
        "linewidth": 2,
    }


@pytest.mark.parametrize(
    "artist, short, long_name",
    [
        (Line2D, "lw", "linewidth"),
        (Line2D, "c", "color"),
        (Line2D, "ls", "linestyle"),
        (Line2D, "ms", "markersize"),
        (Patch, "ec", "edgecolor"),
        (Patch, "fc", "facecolor"),
        (Text, "ha", "horizontalalignment"),
        (Text, "size", "fontsize"),
    ],
)
def test_alias_replaces_the_default(artist, short, long_name):
    options = mpl_styling(None, {long_name: "default"}, {short: "mine"}, artist=artist)
    assert options == {long_name: "mine"}


def test_alias_in_defaults_is_converted_too():
    options = mpl_styling(None, {"lw": 2}, {"linewidth": 5}, artist=Line2D)
    assert options == {"linewidth": 5}


def test_alias_beats_the_format_string():
    options = mpl_styling("r--", {}, {"c": "blue", "ls": ":"}, artist=Line2D)
    assert options == {"color": "blue", "linestyle": ":"}


def test_merged_options_are_accepted_by_matplotlib():
    options = mpl_styling(
        None, {"color": "red", "linewidth": 2}, {"lw": 4}, artist=Line2D
    )
    fig, ax = plt.subplots()
    try:
        (line,) = ax.plot([0, 1], [0, 1], **options)
        assert line.get_linewidth() == 4
        assert line.get_color() == "red"
    finally:
        plt.close(fig)


def test_both_names_given_by_the_caller_still_raises():
    with pytest.raises(TypeError, match="aliases of one another"):
        mpl_styling(None, {}, {"lw": 1, "linewidth": 2}, artist=Line2D)


def test_without_artist_the_names_are_not_converted():
    assert mpl_styling(None, {"linewidth": 2}, {"lw": 4}) == {"linewidth": 2, "lw": 4}


def test_none_values_are_still_dropped():
    assert mpl_styling(None, {"color": "red"}, {"lw": None}, artist=Line2D) == {
        "color": "red"
    }


@pytest.mark.parametrize("name", ["plot_axes", "plot_ellipse", "plot_perimeter"])
def test_blobs_plot_accepts_aliases(name):
    from machinevisiontoolbox import Image

    blobs = Image.Read("sharks.png").blobs()
    fig, ax = plt.subplots()
    try:
        getattr(blobs[0], name)(lw=4, c="blue")
        assert all(line.get_linewidth() == 4 for line in ax.lines)
        assert all(line.get_color() == "blue" for line in ax.lines)
    finally:
        plt.close(fig)


def test_blobs_plot_box_accepts_aliases():
    from machinevisiontoolbox import Image

    blobs = Image.Read("sharks.png").blobs()
    fig, ax = plt.subplots()
    try:
        blobs[0].plot_box(lw=4, ec="blue")
        assert ax.patches[0].get_linewidth() == 4
    finally:
        plt.close(fig)


@pytest.mark.parametrize("fmt", ["--", "o", ":", "r", "b--", "g."])
def test_defaults_not_set_by_the_string_are_kept(fmt):
    defaults = {"color": "red", "linestyle": "-", "marker": "s", "linewidth": 2}
    options = mpl_styling(fmt, defaults, {})
    for key, value in parse_fmt(fmt).items():
        assert options[key] == value  # what the string sets wins
    for key, value in defaults.items():
        if key not in parse_fmt(fmt):
            assert options[key] == value  # the rest is not wiped out


def test_dashes_keep_the_default_color():
    # was: the string's missing color came back as None and wiped the default
    assert mpl_styling("--", {"color": "red"}, {}) == {
        "color": "red",
        "linestyle": "--",
    }


def test_a_marker_in_the_format_string_is_refused_for_a_patch():
    from matplotlib.patches import Patch

    with pytest.raises(ValueError, match="cannot draw"):
        mpl_styling("r.", {}, {}, artist=Patch)
    with pytest.raises(ValueError, match="'o'"):
        mpl_styling(("o",), {}, {}, artist=Patch)  # as *args arrives
    # a line style, a color and a whole color name are fine
    assert mpl_styling("r--", {}, {}, artist=Patch) == {
        "color": "r",
        "linestyle": "--",
    }
    # and a marker is fine for a line, or when no artist is given
    assert mpl_styling("r.", {}, {}, artist=Line2D)["marker"] == "."
    assert mpl_styling("r.", {}, {})["marker"] == "."
