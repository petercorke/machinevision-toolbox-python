"""Compact Matplotlib markers must be consumed before checking leftovers."""

import matplotlib.pyplot as plt
import pytest

from machinevisiontoolbox.base import mpl_styling


@pytest.mark.parametrize(
    "fmt, expected",
    [
        ("o", {"marker": "o"}),
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
    with pytest.raises(ValueError, match="Unrecognized format components"):
        mpl_styling(fmt, {}, {})


def test_keyword_options_override_compact_marker():
    assert mpl_styling("ro--", {"linewidth": 2}, {"marker": "x"}) == {
        "color": "r",
        "marker": "x",
        "linestyle": "--",
        "linewidth": 2,
    }
