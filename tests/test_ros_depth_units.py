"""ROSBag(depth_units=...) returns depth images as float32 in metres.

The bags are synthetic, written into a temporary directory by
``tests/create_ros_fixture.py``.  Pixels that are zero (no measurement) stay zero.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("rosbags")

from create_ros_fixture import (
    RS_COLOR_TOPIC,
    RS_DEPTH_TOPIC,
    RS_DEPTH_UNITS,
    depth_pixels,
    make_realsense_bag,
    realsense_color_frame,
    realsense_depth_frame,
)
from machinevisiontoolbox import ROSBag


def read(path, **kwargs):
    """Depth and color images of a bag, and the RuntimeWarnings issued."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        images = list(ROSBag(path, **kwargs))
    depth = [i for i in images if i.topic == RS_DEPTH_TOPIC]
    color = [i for i in images if i.topic == RS_COLOR_TOPIC]
    return depth, color, [w for w in caught if issubclass(w.category, RuntimeWarning)]


@pytest.mark.parametrize("value", [0, -1, 0.0, float("nan"), float("inf"), "metres", True])
def test_depth_units_must_be_none_auto_or_a_positive_number(tmp_path, value):
    with pytest.raises(ValueError, match="depth_units"):
        ROSBag(tmp_path / "x.bag", depth_units=value)


def test_by_default_depth_is_returned_as_recorded(tmp_path):
    make_realsense_bag(tmp_path / "rs.bag")
    depth, _, caught = read(tmp_path / "rs.bag")
    for k, im in enumerate(depth):
        assert np.issubdtype(im.array.dtype, np.integer)  # not converted
        np.testing.assert_array_equal(im.array, realsense_depth_frame(k))
        assert im.depth_units == pytest.approx(RS_DEPTH_UNITS, rel=1e-6)
    assert not caught


def test_auto_uses_the_footer_of_a_realsense_bag(tmp_path):
    make_realsense_bag(tmp_path / "rs.bag")
    depth, color, caught = read(tmp_path / "rs.bag", depth_units="auto")
    assert len(depth) == 3
    for k, im in enumerate(depth):
        assert im.array.dtype == np.float32
        np.testing.assert_allclose(
            im.array, realsense_depth_frame(k) * RS_DEPTH_UNITS, rtol=1e-6
        )
        assert im.depth_units == 1.0  # now in metres
    # the color stream is never touched
    for k, im in enumerate(color):
        assert im.array.dtype == np.uint8
        np.testing.assert_array_equal(im.array, realsense_color_frame(k))
        assert not hasattr(im, "depth_units")
    assert not caught


def test_zero_stays_zero(tmp_path):
    make_realsense_bag(tmp_path / "rs.bag")
    depth, _, _ = read(tmp_path / "rs.bag", depth_units="auto")
    assert depth[0].array[0, 0] == 0.0  # frame 0 starts at 0


def test_a_number_overrides_the_footer(tmp_path):
    make_realsense_bag(tmp_path / "rs.bag")
    depth, _, _ = read(tmp_path / "rs.bag", depth_units=0.01)
    np.testing.assert_allclose(
        depth[1].array, realsense_depth_frame(1) * 0.01, rtol=1e-6
    )


def test_mono16_without_a_footer_has_no_known_scale(tmp_path):
    """Intel's sample bags have no footer, and mono16 has no unit in REP 118."""
    make_realsense_bag(tmp_path / "rs.bag", footer=False)
    depth, _, caught = read(tmp_path / "rs.bag", depth_units="auto")
    for k, im in enumerate(depth):  # returned as recorded, not guessed
        assert np.issubdtype(im.array.dtype, np.integer)
        np.testing.assert_array_equal(im.array, realsense_depth_frame(k))
    assert len(caught) == 1  # once for the topic, not once for each frame
    assert "not recorded" in str(caught[0].message)
    assert "depth_units=" in str(caught[0].message)


def test_a_number_converts_mono16_without_a_footer(tmp_path):
    make_realsense_bag(tmp_path / "rs.bag", footer=False)
    depth, _, caught = read(tmp_path / "rs.bag", depth_units=0.001)
    np.testing.assert_allclose(depth[2].array, realsense_depth_frame(2) * 0.001, rtol=1e-6)
    assert not caught


def test_16uc1_is_in_millimetres_by_rep_118(tmp_path):
    make_realsense_bag(tmp_path / "rs.bag", footer=False, depth_encoding="16UC1")
    depth, _, caught = read(tmp_path / "rs.bag", depth_units="auto")
    for k, im in enumerate(depth):
        assert im.array.dtype == np.float32
        np.testing.assert_allclose(im.array, realsense_depth_frame(k) * 0.001, rtol=1e-6)
    assert not caught


def test_32fc1_is_already_metres_and_a_number_is_a_gain(tmp_path):
    make_realsense_bag(tmp_path / "rs.bag", footer=False, depth_encoding="32FC1")
    auto, _, _ = read(tmp_path / "rs.bag", depth_units="auto")
    gain, _, _ = read(tmp_path / "rs.bag", depth_units=2.0)
    for k in range(3):
        np.testing.assert_allclose(auto[k].array, depth_pixels(k, "32FC1"), rtol=1e-6)
        np.testing.assert_allclose(gain[k].array, 2 * depth_pixels(k, "32FC1"), rtol=1e-6)
        assert auto[k].array.dtype == np.float32


# ---------------------------------------------------------------------------
# real bags, only in a development checkout (bags/ is not in the repository)
# ---------------------------------------------------------------------------

import itertools
from pathlib import Path

REAL_BAGS = Path(__file__).resolve().parent.parent / "bags"


def _first_depth(name: str, **kwargs):
    path = REAL_BAGS / name
    if not path.exists():
        pytest.skip(f"real bag {name} not available")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for im in itertools.islice(ROSBag(str(path), **kwargs), 100):
            if "depth" in im.topic.lower():
                return im
    raise AssertionError(f"no depth image in {name}")


def test_real_realsense_bag_in_metres():
    raw = _first_depth("20260430_125558.bag")
    metres = _first_depth("20260430_125558.bag", depth_units="auto")
    assert metres.array.dtype == np.float32
    np.testing.assert_allclose(metres.array, raw.array * np.float32(0.001), rtol=1e-6)


def test_real_ros_driver_bag_is_millimetres_by_rep_118():
    raw = _first_depth("2024-12-03-05-40-15.bag")  # 16UC1, no footer
    metres = _first_depth("2024-12-03-05-40-15.bag", depth_units="auto")
    np.testing.assert_allclose(metres.array, raw.array * np.float32(0.001), rtol=1e-6)


def test_real_tum_bag_is_already_metres():
    raw = _first_depth("rgbd_dataset_freiburg1_desk2.bag")  # 32FC1
    metres = _first_depth("rgbd_dataset_freiburg1_desk2.bag", depth_units="auto")
    np.testing.assert_array_equal(metres.array, raw.array)  # NaN where equal


def test_real_intel_sample_bag_scale_is_not_guessed():
    path = REAL_BAGS / "single_depth_color_640x480.bag"
    if not path.exists():
        pytest.skip("real bag single_depth_color_640x480.bag not available")
    with pytest.warns(RuntimeWarning, match="not recorded"):
        images = list(ROSBag(str(path), depth_units="auto"))
    depth = [i for i in images if "Depth" in i.topic]
    assert depth and all(np.issubdtype(i.array.dtype, np.integer) for i in depth)
