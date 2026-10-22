"""ROSBag detects the ROS release by default (``release="auto"``).

``bagtool --release`` already defaulted to ``"auto"``; the ``ROSBag`` class defaulted to
``"ROS1_NOETIC"``, which cannot read ROS 2 (CDR) messages.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rosbags")

from machinevisiontoolbox import ROSBag

DATA = Path(__file__).parent / "data"
ROS1_BAG = DATA / "test_ros1.bag"
ROS2_BAG = DATA / "test_ros2"

# (bag, a release that is known to be right for it)
BAGS = [(ROS1_BAG, "noetic"), (ROS2_BAG, "humble")]


def _need(bag: Path) -> None:
    if not bag.exists():
        pytest.skip(f"fixture {bag.name} not generated")


def test_default_release_is_auto():
    _need(ROS1_BAG)
    assert ROSBag(str(ROS1_BAG)).release == "auto"


@pytest.mark.parametrize("bag, release", BAGS)
def test_default_reads_the_same_images_as_the_right_release(bag, release):
    _need(bag)
    auto = list(ROSBag(str(bag)))
    explicit = list(ROSBag(str(bag), release=release))

    assert len(auto) == len(explicit) > 0
    for a, e in zip(auto, explicit):
        assert a.topic == e.topic and a.timestamp == e.timestamp
        np.testing.assert_array_equal(a.array, e.array)


@pytest.mark.parametrize("bag, release", BAGS)
def test_detected_release_is_reported(bag, release):
    _need(bag)
    reader = ROSBag(str(bag))
    assert reader.release == "auto"
    next(iter(reader))
    assert reader.release != "auto"
    assert reader.release.upper().startswith("ROS")


@pytest.mark.parametrize("bag, release", BAGS)
def test_detection_is_quiet_unless_verbose(bag, release, capsys):
    _need(bag)
    list(ROSBag(str(bag)))
    assert "auto-detect" not in capsys.readouterr().out

    list(ROSBag(str(bag), verbose=True))
    assert "auto-detect" in capsys.readouterr().out


def test_an_explicit_release_still_works_and_is_not_changed():
    _need(ROS1_BAG)
    reader = ROSBag(str(ROS1_BAG), release="noetic")
    assert len(list(reader)) > 0
    assert reader.release == "noetic"  # nothing was detected, nothing to report


def test_default_reads_a_real_ros2_mcap_bag():
    """The old default, ROS1_NOETIC, raised a SerdeError on this recording."""
    path = Path(__file__).parent.parent / "bags" / "ros2_bouncing_ball"
    if not path.exists():
        pytest.skip("real ROS 2 bag not available")
    images = list(ROSBag(str(path)))
    # The bag's metadata claims 132 + 133 images, but its files only hold 39 + 40.
    assert len(images) == 79
