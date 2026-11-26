#!/usr/bin/env python
"""Release auto-detection on ROS 2 bags.

A ROS 2 bag holds CDR messages, but auto-detection used to try the ROS 1 wire format
first.  For a PNG ``CompressedImage`` that trial succeeded by accident with the ROS 1
Noetic store, and the image then failed to decode.  The bags are synthetic, written
into a temporary directory.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

pytest.importorskip("rosbags")

from PIL import Image as PILImage
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from machinevisiontoolbox import ROSBag

RED = np.zeros((8, 8, 3), np.uint8)
RED[..., 0] = 255  # pure red, in R, G, B order


def write_ros2_bag(path, fmt: str) -> None:
    buffer = io.BytesIO()
    PILImage.fromarray(RED, "RGB").save(buffer, fmt.upper(), quality=100)
    ts = get_typestore(Stores.ROS2_HUMBLE)
    T = ts.types
    msgtype = "sensor_msgs/msg/CompressedImage"
    with Writer(path, version=9) as w:
        conn = w.add_connection("/cam", msgtype, typestore=ts)
        header = T["std_msgs/msg/Header"](
            stamp=T["builtin_interfaces/msg/Time"](sec=10, nanosec=0), frame_id="cam"
        )
        msg = T[msgtype](
            header=header,
            format=fmt,
            data=np.frombuffer(buffer.getvalue(), np.uint8),
        )
        w.write(conn, 10_000_000_000, ts.serialize_cdr(msg, msgtype))


@pytest.mark.parametrize("fmt", ["jpeg", "png"])
def test_a_ros2_compressed_image_is_read(tmp_path, fmt):
    write_ros2_bag(tmp_path / "bag", fmt)
    bag = ROSBag(str(tmp_path / "bag"))
    (img,) = list(bag)
    assert bag.release.startswith("ROS2")
    # decoded by OpenCV in B, G, R order, so a red image is large in the last plane
    means = img.array.reshape(-1, 3).mean(axis=0)
    assert means[2] > 250 and means[0] < 5


def test_auto_detect_does_not_try_the_ros1_format_on_a_ros2_bag(tmp_path, capsys):
    write_ros2_bag(tmp_path / "bag", "png")
    list(ROSBag(str(tmp_path / "bag"), verbose=True))
    out = capsys.readouterr().out
    assert "ros1" not in out
    assert "CDR encoding" in out
