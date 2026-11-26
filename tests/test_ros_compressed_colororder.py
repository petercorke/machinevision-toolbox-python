#!/usr/bin/env python
"""ROSBag colors of sensor_msgs/CompressedImage.

The images are encoded with Pillow, not OpenCV, so the test does not depend on the
decoder it checks.  A pure red JPEG must read back as red, whatever its plane order.
The bags are synthetic, written into a temporary directory.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

pytest.importorskip("rosbags")

from PIL import Image as PILImage
from rosbags.rosbag1 import Writer as Writer1
from rosbags.rosbag2 import Writer as Writer2
from rosbags.typesys import Stores, get_typestore

from machinevisiontoolbox import ROSBag


def encode(array: np.ndarray, mode: str, fmt: str) -> bytes:
    buffer = io.BytesIO()
    PILImage.fromarray(array, mode).save(buffer, fmt, quality=100)
    return buffer.getvalue()


def write_bag(path, ros2: bool, images: dict[str, tuple[str, bytes]]) -> None:
    """A bag with one CompressedImage per topic: {topic: (format, bytes)}."""
    ts = get_typestore(Stores.ROS2_HUMBLE if ros2 else Stores.ROS1_NOETIC)
    T = ts.types
    msgtype = "sensor_msgs/msg/CompressedImage"
    writer = Writer2(path, version=9) if ros2 else Writer1(path)
    with writer as w:
        for topic, (fmt, data) in images.items():
            conn = w.add_connection(topic, msgtype, typestore=ts)
            stamp = T["builtin_interfaces/msg/Time"](sec=10, nanosec=0)
            if ros2:
                header = T["std_msgs/msg/Header"](stamp=stamp, frame_id="cam")
            else:
                header = T["std_msgs/msg/Header"](seq=0, stamp=stamp, frame_id="cam")
            msg = T[msgtype](
                header=header, format=fmt, data=np.frombuffer(data, np.uint8)
            )
            serialize = ts.serialize_cdr if ros2 else ts.serialize_ros1
            w.write(conn, 10_000_000_000, serialize(msg, msgtype))


RED = np.zeros((8, 8, 3), np.uint8)
RED[..., 0] = 255  # pure red, in R, G, B order


@pytest.fixture(
    params=["ros1"]
)  # ROS 2 PNG bags need their own fix, see the ROS 2 tests
def make(request, tmp_path):
    def _make(images):
        path = tmp_path / ("bag.bag" if request.param == "ros1" else "bag")
        write_bag(path, request.param == "ros2", images)
        return str(path)

    return _make


def planes(img):
    """Mean of the red and blue planes, by name."""
    return float(img.plane("R").array.mean()), float(img.plane("B").array.mean())


@pytest.mark.parametrize("fmt", ["jpeg", "png"])
def test_a_red_image_is_red(make, fmt):
    data = encode(RED, "RGB", fmt.upper())
    (img,) = list(ROSBag(make({"/cam": (fmt, data)})))
    red, blue = planes(img)
    assert red > 250 and blue < 5
    assert img.colororder_str == "B:G:R"  # as OpenCV decodes it


def test_a_color_order_given_for_all_topics_is_used(make):
    data = encode(RED, "RGB", "PNG")
    path = make({"/cam": ("png", data)})
    (img,) = list(ROSBag(path, colororder="rgb"))
    assert img.colororder_str == "R:G:B"
    red, _ = planes(img)
    assert red < 5  # the caller said the planes are RGB, so the red image is not red


def test_a_color_order_given_per_topic_is_used_and_others_are_kept(make):
    data = encode(RED, "RGB", "PNG")
    path = make({"/a": ("png", data), "/b": ("png", data)})
    images = {i.topic: i for i in ROSBag(path, colororder={"/a": "rgb"})}
    assert images["/a"].colororder_str == "R:G:B"
    assert images["/b"].colororder_str == "B:G:R"  # not listed, so what was decoded
    assert planes(images["/b"])[0] > 250


def test_a_greyscale_image_stays_greyscale(make):
    grey = np.arange(64, dtype=np.uint8).reshape(8, 8)
    (img,) = list(ROSBag(make({"/g": ("png", encode(grey, "L", "PNG"))})))
    assert img.nplanes == 1
    assert img.colororder_str is None
    np.testing.assert_array_equal(img.array, grey)


def test_a_per_topic_dict_leaves_greyscale_topics_alone(make):
    grey = np.arange(64, dtype=np.uint8).reshape(8, 8)
    path = make(
        {
            "/g": ("png", encode(grey, "L", "PNG")),
            "/c": ("png", encode(RED, "RGB", "PNG")),
        }
    )
    images = {i.topic: i for i in ROSBag(path, colororder={"/c": "rgb"})}
    assert images["/g"].nplanes == 1
    assert images["/c"].colororder_str == "R:G:B"


def test_an_image_with_alpha_is_bgra(make):
    rgba = np.zeros((8, 8, 4), np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = 200
    (img,) = list(ROSBag(make({"/a": ("png", encode(rgba, "RGBA", "PNG"))})))
    assert img.colororder_str == "B:G:R:A"
    assert img.plane("R").array.mean() > 250
    assert img.plane("A").array.mean() == pytest.approx(200)
