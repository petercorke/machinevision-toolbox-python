#!/usr/bin/env python
"""ROSBag on message types and encodings that the small fixture bags do not hold:
raw image encodings, PointCloud2, an unsupported encoding, bad messages, and the
``print`` and ``traffic`` summaries.

The bags are synthetic, written into a temporary directory with ``rosbags``.  They show
that ROSBag reads what we believe devices write; they are not recordings of real devices.
"""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest

pytest.importorskip("rosbags")

from rosbags.rosbag1 import Writer as Writer1
from rosbags.rosbag2 import Writer as Writer2
from rosbags.typesys import Stores, get_typestore

from machinevisiontoolbox import ROSBag

try:
    import open3d  # noqa: F401

    open3d_available = True
except ImportError:
    open3d_available = False

H, W = 4, 6


class BagMaker:
    """Write a bag of hand-made messages, as ROS 1 or ROS 2."""

    def __init__(self, path, ros2: bool = False):
        self.path = path
        self.ros2 = ros2
        self.ts = get_typestore(Stores.ROS2_HUMBLE if ros2 else Stores.ROS1_NOETIC)
        self.T = self.ts.types
        self.writer = Writer2(path, version=9) if ros2 else Writer1(path)
        self.writer.__enter__()
        self.conns: dict[str, object] = {}
        self.count = 0

    def header(self, i: int = 0):
        stamp = self.T["builtin_interfaces/msg/Time"](sec=100 + i, nanosec=0)
        if self.ros2:
            return self.T["std_msgs/msg/Header"](stamp=stamp, frame_id="cam")
        return self.T["std_msgs/msg/Header"](seq=i, stamp=stamp, frame_id="cam")

    def add(self, topic: str, msgtype: str, msg) -> None:
        if topic not in self.conns:
            self.conns[topic] = self.writer.add_connection(
                topic, msgtype, typestore=self.ts
            )
        serialize = self.ts.serialize_cdr if self.ros2 else self.ts.serialize_ros1
        t = 100_000_000_000 + self.count * 100_000_000
        self.writer.write(self.conns[topic], t, serialize(msg, msgtype))
        self.count += 1

    def image(self, topic: str, encoding: str, array: np.ndarray, **fields) -> None:
        msgtype = "sensor_msgs/msg/Image"
        data = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
        msg = self.T[msgtype](
            header=self.header(self.count),
            height=H,
            width=W,
            encoding=encoding,
            is_bigendian=0,
            step=len(data) // H,
            data=data,
            **fields,
        )
        self.add(topic, msgtype, msg)

    def cloud(self, topic: str, points: np.ndarray, rgb: np.ndarray | None) -> None:
        """points is Nx3 float32; rgb is Nx3 uint8 or None."""
        PF = self.T["sensor_msgs/msg/PointField"]
        names = [("x", 0), ("y", 4), ("z", 8)]
        fields = [PF(name=n, offset=o, datatype=7, count=1) for n, o in names]
        step = 12
        if rgb is not None:
            fields.append(PF(name="rgb", offset=12, datatype=7, count=1))
            step = 16
        raw = bytearray()
        for k, p in enumerate(points):
            raw += struct.pack("<3f", *p)
            if rgb is not None:
                r, g, b = (int(c) for c in rgb[k])
                raw += struct.pack("<I", (r << 16) | (g << 8) | b)
        msgtype = "sensor_msgs/msg/PointCloud2"
        msg = self.T[msgtype](
            header=self.header(self.count),
            height=1,
            width=len(points),
            fields=fields,
            is_bigendian=False,
            point_step=step,
            row_step=step * len(points),
            data=np.frombuffer(bytes(raw), np.uint8),
            is_dense=True,
        )
        self.add(topic, msgtype, msg)

    def close(self) -> str:
        self.writer.__exit__(None, None, None)
        return str(self.path)


@pytest.fixture
def ros1(tmp_path):
    made = []

    def _make(name="a.bag"):
        m = BagMaker(tmp_path / name)
        made.append(m)
        return m

    yield _make


# --------------------------------------------------------------------------- #
# raw image encodings
# --------------------------------------------------------------------------- #

PIXELS = np.arange(H * W * 4, dtype=np.uint8)

ENCODINGS = [
    # encoding, array dtype, planes, expected colororder string
    ("rgb8", np.uint8, 3, "R:G:B"),
    ("bgr8", np.uint8, 3, "B:G:R"),
    ("rgba8", np.uint8, 4, "R:G:B:A"),
    ("bgra8", np.uint8, 4, "B:G:R:A"),
    ("mono8", np.uint8, 1, None),
    ("mono16", np.uint16, 1, None),
    ("16UC1", np.uint16, 1, None),
    ("32FC1", np.float32, 1, None),
    ("8UC3", np.uint8, 3, "R:G:B"),
    ("16UC3", np.uint16, 3, "R:G:B"),
]


@pytest.mark.parametrize("encoding,dtype,planes,order", ENCODINGS)
def test_raw_image_encodings(ros1, encoding, dtype, planes, order):
    rng = np.random.default_rng(1)
    shape = (H, W, planes) if planes > 1 else (H, W)
    # values above 255, since Image() narrows a 16-bit array that fits in 8 bits
    scale = 60000 if dtype == np.uint16 else 200
    array = (rng.random(shape) * scale).astype(dtype)
    m = ros1()
    m.image("/cam", encoding, array)
    (img,) = list(ROSBag(m.close(), topicfilter="/cam"))
    assert img.array.dtype == dtype
    assert img.array.shape == shape
    np.testing.assert_array_equal(img.array, array)
    assert img.colororder_str == order
    assert img.topic == "/cam"


def test_an_unsupported_encoding_is_an_error(ros1):
    m = ros1()
    m.image("/cam", "yuv422", np.zeros((H, W * 2), np.uint8))
    with pytest.raises(ValueError, match="Unsupported or unknown ROS encoding"):
        list(ROSBag(m.close()))


def test_dtype_and_colororder_can_be_set_per_topic(ros1):
    m = ros1()
    array = np.arange(H * W * 3, dtype=np.uint8).reshape(H, W, 3)
    m.image("/a", "rgb8", array)
    m.image("/b", "rgb8", array)
    path = m.close()
    images = {i.topic: i for i in ROSBag(path, colororder={"/a": "bgr", "/b": "rgb"})}
    assert images["/a"].colororder_str == "B:G:R"
    assert images["/b"].colororder_str == "R:G:B"
    # the same colororder for every topic
    images = {i.topic: i for i in ROSBag(path, colororder="bgr")}
    assert {i.colororder_str for i in images.values()} == {"B:G:R"}
    # a dtype for every topic, and per topic
    for dtype in ("uint8", {"/a": "uint8", "/b": "uint8"}):
        for i in ROSBag(path, dtype=dtype):
            assert i.array.dtype == np.uint8


# --------------------------------------------------------------------------- #
# PointCloud2
# --------------------------------------------------------------------------- #

POINTS = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 3], [1, 1, 4]], dtype=np.float32)
COLORS = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]], np.uint8)


@pytest.mark.skipif(not open3d_available, reason="open3d not installed")
@pytest.mark.parametrize("ros2", [False, True], ids=["ros1", "ros2"])
def test_pointcloud2_without_color(tmp_path, ros2):
    m = BagMaker(tmp_path / ("bag" if ros2 else "a.bag"), ros2=ros2)
    m.cloud("/points", POINTS, None)
    (pc,) = list(ROSBag(m.close(), msgfilter="PointCloud2"))
    assert len(pc) == 4
    np.testing.assert_allclose(pc.points, POINTS.T)
    assert not pc.iscolor
    assert pc.topic == "/points"


@pytest.mark.skipif(not open3d_available, reason="open3d not installed")
@pytest.mark.parametrize("ros2", [False, True], ids=["ros1", "ros2"])
def test_pointcloud2_with_color(tmp_path, ros2):
    m = BagMaker(tmp_path / ("bag" if ros2 else "a.bag"), ros2=ros2)
    m.cloud("/points", POINTS, COLORS)
    (pc,) = list(ROSBag(m.close(), msgfilter="PointCloud2"))
    assert pc.iscolor
    np.testing.assert_allclose(pc.colors, COLORS.T / 255.0)


def test_images_and_clouds_are_filtered_by_message_type(ros1):
    m = ros1()
    m.image("/cam", "mono8", np.zeros((H, W), np.uint8))
    if open3d_available:
        m.cloud("/points", POINTS, None)
    path = m.close()
    assert len(list(ROSBag(path))) == 1  # the default filter is "Image"
    assert len(list(ROSBag(path, msgfilter=None))) == (2 if open3d_available else 1)


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #


def two_topic_bag(m):
    for _ in range(3):
        m.image("/cam", "mono8", np.zeros((H, W), np.uint8))
    for _ in range(2):
        m.image("/depth", "16UC1", np.zeros((H, W), np.uint16))
    return m.close()


def test_traffic_counts_messages_by_type(ros1):
    bag = ROSBag(two_topic_bag(ros1()))
    counts = bag.traffic(progress=False)
    assert counts["sensor_msgs/msg/Image"] == 5


def test_topics_lists_topic_and_type(ros1):
    bag = ROSBag(two_topic_bag(ros1()))
    assert bag.topics() == {
        "/cam": "sensor_msgs/msg/Image",
        "/depth": "sensor_msgs/msg/Image",
    }


def test_print_shows_topics_counts_and_whether_they_pass_the_filter(ros1):
    path = two_topic_bag(ros1())
    out = io.StringIO()
    ROSBag(path, topicfilter="/cam").print(progress=False, file=out)
    text = out.getvalue()
    assert "5 messages" in text
    assert "/cam" in text and "/depth" in text
    assert "sensor_msgs/msg/Image" in text
    assert "allowed" in text
    assert "✓" in text and "✗" in text  # /cam passes, /depth does not

    out = io.StringIO()
    ROSBag(path).print(progress=False, file=out, show_allowed=False)
    assert "allowed" not in out.getvalue()


def test_verbose_open_lists_the_connections(ros1, capsys):
    path = two_topic_bag(ros1())
    list(ROSBag(path, topicfilter="/cam", verbose=True))
    out = capsys.readouterr().out
    assert "2 total connections, 1 pass filter" in out
    assert "'/cam'" in out


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #


def test_a_missing_bag_is_an_error(tmp_path):
    with pytest.raises(Exception):
        list(ROSBag(str(tmp_path / "nothing.bag")))
