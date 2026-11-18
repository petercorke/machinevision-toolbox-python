"""ROSBag and bags recorded with the Intel RealSense SDK.

The SDK appends a 4-byte ``float32`` depth scale to every image message in the bag.
The reader used to guess how many trailing bytes to remove.  It now finds the end of
each image from its layout, so the pixels are exact whether or not there is a footer,
and the footer is kept as ``depth_units``.

The bags are synthetic and written into a temporary directory by
``tests/create_ros_fixture.py``.  Tests that use real bags from the ``bags/`` folder of
a development checkout are skipped when they are not present.
"""

from __future__ import annotations

import io
import itertools
import struct
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rosbags")

from rosbags.typesys import Stores, get_typestore

from create_ros_fixture import (
    RS_COLOR_TOPIC,
    RS_DEPTH_TOPIC,
    RS_DEPTH_UNITS,
    make_realsense_bag,
    realsense_color_frame,
    realsense_depth_frame,
)
from machinevisiontoolbox import ROSBag
from machinevisiontoolbox.Sources import (
    _LENGTH_ERRORS,
    _footer_depth_units,
    _image_payload_end,
)

HERE = Path(__file__).parent
REAL_BAGS = HERE.parent / "bags"


def read_all(path: Path, release: str) -> list:
    with redirect_stdout(io.StringIO()):  # auto-detection prints what it found
        return list(ROSBag(str(path), release=release))


def read_with_warnings(path: Path, release: str) -> tuple[list, list]:
    """All the images in a bag and the RuntimeWarnings issued while reading them."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        images = read_all(path, release)
    return images, [w for w in caught if issubclass(w.category, RuntimeWarning)]


# ---------------------------------------------------------------------------
# finding the end of the pixel data from the layout of the message
# ---------------------------------------------------------------------------

WIRES = ["ros1", "cdr_le", "cdr_be"]


def serialize_image(
    wire: str,
    msgtype: str = "sensor_msgs/msg/Image",
    frame_id: str = "",
    encoding: str = "mono16",
    width: int = 5,
    height: int = 3,
    data_len: int | None = None,
) -> bytes:
    ros1 = wire == "ros1"
    ts = get_typestore(Stores.ROS1_NOETIC if ros1 else Stores.ROS2_HUMBLE)
    Time = ts.types["builtin_interfaces/msg/Time"]
    Header = ts.types["std_msgs/msg/Header"]
    stamp = Time(sec=1, nanosec=2)
    header = (
        Header(seq=9, stamp=stamp, frame_id=frame_id)
        if ros1
        else Header(stamp=stamp, frame_id=frame_id)
    )
    if msgtype.endswith("CompressedImage"):
        msg = ts.types[msgtype](
            header=header,
            format="jpeg",
            data=np.arange(11, dtype=np.uint8),
        )
    else:
        step = width * {"mono16": 2, "rgb8": 3, "mono8": 1}[encoding]
        msg = ts.types[msgtype](
            header=header,
            height=height,
            width=width,
            encoding=encoding,
            is_bigendian=0,
            step=step,
            # normally height * step bytes, a different length is not a valid Image
            data=np.zeros(height * step if data_len is None else data_len, np.uint8),
        )
    if ros1:
        return bytes(ts.serialize_ros1(msg, msgtype))
    return bytes(ts.serialize_cdr(msg, msgtype, little_endian=wire == "cdr_le"))


@pytest.mark.parametrize("wire", WIRES)
@pytest.mark.parametrize("msgtype", ["sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"])
@pytest.mark.parametrize("frame_id", ["", "c", "cam", "camera_1"])  # alignment
@pytest.mark.parametrize("encoding", ["mono16", "rgb8", "mono8"])
@pytest.mark.parametrize("extra", [b"", b"\x00" * 4, b"\x6f\x12\x83\x3a", b"\x01" * 8])
def test_payload_end_ignores_trailing_bytes(wire, msgtype, frame_id, encoding, extra):
    raw = serialize_image(wire, msgtype, frame_id, encoding)
    end = _image_payload_end(raw + extra, msgtype, cdr=wire != "ros1")
    assert end == len(raw)


@pytest.mark.parametrize("wire", WIRES)
def test_payload_end_rejects_what_is_not_an_image(wire):
    cdr = wire != "ros1"
    msgtype = "sensor_msgs/msg/Image"
    raw = serialize_image(wire)
    assert _image_payload_end(b"", msgtype, cdr) is None
    assert _image_payload_end(b"\x00" * 3, msgtype, cdr) is None
    assert _image_payload_end(raw[:-1], msgtype, cdr) is None  # data array cut short
    assert _image_payload_end(raw[:20], msgtype, cdr) is None  # inside the header
    # the data array is not height * step bytes long, so this is not a valid Image
    for data_len in (3 * 10 - 1, 3 * 10 + 1, 0):
        assert _image_payload_end(serialize_image(wire, data_len=data_len), msgtype, cdr) is None
    assert _image_payload_end(bytes(range(200)), msgtype, cdr) is None


# ---------------------------------------------------------------------------
# the footer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ["<", ">"])
def test_footer_is_a_positive_finite_float32(order):
    assert _footer_depth_units(struct.pack(order + "f", 0.001), order) == pytest.approx(
        0.001, rel=1e-6
    )
    for bad in (0.0, -0.001, float("nan"), float("inf")):
        assert _footer_depth_units(struct.pack(order + "f", bad), order) is None


@pytest.mark.parametrize("footer", [b"", b"\x00", b"\x6f\x12\x83", b"\x6f\x12\x83\x3a\x00"])
def test_footer_must_be_four_bytes(footer):
    assert _footer_depth_units(footer) is None


# ---------------------------------------------------------------------------
# bags
# ---------------------------------------------------------------------------

CASES = [
    # (wire, footer, streaminfo, release)
    *itertools.product(["ros1"], [True, False], [True, False], ["ROS1_NOETIC", "auto"]),
    # a CDR message in a ROS 1 bag can only be found by auto-detection
    *itertools.product(["cdr_le", "cdr_be"], [True, False], [True, False], ["auto"]),
]


@pytest.mark.parametrize("wire, footer, streaminfo, release", CASES)
def test_realsense_style_bag_is_read_exactly(tmp_path, wire, footer, streaminfo, release):
    path = tmp_path / "rs.bag"
    make_realsense_bag(path, footer=footer, wire=wire, streaminfo=streaminfo, n=3)

    images, caught = read_with_warnings(path, release)

    depth = [i for i in images if i.topic == RS_DEPTH_TOPIC]
    color = [i for i in images if i.topic == RS_COLOR_TOPIC]
    assert (len(depth), len(color)) == (3, 3)  # nothing silently dropped
    for k, im in enumerate(depth):
        np.testing.assert_array_equal(im.array, realsense_depth_frame(k))
    for k, im in enumerate(color):
        np.testing.assert_array_equal(im.array, realsense_color_frame(k))

    # 4 bytes are only taken to be a depth scale in a bag recognised as RealSense
    for im in depth:
        if footer and streaminfo:
            assert im.depth_units == pytest.approx(RS_DEPTH_UNITS, rel=1e-6)
        else:
            assert not hasattr(im, "depth_units")
    for im in color:  # the color stream's footer is 0.0, which is not a scale
        assert not hasattr(im, "depth_units")

    # bytes that are not understood are reported, once for each topic
    if footer and not streaminfo:
        assert len(caught) == 2
        assert all("not recognised" in str(w.message) for w in caught)
    else:
        assert not caught


@pytest.mark.parametrize(
    "footer",
    [
        b"\x01" * 8,  # the wrong length
        b"\x00\x00\x80\xbf",  # -1.0
        b"\x00\x00\xc0\x7f",  # NaN
        b"\x00\x00\x80\x7f",  # infinity
        b"\x6f\x12\x83",  # 3 bytes
    ],
)
def test_unexpected_bytes_after_an_image_are_ignored_and_reported(tmp_path, footer):
    path = tmp_path / "rs.bag"
    make_realsense_bag(path, footer=footer, streaminfo=True, n=3)

    images, caught = read_with_warnings(path, "auto")

    assert len(images) == 6  # the images are still read, exactly
    depth = [i for i in images if i.topic == RS_DEPTH_TOPIC]
    for k, im in enumerate(depth):
        np.testing.assert_array_equal(im.array, realsense_depth_frame(k))
    assert not any(hasattr(i, "depth_units") for i in images)
    assert len(caught) == 2  # one for each of the two topics, not one per image
    assert all("not a RealSense depth scale" in str(w.message) for w in caught)


@pytest.mark.parametrize("padding", [b"\x00", b"\x00\x00", b"\x00\x00\x00"])
def test_cdr_alignment_padding_after_an_image_is_silent(tmp_path, padding):
    path = tmp_path / "rs.bag"
    make_realsense_bag(path, footer=padding, wire="cdr_le", streaminfo=True, n=3)

    images, caught = read_with_warnings(path, "auto")

    assert len(images) == 6 and not caught
    assert not any(hasattr(i, "depth_units") for i in images)


def test_realsense_bag_without_footer_is_not_truncated(tmp_path):
    """Intel's sample bag has realsense_msgs connections but no footers.

    The reader used to see realsense_msgs, assume a 4-byte footer, remove 4 real bytes
    of every image and, with ``release="auto"``, return no images at all.
    """
    path = tmp_path / "sample.bag"
    make_realsense_bag(path, footer=False, streaminfo=True)
    images = read_all(path, "auto")
    assert len(images) == 6
    np.testing.assert_array_equal(images[-1].array.shape[:2], (3, 5))


def test_ordinary_bag_has_no_depth_units():
    ros1 = HERE / "data" / "test_ros1.bag"
    if not ros1.exists():
        pytest.skip("fixture bag not generated")
    for im in read_all(ros1, "ROS1_NOETIC"):
        assert not hasattr(im, "depth_units")


# ---------------------------------------------------------------------------
# messages that are not images must be exactly the length their type says
# ---------------------------------------------------------------------------


def _float32_message(tmp_path):
    path = tmp_path / "rs.bag"
    make_realsense_bag(path, footer=False)
    bag = ROSBag(str(path), release="ROS1_NOETIC")
    ts = bag.typestore
    msgtype = "std_msgs/msg/Float32"
    raw = bytes(ts.serialize_ros1(ts.types[msgtype](data=1.5), msgtype))
    return bag, SimpleNamespace(msgtype=msgtype, id=7), raw


def test_non_image_message_is_decoded_at_its_exact_length(tmp_path):
    bag, conn, raw = _float32_message(tmp_path)
    msg, rest = bag._decode(conn, raw)
    assert msg.data == 1.5
    assert rest == b""


@pytest.mark.parametrize("extra", [b"\x00" * 4, b"\x00" * 8])
def test_non_image_message_with_extra_bytes_is_an_error(tmp_path, extra):
    """Nothing is guessed: a message of the wrong length is not decoded."""
    bag, conn, raw = _float32_message(tmp_path)
    with pytest.raises(_LENGTH_ERRORS):
        bag._decode(conn, raw + extra)


# ---------------------------------------------------------------------------
# real bags, only in a development checkout (bags/ is not in the repository)
# ---------------------------------------------------------------------------


def _real(name: str) -> Path:
    path = REAL_BAGS / name
    if not path.exists():
        pytest.skip(f"real bag {name} not available")
    return path


@pytest.mark.parametrize("release", ["ROS1_NOETIC", "auto"])
def test_real_realsense_bag(release):
    path = _real("20260430_125558.bag")
    with redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # a real bag raises no warnings
        images = list(itertools.islice(ROSBag(str(path), release=release), 10))
    assert len(images) == 10
    depth = [i for i in images if "Depth" in i.topic]
    color = [i for i in images if "Color" in i.topic]
    assert depth and color
    assert depth[0].shape == (720, 1280) and depth[0].dtype == np.uint16
    assert color[0].shape == (720, 1280, 3)
    assert all(i.depth_units == pytest.approx(0.001, rel=1e-6) for i in depth)
    assert not any(hasattr(i, "depth_units") for i in color)


@pytest.mark.parametrize("release", ["ROS1_NOETIC", "auto"])
def test_real_intel_sample_bag(release):
    path = _real("single_depth_color_640x480.bag")
    images = read_all(path, release)
    assert len(images) == 3  # 2 depth, 1 color
    assert sorted(i.shape for i in images) == [(480, 640), (480, 640), (480, 640, 3)]
    assert not any(hasattr(i, "depth_units") for i in images)


@pytest.mark.parametrize("release", ["humble", "auto"])
def test_real_ros2_mcap_bag(release):
    """A ROS 2 recording (rosbag2, MCAP, CDR): no footers, so nothing is reported."""
    path = _real("ros2_bouncing_ball")
    with redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        images = list(ROSBag(str(path), release=release))
    # The bag's metadata claims 132 + 133 images, but its files only hold 39 + 40.
    assert len(images) == 79
    shapes = {i.topic: i.shape for i in images}
    assert shapes == {"/image": (200, 320, 3), "/modified_image": (200, 320)}
    assert not any(hasattr(i, "depth_units") for i in images)
