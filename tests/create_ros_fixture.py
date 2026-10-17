#!/usr/bin/env python
"""
Create synthetic ROS bag fixtures for the MVTB test suite.

Usage::

    $ python tests/create_ros_fixture.py

Creates:

    tests/data/test_ros1.bag             – ROS 1 Noetic bag
    tests/data/test_ros1_realsense.bag   – ROS 1 bag with RealSense SDK footers
    tests/data/test_ros2/                – ROS 2 Humble bag directory

Contents (both ``test_ros1`` and ``test_ros2`` carry the same messages):

    /camera/compressed   sensor_msgs/CompressedImage   3 frames (JPEG, rgb)
    /camera/raw          sensor_msgs/Image             3 frames (rgb8 encoding)
    /imu/data            sensor_msgs/Imu               3 messages

The image data are tiny (8 × 6 px) so the files stay small.  The IMU topic
provides a non-image message for testing that the ``msgfilter`` correctly
accepts or rejects non-image types and that the generic pass-through path
attaches ``timestamp`` and ``topic`` attributes.

``test_ros1_realsense.bag`` mimics a recording made by the Intel RealSense
SDK, which appends a 4-byte ``float32`` footer (the depth scale) to every
image message:

    /camera/depth/image_rect_raw   sensor_msgs/Image   3 frames (16UC1)
    /camera/color/image_raw        sensor_msgs/Image   3 frames (rgb8)
    /device_0/stream_info          realsense_msgs/StreamInfo   1 message

The ``realsense_msgs`` connection is what identifies the bag as a RealSense
recording.  ``test_ros1_realsense_notrailer.bag`` holds the same topics without
the footer, matching bags such as Intel's ``single_depth_color_640x480.bag``
where the messages are self-contained.

Timestamps begin at 2023-11-14 22:13:20 UTC and step by 0.1 s.
"""

from __future__ import annotations

import io
import shutil
import struct
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from rosbags.rosbag1 import Writer as Writer1
from rosbags.rosbag2 import Writer as Writer2
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

HERE = Path(__file__).parent
DATADIR = HERE / "data"
ROS1_BAG = DATADIR / "test_ros1.bag"
ROS1_REALSENSE_BAG = DATADIR / "test_ros1_realsense.bag"
ROS1_REALSENSE_NOTRAILER_BAG = DATADIR / "test_ros1_realsense_notrailer.bag"
ROS2_BAG = DATADIR / "test_ros2"

N = 3  # messages per topic
STAMP0_NS = 1_700_000_000_000_000_000  # 2023-11-14 22:13:20 UTC
STEP_NS = 100_000_000  # 0.1 s between messages
W, H = 8, 6  # image width, height (pixels)

# Intel RealSense default depth scale, metres per depth unit
DEPTH_UNITS = 0.001
COLOR_UNITS = 0.0

# Minimal realsense_msgs/StreamInfo definition: enough for the bag reader to
# recognise a RealSense recording without a RealSense installation.
STREAMINFO_MSGDEF = """\
std_msgs/Header header
string name
uint32 width
uint32 height
uint32 encoding
uint32 fps
uint32 bytes_per_pixel
"""


def _jpeg_bytes(frame_idx: int) -> bytes:
    """Return JPEG bytes for a small colour test image."""
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    arr[:, :, 0] = frame_idx * 40  # vary red channel per frame
    arr[:, :, 1] = 128
    arr[:, :, 2] = 200
    buf = io.BytesIO()
    PILImage.fromarray(arr, "RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _rgb_bytes(frame_idx: int) -> bytes:
    """Return raw rgb8 pixel bytes for a small test image."""
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    arr[:, :, 0] = frame_idx * 40
    arr[:, :, 1] = 100
    arr[:, :, 2] = 50
    return arr.tobytes()


def _depth_bytes(frame_idx: int) -> bytes:
    """Return raw 16UC1 pixel bytes for a small test depth image."""
    arr = np.full((H, W), 1000 + frame_idx * 500, dtype=np.uint16)
    arr[:, 0] = 900
    return arr.tobytes()


def _footer(depth_units: float) -> bytes:
    """Return the 4-byte RealSense footer that follows an image message."""
    return struct.pack("<f", depth_units)


def _make_ros1_realsense(path: Path, trailer: bool = True) -> None:
    """Write a ROS 1 bag that mimics a recording made by the RealSense SDK.

    The bag always carries a ``realsense_msgs/StreamInfo`` connection, which is
    how the bag reader identifies RealSense recordings.  When ``trailer`` is
    true every image message is followed by the 4-byte little-endian
    ``float32`` footer holding the depth scale, as written by the SDK.  When it
    is false the messages are self-contained, matching bag files that contain
    RealSense topics but no trailer.
    """
    ts = get_typestore(Stores.ROS1_NOETIC)
    ts.register(
        get_types_from_msg(STREAMINFO_MSGDEF, "realsense_msgs/msg/StreamInfo")
    )

    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    Image = ts.types["sensor_msgs/msg/Image"]
    StreamInfo = ts.types["realsense_msgs/msg/StreamInfo"]

    path.unlink(missing_ok=True)
    with Writer1(path) as writer:
        conn_depth = writer.add_connection(
            "/camera/depth/image_rect_raw",
            "sensor_msgs/msg/Image",
            typestore=ts,
        )
        conn_rgb = writer.add_connection(
            "/camera/color/image_raw",
            "sensor_msgs/msg/Image",
            typestore=ts,
        )
        conn_info = writer.add_connection(
            "/device_0/stream_info",
            "realsense_msgs/msg/StreamInfo",
            typestore=ts,
        )

        for i in range(N):
            t_ns = STAMP0_NS + i * STEP_NS
            sec = t_ns // 1_000_000_000
            nanosec = t_ns % 1_000_000_000
            stamp = Time(sec=sec, nanosec=nanosec)
            header = Header(seq=i, stamp=stamp, frame_id="camera")

            # Depth frame: 16-bit units, footer carries the depth scale
            depth = Image(
                header=header,
                height=H,
                width=W,
                encoding="16UC1",
                is_bigendian=0,
                step=W * 2,
                data=np.frombuffer(_depth_bytes(i), dtype=np.uint8),
            )
            payload = bytes(ts.serialize_ros1(depth, "sensor_msgs/msg/Image"))
            if trailer:
                payload += _footer(DEPTH_UNITS)
            writer.write(conn_depth, t_ns, payload)

            # Colour frame: footer is present but zero, as recorded by the SDK
            rgb = Image(
                header=header,
                height=H,
                width=W,
                encoding="rgb8",
                is_bigendian=0,
                step=W * 3,
                data=np.frombuffer(_rgb_bytes(i), dtype=np.uint8),
            )
            payload = bytes(ts.serialize_ros1(rgb, "sensor_msgs/msg/Image"))
            if trailer:
                payload += _footer(COLOR_UNITS)
            writer.write(conn_rgb, t_ns, payload)

        info = StreamInfo(
            header=Header(
                seq=0,
                stamp=Time(sec=STAMP0_NS // 1_000_000_000, nanosec=0),
                frame_id="camera",
            ),
            name="Depth",
            width=W,
            height=H,
            encoding=0,
            fps=30,
            bytes_per_pixel=2,
        )
        writer.write(
            conn_info,
            STAMP0_NS,
            ts.serialize_ros1(info, "realsense_msgs/msg/StreamInfo"),
        )


def _make_ros1() -> None:
    ts = get_typestore(Stores.ROS1_NOETIC)

    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    Image = ts.types["sensor_msgs/msg/Image"]
    CompressedImage = ts.types["sensor_msgs/msg/CompressedImage"]
    Imu = ts.types["sensor_msgs/msg/Imu"]
    Vec3 = ts.types["geometry_msgs/msg/Vector3"]
    Quat = ts.types["geometry_msgs/msg/Quaternion"]

    ROS1_BAG.unlink(missing_ok=True)
    with Writer1(ROS1_BAG) as writer:
        conn_ci = writer.add_connection(
            "/camera/compressed",
            "sensor_msgs/msg/CompressedImage",
            typestore=ts,
        )
        conn_raw = writer.add_connection(
            "/camera/raw",
            "sensor_msgs/msg/Image",
            typestore=ts,
        )
        conn_imu = writer.add_connection(
            "/imu/data",
            "sensor_msgs/msg/Imu",
            typestore=ts,
        )

        for i in range(N):
            t_ns = STAMP0_NS + i * STEP_NS
            sec = t_ns // 1_000_000_000
            nanosec = t_ns % 1_000_000_000
            stamp = Time(sec=sec, nanosec=nanosec)
            header = Header(seq=i, stamp=stamp, frame_id="camera")

            # CompressedImage
            ci = CompressedImage(
                header=header,
                format="jpeg",
                data=np.frombuffer(_jpeg_bytes(i), dtype=np.uint8),
            )
            writer.write(
                conn_ci,
                t_ns,
                ts.serialize_ros1(ci, "sensor_msgs/msg/CompressedImage"),
            )

            # Raw Image (rgb8)
            raw = Image(
                header=header,
                height=H,
                width=W,
                encoding="rgb8",
                is_bigendian=0,
                step=W * 3,
                data=np.frombuffer(_rgb_bytes(i), dtype=np.uint8),
            )
            writer.write(
                conn_raw,
                t_ns,
                ts.serialize_ros1(raw, "sensor_msgs/msg/Image"),
            )

            # IMU
            zero9 = np.zeros(9, dtype=np.float64)
            imu = Imu(
                header=header,
                orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
                orientation_covariance=zero9,
                angular_velocity=Vec3(x=0.0, y=0.0, z=float(i) * 0.1),
                angular_velocity_covariance=zero9,
                linear_acceleration=Vec3(x=0.0, y=0.0, z=9.81),
                linear_acceleration_covariance=zero9,
            )
            writer.write(
                conn_imu,
                t_ns,
                ts.serialize_ros1(imu, "sensor_msgs/msg/Imu"),
            )


def _make_ros2() -> None:
    ts = get_typestore(Stores.ROS2_HUMBLE)

    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    Image = ts.types["sensor_msgs/msg/Image"]
    CompressedImage = ts.types["sensor_msgs/msg/CompressedImage"]
    Imu = ts.types["sensor_msgs/msg/Imu"]
    Vec3 = ts.types["geometry_msgs/msg/Vector3"]
    Quat = ts.types["geometry_msgs/msg/Quaternion"]

    if ROS2_BAG.exists():
        shutil.rmtree(ROS2_BAG)
    with Writer2(ROS2_BAG, version=9) as writer:
        conn_ci = writer.add_connection(
            "/camera/compressed",
            "sensor_msgs/msg/CompressedImage",
            typestore=ts,
        )
        conn_raw = writer.add_connection(
            "/camera/raw",
            "sensor_msgs/msg/Image",
            typestore=ts,
        )
        conn_imu = writer.add_connection(
            "/imu/data",
            "sensor_msgs/msg/Imu",
            typestore=ts,
        )

        for i in range(N):
            t_ns = STAMP0_NS + i * STEP_NS
            sec = t_ns // 1_000_000_000
            nanosec = t_ns % 1_000_000_000
            stamp = Time(sec=sec, nanosec=nanosec)
            # ROS 2 Header has no seq field
            header = Header(stamp=stamp, frame_id="camera")

            ci = CompressedImage(
                header=header,
                format="jpeg",
                data=np.frombuffer(_jpeg_bytes(i), dtype=np.uint8),
            )
            writer.write(
                conn_ci,
                t_ns,
                ts.serialize_cdr(ci, "sensor_msgs/msg/CompressedImage"),
            )

            raw = Image(
                header=header,
                height=H,
                width=W,
                encoding="rgb8",
                is_bigendian=0,
                step=W * 3,
                data=np.frombuffer(_rgb_bytes(i), dtype=np.uint8),
            )
            writer.write(
                conn_raw,
                t_ns,
                ts.serialize_cdr(raw, "sensor_msgs/msg/Image"),
            )

            zero9 = np.zeros(9, dtype=np.float64)
            imu = Imu(
                header=header,
                orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
                orientation_covariance=zero9,
                angular_velocity=Vec3(x=0.0, y=0.0, z=float(i) * 0.1),
                angular_velocity_covariance=zero9,
                linear_acceleration=Vec3(x=0.0, y=0.0, z=9.81),
                linear_acceleration_covariance=zero9,
            )
            writer.write(
                conn_imu,
                t_ns,
                ts.serialize_cdr(imu, "sensor_msgs/msg/Imu"),
            )


if __name__ == "__main__":
    DATADIR.mkdir(exist_ok=True)
    print(f"Writing ROS1 bag  → {ROS1_BAG}")
    _make_ros1()
    print(f"Writing ROS1 RealSense bag  → {ROS1_REALSENSE_BAG}")
    _make_ros1_realsense(ROS1_REALSENSE_BAG)
    print(f"Writing ROS1 RealSense bag (no trailer)  → {ROS1_REALSENSE_NOTRAILER_BAG}")
    _make_ros1_realsense(ROS1_REALSENSE_NOTRAILER_BAG, trailer=False)
    print(f"Writing ROS2 bag  → {ROS2_BAG}/")
    _make_ros2()
    print("Done.")
