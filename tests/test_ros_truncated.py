#!/usr/bin/env python
"""ROSBag on a message that ends before its type says it should, as at the end of a
recording that was cut short.  The bags are synthetic, written into a temporary
directory."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("rosbags")

from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

from machinevisiontoolbox import ROSBag

H, W = 4, 6


def write_bag(path, n_good: int = 1, n_truncated: int = 2) -> str:
    """Good mono8 images on /cam, then truncated ones on the same topic."""
    ts = get_typestore(Stores.ROS1_NOETIC)
    T = ts.types
    msgtype = "sensor_msgs/msg/Image"
    with Writer(path) as w:
        conn = w.add_connection("/cam", msgtype, typestore=ts)
        for i in range(n_good + n_truncated):
            header = T["std_msgs/msg/Header"](
                seq=i,
                stamp=T["builtin_interfaces/msg/Time"](sec=10 + i, nanosec=0),
                frame_id="cam",
            )
            msg = T[msgtype](
                header=header,
                height=H,
                width=W,
                encoding="mono8",
                is_bigendian=0,
                step=W,
                data=np.full(H * W, i, np.uint8),
            )
            raw = ts.serialize_ros1(msg, msgtype)
            if i >= n_good:
                raw = raw[:12]  # ends inside the header
            w.write(conn, (10 + i) * 1_000_000_000, raw)
    return str(path)


def test_a_truncated_message_is_skipped_with_one_warning_per_topic(tmp_path):
    path = write_bag(tmp_path / "a.bag", n_good=2, n_truncated=3)
    with pytest.warns(RuntimeWarning, match="truncated") as caught:
        images = list(ROSBag(path, release="noetic"))
    assert len(images) == 2  # the good ones are kept
    assert [int(i.array[0, 0]) for i in images] == [0, 1]
    assert len(caught) == 1  # not one for each truncated message
    text = str(caught[0].message)
    assert "/cam" in text and "sensor_msgs/msg/Image" in text
    assert "verbose" in text


def test_verbose_lists_each_truncated_message_and_does_not_warn(tmp_path, capsys):
    path = write_bag(tmp_path / "a.bag", n_good=1, n_truncated=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        images = list(ROSBag(path, release="noetic", verbose=True))
    assert len(images) == 1
    out = capsys.readouterr().out
    assert out.count("Truncated sensor_msgs/msg/Image on '/cam', skipped") == 2


def test_a_bag_with_no_truncated_message_does_not_warn(tmp_path):
    path = write_bag(tmp_path / "a.bag", n_good=3, n_truncated=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(list(ROSBag(path, release="noetic"))) == 3
