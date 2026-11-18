"""Video-file initialisation failures and capture lifetime."""

from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

import machinevisiontoolbox.Sources as sources
from machinevisiontoolbox import VideoFile


@pytest.mark.parametrize("contents", [b"", b"not a video"])
def test_invalid_video_file(tmp_path: Path, contents: bytes) -> None:
    filename = tmp_path / "invalid.avi"
    filename.write_bytes(contents)

    with pytest.raises(ValueError, match="Cannot open video file") as error:
        VideoFile(str(filename))

    assert repr(str(filename)) in str(error.value)


@pytest.mark.parametrize(
    "opened, readable, message",
    [
        (False, False, "Cannot open video file"),
        (True, False, "Cannot read the first frame"),
        (True, True, "Cannot read the first frame"),
    ],
)
def test_failed_video_probe_releases_capture(
    tmp_path: Path, opened: bool, readable: bool, message: str
) -> None:
    filename = tmp_path / "invalid.avi"
    filename.touch()
    capture = Mock(spec=cv2.VideoCapture)
    capture.isOpened.return_value = opened
    capture.read.return_value = (readable, None)
    capture.get.return_value = 0

    with patch.object(sources.cv2, "VideoCapture", return_value=capture):
        with pytest.raises(ValueError, match=message) as error:
            VideoFile(str(filename))

    assert repr(str(filename)) in str(error.value)
    capture.release.assert_called_once_with()
    capture.get.assert_not_called()
    if not opened:
        capture.read.assert_not_called()


def test_valid_video_probe_preserves_iteration(tmp_path: Path) -> None:
    filename = tmp_path / "valid.avi"
    writer = cv2.VideoWriter(
        str(filename), cv2.VideoWriter_fourcc(*"MJPG"), 10, (16, 16)
    )
    if not writer.isOpened():
        writer.release()
        pytest.skip("MJPG video writer unavailable")
    try:
        for intensity in (40, 200):
            writer.write(np.full((16, 16, 3), intensity, dtype=np.uint8))
    finally:
        writer.release()

    with VideoFile(str(filename)) as video:
        assert video.shape == (16, 16, 3)
        assert video.nframes == 2
        assert video.fps == 10
        frames = list(video)

    assert [frame.id for frame in frames] == [0, 1]
    np.testing.assert_allclose(frames[0].array, 40, atol=2)
    np.testing.assert_allclose(frames[1].array, 200, atol=2)
