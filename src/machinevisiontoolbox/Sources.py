"""
Video, camera, image-collection, and ZIP-archive sources for streaming images.
"""

from __future__ import annotations

import fnmatch
import io
import json
import base64
import pathlib
import shutil
import subprocess
import sys
import tarfile
import threading
from dataclasses import dataclass
from http import client
import os
import tempfile
import time
import urllib.request
import warnings
import zipfile
from collections import Counter, deque
from collections.abc import Iterator
from numpy.char import array
from tqdm import tqdm
from datetime import datetime, timezone
from typing import TYPE_CHECKING, IO, Any, Literal, cast

if TYPE_CHECKING:
    import torch
    from numpy.typing import DTypeLike

# from numpy.lib.arraysetops import isin
from abc import ABC, abstractmethod

import cv2
import numpy as np
from ansitable import ANSITable, Column
from spatialmath import Polygon2

from machinevisiontoolbox.ImageCore import Image
from machinevisiontoolbox.PointCloud import PointCloud
from machinevisiontoolbox.base import mvtb_path_to_datafile
from machinevisiontoolbox.base.imageio import convert, iread, iread_iter

try:
    from rosbags.rosbag1 import Reader as _RosBagReader1
    from rosbags.rosbag2 import Reader as _RosBagReader2
    from rosbags.typesys import Stores as _Stores, get_typestore as _get_typestore

    _rosbags_available = True
except ImportError:
    _RosBagReader1 = None
    _RosBagReader2 = None
    _Stores = None
    _get_typestore = None
    _rosbags_available = False

try:
    import roslibpy

    _roslibpy_available = True
except ImportError:
    roslibpy = None
    _roslibpy_available = False

try:
    import open3d as o3d

    _open3d_available = True
except ImportError:
    o3d = None
    _open3d_available = False

try:
    import py7zr as _py7zr

    _py7zr_available = True
except ImportError:
    _py7zr = None
    _py7zr_available = False

try:
    import rarfile as _rarfile

    _rarfile_available = True
except ImportError:
    _rarfile = None
    _rarfile_available = False

_unar_available: bool = (
    shutil.which("unar") is not None and shutil.which("lsar") is not None
)


def _wants_mono(kwargs: dict[str, Any]) -> bool:
    """Return True if convert kwargs request a single-plane output image."""
    return bool(kwargs.get("mono") or kwargs.get("grey") or kwargs.get("gray"))


def _make_image(*args: Any, **kwargs: Any) -> Image:
    """Construct an Image instance while keeping static type checking practical."""
    return Image(*args, **kwargs)  # pyright: ignore[reportAbstractUsage]


def _set_sample_metadata(sample: Any, timestamp: int, topic: str) -> None:
    """Attach dynamic timestamp/topic metadata used by stream sources."""
    setattr(sample, "timestamp", timestamp)
    setattr(sample, "topic", topic)


class ImageSource(ABC):
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Base constructor for image sources.

        :param kwargs: source-specific keyword arguments
        :type kwargs: Any
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of images in this source."""
        raise NotImplementedError(f"{type(self).__name__} does not support len()")

    @abstractmethod
    def __iter__(self) -> Iterator:
        """Return an iterator over images in this source."""
        raise NotImplementedError(f"{type(self).__name__} does not support iteration")

    def tensor(
        self,
        device: str = "cpu",
        normalize="imagenet",
        dtype: "torch.dtype | None" = None,
    ) -> "torch.Tensor":
        """
        Convert all images from this source into a single 4D PyTorch tensor.

        :param device: target PyTorch device, e.g. ``"cpu"``, ``"cuda"`` or ``"mps"``,
            defaults to ``"cpu"``
        :type device: str, optional
        :param normalize: normalisation to apply to each frame; passed directly
            to :meth:`Image.tensor`, defaults to ``"imagenet"``
        :type normalize: str, tuple, or None, optional
        :param dtype: output tensor dtype, for example ``torch.float32``;
            passed directly to :meth:`Image.tensor`, defaults to None
        :type dtype: torch.dtype or None, optional
        :raises ImportError: if PyTorch is not installed
        :raises TypeError: if the source is not finite (no ``__len__``)
        :raises TypeError: if any yielded item is not an :class:`~machinevisiontoolbox.Image`
        :raises ValueError: if any frame has a different shape to the first
        :return: tensor of shape ``(N, C, H, W)``
        :rtype: torch.Tensor

        Images are decoded one at a time and written directly into a
        pre-allocated tensor, so peak memory is one decoded frame plus the
        output tensor — the full source need not reside in memory at once.
        This is particularly useful for :class:`VideoFile` and
        :class:`FileArchive`.

        Example:

            .. code-block:: pycon

            >>> from machinevisiontoolbox import VideoFile
            >>> t = VideoFile("traffic_sequence.mpg").tensor(normalize=None)
            >>> t.shape
            torch.Size([N, 3, H, W])

        """
        try:
            import torch as _torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for tensor(). "
                "Install it with: pip install torch "
                "or pip install machinevision-toolbox-python[torch]"
            )

        if not hasattr(self, "__len__"):
            raise TypeError(
                f"{type(self).__name__} is not a finite source; "
                "tensor() requires a source with a known length"
            )

        n = len(self)
        it = iter(self)

        # decode the first frame to learn shape and dtype
        first = next(it)
        if not isinstance(first, Image):
            raise TypeError(
                f"Expected Image, got {type(first).__name__}; "
                "use a msgfilter or topicfilter to select only image topics"
            )
        first_t = first.tensor(device=device, normalize=normalize, dtype=dtype).squeeze(
            0
        )  # (C, H, W)
        expected_shape = first_t.shape

        # pre-allocate the full tensor
        out = _torch.empty((n,) + expected_shape, dtype=first_t.dtype, device=device)
        out[0] = first_t

        for i in range(1, n):
            img = next(it)
            if not isinstance(img, Image):
                raise TypeError(
                    f"Frame {i}: expected Image, got {type(img).__name__}; "
                    "use a msgfilter or topicfilter to select only image topics"
                )
            t = img.tensor(device=device, normalize=normalize, dtype=dtype).squeeze(0)
            if t.shape != expected_shape:
                raise ValueError(
                    f"Frame {i} shape {tuple(t.shape)} differs from "
                    f"frame 0 shape {tuple(expected_shape)}"
                )
            out[i] = t

        return out

    def disp(
        self,
        animate: bool = False,
        fps: float | None = None,
        title: str | None = None,
        loop: bool = False,
        **kwargs,
    ) -> None:
        """
        Display images from the source interactively.

        :param animate: if ``True``, play as timed animation; if ``False``
            (default), step through one frame at a time
        :type animate: bool, optional
        :param fps: playback rate in frames per second, used when
            ``animate=True``, defaults to None.  If not None ``animate`` mode is assumed.
        :type fps: float, optional
        :param title: window title; defaults to the ``topic`` attribute of each
            frame when present
        :type title: str or None, optional
        :param loop: restart when the source is exhausted (finite sources only),
            defaults to ``False``
        :type loop: bool, optional
        :param kwargs: additional keyword arguments passed to
            :meth:`Image.disp` for each frame, e.g. ``grid=True``

        Display images from the source.  For example::

            VideoFile("traffic_sequence.mpg").disp(fps=10)

        will display an animation of the video at 10 frames per second.

        By default, the display is

        **Keys — step-through mode** (``animate=False``):

        - ``[space]`` — next frame
        - ``[1-9]`` / ``[0]`` — jump 1–9 or 10 frames (finite sources only)
        - ``[l]`` / ``[c]`` / ``[d]`` — jump 50 / 100 / 500 frames (finite sources only)
        - ``[q]`` / ``[x]`` — quit

        **Keys — animate mode** (``animate=True``):

        - ``[space]`` — pause / resume
        - ``[+]`` / ``[=]`` — increase playback speed
        - ``[-]`` / ``[_]`` — decrease playback speed
        - ``[q]`` / ``[x]`` — quit

        .. note:: Jump keys and ``loop`` are only available for finite sources
            that implement ``__len__``.  For live streams such as
            :class:`VideoCamera` or :class:`ROSTopic` those controls are
            disabled automatically.

        :seealso: :class:`ImageSequence`
        """
        import matplotlib.pyplot as plt

        _finite = hasattr(self, "__len__")

        _jump = {
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "0": 10,
            "l": 50,
            "c": 100,
            "d": 500,
        }

        def _label(item, i):
            ts = getattr(item, "timestamp", None)
            if ts is not None:
                dt = datetime.fromtimestamp(ts / 1e9, tz=timezone.utc).astimezone()
                return dt.isoformat(timespec="milliseconds") + f" [{i}]"
            return f"[{i}]"

        def _make_overlay(fig):
            return fig.text(
                0,
                0,
                "",
                backgroundcolor="black",
                color="white",
                horizontalalignment="left",
                verticalalignment="bottom",
            )

        def _win_title(item):
            return title or getattr(item, "topic", None) or ""

        fig = None
        ax = None
        ts_text = None
        if fps is not None:
            animate = True

        if animate:
            state = {"fps": fps, "paused": False, "quit": False}

            def on_key(event, s=state):
                if event.key in ("q", "x"):
                    s["quit"] = True
                elif event.key == " ":
                    s["paused"] = not s["paused"]
                elif event.key in ("+", "="):
                    s["fps"] = s["fps"] * 1.5
                elif event.key in ("-", "_"):
                    s["fps"] = max(1.0, s["fps"] / 1.5)

            it = iter(self)
            i = 0
            while True:
                try:
                    x = next(it)
                except StopIteration:
                    if loop and _finite:
                        it = iter(self)
                        i = 0
                        continue
                    break
                if state["quit"]:
                    break
                while state["paused"] and not state["quit"]:
                    plt.pause(0.05)
                if state["quit"]:
                    break
                if fig is None:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    fig.canvas.mpl_connect("key_press_event", on_key)
                    ts_text = _make_overlay(fig)
                    print(
                        "\nKeys: [space] pause/resume  [+] faster  [-] slower  [q/x] quit"
                    )
                x.disp(
                    fps=state["fps"], reuse=True, title=_win_title(x), ax=ax, **kwargs
                )
                assert ts_text is not None
                ts_text.set_text(_label(x, i))
                i += 1

            if fig is not None:
                if not state["quit"]:
                    plt.show(block=True)
                plt.close(fig)

        else:
            view_state = {"next": False, "quit": False, "skip": 0}

            def on_key(event, s=view_state):
                if event.key == " ":
                    s["next"] = True
                elif event.key in ("q", "x"):
                    s["quit"] = True
                elif _finite and event.key in _jump:
                    s["skip"] = _jump[event.key]
                    s["next"] = True

            it = iter(self)
            i = 0
            while True:
                try:
                    x = next(it)
                except StopIteration:
                    break
                if view_state["quit"]:
                    break
                if fig is None:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ts_text = _make_overlay(fig)
                    fig.canvas.mpl_connect("key_press_event", on_key)
                    if _finite:
                        print(
                            "\nKeys: [space] next  [1-9] jump N frames  [0] jump 10"
                            "  [l/c/d] jump 50/100/500  [q/x] quit"
                        )
                    else:
                        print("\nKeys: [space] next frame  [q/x] quit")
                x.disp(title=_win_title(x), ax=ax, reuse=True, **kwargs)
                assert ts_text is not None
                ts_text.set_text(_label(x, i))
                view_state["next"] = False
                while not view_state["next"] and not view_state["quit"]:
                    plt.pause(0.05)
                skip = view_state["skip"]
                view_state["skip"] = 0
                for _ in range(skip):
                    try:
                        next(it)
                        i += 1
                    except StopIteration:
                        break
                i += 1

            if fig is not None:
                plt.close(fig)


class VideoFile(ImageSource):
    """
    Iterate images from a video file

    :param filename: Path to video file
    :type filename: str
    :param kwargs: options applied to image frames, see :func:`~machinevisiontoolbox.base.imageio.convert`

    The resulting object is an iterator over the frames of the video file.
    The iterator returns :class:`~machinevisiontoolbox.Image` objects where:

    - the ``name`` attribute is the name of the video file
    - the ``id`` attribute is the frame number within the file

    If the path is not absolute, the video file is first searched for
    relative to the current directory, and if not found, it is searched for
    in the ``images`` folder of the ``mvtb-data`` package, installed as a
    Toolbox dependency.

    Example:

        .. code-block:: python

            from machinevisiontoolbox import VideoFile
            video = VideoFile("traffic_sequence.mpg")
            len(video)
            for im in video:
                pass


    or using a context manager to ensure the file handle is always released:

        .. code-block:: python

            with VideoFile("traffic_sequence.mpg") as video:
                for im in video:
                    pass


    :references:
        - |RVC3|, Section 11.1.4.

    :seealso: :func:`~machinevisiontoolbox.base.imageio.convert`
        `opencv.VideoCapture <https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80>`_
    """

    filename: str
    nframes: int
    shape: tuple
    fps: int
    args: dict
    cap: cv2.VideoCapture | None
    i: int

    def __init__(self, filename: str, **kwargs: Any) -> None:

        self.filename = str(mvtb_path_to_datafile("images", filename))

        # get the number of frames in the video
        #  not sure it's always correct
        cap = cv2.VideoCapture(self.filename)
        ret, frame = cap.read()
        self.nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.shape = frame.shape
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.args = kwargs
        cap.release()
        self.cap = None
        self.i = 0

    def __iter__(self) -> VideoFile:
        self.i = 0
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.filename)
        return self

    def __next__(self) -> Image:
        if self.cap is None:
            self.__iter__()
        assert self.cap is not None
        ret, frame = self.cap.read()
        if ret is False:
            self.cap.release()
            raise StopIteration
        else:
            opts: dict[str, Any] = {"rgb": True}
            opts.update(self.args)
            if frame.ndim == 3 and not _wants_mono(opts):
                im = _make_image(
                    frame, id=self.i, name=self.filename, colororder="RGB", **opts
                )
            else:
                im = _make_image(frame, id=self.i, name=self.filename, **opts)
            self.i += 1
            return im

    def __len__(self) -> int:
        return self.nframes

    def __str__(self) -> str:
        return f"VideoFile({os.path.basename(self.filename)}) {self.shape[1]} x {self.shape[0]}, {self.nframes} frames @ {self.fps}fps"

    def __repr__(self) -> str:
        return f"VideoFile(file={os.path.basename(self.filename)}, size=({self.shape[1]}, {self.shape[0]}), nframes={self.nframes}, fps={self.fps})"

    def __enter__(self) -> VideoFile:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VideoCamera(ImageSource):
    """
    Iterate images from a local video camera

    :param id: Identity of local camera
    :type id: int
    :param kwargs: options applied to image frames, see :func:`~machinevisiontoolbox.base.imageio.convert`

    Connect to a local video camera.  For some cameras this will cause
    the recording light to come on.

    The resulting object is an iterator over the frames from the video
    camera. The iterator returns :class:`~machinevisiontoolbox.Image` objects.

    Example:

        .. code-block:: python

            from machinevisiontoolbox import VideoCamera
            video = VideoCamera(0)
            for im in video:
                pass


    alternatively:

        .. code-block:: python

            img = next(video)


    or using a context manager to ensure the camera is released:

        .. code-block:: python

            with VideoCamera(0) as camera:
                for im in camera:
                    pass


    .. note::

        The value of ``id`` is system specific but generally 0 is the first
        attached video camera.  On a Mac running 13.0 (Ventura) or later and
        an iPhone with iOS 16 or later, the Continuity Camera feature allows
        the phone camera to be used as a local video camera, and it will
        appear as a separate camera with its own ID.

        OpenCV does not expose a portable API for mapping integer ``id``
        values to human-readable camera names.  Use :meth:`list` to
        enumerate the cameras that are present on the current machine
        together with their best-available names.

    :references:
        - |RVC3|, Section 11.1.3.

    :seealso: :meth:`list` :func:`~machinevisiontoolbox.base.imageio.convert`
        `cv2.VideoCapture <https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80>`_
    """

    id: int
    cap: cv2.VideoCapture
    args: dict
    rgb: bool
    i: int

    def __init__(self, id: int = 0, rgb: bool = True, **kwargs: Any) -> None:

        self.id = id
        self.cap = cv2.VideoCapture(id)
        self.args = kwargs
        self.rgb = rgb
        self.i = 0

    def __iter__(self) -> VideoCamera:
        self.i = 0
        self.cap.release()
        self.cap = cv2.VideoCapture(self.id)
        return self

    def __next__(self) -> Image:
        ret, frame = self.cap.read()  # frame will be in BGR order
        if ret is False:
            self.cap.release()
            raise StopIteration
        else:
            opts: dict[str, Any] = {"rgb": self.rgb, "copy": True}
            opts.update(self.args)
            if self.rgb:
                if frame.ndim == 3 and not _wants_mono(opts):
                    img = _make_image(frame, id=self.i, colororder="RGB", **opts)
                else:
                    img = _make_image(frame, id=self.i, **opts)
            else:
                if frame.ndim == 3 and not _wants_mono(opts):
                    img = _make_image(frame, id=self.i, colororder="BGR", **opts)
                else:
                    img = _make_image(frame, id=self.i, **opts)

            self.i += 1
            return img

    def grab(self) -> Image:
        """
        Grab single frame from camera

        :return: next frame from the camera
        :rtype: :class:`~machinevisiontoolbox.Image`

        .. deprecated:: 0.11.4
            Use :func:`next` on the iterator instead, for example ``next(camera)``.
        """
        warnings.warn(
            "VideoCamera.grab() is deprecated; use next(camera) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return next(self)

    def release(self) -> None:
        """
        Release the camera

        Disconnect from the local camera, and for cameras with a recording
        light, turn off that light.
        """
        self.cap.release()

    def __str__(self) -> str:
        backend = self.cap.getBackendName()
        return f"VideoCamera({self.id}) {self.width} x {self.height} @ {self.framerate}fps using {backend}"

    def __repr__(self) -> str:
        backend = self.cap.getBackendName()
        return f"VideoCamera(id={self.id}, size=({self.width}, {self.height}), fps={self.framerate}, backend={backend})"

    # see https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    properties: dict[str, int] = {
        "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "contrast": cv2.CAP_PROP_CONTRAST,
        "saturation": cv2.CAP_PROP_SATURATION,
        "hue": cv2.CAP_PROP_HUE,
        "gain": cv2.CAP_PROP_GAIN,
        "exposure": cv2.CAP_PROP_EXPOSURE,
        "auto-exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
        "gamma": cv2.CAP_PROP_GAMMA,
        "temperature": cv2.CAP_PROP_TEMPERATURE,
        "auto-whitebalance": cv2.CAP_PROP_AUTO_WB,
        "whitebalance-temperature": cv2.CAP_PROP_WB_TEMPERATURE,
        "ios:exposure": cv2.CAP_PROP_IOS_DEVICE_EXPOSURE,
        "ios:whitebalance": cv2.CAP_PROP_IOS_DEVICE_WHITEBALANCE,
    }

    def get(self, property: str | None = None) -> float | dict[str, float]:
        """
        Get camera property

        :param prop: camera property name
        :type prop: str
        :return: parameter value
        :rtype: float

        Get value for the specified property. Value 0 is returned when querying a property that is not supported by the backend used by the VideoCapture instance.

        ==============================  =========================================================
        Property                        description
        ==============================  =========================================================
        ``"brightness"``                image brightness (offset)
        ``"contrast"``                  contrast of the image
        ``"saturation"``                saturation of the image
        ``"hue"``                       hue of the image
        ``"gain"``                      gain of the image
        ``"exposure"``                  exposure of image
        ``"auto-exposure"``             exposure control by camera
        ``"gamma"``                     gamma of image
        ``"temperature"``               color temperature
        ``"auto-whitebalance"``         enable/ disable auto white-balance
        ``"whitebalance-temperature"``  white-balance color temperature
        ``"ios:exposure"``              exposure of image for Apple AVFOUNDATION backend
        ``"ios:whitebalance"``          white balance of image for Apple AVFOUNDATION backend
        ==============================  =========================================================


        :seealso: :meth:`set`
        """
        if property is not None:
            return self.cap.get(self.properties[property])
        else:
            return {
                property: self.cap.get(self.properties[property])
                for property in self.properties
            }

    def set(self, property: str, value: float) -> float:
        """
        Set camera property

        :param prop: camera property name
        :type prop: str
        :param value: new property value
        :type value: float
        :return: parameter value
        :rtype: float

        Set new value for the specified property. Value 0 is returned when querying a property that is not supported by the backend used by the VideoCapture instance.

        ==============================  =========================================================
        Property                        description
        ==============================  =========================================================
        ``"brightness"``                image brightness (offset)
        ``"contrast"``                  contrast of the image
        ``"saturation"``                saturation of the image
        ``"hue"``                       hue of the image
        ``"gain"``                      gain of the image
        ``"exposure"``                  exposure of image
        ``"auto-exposure"``             exposure control by camera
        ``"gamma"``                     gamma of image
        ``"temperature"``               color temperature
        ``"auto-whitebalance"``         enable/ disable auto white-balance
        ``"whitebalance-temperature"``  white-balance color temperature
        ``"ios:exposure"``              exposure of image for Apple AVFOUNDATION backend
        ``"ios:whitebalance"``          white balance of image for Apple AVFOUNDATION backend
        ==============================  =========================================================

        :seealso: :meth:`get`
        """
        return self.cap.set(self.properties[property], value)

    @property
    def width(self) -> int:
        """
        Width of video frame

        :return: width of video frame in pixels
        :rtype: int

        :seealso: :meth:`height` :meth:`shape`
        """
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """
        Height of video frame

        :return: height of video frame in pixels
        :rtype: int

        :seealso: :meth:`width` :meth:`shape`
        """
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def framerate(self) -> int:
        """
        Camera frame rate

        :return: camera frame rate in frames per second
        :rtype: int

        .. note:: If frame rate cannot be determined return -1
        """
        try:
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        except ValueError:
            fps = -1
        return fps

    @property
    def shape(self) -> tuple[int, int]:
        """
        Shape of video frame

        :return: height and width of video frame in pixels
        :rtype: int, int

        :seealso: :meth:`height` :meth:`width`
        """
        return (self.height, self.width)

    @classmethod
    def list(cls) -> list[dict]:
        """
        Enumerate available local video cameras

        :return: list of dicts, one per camera, each with keys ``id``,
            ``width``, ``height``, ``fps``, and ``name`` (best-effort).
            On macOS, ``width``, ``height``, and ``fps`` are ``None`` if the
            process has not yet been granted camera authorisation.
        :rtype: list[dict]

        Probes integer indices 0, 1, 2, … until no camera responds, and
        returns a list describing each one.  Camera names are obtained by
        a platform-specific method:

        - **macOS** — ``system_profiler SPCameraDataType -json`` (no camera
          authorisation required)
        - **Linux** — ``/sys/class/video4linux/videoN/name`` sysfs entry
        - **Windows** — not available; ``name`` is ``None``

        The name mapping is best-effort: if the platform query fails the
        ``name`` key is ``None``.

        Example::

            $ python
            >>> from machinevisiontoolbox import VideoCamera
            >>> for cam in VideoCamera.list():
            ...     print(cam)
            {'id': 0, 'width': 1280, 'height': 720, 'fps': 30, 'name': 'FaceTime HD Camera'}
            {'id': 1, 'width': 1920, 'height': 1080, 'fps': 30, 'name': 'iPhone Camera'}

        :seealso: :class:`VideoCamera`
        """

        def _query_macos() -> list[str]:
            """Return ordered list of camera names from system_profiler JSON."""
            try:
                out = subprocess.check_output(
                    ["system_profiler", "SPCameraDataType", "-json"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                data = json.loads(out)
                return [item["_name"] for item in data.get("SPCameraDataType", [])]
            except (
                FileNotFoundError,
                subprocess.CalledProcessError,
                KeyError,
                json.JSONDecodeError,
            ):
                return []

        def _names_linux() -> dict[int, str]:
            """Read /sys/class/video4linux/videoN/name on Linux."""
            result = {}
            sysfs = pathlib.Path("/sys/class/video4linux")
            if not sysfs.exists():
                return result
            for entry in sorted(sysfs.iterdir()):
                name_file = entry / "name"
                try:
                    idx = int(entry.name.replace("video", ""))
                    result[idx] = name_file.read_text().strip()
                except (ValueError, OSError):
                    pass
            return result

        cameras = []
        platform = sys.platform

        if platform == "darwin":
            # Enumerate from system_profiler — works even without camera
            # authorisation.  OpenCV properties are best-effort: if the
            # process hasn't been granted camera access yet, width/height/fps
            # will be None for those cameras.
            names = _query_macos()
            for idx, name in enumerate(names):
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    entry = {
                        "id": idx,
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
                        "name": name,
                    }
                    cap.release()
                else:
                    cap.release()
                    entry = {
                        "id": idx,
                        "width": None,
                        "height": None,
                        "fps": None,
                        "name": name,
                    }
                cameras.append(entry)
        else:
            name_map = _names_linux() if platform.startswith("linux") else {}
            for idx in range(32):  # practical upper bound
                cap = cv2.VideoCapture(idx)
                if not cap.isOpened():
                    cap.release()
                    break
                entry = {
                    "id": idx,
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": int(cap.get(cv2.CAP_PROP_FPS)),
                    "name": name_map.get(idx),
                }
                cap.release()
                cameras.append(entry)

        return cameras

    def __enter__(self) -> VideoCamera:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class FileCollection(ImageSource):
    """
    Iterate images from a collection of files

    :param filename: wildcard path to image files
    :type filename: str
    :param loop: Endlessly loop over the files, defaults to False
    :type loop: bool, optional
    :param kwargs: options applied to image frames, see :func:`~machinevisiontoolbox.base.imageio.convert`

    The resulting object is an iterator over the image files that match the
    wildcard description. The iterator returns :class:`~machinevisiontoolbox.Image` objects where
    the ``name`` attribute is the name of the image file.

    **Eager mode** (default): All images are decoded at construction time.
    This makes random access (``files[i]``), slicing, and ``len(files)`` fast
    but can be slow and memory-intensive for large collections.

    .. code-block:: python

        from machinevisiontoolbox import FileCollection
        images = FileCollection('campus/*.png')
        len(images)                      # fast
        img = images[5]                  # fast, in-memory
        for image in images:             # iterate in-memory
            pass

    **Lazy mode** (via context manager): Files are decoded on-demand during
    iteration. This avoids the startup cost of decoding all images upfront.
    Trade-off: random access (``__getitem__``) is not available; use only for
    sequential iteration.

    .. code-block:: python

        from machinevisiontoolbox import FileCollection
        with FileCollection('campus/*.png') as images:
            for image in images:         # decode on-demand
                pass

    If the path is not absolute, the file is first searched for
    relative to the current directory, and if not found, it is searched for
    in the ``images`` folder of the ``mvtb-data`` package, installed as a
    Toolbox dependency.

    Example:

        .. code-block:: python

            from machinevisiontoolbox import FileCollection
            images = FileCollection('campus/*.png')
            len(images)
            for image in images:  # iterate over images
                pass


    alternatively:

        .. code-block:: python

            img = files[i]  # load i'th file from the collection


    or using a context manager for memory-efficient streaming:

        .. code-block:: python

            with FileCollection('campus/*.png') as images:
                for image in images:
                    pass


    :references:
        - |RVC3|, Section 11.1.2.

    :seealso: :func:`~machinevisiontoolbox.base.imageio.convert`
        `cv2.imread <https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56>`_
    """

    images: list
    names: list
    args: dict
    loop: bool
    i: int
    _lazy: bool
    _filename: str | None
    _lazy_iter: Iterator | None
    _loaded: bool

    def __init__(
        self, filename: str | None = None, loop: bool = False, **kwargs: Any
    ) -> None:

        self.images = []
        self.names = []
        self._lazy = False
        self._filename = filename
        self.args = kwargs
        self.loop = loop
        self.i = 0
        self._lazy_iter = None
        self._loaded = False

    def _load_eager(self) -> None:
        """Load and decode all images eagerly."""
        if self._loaded or self._filename is None:
            return
        images, names = iread(self._filename, rgb=True)
        if isinstance(images, np.ndarray):
            self.images = [images]
            self.names = [names]
        else:
            self.images = images
            self.names = cast(list[str], names)
        self._loaded = True

    def __getitem__(self, i: int | slice) -> FileCollection | Image:

        if self._lazy:
            raise TypeError(
                "Random access is not supported in lazy mode. "
                "Iterate sequentially or exit context manager to use eager mode."
            )

        self._load_eager()

        if isinstance(i, slice):
            # slice of a collection -> FileCollection
            new = self.__class__()
            new.images = self.images[i]
            new.names = self.names[i]
            new.args = self.args
            return new
        else:
            # element of a collection -> Image
            data = self.images[i]
            if data.ndim == 3 and not _wants_mono(self.args):
                return _make_image(
                    data, name=self.names[i], id=i, colororder="RGB", **self.args
                )
            else:
                return _make_image(data, id=i, name=self.names[i], **self.args)

    def __iter__(self) -> FileCollection:
        self.i = 0
        if self._lazy:
            self._lazy_iter = None
        else:
            # Ensure eager load before iterating in non-CM mode
            self._load_eager()
        return self

    def __str__(self) -> str:
        if self._lazy:
            return f"FileCollection(lazy mode)"
        return "\n".join([str(f) for f in self.names])

    def __repr__(self) -> str:
        if self._lazy:
            return f"FileCollection(lazy mode)"
        return f"FileCollection(nimages={len(self.images)})"

    def __next__(self) -> Image:
        if self._lazy:
            if self._filename is None:
                raise StopIteration

            # Lazily create the on-demand image iterator
            if self._lazy_iter is None:
                self._lazy_iter = iread_iter(self._filename, rgb=True)

            try:
                if self.loop:
                    try:
                        data, name = next(self._lazy_iter)
                    except StopIteration:
                        self._lazy_iter = iread_iter(self._filename, rgb=True)
                        data, name = next(self._lazy_iter)
                else:
                    data, name = next(self._lazy_iter)
            except StopIteration:
                raise StopIteration

            if data.ndim == 3 and not _wants_mono(self.args):
                im = _make_image(
                    data, id=self.i, name=name, colororder="RGB", **self.args
                )
            else:
                im = _make_image(data, id=self.i, name=name, **self.args)
            self.i += 1
            return im
        else:
            # Eager iteration: use pre-loaded images
            if self.i >= len(self.names):
                if self.loop:
                    self.i = 0
                else:
                    raise StopIteration
            data = self.images[self.i]
            if data.ndim == 3 and not _wants_mono(self.args):
                im = _make_image(
                    data,
                    id=self.i,
                    name=self.names[self.i],
                    colororder="RGB",
                    **self.args,
                )
            else:
                im = _make_image(data, id=self.i, name=self.names[self.i], **self.args)
            self.i += 1
            return im

    def __len__(self) -> int:
        if self._lazy:
            raise TypeError(
                "__len__() is not reliable in lazy mode. "
                "Exit context manager to use eager mode with known length."
            )
        self._load_eager()
        return len(self.images)

    def __enter__(self) -> FileCollection:
        # Switch to lazy mode - don't load anything yet, wait for __iter__
        self._lazy = True
        self.i = 0
        self._lazy_iter = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Exit lazy mode
        self._lazy = False
        self._lazy_iter = None


class ImageSequence(ImageSource):
    """
    An in-memory sequence of :class:`~machinevisiontoolbox.Image` objects with interactive display.

    :param images: sequence of images
    :type images: iterable of :class:`~machinevisiontoolbox.Image`

    The sequence is materialised into a list on construction.  Items are
    expected to be :class:`~machinevisiontoolbox.Image` instances; a ``timestamp`` attribute (ROS
    nanosecond epoch) and ``topic`` attribute are used for the display overlay
    when present.

    Example:

        .. code-block:: python

            from machinevisiontoolbox import ROSBag, ImageSequence
            bag = ROSBag("mybag.bag", msgfilter="Image")
            seq = ImageSequence(bag)
            seq.disp()                        # step through one frame at a time
            seq.disp(animate=True, fps=5)     # timed playback


    :seealso: :class:`PointCloudSequence`
    """

    _frames: list
    _i: int

    def __init__(self, images: Any) -> None:
        self._frames = list(images)
        self._i = 0

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, i: int | slice) -> Image | list[Image]:
        return self._frames[i]

    def __iter__(self) -> ImageSequence:
        self._i = 0
        return self

    def __next__(self) -> Image:
        if self._i >= len(self._frames):
            raise StopIteration
        frame = self._frames[self._i]
        self._i += 1
        return frame

    def __enter__(self) -> ImageSequence:
        return self

    def __exit__(self, *_):
        pass

    def __repr__(self) -> str:
        return f"ImageSequence(nimages={len(self._frames)})"


class _ZipAdapter:
    """Zip archive backend for :class:`FileArchive`."""

    def __init__(self, filename: pathlib.Path) -> None:
        self._zf = zipfile.ZipFile(filename, "r")

    def namelist(self) -> list[str]:
        return self._zf.namelist()

    def read(self, name: str) -> bytes:
        return self._zf.read(name)

    def open(self, name: str) -> IO[bytes]:
        return self._zf.open(name)

    def close(self) -> None:
        self._zf.close()


class _TarAdapter:
    """Tar archive backend (plain, .gz, .bz2, .xz) for :class:`FileArchive`."""

    def __init__(self, filename: pathlib.Path) -> None:
        self._tf = tarfile.open(str(filename), "r:*")

    def namelist(self) -> list[str]:
        return [m.name for m in self._tf.getmembers() if m.isfile()]

    def read(self, name: str) -> bytes:
        f = self._tf.extractfile(name)
        if f is None:
            return b""
        return f.read()

    def open(self, name: str) -> IO[bytes]:
        f = self._tf.extractfile(name)
        if f is None:
            raise KeyError(f"{name!r} is not a regular file in this archive")
        return f

    def close(self) -> None:
        self._tf.close()


class _SevenZAdapter:
    """7-Zip archive backend for :class:`FileArchive` (requires ``py7zr``)."""

    def __init__(self, filename: pathlib.Path) -> None:
        if not _py7zr_available:
            raise ImportError(
                "py7zr is required to read 7-Zip archives. "
                "Install it with: pip install py7zr"
            )
        assert _py7zr is not None
        self._filename = str(filename)
        with _py7zr.SevenZipFile(self._filename, mode="r") as a:
            self._names = [f.filename for f in a.list() if not f.is_directory]

    def namelist(self) -> list[str]:
        return self._names

    def read(self, name: str) -> bytes:
        assert _py7zr is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            with _py7zr.SevenZipFile(self._filename, mode="r") as a:
                a.extract(path=tmpdir, targets=[name])

            target = pathlib.Path(tmpdir) / name
            if target.exists():
                return target.read_bytes()

            # Some archives may store member names with different separators.
            matches = list(pathlib.Path(tmpdir).rglob(pathlib.Path(name).name))
            return matches[0].read_bytes() if matches else b""

    def open(self, name: str) -> IO[bytes]:
        return io.BytesIO(self.read(name))

    def close(self) -> None:
        pass  # no persistent handle


class _UnarAdapter:
    """Shell-out adapter using ``unar``/``lsar`` (The Unarchiver) for :class:`FileArchive`.

    Does not require the ``rarfile`` Python package; needs only the ``unar`` and
    ``lsar`` command-line tools in PATH (``brew install unar`` on macOS).
    """

    def __init__(self, filename: pathlib.Path) -> None:
        unar = shutil.which("unar")
        lsar = shutil.which("lsar")
        if unar is None or lsar is None:
            raise ImportError(
                "unar and lsar are required to read this archive but were not found.\n"
                "Install with: brew install unar  (macOS)\n"
                "          or: sudo apt install unar  (Debian/Ubuntu)"
            )
        self._unar = unar
        self._lsar = lsar
        self._filename = str(filename)
        result = subprocess.run(
            [self._lsar, "-j", self._filename],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        self._names = [
            e["XADFileName"]
            for e in data.get("lsarContents", [])
            if not e.get("XADIsDirectory", False) and not e["XADFileName"].endswith("/")
        ]

    def namelist(self) -> list[str]:
        return self._names

    def read(self, name: str) -> bytes:
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [self._unar, "-q", "-f", "-o", tmpdir, self._filename, name],
                check=True,
                capture_output=True,
            )
            # unar recreates the internal directory structure inside tmpdir
            target = pathlib.Path(tmpdir) / name
            if target.exists():
                return target.read_bytes()
            # fall back: search by basename (handles path separator differences)
            matches = list(pathlib.Path(tmpdir).rglob(pathlib.Path(name).name))
            return matches[0].read_bytes() if matches else b""

    def open(self, name: str) -> IO[bytes]:
        return io.BytesIO(self.read(name))

    def close(self) -> None:
        pass  # stateless shell-out; nothing to close


class _RarAdapter:
    """RAR archive backend for :class:`FileArchive` (requires ``rarfile`` + ``unrar``/``unar``)."""

    def __init__(self, filename: pathlib.Path) -> None:
        if not _rarfile_available:
            raise ImportError(
                "rarfile is required to read RAR archives. "
                "Install it with: pip install rarfile"
            )
        assert _rarfile is not None
        # auto-detect available extraction tool (unrar, unar, bsdtar, 7z)
        _rarfile.tool_setup()
        try:
            self._rf = _rarfile.RarFile(str(filename), "r")
        except _rarfile.RarCannotExec:
            raise ImportError(
                "No RAR extraction tool found. Install one of:\n"
                "  macOS:         brew install unar\n"
                "  Debian/Ubuntu: sudo apt install unar\n"
                "  (or)           pip install rarfile  +  brew install rar"
            )

    def namelist(self) -> list[str]:
        return [info.filename for info in self._rf.infolist() if not info.is_dir()]

    def read(self, name: str) -> bytes:
        return self._rf.read(name)

    def open(self, name: str) -> IO[bytes]:
        return cast(IO[bytes], self._rf.open(name))

    def close(self) -> None:
        self._rf.close()


class FileArchive(ImageSource):
    """
    Iterate images from a compressed archive

    :param filename: path to archive file
    :type filename: str
    :param filter: a Unix shell-style wildcard that specifies which files
        to include when iterating over the archive
    :type filter: str, optional
    :param loop: endlessly loop over the files, defaults to False
    :type loop: bool, optional
    :param kwargs: options applied to image frames, see :func:`~machinevisiontoolbox.base.imageio.convert`

    The resulting object is an iterator over the files within the archive.
    The iterator returns the file as a :class:`~machinevisiontoolbox.Image` instance if it is an
    image (the ``name`` attribute is the filename within the archive), or a
    :class:`bytes` object for non-image files.

        The following archive formats are supported:

        - ``.zip``: stdlib, always available
        - ``.tar``: stdlib, always available
        - ``.tar.gz`` / ``.tgz``: stdlib, always available
        - ``.tar.bz2``: stdlib, always available
        - ``.tar.xz``: stdlib, always available
        - ``.7z``: requires ``pip install py7zr``
        - ``.rar``: requires ``brew install unar`` (macOS) or ``apt install unar``;
            alternatively ``pip install rarfile`` with ``unrar``/``unar`` in PATH

    If the path is not absolute it is first searched for relative to the
    current directory, and if not found, it is searched for in the
    ``images`` folder of the ``mvtb-data`` package, installed as a
    Toolbox dependency.

    To read just the image files within the archive, use a ``filter`` such as
    ``"*.png"`` or ``"*.pgm"``.  Note that ``filter`` is a Unix shell style
    wildcard expression, not a Python regexp.

    Example:

        .. code-block:: python

            from machinevisiontoolbox import FileArchive
            images = FileArchive('bridge-l.zip')
            len(images)
            for image in images:  # iterate over files
                pass


    alternatively:

        .. code-block:: python

            image = images[i]  # load i'th file from the archive


    or using a context manager to ensure the archive is closed:

        .. code-block:: python

            with FileArchive('bridge-l.zip') as images:
                for image in images:
                    pass


    :references:
        - |RVC3|, Section 11.1.2.

    :seealso: :meth:`open` :func:`~machinevisiontoolbox.base.imageio.convert`
        `cv2.imread <https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56>`_
    """

    files: list[str]
    args: dict
    loop: bool
    i: int

    def __init__(
        self,
        filename: str,
        filter: str | None = None,
        loop: bool = False,
        **kwargs: Any,
    ) -> None:

        path = pathlib.Path(mvtb_path_to_datafile("images", filename))
        suffixes = path.suffixes
        suffix = path.suffix.lower()

        if suffix == ".zip":
            self._archive: (
                _ZipAdapter | _TarAdapter | _SevenZAdapter | _UnarAdapter | _RarAdapter
            ) = _ZipAdapter(path)
        elif ".tar" in [s.lower() for s in suffixes] or suffix == ".tgz":
            self._archive = _TarAdapter(path)
        elif suffix == ".7z":
            self._archive = _SevenZAdapter(path)
        elif suffix == ".rar":
            # prefer unar (shell-out, no extra pip dep); fall back to rarfile
            if _unar_available:
                self._archive = _UnarAdapter(path)
            else:
                self._archive = _RarAdapter(path)
        else:
            # unknown extension — probe zip then tar
            try:
                self._archive = _ZipAdapter(path)
            except zipfile.BadZipFile:
                try:
                    self._archive = _TarAdapter(path)
                except tarfile.TarError:
                    raise ValueError(
                        f"Unrecognised archive format for {pathlib.Path(path).name!r}. "
                        "Supported: .zip, .tar, .tar.gz, .tgz, .tar.bz2, .tar.xz, "
                        ".7z (requires py7zr), .rar (requires unar/lsar or rarfile + extractor)"
                    )

        if filter is None:
            files = [f for f in self._archive.namelist() if not f.endswith("/")]
        else:
            files = fnmatch.filter(self._archive.namelist(), filter)
        self.files = sorted(files)
        self.args = kwargs
        self.loop = loop
        self.i = 0

    def open(self, name: str) -> IO[bytes]:
        """
        Open a file from the archive

        :param name: file name
        :type name: str
        :return: read-only handle to the named file
        :rtype: file object

        Opens the specified file within the archive.  Typically the
        ``FileArchive`` instance is used as an iterator over the image files
        within, but this method can be used to access non-image data such as
        camera calibration data etc. that might also be contained within the
        archive and is excluded by the ``filter``.
        """
        return self._archive.open(name)

    def ls(self) -> None:
        """
        List all files within the archive to stdout.
        """
        for name in self._archive.namelist():
            print(name)

    def __getitem__(self, i: int) -> Image | bytes:
        im = self._read(i)
        if isinstance(im, np.ndarray):
            if im.ndim == 3 and not _wants_mono(self.args):
                return _make_image(
                    im, name=self.files[i], id=i, colororder="BGR", **self.args
                )
            else:
                return _make_image(im, id=i, name=self.files[i], **self.args)
        else:
            # not an image file, just return the contents
            return im

    def __iter__(self) -> FileArchive:
        self.i = 0
        return self

    def __str__(self) -> str:
        return "FileArchive(\n  " + "\n  ".join(self.files) + "\n)"

    def __repr__(self) -> str:
        return f"FileArchive(nfiles={len(self.files)})"

    def __next__(self) -> Image | bytes:
        if self.i >= len(self.files):
            if self.loop:
                self.i = 0
            else:
                raise StopIteration

        im = self._read(self.i)
        if isinstance(im, np.ndarray):
            if im.ndim == 3 and not _wants_mono(self.args):
                im = _make_image(
                    im,
                    id=self.i,
                    name=self.files[self.i],
                    colororder="BGR",
                    **self.args,
                )
            else:
                im = _make_image(im, id=self.i, name=self.files[self.i], **self.args)
        self.i += 1
        return im

    def __len__(self) -> int:
        return len(self.files)

    def _read(self, i: int) -> np.ndarray | bytes:
        data = self._archive.read(self.files[i])
        img = cv2.imdecode(
            np.frombuffer(data, np.uint8), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED
        )
        if img is None:
            # not an image file, just return the contents
            return data
        else:
            return img

    def __enter__(self) -> FileArchive:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._archive.close()


class ImageCollection(FileCollection):
    """Deprecated alias for :class:`FileCollection`.

    :param kwargs: forwarded to :class:`FileCollection`
    :type kwargs: Any

    .. deprecated:: 1.1.0
        Use :class:`FileCollection` instead.
    """

    def __init__(
        self, filename: str | None = None, loop: bool = False, **kwargs: Any
    ) -> None:
        warnings.warn(
            "Deprecated in 1.1.0: use FileCollection instead of ImageCollection.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(filename=filename, loop=loop, **kwargs)


class ZipArchive(FileArchive):
    """Deprecated alias for :class:`FileArchive`.

    :param kwargs: forwarded to :class:`FileArchive`
    :type kwargs: Any

    .. deprecated:: 1.1.0
        Use :class:`FileArchive` instead.
    """

    def __init__(
        self,
        filename: str,
        filter: str | None = None,
        loop: bool = False,
        **kwargs: Any,
    ) -> None:
        warnings.warn(
            "Deprecated in 1.1.0: use FileArchive instead of ZipArchive.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(filename=filename, filter=filter, loop=loop, **kwargs)


class WebCam(ImageSource):
    """
    Iterate images from an internet web camera

    :param url: URL of the camera
    :type url: str
    :param kwargs: options applied to image frames, see :func:`~machinevisiontoolbox.base.imageio.convert`

    The resulting object is an iterator over the frames returned from the
    remote camera. The iterator returns :class:`~machinevisiontoolbox.Image` objects.

    Example:

        .. code-block:: python

            from machinevisiontoolbox import WebCam
            webcam = WebCam('https://webcam.dartmouth.edu/webcam/image.jpg')
            for image in webcam:  # iterate over frames
                pass


    alternatively:

        .. code-block:: python

            img = next(webcam)  # get next frame


    or using a context manager to ensure the connection is released:

        .. code-block:: python

            with WebCam('https://webcam.dartmouth.edu/webcam/image.jpg') as webcam:
                for image in webcam:
                    pass


    .. note:: Manu webcameras accept a query string in the URL to specify
        image resolution, image format, codec and other parameters. There
        is no common standard for this, see the manufacturer's datasheet
        for details.

    :references:
        - |RVC3|, Section 11.1.5.

    :seealso: :func:`~machinevisiontoolbox.base.imageio.convert`
        `cv2.VideoCapture <https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80>`_
    """

    url: str
    args: dict
    cap: cv2.VideoCapture | None

    def __init__(self, url: str, **kwargs: Any) -> None:

        self.url = url
        self.args = kwargs
        self.cap = None

    def __iter__(self) -> WebCam:
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.url)
        return self

    def __next__(self) -> Image:
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.url)
        assert self.cap is not None
        ret, frame = self.cap.read()
        if ret is False:
            self.cap.release()
            raise StopIteration
        else:
            opts: dict[str, Any] = {"rgb": True}
            opts.update(self.args)
            if frame.ndim == 3 and not _wants_mono(opts):
                return _make_image(frame, colororder="RGB", **opts)
            else:
                return _make_image(frame, **opts)

    def grab(self) -> Image:
        """
        Grab frame from web camera

        :return: next frame from the web camera
        :rtype: :class:`~machinevisiontoolbox.Image`

        .. deprecated:: 0.11.4
            Use :func:`next` on the iterator instead, for example ``next(webcam)``.
        """
        warnings.warn(
            "WebCam.grab() is deprecated; use next(webcam) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return next(self)

    def __enter__(self) -> WebCam:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __repr__(self) -> str:
        return f"WebCam({self.url})"


# dartmouth = WebCam('https://webcam.dartmouth.edu/webcam/image.jpg')


class EarthView(ImageSource):
    """
    Iterate images from GoogleEarth

    :param key: Google API key, defaults to None
    :type key: str
    :param type: type of map (API ``maptype``): 'satellite' [default], 'map', 'roads', 'hybrid', and 'terrain'.
    :type type: str, optional
    :param zoom: map zoom, defaults to 18
    :type zoom: int, optional
    :param scale: image scale factor: 1 [default] or 2
    :type scale: int, optional
    :param shape: image size (API ``size``), defaults to (500, 500)
    :type shape: tuple, optional
    :param kwargs: options applied to image frames, see :func:`~machinevisiontoolbox.base.imageio.convert`

    The resulting object has a ``grab`` method that returns :class:`~machinevisiontoolbox.Image`
    objects for a specified position on the planet.
    ``zoom`` varies from 1 (whole world) to a maximum of 18.

    The ``type`` argument controls the type of map returned:

    ===============  ========================================================================
    ``type``         Returned image
    ===============  ========================================================================
    ``"satellite"``  satellite color image from space
    ``"roadmap"``    a standard roadmap image as normally shown on the Google Maps website
    ``"map"``        synonym for ``"roadmap"``
    ``"hybrid"``     hybrid of the satellite with overlay of roadmap image
    ``"terrain"``    specifies a physical relief map image, showing terrain and vegetation
    ``"roads"``      a binary image which is an occupancy grid, roads are free space
    ===============  ========================================================================

    Example:

        .. code-block:: python

            from machinevisiontoolbox import EarthView
            earth = EarthView()  # create an Earth viewer
            image = earth.grab(-27.475722, 153.0285, zoom=17)


    .. warning:: You must have a Google account and a valid key, backed
        by a credit card, to access this service.
        `Getting started <https://developers.google.com/maps/documentation/maps-static>`_

    .. note::
        - If the key is not passed in, a value is sought from the
            environment variable ``GOOGLE_KEY``.
        - Uses the `Google Maps Static API <https://developers.google.com/maps/documentation/maps-static/start>`_

    :references:
        - |RVC3|, Section 11.1.6.

    :seealso: :meth:`grab` :func:`~machinevisiontoolbox.base.imageio.convert`
    """

    key: str | None
    type: str
    scale: int
    zoom: int
    shape: tuple
    args: dict

    def __init__(
        self,
        key: str | None = None,
        type: str = "satellite",
        zoom: int = 18,
        scale: int = 1,
        shape: tuple = (500, 500),
        **kwargs: Any,
    ) -> None:

        if key is None:
            self.key = os.getenv("GOOGLE_KEY")
        else:
            self.key = key

        self.type = type
        self.scale = scale
        self.zoom = zoom
        self.shape = shape
        self.args = kwargs

    def grab(
        self,
        lat: float,
        lon: float,
        zoom: int | None = None,
        type: str | None = None,
        scale: int | None = None,
        shape: tuple | None = None,
        roadnames: bool = False,
        placenames: bool = False,
    ) -> Image:
        """
        Google map view as an image

        :param lat: latitude (degrees)
        :type lat: float
        :param lon: longitude (degrees)
        :type lon: float
        :param type: map type, "roadmap", "satellite" [default], "hybrid", and "terrain".
        :type type: str, optional
        :param zoom: image zoom factor, defaults to None
        :type zoom: int, optional
        :param scale: image scale factor: 1 [default] or 2
        :type scale: int, optional
        :param shape: image shape (width, height), defaults to None
        :type shape: array_like(2), optional
        :param roadnames: show roadnames, defaults to False
        :type roadnames: bool, optional
        :param placenames: show place names, defaults to False
        :type placenames: bool, optional
        :return: Google map view
        :rtype: :class:`~machinevisiontoolbox.Image`

        If parameters are not given the values provided to the constructor
        are taken as defaults.

        .. note:: The returned image may have an alpha plane.
        """
        if type is None:
            type = self.type
        if scale is None:
            scale = self.scale
        if zoom is None:
            zoom = self.zoom
        if shape is None:
            shape = self.shape

        # type is one of: satellite map hybrid terrain roadmap roads
        occggrid = False
        if type == "map":
            type = "roadmap"
        elif type == "roads":
            type = "roadmap"
            occggrid = True

        # https://developers.google.com/maps/documentation/maps-static/start#URL_Parameters

        # now read the map
        url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size={shape[0]}x{shape[1]}&scale={scale}&format=png&maptype={type}&key={self.key}&sensor=false"

        opturl = []

        if roadnames:
            opturl.append("style=feature:road|element:labels|visibility:off")
        if placenames:
            opturl.append(
                "style=feature:administrative|element:labels.text|visibility:off&style=feature:poi|visibility:off"
            )

        if occggrid:
            opturl.extend(
                [
                    "style=feature:landscape|element:geometry.fill|color:0x000000|visibility:on",
                    "style=feature:landscape|element:labels|visibility:off",
                    "style=feature:administrative|visibility:off",
                    "style=feature:road|element:geometry|color:0xffffff|visibility:on",
                    "style=feature:road|element:labels|visibility:off",
                    "style=feature:poi|element:all|visibility:off",
                    "style=feature:transit|element:all|visibility:off",
                    "style=feature:water|element:all|visibility:off",
                ]
            )

        if len(opturl) > 0:
            url += "&" + "&".join(opturl)
        image_data, _ = iread(url)
        if isinstance(image_data, list):
            image = image_data[0]
        else:
            image = image_data

        if image.shape[2] == 4:
            colororder = "RGBA"
        elif image.shape[2] == 3:
            colororder = "RGB"
        else:
            colororder = None
        if colororder is not None and not _wants_mono(self.args):
            return _make_image(image, colororder=colororder, **self.args)
        return _make_image(image, **self.args)

    def __repr__(self) -> str:
        return f"EarthView(type={self.type}, zoom={self.zoom}, scale={self.scale}, shape={tuple(self.shape)})"


# ROS PointCloud2 field datatype constants → (numpy dtype, byte size)
_PC2_DTYPE = {
    1: (np.int8, 1),
    2: (np.uint8, 1),
    3: (np.int16, 2),
    4: (np.uint16, 2),
    5: (np.int32, 4),
    6: (np.uint32, 4),
    7: (np.float32, 4),
    8: (np.float64, 8),
}


def _resolve_typestore(release: str):
    """Return a typestore for the given ROS release name.

    The *release* string is matched case-insensitively against
    :class:`~rosbags.typesys.Stores` member names, so short names such as
    ``"noetic"`` or ``"humble"`` are accepted in addition to the full enum
    name (e.g. ``"ROS1_NOETIC"``).

    :param release: release name, or substring thereof
    :type release: str
    :raises ImportError: if the ``rosbags`` package is not installed
    :raises ValueError: if the name matches no enum member, or is ambiguous
    :return: configured typestore
    """
    if not _rosbags_available:
        raise ImportError(
            "rosbags is required for ROS bag support. "
            "Install it with: pip install rosbags "
            "or pip install machinevision-toolbox-python[ros]"
        )
    assert _Stores is not None
    assert _get_typestore is not None
    key = release.upper()
    matches = [s for s in _Stores if key in s.name]
    if not matches:
        valid = [s.name for s in _Stores]
        raise ValueError(f"Unknown release {release!r}. Valid options: {valid}")
    if len(matches) > 1:
        exact = [s for s in matches if s.name == key]
        if exact:
            return _get_typestore(exact[0])
        ambiguous = [s.name for s in matches]
        raise ValueError(f"Ambiguous release {release!r}, matches: {ambiguous}")
    return _get_typestore(matches[0])


@dataclass(slots=True, frozen=True)
class ROSMessage:
    """
    Normalised ROS message sample.

    :param topic: topic name
    :type topic: str
    :param msgtype: ROS message type string
    :type msgtype: str
    :param timestamp: timestamp in nanoseconds since Unix epoch
    :type timestamp: int
    :param data: decoded message payload
    :type data: Any
    """

    topic: str
    msgtype: str
    timestamp: int
    data: Any


class ROSTopic(ImageSource):
    """
    Iterate images from a live ROS topic via a rosbridge WebSocket.

    :param topic: ROS topic name, e.g. ``"/camera/image/compressed"``
    :type topic: str
    :param message: ROS message type, defaults to
        ``"sensor_msgs/CompressedImage"``
    :type message: str, optional
    :param host: hostname or IP address of the rosbridge server, defaults to
        ``"localhost"``
    :type host: str, optional
    :param port: rosbridge WebSocket port, defaults to 9090
    :type port: int, optional
    :param subscribe: if ``True`` (default) subscribe to incoming messages;
        set ``False`` for publish-only use
    :type subscribe: bool, optional
    :param output: output mode, ``"image"`` yields :class:`~machinevisiontoolbox.Image` and
        ``"message"`` yields :class:`ROSMessage`, defaults to ``"image"``
    :type output: str, optional
    :param blocking: if ``True`` (default) :meth:`__next__` blocks until a new
        frame arrives; if ``False`` it returns the most recently received frame
        immediately (or blocks until the very first frame is available)
    :type blocking: bool, optional
    :param rgb: if ``True`` (default) return RGB images; if ``False`` return BGR
    :type rgb: bool, optional
    :param kwargs: options applied to image frames, see
        :func:`~machinevisiontoolbox.base.imageio.convert`
    :raises ImportError: if the ``roslibpy`` package is not installed

    In subscribe mode, the object is an iterator that yields :class:`~machinevisiontoolbox.Image`
    instances (``output="image"``) or :class:`ROSMessage` instances
    (``output="message"``) as they arrive from the topic.  Use it as a context
    manager to ensure the connection is always closed:

        .. code-block:: python

            with ROSTopic("/camera/image/compressed", host="192.168.1.10") as stream:
                for img in stream:
                    img.disp()


    alternatively fetch a single frame with :func:`next`:

        .. code-block:: python

            stream = ROSTopic("/camera/image/compressed")
            img = next(stream)
            stream.release()


    For publish-only use, disable subscription setup and call :meth:`publish`:

        .. code-block:: python

            pub = ROSTopic("/cmd_topic", message="std_msgs/String", subscribe=False)
            pub.publish({"data": "hello"})
            pub.release()


    **Message timing**

    Messages are timestamped at the ROS publisher (header stamp), received by
    rosbridge, then forwarded over WebSocket to this object.  The callback
    stores only the most recent decoded frame and timestamp.

    With ``blocking=True``, each :meth:`__next__` call waits until a newer
    message arrives and then returns it.

    With ``blocking=False``, :meth:`__next__` returns immediately after the
    first message has arrived, returning the current most recent message,
    which may be the same frame as a previous call if no new message has
    arrived in the meantime.

    .. note:: This class use ``roslibpy``, which uses WebSockets to connect to a running rosbridge 2.0 server.
      It does not require a local ROS environment, allowing usage from platforms other than Linux.

    """

    host: str
    topic: str
    message: str
    port: int
    args: dict

    def __init__(
        self,
        topic: str,
        message: str = "sensor_msgs/CompressedImage",
        host: str = "localhost",
        port: int = 9090,
        subscribe: bool = True,
        output: Literal["image", "message"] = "image",
        blocking: bool = True,
        rgb: bool = True,
        **kwargs: Any,
    ) -> None:

        if not _roslibpy_available:
            raise ImportError(
                "roslibpy is required for ROS streaming support. "
                "Install it with: pip install roslibpy "
                "or pip install machinevision-toolbox-python[ros]"
            )
        assert roslibpy is not None

        self.host = host
        self.topic = topic
        self.message = message
        self.port = port
        self._subscribe = subscribe
        self._output = output
        self._blocking = blocking
        self._rgb = rgb
        self.args = kwargs
        self._compressed = "Compressed" in message
        if output not in {"image", "message"}:
            raise ValueError("output must be 'image' or 'message'")
        if output == "image" and not self._is_image_message_type(message):
            raise ValueError(
                "output='image' requires sensor_msgs/Image or "
                "sensor_msgs/CompressedImage"
            )
        self._latest_frame: np.ndarray | None = None
        self._latest_message: ROSMessage | None = None
        self._latest_timestamp: int | None = None
        self._frame_event = threading.Event() if subscribe else None
        self._advertised = False
        self.i = 0

        self.client = roslibpy.Ros(host=host, port=port)
        self.client.run()
        self._topic = roslibpy.Topic(self.client, topic, message)
        if subscribe:
            self._topic.subscribe(self._process_frame)

    @staticmethod
    def _is_image_message_type(message: str) -> bool:
        return message in {
            "sensor_msgs/CompressedImage",
            "sensor_msgs/Image",
            "sensor_msgs/msg/CompressedImage",
            "sensor_msgs/msg/Image",
        }

    @staticmethod
    def _is_pointcloud_message_type(message: str) -> bool:
        return message in {
            "sensor_msgs/PointCloud2",
            "sensor_msgs/msg/PointCloud2",
        }

    @staticmethod
    def _stamp_dict(timestamp_ns: int | None = None) -> dict[str, int]:
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()
        sec, nsec = divmod(int(timestamp_ns), 1_000_000_000)
        return {"secs": int(sec), "nsecs": int(nsec)}

    @staticmethod
    def _image_from_mvtb(img: Image) -> tuple[np.ndarray, str]:
        arr = img.array
        if arr.ndim == 2:
            return np.ascontiguousarray(arr), "mono"

        if arr.ndim != 3:
            raise ValueError(f"Image array must be 2D or 3D, got shape {arr.shape}")

        if img.isrgb:
            return np.ascontiguousarray(img.rgb), "rgb"
        if img.isbgr:
            return np.ascontiguousarray(img.bgr), "bgr"

        if arr.shape[2] == 3:
            return np.ascontiguousarray(arr), "rgb"
        if arr.shape[2] == 4:
            return np.ascontiguousarray(arr), "rgba"

        raise ValueError(
            "Unsupported color image format; expected 3-plane RGB/BGR "
            "or 4-plane RGBA/BGRA"
        )

    def _image_to_ros_message(
        self, img: Image, timestamp_ns: int | None = None
    ) -> dict:
        arr, order = self._image_from_mvtb(img)
        if timestamp_ns is None:
            timestamp_ns = getattr(img, "timestamp", None)
        stamp = self._stamp_dict(timestamp_ns)

        if self._compressed:
            if arr.ndim == 2:
                bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                encoding = "mono8" if arr.dtype == np.uint8 else "mono16"
            elif arr.shape[2] == 4:
                if order == "rgba":
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                    encoding = "rgba8"
                else:
                    bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
                    encoding = "bgra8"
            elif order == "rgb":
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                encoding = "rgb8"
            else:
                bgr = arr
                encoding = "bgr8"

            ok, enc = cv2.imencode(".jpg", bgr)
            if not ok:
                raise ValueError("Failed to JPEG-compress Image for ROS publish")

            return {
                "header": {"stamp": stamp, "frame_id": self.topic},
                "format": f"jpeg; {encoding}",
                "data": base64.b64encode(enc.tobytes()).decode("ascii"),
            }

        h, w = arr.shape[:2]
        channels = 1 if arr.ndim == 2 else int(arr.shape[2])
        dtype = arr.dtype

        if channels == 1:
            if dtype == np.uint8:
                encoding = "mono8"
            elif dtype == np.uint16:
                encoding = "mono16"
            elif dtype == np.float32:
                encoding = "32FC1"
            else:
                raise ValueError(f"Unsupported greyscale dtype for ROS Image: {dtype}")
        elif channels == 3:
            if dtype != np.uint8:
                raise ValueError("ROS 3-plane images must be uint8")
            encoding = "rgb8" if order == "rgb" else "bgr8"
        elif channels == 4:
            if dtype != np.uint8:
                raise ValueError("ROS 4-plane images must be uint8")
            encoding = "rgba8" if order == "rgba" else "bgra8"
        else:
            raise ValueError(f"Unsupported channel count for ROS Image: {channels}")

        arr_contig = np.ascontiguousarray(arr)
        step = int(arr_contig.strides[0])
        return {
            "header": {"stamp": stamp, "frame_id": self.topic},
            "height": int(h),
            "width": int(w),
            "encoding": encoding,
            "is_bigendian": int(arr_contig.dtype.byteorder == ">"),
            "step": step,
            "data": base64.b64encode(arr_contig.tobytes()).decode("ascii"),
        }

    def _pointcloud_to_ros_message(
        self, pc: PointCloud, timestamp_ns: int | None = None
    ) -> dict:
        if not _open3d_available:
            raise ImportError(
                "open3d is required for PointCloud publish support. "
                "Install it with: pip install open3d-python "
                "or pip install machinevision-toolbox-python[open3d]"
            )
            assert o3d is not None

        points = np.asarray(pc._pcd.points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("PointCloud must contain Nx3 points")

        n = int(points.shape[0])
        if timestamp_ns is None:
            try:
                timestamp_ns = pc.timestamp
            except (AttributeError, ValueError):
                timestamp_ns = None
        if not isinstance(timestamp_ns, (int, np.integer)):
            timestamp_ns = None
        stamp = self._stamp_dict(
            int(timestamp_ns) if timestamp_ns is not None else None
        )

        has_color = pc._pcd.has_colors()
        if has_color:
            colors = np.asarray(pc._pcd.colors)
            if colors.shape != points.shape:
                raise ValueError("PointCloud color array must match point array shape")

            cloud = np.empty(
                n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")]
            )
            cloud["x"] = points[:, 0].astype(np.float32, copy=False)
            cloud["y"] = points[:, 1].astype(np.float32, copy=False)
            cloud["z"] = points[:, 2].astype(np.float32, copy=False)
            rgb_u8 = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
            rgb_packed = (
                (rgb_u8[:, 0].astype(np.uint32) << 16)
                | (rgb_u8[:, 1].astype(np.uint32) << 8)
                | rgb_u8[:, 2].astype(np.uint32)
            )
            cloud["rgb"] = rgb_packed.view(np.float32)
            fields = [
                {"name": "x", "offset": 0, "datatype": 7, "count": 1},
                {"name": "y", "offset": 4, "datatype": 7, "count": 1},
                {"name": "z", "offset": 8, "datatype": 7, "count": 1},
                {"name": "rgb", "offset": 12, "datatype": 7, "count": 1},
            ]
        else:
            cloud = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
            cloud["x"] = points[:, 0].astype(np.float32, copy=False)
            cloud["y"] = points[:, 1].astype(np.float32, copy=False)
            cloud["z"] = points[:, 2].astype(np.float32, copy=False)
            fields = [
                {"name": "x", "offset": 0, "datatype": 7, "count": 1},
                {"name": "y", "offset": 4, "datatype": 7, "count": 1},
                {"name": "z", "offset": 8, "datatype": 7, "count": 1},
            ]

        point_step = int(cloud.dtype.itemsize)
        data = cloud.tobytes()
        return {
            "header": {"stamp": stamp, "frame_id": self.topic},
            "height": 1,
            "width": n,
            "fields": fields,
            "is_bigendian": False,
            "point_step": point_step,
            "row_step": point_step * n,
            "data": base64.b64encode(data).decode("ascii"),
            "is_dense": bool(np.isfinite(points).all()),
        }

    def _process_frame(self, msg: dict) -> None:
        """Callback invoked by roslibpy on each incoming message."""
        stamp = msg.get("header", {}).get("stamp", {})
        sec = stamp.get("secs", stamp.get("sec", 0))
        nsec = stamp.get("nsecs", stamp.get("nanosec", 0))
        timestamp = int(sec) * 1_000_000_000 + int(nsec)
        if timestamp == 0:
            timestamp = time.time_ns()
        self._latest_timestamp = timestamp

        if self._output == "message":
            self.i += 1
            self._latest_message = ROSMessage(
                topic=self.topic,
                msgtype=self.message,
                timestamp=timestamp,
                data=msg,
            )
            if self._frame_event is not None:
                self._frame_event.set()
            return

        img_data = base64.b64decode(msg["data"])
        if not self._compressed:
            # Uncompressed sensor_msgs/Image: raw pixel data with metadata
            h = msg["height"]
            w = msg["width"]
            encoding = msg["encoding"]
            # Map ROS encoding to numpy dtype and channel count
            _enc = {
                "rgb8": (np.uint8, 3),
                "bgr8": (np.uint8, 3),
                "rgba8": (np.uint8, 4),
                "bgra8": (np.uint8, 4),
                "mono8": (np.uint8, 1),
                "mono16": (np.uint16, 1),
                "16UC1": (np.uint16, 1),
                "32FC1": (np.float32, 1),
            }
            dtype, channels = _enc.get(encoding, (np.uint8, 3))
            arr = np.frombuffer(img_data, dtype=dtype)
            frame = (
                arr.reshape((h, w, channels)) if channels > 1 else arr.reshape((h, w))
            )
            # Normalise to BGR uint8 for consistency with the compressed path
            if encoding in ("rgb8", "rgba8"):
                frame = cv2.cvtColor(
                    frame, cv2.COLOR_RGB2BGR if channels == 3 else cv2.COLOR_RGBA2BGR
                )
            elif encoding == "mono8":
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif encoding == "mono16":
                # Preserve full 16-bit depth precision.
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            # Compressed sensor_msgs/CompressedImage: JPEG or PNG bytes
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR ndarray
        if frame is None:
            return
        self.i += 1
        self._latest_frame = frame
        if self._frame_event is not None:
            self._frame_event.set()

    def __iter__(self) -> ROSTopic:
        if not self._subscribe:
            raise TypeError("ROSTopic is publish-only (subscribe=False)")
        self.i = 0
        return self

    def __len__(self) -> int:
        """Return length of stream.

        Live ROS topics are unbounded streams, so they do not have a finite
        length.
        """
        raise TypeError("ROSTopic is a live stream and has no finite length")

    def __next__(self) -> Image | ROSMessage:
        if not self._subscribe or self._frame_event is None:
            raise TypeError("ROSTopic is publish-only (subscribe=False)")
        if self._blocking:
            # wait for a *new* frame to arrive, then clear for the next call
            self._frame_event.wait()
            self._frame_event.clear()
        else:
            # non-blocking: return immediately once the first frame has arrived
            self._frame_event.wait()

        if self._output == "message":
            if self._latest_message is None:
                raise StopIteration
            return self._latest_message

        if self._latest_frame is None:
            raise StopIteration
        opts: dict[str, Any] = {"rgb": self._rgb, "copy": True}
        opts.update(self.args)
        if self._rgb:
            if self._latest_frame.ndim == 3 and not _wants_mono(opts):
                img = _make_image(
                    self._latest_frame, id=self.i, colororder="RGB", **opts
                )
            else:
                img = _make_image(self._latest_frame, id=self.i, **opts)
        else:
            if self._latest_frame.ndim == 3 and not _wants_mono(opts):
                img = _make_image(
                    self._latest_frame, id=self.i, colororder="BGR", **opts
                )
            else:
                img = _make_image(self._latest_frame, id=self.i, **opts)
        if self._latest_timestamp is None:
            raise StopIteration
        _set_sample_metadata(img, int(self._latest_timestamp), self.topic)
        return img

    def publish(
        self, msg: dict | Image | PointCloud, timestamp_ns: int | None = None
    ) -> None:
        """
        Publish a message on the configured ROS topic.

        :param msg: ROS payload as a message dictionary, :class:`~machinevisiontoolbox.Image`, or
            :class:`PointCloud`
        :type msg: dict, :class:`~machinevisiontoolbox.Image`, or :class:`PointCloud`
        :param timestamp_ns: optional timestamp in nanoseconds since Unix epoch;
            used for generated ROS headers. If None, message/object timestamp is
            used when present, otherwise current wall-clock time is used.
        :type timestamp_ns: int or None

        Dictionary payloads are forwarded directly and must match the configured
        ROS message type.

        If ``msg`` is an :class:`~machinevisiontoolbox.Image`, it is serialised according to the
        configured topic message type:

        - ``sensor_msgs/Image`` or ``sensor_msgs/msg/Image`` for raw image data
        - ``sensor_msgs/CompressedImage`` or ``sensor_msgs/msg/CompressedImage``
          for JPEG-compressed image data

        Colour plane order is encoded in the ROS ``encoding`` field for raw
        images, and in the compressed-message ``format`` string for compressed
        images.

        If ``msg`` is a :class:`PointCloud`, it is published as
        ``sensor_msgs/PointCloud2`` with ``x``, ``y``, ``z`` fields and an
        ``rgb`` field when color data is present.

        Example publishing a ``std_msgs/String`` message:

            .. code-block:: python

                pub = ROSTopic("/cmd_text", message="std_msgs/String", subscribe=False)
                pub.publish({"data": "hello"})
                pub.release()


        Example publishing a ``geometry_msgs/Twist`` message:

            .. code-block:: python

                pub = ROSTopic("/cmd_vel", message="geometry_msgs/Twist", subscribe=False)
                pub.publish(
                    {
                        "linear": {"x": 0.2, "y": 0.0, "z": 0.0},
                        "angular": {"x": 0.0, "y": 0.0, "z": 0.5},
                    }
                )
                pub.release()


        Example publishing an :class:`~machinevisiontoolbox.Image` with an explicit timestamp:

            .. code-block:: python

                img = _make_image(np.zeros((240, 320, 3), dtype=np.uint8), colororder="RGB")
                pub = ROSTopic("/camera/image_raw", message="sensor_msgs/Image", subscribe=False)
                pub.publish(img, timestamp_ns=1_700_000_000_123_456_789)
                pub.release()


        Example publishing a :class:`PointCloud` with an explicit timestamp:

            .. code-block:: python

                points = np.array([[0.0, 1.0], [0.0, 0.2], [1.0, 1.2]], dtype=np.float32)
                pc = PointCloud(points)
                pub = ROSTopic("/cloud", message="sensor_msgs/PointCloud2", subscribe=False)
                pub.publish(pc, timestamp_ns=1_700_000_000_223_456_789)
                pub.release()


        The topic is advertised lazily on first publish.
        """
        if isinstance(msg, Image):
            if not self._is_image_message_type(self.message):
                raise TypeError(
                    "Image payload requires sensor_msgs/Image or "
                    "sensor_msgs/CompressedImage topic type"
                )
            payload = self._image_to_ros_message(msg, timestamp_ns=timestamp_ns)
        elif isinstance(msg, PointCloud):
            if not self._is_pointcloud_message_type(self.message):
                raise TypeError(
                    "PointCloud payload requires sensor_msgs/PointCloud2 topic type"
                )
            payload = self._pointcloud_to_ros_message(msg, timestamp_ns=timestamp_ns)
        elif isinstance(msg, dict):
            payload = msg
        else:
            raise TypeError("msg must be a dict, Image, or PointCloud")

        if not self._advertised:
            self._topic.advertise()
            self._advertised = True
        self._topic.publish(payload)

    def release(self) -> None:
        """
        Close the ROS topic and terminate the WebSocket connection.
        """
        if self._subscribe:
            self._topic.unsubscribe()
        if self._advertised:
            self._topic.unadvertise()
            self._advertised = False
        self.client.terminate()

    def grab(self) -> Image | ROSMessage:
        """
        Grab a single frame from the ROS topic.

        :return: next frame from the topic
        :rtype: :class:`~machinevisiontoolbox.Image` or :class:`ROSMessage`

        .. deprecated:: 0.11.4
            Use :func:`next` on the iterator instead, for example ``next(stream)``.
        """
        if not self._subscribe:
            raise TypeError("ROSTopic is publish-only (subscribe=False)")
        warnings.warn(
            "ROSTopic.grab() is deprecated; use next(stream) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return next(self)

    def __enter__(self) -> ROSTopic:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __repr__(self) -> str:
        if self._subscribe:
            mode = "blocking" if self._blocking else "latest"
        else:
            mode = "publish"
        return f"ROSTopic({self.topic!r}, host={self.host!r}, port={self.port}, {mode})"

    @property
    def width(self) -> int | None:
        """
        Width of the most recently received frame.

        :return: width in pixels, or ``None`` if no frame has been received yet
        :rtype: int or None

        :seealso: :meth:`height` :meth:`shape`
        """
        if self._latest_frame is None:
            return None
        return self._latest_frame.shape[1]

    @property
    def height(self) -> int | None:
        """
        Height of the most recently received frame.

        :return: height in pixels, or ``None`` if no frame has been received yet
        :rtype: int or None

        :seealso: :meth:`width` :meth:`shape`
        """
        if self._latest_frame is None:
            return None
        return self._latest_frame.shape[0]

    @property
    def shape(self) -> tuple[int, int] | None:
        """
        Shape of the most recently received frame.

        :return: ``(height, width)`` in pixels, or ``None`` if no frame has
            been received yet
        :rtype: tuple(int, int) or None

        :seealso: :meth:`height` :meth:`width`
        """
        if self._latest_frame is None:
            return None
        return self._latest_frame.shape[:2]  # (height, width)


class SyncROSStreams:
    """
    Synchronise multiple :class:`ROSTopic` objects by timestamp.

    :param streams: two or more ROS streams to synchronise
    :type streams: list of :class:`ROSTopic`
    :param tolerance: maximum timestamp mismatch in seconds for a matched set,
        defaults to 0.02
    :type tolerance: float, optional
    :raises ValueError: if fewer than two streams are provided
    :raises ValueError: if ``tolerance`` is not positive
    :raises TypeError: if any stream item does not have a ``timestamp``
        attribute in nanoseconds
    :return: iterator yielding tuples of time-aligned items
    :rtype: tuple

    The matcher uses approximate-time synchronisation.  At each step it compares
    the oldest buffered frame from each stream; if their timestamps are within
    ``tolerance`` they are emitted together.  Otherwise it discards the single
    oldest frame and continues.

    Example:

        .. code-block:: python

            rgb = ROSTopic("/camera/color/image_raw/compressed")
            depth = ROSTopic("/camera/depth/image_rect_raw/compressed")
            with SyncROSStreams([rgb, depth], tolerance=0.03) as sync:
                for rgb_im, depth_im in sync:
                    # process aligned pair
                    pass


    **Interaction with ``ROSTopic`` blocking mode**

    For time-step synchronisation, set each input stream to ``blocking=True``.
    This gives one newly arrived frame per stream pull and avoids repeated
    reuse of stale frames.

    If any stream uses ``blocking=False``, repeated calls can return the same
    latest frame multiple times, which may lead to duplicate tuples, tighter
    polling loops, and weaker one-sample-per-time-step behaviour.  It remains
    useful for low-latency latest-state fusion, but is less suitable for
    strict frame-by-frame synchronisation.
    """

    streams: list[ROSTopic]
    tolerance: float
    _tol_ns: int

    def __init__(self, streams: list[ROSTopic], tolerance: float = 0.02) -> None:
        if len(streams) < 2:
            raise ValueError("SyncROSStreams requires at least two streams")
        if tolerance <= 0:
            raise ValueError("tolerance must be > 0")
        if not all(stream._subscribe for stream in streams):
            raise ValueError("All streams must be in subscribe mode (subscribe=True)")

        self.streams = streams
        self.tolerance = tolerance
        self._tol_ns = int(round(tolerance * 1_000_000_000))

        self._iters = []
        self._buffers = []

    @staticmethod
    def _timestamp_ns(item: Any) -> int:
        ts = getattr(item, "timestamp", None)
        if ts is None:
            raise TypeError("Stream item has no timestamp attribute")
        return int(ts)

    def __iter__(self) -> SyncROSStreams:
        self._iters = [iter(stream) for stream in self.streams]
        self._buffers = [deque() for _ in self.streams]
        return self

    def _fill_one(self, i: int) -> None:
        if not self._buffers[i]:
            item = next(self._iters[i])
            self._timestamp_ns(item)
            self._buffers[i].append(item)

    def __next__(self) -> tuple[Any, ...]:
        while True:
            for i in range(len(self.streams)):
                self._fill_one(i)

            heads = [buf[0] for buf in self._buffers]
            times = [self._timestamp_ns(item) for item in heads]

            t_min = min(times)
            t_max = max(times)

            if t_max - t_min <= self._tol_ns:
                matched = tuple(buf.popleft() for buf in self._buffers)
                return matched

            oldest_index = times.index(t_min)
            self._buffers[oldest_index].popleft()

    def __enter__(self) -> SyncROSStreams:
        for stream in self.streams:
            stream.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for stream in self.streams:
            stream.__exit__(exc_type, exc, tb)

    def __str__(self) -> str:
        topics = ", ".join(
            getattr(stream, "topic", "<unknown>") for stream in self.streams
        )
        return f"SyncROSStreams([{topics}])"

    def __repr__(self) -> str:
        return (
            f"SyncROSStreams(nstreams={len(self.streams)}, "
            f"tolerance={self.tolerance:.6f}s)"
        )


class ROSBag(ImageSource):
    """
    Iterate images and point clouds from a ROS 1 bag file.

    :param filename: path to a ``.bag`` file, or an ``http(s)://`` URL
    :type filename: str or Path
    :param release: ROS release name (or unique substring), e.g. ``"noetic"``, defaults to ``"ROS1_NOETIC"``; controls the message definitions used to parse the bag file
    :type release: str
    :param topicfilter: only yield messages from this topic or list of topics;
        ``None`` accepts all topics
    :type topicfilter: str or list of str or None
    :param msgfilter: only yield messages whose type contains this substring
        or matches any entry in the list; ``None`` accepts all types,
        defaults to ``"Image"``
    :type msgfilter: str or list of str or None
    :param dtype: override the numpy dtype for all image pixel data, or by topic ``{topic: dtype}``
    :type dtype: str or dict
    :param colororder: override the color-plane order for all image, or by topic
        ``{topic: colororder}``
    :type colororder: str or dict or None
    :param kwargs: options applied to image frames, see
        :func:`~machinevisiontoolbox.base.imageio.convert`
    :raises ImportError: if the ``rosbags`` package is not installed

    The resulting object is an iterator that yields:

    - :class:`~machinevisiontoolbox.Image` for messages whose type ends in ``Image``, which includes ``CompressedImage``.
    - :class:`PointCloud` for ``PointCloud2`` messages (requires ``open3d``)
    - the raw deserialised message object for all other types

    Each yielded object carries a ``timestamp`` attribute (ROS nanosecond
    epoch from the message header) and a ``topic`` attribute (the topic on which it was published).

    **Usage modes**

    *Implicit* — iterating directly over the object opens and closes the bag
    file automatically around the loop:

        .. code-block:: python

            from machinevisiontoolbox import ROSBag
            for img in ROSBag("mybag.bag", release="noetic"):
                img.disp()


    *Explicit context manager* — use a ``with`` statement when you need to
    make multiple passes over the bag, call helper methods such as
    :meth:`topics` or :meth:`print`, or simply want a guaranteed close even
    if an exception is raised:

        .. code-block:: python

            bag = ROSBag("mybag.bag", release="noetic", msgfilter=None)
            with bag:
                bag.print()                     # inspect topics
                for msg in bag:                 # iterate messages
                    print(msg.topic, msg.timestamp)


    .. note::
        ``filename`` may be an ``http://`` or ``https://`` URL, in which case
        the bag file is downloaded to a temporary file on first use and that
        file is reused for the lifetime of the ``ROSBag`` object.  The
        temporary file is deleted automatically when the object is garbage
        collected or when the script exits.

    .. note::
        If ``filename`` is a relative path that does not exist in the current
        working directory, it is looked up in the ``mvtb-data`` companion
        package automatically.  Bag files placed there can therefore be
        referenced by their bare name, e.g. ``ROSBag("forest.bag")``.

    .. note::
        The ``release`` argument controls the ROS message definitions used to
        parse the bag file.  This is important because message definitions can
        change between ROS releases, and using the wrong definitions can lead to
        incorrect parsing of the data.  The ``release`` string is matched
        case-insensitively against the member names of :class:`~rosbags.typesys.Stores`,
        so short names such as ``"noetic"`` or ``"humble"`` are accepted in addition to the full enum name (e.g. ``"ROS1_NOETIC"``).

    """

    filename: str
    release: str
    dtype: np.dtype | str | dict | None
    colororder: str | dict | None
    verbose: bool
    args: dict

    def __init__(
        self,
        filename: str,
        release: str = "ROS1_NOETIC",
        topicfilter: str | list[str] | None = None,
        msgfilter: str | list[str] | None = "Image",
        dtype: np.dtype | str | dict | None = None,
        colororder: str | dict | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        if not _rosbags_available:
            raise ImportError(
                "rosbags is required for ROS bag support. "
                "Install it with: pip install rosbags "
                "or pip install machinevision-toolbox-python[ros]"
            )
        self.filename = filename
        self.release = release
        self._topic_filter = topicfilter
        self._msgfilter = msgfilter
        self.verbose = verbose
        self.reader = None
        self.connections = []
        self.typestore = _resolve_typestore(release)
        self.dtype = dtype
        self.colororder = colororder
        self.args = kwargs
        self._tmpfile: str | None = None
        self._is_ros2: bool = False

    def __repr__(self) -> str:
        return (
            f"ROSBag({str(self.filename)!r}, release={self.release!r}, "
            f"topicfilter={self._topic_filter!r}, msgfilter={self._msgfilter!r})"
        )

    def __str__(self) -> str:
        return f"ROSBag({str(self.filename)!r})"

    @staticmethod
    def format_duration(duration_ns: int) -> str:
        """Format a duration in nanoseconds as ``hh:mm:ss``.

        :param duration_ns: duration in nanoseconds
        :type duration_ns: int
        :return: formatted duration string
        :rtype: str
        """
        total_s = int(duration_ns // 1_000_000_000)
        h = total_s // 3600
        m = total_s % 3600 // 60
        s = total_s % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def format_local_time(timestamp_ns: int, fmt: str | None = None) -> str:
        """Format a ROS timestamp to a local-time string.

        :param timestamp_ns: ROS timestamp in nanoseconds since the Unix epoch (UTC)
        :type timestamp_ns: int
        :param fmt: :func:`~datetime.datetime.strftime` format string;
            defaults to ISO 8601 with millisecond precision
        :type fmt: str or None
        :return: formatted time string
        :rtype: str

        :seealso: `strftime format codes <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_
        """
        dt = datetime.fromtimestamp(
            timestamp_ns / 1_000_000_000, tz=timezone.utc
        ).astimezone()
        return dt.strftime(fmt) if fmt else dt.isoformat(timespec="milliseconds")

    @property
    def topicfilter(self) -> str | list[str] | None:
        """Topic filter applied when iterating.

        The iterator only returns messages from topics that match the filter.
        A string matches any topic whose name *contains* that string.
        A list matches any topic containing *any* of the strings.
        ``None`` accepts all topics.

        :type: str, list of str, or None
        :seealso: :attr:`msgfilter`
        """
        return self._topic_filter

    @topicfilter.setter
    def topicfilter(self, topicfilter: str | list[str] | None) -> None:
        self._topic_filter = topicfilter

    @property
    def msgfilter(self) -> str | list[str] | None:
        """Message-type filter applied when iterating.

        The iterator only returns messages from message types that match the filter.
        A string matches any message type whose name *contains* that string.
        A list matches any message type containing *any* of the strings.
        ``None`` accepts all message types.

        :type: str, list of str, or None
        :seealso: :attr:`topicfilter`
        """
        return self._msgfilter

    @msgfilter.setter
    def msgfilter(self, msgfilter: str | list[str] | None) -> None:
        self._msgfilter = msgfilter

    def _allowed(self, x) -> bool:
        msg_ok = (
            self._msgfilter is None
            or (
                isinstance(self._msgfilter, list)
                and any(f in x.msgtype for f in self._msgfilter)
            )
            or (isinstance(self._msgfilter, str) and self._msgfilter in x.msgtype)
        )
        topic_ok = (
            self._topic_filter is None
            or (
                isinstance(self._topic_filter, list)
                and any(f in x.topic for f in self._topic_filter)
            )
            or (isinstance(self._topic_filter, str) and self._topic_filter in x.topic)
        )
        return msg_ok and topic_ok

    @staticmethod
    def _stamp_to_ns(stamp) -> int:
        """
        Robustly converts a ROS1 or ROS2 stamp to total nanoseconds as an integer.
        Handles 'sec/nanosec' (ROS2) and 'secs/nsecs' (ROS1) naming.
        """
        # Try ROS2/rosbags naming first, fall back to ROS1 naming
        sec = getattr(stamp, "sec", getattr(stamp, "secs", 0))
        nanosec = getattr(stamp, "nanosec", getattr(stamp, "nsecs", 0))

        return int(sec) * 1_000_000_000 + int(nanosec)

    def __del__(self) -> None:
        self._close_reader()
        tmpfile = getattr(self, "_tmpfile", None)
        if tmpfile is not None:
            from pathlib import Path as _Path

            _Path(tmpfile).unlink(missing_ok=True)
            self._tmpfile = None

    def _close_reader(self) -> None:
        reader = getattr(self, "reader", None)
        if reader is not None:
            reader.close()
            self.reader = None

    def _open_reader(self):
        if self.reader is not None:
            return self.reader

        filename = str(self.filename)
        if filename.startswith(("http://", "https://")):
            if self._tmpfile is None:
                with tempfile.NamedTemporaryFile(suffix=".bag", delete=False) as tmp:
                    self._tmpfile = tmp.name
                print(f"Downloading {filename} ...")
                urllib.request.urlretrieve(filename, self._tmpfile)
            path = self._tmpfile
        else:
            path = self.filename

        from pathlib import Path as _BagPath

        # Resolve via mvtb_path_to_datafile: returns path unchanged if the file
        # exists locally, otherwise searches the mvtb-data companion package.
        try:
            path = mvtb_path_to_datafile("data", path)
        except (ValueError, ModuleNotFoundError):
            pass  # fall through; rosbags will raise a clear error on open

        self._is_ros2 = _BagPath(path).is_dir()
        assert _RosBagReader1 is not None
        assert _RosBagReader2 is not None
        self.reader = (_RosBagReader2 if self._is_ros2 else _RosBagReader1)(path)
        self.reader.open()
        self.connections = [x for x in self.reader.connections if self._allowed(x)]
        return self.reader

    def topics(self) -> dict[str, str]:
        """Return topics found in the ROS bag.

        :return: mapping of topic name to message type
        :rtype: dict

        Returns a dictionary mapping topic names to message types for all topics found
        in the bag file.  The message types are returned as strings, e.g.
        ``"sensor_msgs/Image"``.
        """
        reader = self._open_reader()
        topicdict = {conn.topic: conn.msgtype for conn in reader.connections}
        self._close_reader()
        return topicdict

    def traffic(self, progress: bool = True) -> Counter:
        """Message counts by type found in the ROS bag.

        :param progress: show a tqdm progress bar while scanning, defaults to ``True``
        :type progress: bool
        :return: mapping of message type to count
        :rtype: :class:`~collections.Counter`

        Scans through the bag file and counts the number of messages of each type.  For
        a large bag file this can take some time, so a progress bar is shown if
        ``progress`` is ``True``.
        """
        reader = self._open_reader()
        counts = Counter(
            conn.msgtype
            for conn, _ts, _raw in tqdm(
                reader.messages(),
                total=reader.message_count,
                desc="scanning",
                unit="msg",
                disable=not progress,
            )
        )
        self._close_reader()
        return counts

    def print(self, progress: bool = True, file=None) -> None:
        """Print a summary table of topics in the ROS bag.

        :param progress: show a tqdm progress bar while scanning, defaults to ``True``
        :type progress: bool
        :param file: file to write output to, defaults to ``None``
        :type file: file-like object, optional

        Print a human-readable summary of the topics in the ROS bag, showing the message type, total message count, and
        whether it passes the current topic and message filters.
        """
        reader = self._open_reader()
        print(
            f"recorded on {self.format_local_time(reader.start_time)}, duration {self.format_duration(reader.duration)}, {reader.message_count} messages",
            file=file,
        )
        table = ANSITable(
            Column("topic", colalign="<", headalign="^"),
            Column("msgtype", colalign="<", headalign="^"),
            Column("count", colalign=">", headalign="^"),
            Column("allowed", colalign="^", headalign="^"),
            border="thin",
        )
        counts = self.traffic(progress=progress)
        for topic, msgtype in self.topics().items():
            conn_stub = type("_C", (), {"topic": topic, "msgtype": msgtype})
            allowed = self._allowed(conn_stub)
            table.row(
                topic,
                msgtype,
                counts[msgtype],
                "✓" if allowed else "✗",
                style="bold" if allowed else None,
            )
        print(table, file=file)

    def __enter__(self) -> ROSBag:
        self._open_reader()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._close_reader()

    # Map of ROS encoding strings to (numpy_dtype, channels, color_plane_order)
    _ROS_ENCODING_MAP = {
        # Common Color Encodings
        "rgb8": (np.uint8, 3, "rgb"),
        "rgba8": (np.uint8, 4, "rgba"),
        "rgb16": (np.uint16, 3, "rgb"),
        "rgba16": (np.uint16, 4, "rgba"),
        "bgr8": (np.uint8, 3, "bgr"),
        "bgra8": (np.uint8, 4, "bgra"),
        "bgr16": (np.uint16, 3, "bgr"),
        "bgra16": (np.uint16, 4, "bgra"),
        "mono8": (np.uint8, 1, None),
        "mono16": (np.uint16, 1, None),
        # Generic Encodings (Bits, Type, Channels)
        # 8-bit
        "8UC1": (np.uint8, 1, None),
        "8UC2": (np.uint8, 2, None),  # Usually complex or stereo
        "8UC3": (np.uint8, 3, "rgb"),  # Standard ROS assumption
        "8UC4": (np.uint8, 4, "rgba"),
        "8SC1": (np.int8, 1, None),
        # 16-bit
        "16UC1": (np.uint16, 1, None),
        "16UC2": (np.uint16, 2, None),
        "16UC3": (np.uint16, 3, "rgb"),
        "16UC4": (np.uint16, 4, "rgba"),
        "16SC1": (np.int16, 1, None),
        # 32-bit (Freiburg Depth standard)
        "32SC1": (np.int32, 1, None),
        "32FC1": (np.float32, 1, None),  # Depth (m)
        "32FC2": (np.float32, 2, None),
        "32FC3": (np.float32, 3, "rgb"),
        "32FC4": (np.float32, 4, "rgba"),
        # 64-bit
        "64FC1": (np.float64, 1, None),
        # Bayer Patterns (Single-channel raw data)
        "bayer_rggb8": (np.uint8, 1, None),
        "bayer_bggr8": (np.uint8, 1, None),
        "bayer_gbrg8": (np.uint8, 1, None),
        "bayer_grbg8": (np.uint8, 1, None),
        "bayer_rggb16": (np.uint16, 1, None),
        "bayer_bggr16": (np.uint16, 1, None),
        "bayer_gbrg16": (np.uint16, 1, None),
        "bayer_grbg16": (np.uint16, 1, None),
    }

    def __iter__(self) -> Iterator[Any]:
        self._open_reader()
        assert self.reader is not None
        try:
            for connection, timestamp, rawdata in self.reader.messages(
                connections=self.connections
            ):
                try:
                    _deser = (
                        self.typestore.deserialize_cdr
                        if self._is_ros2
                        else self.typestore.deserialize_ros1
                    )
                    msg = cast(Any, _deser(rawdata, connection.msgtype))
                except KeyError as e:
                    if self.verbose:
                        print(
                            f"{self.format_local_time(timestamp)} Error occurred while deserializing message: {e}"
                        )
                    continue

                # check if message and topic pass the filters before doing any expensive processing
                if not self._allowed(connection):
                    continue

                # now check if it's an image or point cloud we can convert
                if connection.msgtype.endswith("CompressedImage"):
                    # Compressed sensor_msgs/CompressedImage: JPEG or PNG bytes
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    img_array = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                    if img_array is None:
                        raise RuntimeError("Failed to decode compressed image")

                    # OpenCV decodes color images as BGR by default.
                    if img_array.ndim == 2:
                        colororder = None
                    elif img_array.shape[2] == 4:
                        colororder = "bgra"
                    else:
                        colororder = "bgr"

                    # check for a per-topic override
                    if isinstance(self.colororder, dict):
                        colororder = self.colororder.get(connection.topic, None)
                    else:
                        colororder = self.colororder

                    if colororder is not None and not _wants_mono(self.args):
                        img = _make_image(
                            img_array,
                            colororder=colororder.upper(),
                            **self.args,
                        )
                    else:
                        img = _make_image(img_array, **self.args)
                    _set_sample_metadata(
                        img, self._stamp_to_ns(msg.header.stamp), connection.topic
                    )
                    yield img

                elif connection.msgtype.endswith("Image"):

                    if hasattr(msg, "encoding"):
                        encoding = msg.encoding
                        if encoding not in self._ROS_ENCODING_MAP:
                            raise ValueError(
                                f"Unsupported or unknown ROS encoding: {encoding}"
                            )

                        dtype, channels, colororder = self._ROS_ENCODING_MAP[encoding]
                    else:
                        # Fallback if no encoding is provided, assume 8-bit RGB or mono based on channels
                        channels = msg.data.shape[0] // (msg.height * msg.width)
                        dtype = np.uint8
                        colororder = "rgb" if channels == 3 else None

                    # apply per-topic overrides if specified
                    if self.dtype is not None:
                        if isinstance(self.dtype, dict):
                            dtype = self.dtype.get(connection.topic, "uint8")
                        else:
                            dtype = self.dtype or "uint8"

                    if self.colororder is not None:
                        # command line override
                        if isinstance(self.colororder, dict):
                            colororder = self.colororder.get(connection.topic, None)
                        else:
                            colororder = self.colororder

                    # Use frombuffer for zero-copy efficiency where possible
                    data = np.frombuffer(msg.data, dtype=dtype)

                    # Reshape to (H, W, C) or (H, W)
                    if channels > 1:
                        img_array = data.reshape((msg.height, msg.width, channels))
                    else:
                        img_array = data.reshape((msg.height, msg.width))

                    if colororder is not None and not _wants_mono(self.args):
                        img = _make_image(
                            img_array,
                            colororder=colororder.upper(),
                            **self.args,
                        )
                    else:
                        img = _make_image(img_array, **self.args)
                    _set_sample_metadata(
                        img, self._stamp_to_ns(msg.header.stamp), connection.topic
                    )
                    yield img

                elif connection.msgtype.endswith("PointCloud2"):
                    # Converts a sensor_msgs/PointCloud2 message to an Open3D PointCloud.
                    # Handles both colored and uncolored clouds.
                    if not _open3d_available:
                        raise ImportError(
                            "open3d is required to read PointCloud2 messages. "
                            "Install it with: pip install open3d-python "
                            "or pip install machinevision-toolbox-python[open3d]"
                        )
                    assert o3d is not None

                    # 1. Map ROS datatypes to numpy dtypes
                    # 1:INT8, 2:UINT8, 3:INT16, 4:UINT16, 5:INT32, 6:UINT32, 7:FLOAT32, 8:FLOAT64
                    type_map = {
                        1: np.int8,
                        2: np.uint8,
                        3: np.int16,
                        4: np.uint16,
                        5: np.int32,
                        6: np.uint32,
                        7: np.float32,
                        8: np.float64,
                    }

                    # 2. Build structured numpy dtype from message fields
                    formats = []
                    offsets = []
                    names = []
                    for field in msg.fields:
                        names.append(field.name)
                        formats.append(type_map[field.datatype])
                        offsets.append(field.offset)

                    # itemsize must match point_step to handle potential padding bytes
                    dtype = np.dtype(
                        {
                            "names": names,
                            "formats": formats,
                            "offsets": offsets,
                            "itemsize": msg.point_step,
                        }
                    )

                    # 3. Create the structured array from raw bytes
                    cloud_arr = np.frombuffer(msg.data, dtype=dtype)

                    # 4. Extract XYZ coordinates
                    # Open3D requires float64 for its Vector3dVector
                    points = np.zeros((len(cloud_arr), 3), dtype=np.float64)
                    points[:, 0] = cloud_arr["x"]
                    points[:, 1] = cloud_arr["y"]
                    points[:, 2] = cloud_arr["z"]

                    # 5. Initialize Open3D object
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)

                    # 6. Extract Color (if available)
                    if "rgb" in names:
                        # Step A: Interpret bytes as uint32 (packing is usually A-R-G-B)
                        # Note: Even if field.datatype says FLOAT32, we 'view' the bits as UINT32
                        rgb_data = cloud_arr["rgb"].view(np.uint32)

                        # Step B: Bit-shift to extract 8-bit channels
                        r = (rgb_data >> 16) & 0xFF
                        g = (rgb_data >> 8) & 0xFF
                        b = (rgb_data) & 0xFF

                        # Step C: Normalize to [0, 1] for Open3D
                        colors = np.column_stack((r, g, b)).astype(np.float64) / 255.0
                        pcd.colors = o3d.utility.Vector3dVector(colors)

                    pc = PointCloud(pcd)
                    _set_sample_metadata(
                        pc, self._stamp_to_ns(msg.header.stamp), connection.topic
                    )
                    yield pc

                else:
                    # any other sort of message, just yield the deserialized object with timestamp and topic attributes
                    _set_sample_metadata(msg, timestamp, connection.topic)
                    yield msg
        finally:
            self._close_reader()

    def __len__(self) -> int:
        """Return number of messages that pass current filters.

        If no topic or message-type filter is set, this returns the bag's
        total message count. Otherwise, it counts messages on the filtered
        connections.
        """
        was_open = self.reader is not None
        reader = self._open_reader()

        if self._topic_filter is None and self._msgfilter is None:
            n = reader.message_count
        else:
            n = sum(1 for _ in reader.messages(connections=self.connections))

        if not was_open:
            self._close_reader()
        return n


class PointCloudSequence:
    """
    An in-memory sequence of :class:`PointCloud` objects with interactive display.

    :param clouds: sequence of point clouds
    :type clouds: iterable of :class:`PointCloud`

    The sequence is materialised into a list on construction.  Items are
    expected to be :class:`PointCloud` instances; a ``timestamp`` attribute
    (ROS nanosecond epoch) and ``topic`` attribute are used for the display
    overlay when present.

    Requires the ``open3d`` package.

    Example:

        .. code-block:: python

            from machinevisiontoolbox import ROSBag, PointCloudSequence
            bag = ROSBag("mybag.bag", msgfilter="PointCloud2")
            seq = PointCloudSequence(bag)
            seq.disp()                       # step through one cloud at a time
            seq.disp(animate=True, fps=10)   # timed playback


    :seealso: :class:`ImageSequence`
    """

    _clouds: list
    _i: int

    def __init__(self, clouds: Any) -> None:
        self._clouds = list(clouds)
        self._i = 0

    def __len__(self) -> int:
        return len(self._clouds)

    def __getitem__(self, i: int | slice) -> PointCloud | list[PointCloud]:
        return self._clouds[i]

    def __iter__(self) -> PointCloudSequence:
        self._i = 0
        return self

    def __next__(self) -> PointCloud:
        if self._i >= len(self._clouds):
            raise StopIteration
        cloud = self._clouds[self._i]
        self._i += 1
        return cloud

    def __repr__(self) -> str:
        return f"PointCloudSequence({len(self._clouds)} clouds)"

    def disp(
        self,
        animate: bool = False,
        fps: float = 10.0,
        title: str | None = None,
        loop: bool = False,
    ) -> None:
        """
        Display the point-cloud sequence interactively using Open3D GUI.

        :param animate: if ``True``, play as timed animation; if ``False``
            (default), step through one cloud at a time
        :type animate: bool, optional
        :param fps: playback rate in frames per second, used when
            ``animate=True``, defaults to 10.0
        :type fps: float, optional
        :param title: window title, defaults to ``"PointCloudSequence"``
        :type title: str or None, optional
        :param loop: restart when the sequence ends (animate mode only),
            defaults to ``False``
        :type loop: bool, optional

        **Keys — step-through mode** (``animate=False``):

        - ``[space]`` — next cloud
        - ``[1-9]`` / ``[0]`` — jump 1–9 or 10 clouds
        - ``[l]`` / ``[c]`` / ``[d]`` — jump 50 / 100 / 500 clouds
        - ``[q]`` / ``[x]`` — quit

        **Keys — animate mode** (``animate=True``):

        - ``[space]`` — pause / resume
        - ``[=]`` — increase playback speed
        - ``[-]`` — decrease playback speed
        - ``[q]`` / ``[x]`` — quit

        :seealso: :class:`ImageSequence`
        """
        import threading
        import time
        import open3d.visualization.gui as gui  # type: ignore[import-not-found]
        import open3d.visualization.rendering as rendering  # type: ignore[import-not-found]

        _jump = {
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "0": 10,
            "l": 50,
            "c": 100,
            "d": 500,
        }

        def _label(item, i):
            ts = getattr(item, "timestamp", None)
            if ts is not None:
                dt = datetime.fromtimestamp(ts / 1e9, tz=timezone.utc).astimezone()
                return dt.isoformat(timespec="milliseconds") + f" [{i}]"
            return f"[{i}]"

        app = gui.Application.instance
        app.initialize()

        win_title = title or "PointCloudSequence"
        win = app.create_window(win_title, 1280, 720)
        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(win.renderer)
        scene.scene.set_background([0.15, 0.15, 0.15, 1.0])
        label = gui.Label("")

        def on_layout(ctx):
            r = win.content_rect
            scene.frame = r
            em = ctx.theme.font_size
            label.frame = gui.Rect(
                r.x + 8, r.get_bottom() - em - 8, r.width - 16, em + 4
            )

        win.set_on_layout(on_layout)
        win.add_child(scene)
        win.add_child(label)

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 2.0

        def _update_cloud(pcd, label_text, first=False):
            if scene.scene.has_geometry("cloud"):
                scene.scene.remove_geometry("cloud")
            scene.scene.add_geometry("cloud", pcd, mat)
            if first:
                bb = pcd.get_axis_aligned_bounding_box()
                scene.setup_camera(60.0, bb, bb.get_center())
            label.text = label_text
            win.post_redraw()

        if animate:
            state = {"fps": fps, "paused": False, "quit": False}

            def on_key(evt):
                if evt.type == gui.KeyEvent.DOWN:
                    k = evt.key
                    if k in (ord("q"), ord("x")):
                        state["quit"] = True
                        win.close()
                        return gui.Widget.EventCallbackResult.HANDLED
                    elif k == gui.KeyName.SPACE:
                        state["paused"] = not state["paused"]
                        return gui.Widget.EventCallbackResult.HANDLED
                    elif k == ord("="):
                        state["fps"] = state["fps"] * 1.5
                        return gui.Widget.EventCallbackResult.HANDLED
                    elif k == ord("-"):
                        state["fps"] = max(1.0, state["fps"] / 1.5)
                        return gui.Widget.EventCallbackResult.HANDLED
                return gui.Widget.EventCallbackResult.IGNORED

            win.set_on_key(on_key)
            print("\nKeys: [space] pause/resume  [=] faster  [-] slower  [q/x] quit")

            first = [True]

            def run():
                i = 0
                while True:
                    if i >= len(self._clouds):
                        if loop:
                            i = 0
                        else:
                            break
                    x = self._clouds[i]
                    while state["paused"] and not state["quit"]:
                        time.sleep(0.05)
                    if state["quit"]:
                        break
                    lbl = _label(x, i)
                    pcd = x._pcd
                    is_first = first[0]
                    first[0] = False
                    app.post_to_main_thread(
                        win,
                        lambda pcd=pcd, lbl=lbl, f=is_first: _update_cloud(pcd, lbl, f),
                    )
                    time.sleep(1.0 / state["fps"])
                    i += 1
                if not state["quit"]:
                    app.post_to_main_thread(win, win.close)

        else:
            view_state = {"next": False, "quit": False, "skip": 0}

            def on_key(evt):
                if evt.type == gui.KeyEvent.DOWN:
                    k = evt.key
                    if k in (ord("q"), ord("x")):
                        view_state["quit"] = True
                        win.close()
                        return gui.Widget.EventCallbackResult.HANDLED
                    elif k == gui.KeyName.SPACE:
                        view_state["next"] = True
                        return gui.Widget.EventCallbackResult.HANDLED
                    else:
                        ch = chr(k) if 32 <= k < 128 else ""
                        if ch in _jump:
                            view_state["skip"] = _jump[ch]
                            view_state["next"] = True
                            return gui.Widget.EventCallbackResult.HANDLED
                return gui.Widget.EventCallbackResult.IGNORED

            win.set_on_key(on_key)
            print(
                "\nKeys: [space] next  [1-9] jump N frames  [0] jump 10"
                "  [l/c/d] jump 50/100/500  [q/x] quit"
            )

            first = [True]

            def run():
                i = 0
                while i < len(self._clouds):
                    if view_state["quit"]:
                        break
                    x = self._clouds[i]
                    lbl = _label(x, i)
                    pcd = x._pcd
                    is_first = first[0]
                    first[0] = False
                    app.post_to_main_thread(
                        win,
                        lambda pcd=pcd, lbl=lbl, f=is_first: _update_cloud(pcd, lbl, f),
                    )
                    view_state["next"] = False
                    while not view_state["next"] and not view_state["quit"]:
                        time.sleep(0.05)
                    skip = view_state["skip"]
                    view_state["skip"] = 0
                    i += 1 + skip

        t = threading.Thread(target=run, daemon=True)
        t.start()
        app.run()


class TensorStack(ImageSource):
    """
    Lazy image source from a PyTorch batch tensor.

    Each frame is a zero-copy view into the batch tensor, providing
    memory-efficient iteration over model outputs or other batch-processed tensors.

    :param tensor: tensor of shape ``(B, C, H, W)`` or ``(B, H, W)``
    :type tensor: torch.Tensor
    :param colororder: color plane order for multi-channel tensors,
        e.g. ``"RGB"`` or ``"BGR"``, defaults to None
    :type colororder: str, optional
    :param logits: if True, take argmax over the channel dimension to
        convert per-class logits to a class label image, defaults to False
    :type logits: bool, optional
    :param dtype: data type for output image arrays, e.g. ``np.uint8`` or
        ``np.float32``; if None, dtype is inferred from the tensor data,
        defaults to None
    :type dtype: numpy dtype or None, optional
    :param kwargs: options applied to image frames, see
        :func:`~machinevisiontoolbox.base.imageio.convert`

    Example:

        .. code-block:: pycon

        >>> from machinevisiontoolbox import TensorStack
        >>> import torch
        >>> batch = torch.randn(10, 3, 64, 64)  # 10 RGB images
        >>> source = TensorStack(batch, colororder="RGB")
        >>> len(source)
        10
        >>> img = source[7]  # Returns Image wrapping tensor[7] (zero-copy view)
        >>> for img in source:
        ...     features = extract(img)  # Process each image lazily


    :seealso: :meth:`Image.Tensor`
    """

    def __init__(
        self,
        tensor: "torch.Tensor",
        colororder: str | None = None,
        logits: bool = False,
        dtype: "DTypeLike | None" = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize TensorStack from a batch tensor.

        :param tensor: batch tensor of shape ``(B, C, H, W)`` or ``(B, H, W)``
        :param colororder: color plane order for display/export
        :param logits: if True, argmax the channel dimension for segmentation masks
        :param dtype: output array dtype passed to Image constructor
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for TensorStack. "
                "Install it with: pip install torch "
                "or pip install machinevision-toolbox-python[torch]"
            )

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

        if tensor.ndim not in (4, 3):
            raise ValueError(
                f"Expected tensor of shape (B, C, H, W) or (B, H, W), got {tensor.shape}"
            )

        # Convert to CPU numpy view (keeps shared memory)
        self._tensor = tensor.detach().cpu()
        self._array = self._tensor.numpy()
        self._colororder = colororder
        self._logits = logits
        self._dtype = dtype
        self.args = kwargs
        self._batch_size = self._array.shape[0]
        self._i = 0

    def __len__(self) -> int:
        """Return number of images in the batch."""
        return self._batch_size

    def __iter__(self) -> TensorStack:
        """Iterate over images as views into the batch."""
        self._i = 0
        return self

    def __next__(self) -> Image:
        if self._i >= self._batch_size:
            raise StopIteration
        image = self[self._i]
        self._i += 1
        return image

    def __getitem__(self, index: int) -> Image:
        """
        Get image at index as a zero-copy view.

        :param index: image index in range [0, batch_size)
        :return: Image wrapping frame at index
        :rtype: Image
        """
        if not isinstance(index, int) or index < 0 or index >= self._batch_size:
            raise IndexError(f"Index {index} out of range [0, {self._batch_size})")

        # Extract slice as view (B, C, H, W) → (C, H, W) or (B, H, W) → (H, W)
        frame = self._array[index]

        if self._logits:
            # Argmax over channel dimension for segmentation masks
            if frame.ndim == 3:
                frame = np.argmax(frame, axis=0)
        else:
            # Permute (C, H, W) → (H, W, C) for color images
            if frame.ndim == 3:
                frame = np.transpose(frame, (1, 2, 0))

        opts: dict[str, Any] = {"dtype": self._dtype}
        opts.update(self.args)
        if self._colororder is not None and not _wants_mono(opts):
            return _make_image(frame, colororder=self._colororder, **opts)
        return _make_image(frame, **opts)

    def __str__(self) -> str:
        """Return string representation."""
        shape = self._array.shape
        dtype = self._array.dtype
        return (
            f"TensorStack({self._batch_size} frames, shape {shape[1:]}, dtype {dtype})"
        )

    def __repr__(self) -> str:
        """Return string representation."""
        shape = self._array.shape
        dtype = self._array.dtype
        return (
            f"TensorStack(nframes={self._batch_size}, shape={shape[1:]}, dtype={dtype})"
        )


class LabelMeReader:
    """
    Read annotations from a LabelMe JSON file.

    :param filename: path to LabelMe JSON file, or an ``http(s)://`` URL
    :type filename: str

    LabelMe is a popular annotation tool that saves annotations in a JSON format.  This
    class reads the JSON file and extracts the image, polygonal shapes, and associated
    flags. Other popular tools like CVAT and LabelStudio can export to LabelMe format,
    so this class can be used as a lightweight reader for those formats as well, albeit
    with some limitations (e.g. only polygonal shapes are supported).

    Methods of this class return:

    - a list of :class:`Polygon2` instances for all shapes
    - file-level ``flags`` as a dictionary
    - the labeled image as a :class:`~machinevisiontoolbox.Image` instance

    For each returned polygon, additional attributes are attached:

    - ``group_id: int | None`` from the shape entry
    - ``flags: dict`` from the shape entry as a dictionary

    Rectangle shapes are converted to 4-vertex polygons.

    Example:

        .. code-block:: python

            from machinevisiontoolbox import LabelMeReader
            lme = LabelMeReader("https://github.com/wkentaro/labelme/raw/main/examples/tutorial/apc2016_obj3.json")
            len(lme.shapes)  # number of annotated shapes
            flags = lme.flags  # file-level flags
            image = lme.image  # labeled image (if available)
            for polygon in lme.shapes:
                print(polygon.points)  # polygon vertices
                print(polygon.group_id)  # group ID (if any)
                print(polygon.flags)  # shape-level flags

    .. note:: The ``labelme`` package provides a more comprehensive interface to
        LabelMe annotations, including support for all shape types and attributes, but
        it is quite bloated with many package dependencies. This class is a lightweight
        alternative focused on polygonal shapes and basic flags.

    :seealso: :meth:`pixels_mask` :class:`~machinevisiontoolbox.Image.Polygons` :class:`~machinevisiontoolbox.Image`, :class:`Polygon2`
    """

    filename: str
    _nshapes: int | None

    def __init__(self, filename: str) -> None:
        self.filename = filename

        try:
            if str(filename).startswith(("http://", "https://")):
                import urllib.request

                with urllib.request.urlopen(filename) as response:  # noqa: S310
                    self.data = json.loads(response.read().decode("utf-8"))
            else:
                with open(self.filename, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
        self._nshapes = len(self.data.get("shapes", []))
        self._version = self.data.get("version")

    def __repr__(self) -> str:
        """Return a concise summary with filename and number of shapes."""
        return f"LabelMeReader(filename={self.filename!r}, version={self._version}, nshapes={self._nshapes})"

    @staticmethod
    def _rectangle_points(points: list[list[float]]) -> list[tuple[float, float]]:
        """Convert LabelMe rectangle (2 corner points) to 4 polygon points."""
        if len(points) != 2:
            raise ValueError("Rectangle shape must have exactly 2 points")

        (x1, y1), (x2, y2) = points
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    @property
    def image(self) -> Image | None:
        """Return the image from the LabelMe JSON, or None if not available."""
        import base64
        import binascii
        from io import BytesIO
        from PIL import Image as PILImage, UnidentifiedImageError

        imdata = self.data.get("imageData")
        array = None
        if imdata is not None:
            try:
                # Convert from base64-encoded string to image bytes.
                imbytes = base64.b64decode(imdata)
                imfile = BytesIO(imbytes)

                # Force decode now so image-corruption errors are caught here.
                with PILImage.open(imfile) as pil_image:
                    pil_image.load()
                    array = np.array(pil_image)
            except (binascii.Error, OSError, TypeError, UnidentifiedImageError):
                return None

        if array is not None and array.ndim == 3:
            if array.shape[2] == 3:
                colororder = "RGB"
            elif array.shape[2] == 4:
                colororder = "RGBA"
            else:
                colororder = None
        else:
            colororder = None

        return _make_image(array, colororder=colororder)

    @property
    def flags(self) -> dict:
        """Return file-level flags from the LabelMe JSON as a dictionary."""
        return dict(self.data.get("flags", {}) or {})

    @property
    def shapes(self) -> list[Polygon2]:
        """
        Return list of shape polygons .

        :return: list of :class:`Polygon2` instances for all shapes
        :rtype: list[Polygon2]

        The polygons in the list have additional attributes:
        - ``group_id`` from the shape entry
        - ``flags`` from the shape entry as a dictionary

        """
        polygons: list[Polygon2] = []
        for shape in self.data.get("shapes", []):
            points = shape.get("points", [])
            shape_type = shape.get("shape_type", "polygon")

            if shape_type == "rectangle":
                polygon_points = self._rectangle_points(points)
            else:
                if len(points) < 2:
                    continue
                polygon_points = [tuple(p) for p in points]

            polygon = Polygon2(np.array(polygon_points, dtype=float).T, close=True)
            polygon.group_id = shape.get("group_id")  # type: ignore[attr-defined]
            polygon.flags = dict(shape.get("flags", {}))  # type: ignore[attr-defined]
            polygons.append(polygon)

        return polygons


if __name__ == "__main__":
    from pathlib import Path

    import pytest

    pytest.main(
        [
            str(
                Path(__file__).parent.parent.parent / "tests" / "test_image_sources.py"
            ),
            "-v",
        ]
    )
