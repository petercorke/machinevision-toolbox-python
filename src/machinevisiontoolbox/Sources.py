"""
Video, camera, image-collection, and ZIP-archive sources for streaming images.
"""

from __future__ import annotations

import fnmatch
import json
import threading
from dataclasses import dataclass
from http import client
import io
import os
import tempfile
import time
import urllib.request
import zipfile
from collections import Counter, deque
from tqdm import tqdm
from datetime import datetime, timezone
from typing import Any, Literal
from PIL import Image as PILImage

# from numpy.lib.arraysetops import isin
from abc import ABC, abstractmethod

import cv2 as cv
import numpy as np
from ansitable import ANSITable, Column
from spatialmath import Polygon2

from machinevisiontoolbox import Image, PointCloud
from machinevisiontoolbox.base import convert, iread, mvtb_path_to_datafile

try:
    from rosbags.rosbag1 import Reader as _RosBagReader1
    from rosbags.rosbag2 import Reader as _RosBagReader2
    from rosbags.typesys import Stores as _Stores, get_typestore as _get_typestore

    _rosbags_available = True
except ImportError:
    _rosbags_available = False

try:
    import roslibpy
    import base64  # needed for roslibpy image decoding

    _roslibpy_available = True
except ImportError:
    _roslibpy_available = False

try:
    import open3d as o3d

    _open3d_available = True
except ImportError:
    _open3d_available = False

try:
    import pytesseract as _pytesseract

    _pytesseract_available = True
except ImportError:
    _pytesseract = None
    _pytesseract_available = False


class ImageSource(ABC):
    @abstractmethod
    def __init__():
        pass

    def torch(self, device: str = "cpu", normalize="imagenet") -> "torch.Tensor":
        """
        Convert all images from this source into a single 4D PyTorch tensor.

        :param device: target PyTorch device, e.g. ``"cpu"`` or ``"cuda"``,
            defaults to ``"cpu"``
        :type device: str, optional
        :param normalize: normalisation to apply to each frame; passed directly
            to :meth:`Image.torch`, defaults to ``"imagenet"``
        :type normalize: str, tuple, or None, optional
        :raises ImportError: if PyTorch is not installed
        :raises TypeError: if the source is not finite (no ``__len__``)
        :raises TypeError: if any yielded item is not an :class:`Image`
        :raises ValueError: if any frame has a different shape to the first
        :return: tensor of shape ``(N, C, H, W)``
        :rtype: torch.Tensor

        Images are decoded one at a time and written directly into a
        pre-allocated tensor, so peak memory is one decoded frame plus the
        output tensor — the full source need not reside in memory at once.
        This is particularly useful for :class:`VideoFile` and
        :class:`ZipArchive`.

        Example::

            >>> from machinevisiontoolbox import VideoFile
            >>> t = VideoFile("traffic_sequence.mpg").torch(normalize=None)
            >>> t.shape
            torch.Size([N, 3, H, W])
        """
        try:
            import torch as _torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for torch(). " "Install it with: pip install torch"
            )

        if not hasattr(self, "__len__"):
            raise TypeError(
                f"{type(self).__name__} is not a finite source; "
                "torch() requires a source with a known length"
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
        first_t = first.torch(device=device, normalize=normalize).squeeze(
            0
        )  # (C, H, W)
        expected_shape = first_t.shape

        # pre-allocate the full tensor
        out = _torch.empty((n,) + expected_shape, dtype=first_t.dtype, device=device)
        out[0] = first_t

        for i, img in enumerate(it, start=1):
            if not isinstance(img, Image):
                raise TypeError(
                    f"Frame {i}: expected Image, got {type(img).__name__}; "
                    "use a msgfilter or topicfilter to select only image topics"
                )
            t = img.torch(device=device, normalize=normalize).squeeze(0)
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
            :class:`VideoCamera` or :class:`RosStream` those controls are
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
    The iterator returns :class:`Image` objects where:

    - the ``name`` attribute is the name of the video file
    - the ``id`` attribute is the frame number within the file

    If the path is not absolute, the video file is first searched for
    relative to the current directory, and if not found, it is searched for
    in the ``images`` folder of the ``mvtb-data`` package, installed as a
    Toolbox dependency.

    Example::

        >>> from machinevisiontoolbox import VideoFile
        >>> video = VideoFile("traffic_sequence.mpg")
        >>> len(video)
        >>> for im in video:
        >>>   # process image

    or using a context manager to ensure the file handle is always released::

        >>> with VideoFile("traffic_sequence.mpg") as video:
        ...     for im in video:
        ...         # process image

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
    cap: cv.VideoCapture | None
    i: int

    def __init__(self, filename: str, **kwargs: Any) -> None:

        self.filename = str(mvtb_path_to_datafile("images", filename))

        # get the number of frames in the video
        #  not sure it's always correct
        cap = cv.VideoCapture(self.filename)
        ret, frame = cap.read()
        self.nframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.shape = frame.shape
        self.fps = int(cap.get(cv.CAP_PROP_FPS))
        self.args = kwargs
        cap.release()
        self.cap = None
        self.i = 0

    def __iter__(self) -> VideoFile:
        self.i = 0
        if self.cap is not None:
            self.cap.release()
        self.cap = cv.VideoCapture(self.filename)
        return self

    def __next__(self) -> Image:
        assert self.cap is not None
        ret, frame = self.cap.read()
        if ret is False:
            self.cap.release()
            raise StopIteration
        else:
            im = convert(frame, **self.args)
            if im.ndim == 3:
                im = Image(im, id=self.i, name=self.filename, colororder="RGB")
            else:
                im = Image(im, id=self.i, name=self.filename)
            self.i += 1
            return im

    def __len__(self) -> int:
        return self.nframes

    def __repr__(self) -> str:
        return f"VideoFile({os.path.basename(self.filename)}) {self.shape[1]} x {self.shape[0]}, {self.nframes} frames @ {self.fps}fps"

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
    camera. The iterator returns :class:`Image` objects.

    Example::

        >>> from machinevisiontoolbox import VideoCamera
        >>> video = VideoCamera(0)
        >>> for im in video:
        >>>   # process image

    alternatively::

        >>> img = video.grab()

    or using a context manager to ensure the camera is released::

        >>> with VideoCamera(0) as camera:
        ...     for im in camera:
        ...         # process image

    .. note:: The value of ``id`` is system specific but generally 0 is the
        first attached video camera.  On a Mac running 13.0 (Ventura) or later and an iPhone with
        iOS16 or later, the Contuinity Camera feature allows the phone camera to be used as
        a local video camera, and it will appear as a separate camera with its own ID.


    :references:
        - |RVC3|, Section 11.1.3.

    :seealso: :func:`~machinevisiontoolbox.base.imageio.convert`
        `cv2.VideoCapture <https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80>`_,
    """

    id: int
    cap: cv.VideoCapture
    args: dict
    rgb: bool
    i: int

    def __init__(self, id: int = 0, rgb: bool = True, **kwargs: Any) -> None:

        self.id = id
        self.cap = cv.VideoCapture(id)
        self.args = kwargs
        self.rgb = rgb
        self.i = 0

    def __iter__(self) -> VideoCamera:
        self.i = 0
        self.cap.release()
        self.cap = cv.VideoCapture(self.id)
        return self

    def __next__(self) -> Image:
        ret, frame = self.cap.read()  # frame will be in BGR order
        if ret is False:
            self.cap.release()
            raise StopIteration
        else:
            if self.rgb:
                # RGB required, invert the planes
                im = convert(frame, rgb=True, copy=True, **self.args)
                img = Image(im, id=self.i, colororder="RGB")
            else:
                # BGR required
                im = convert(frame, rgb=False, colororder="BGR", copy=True, **self.args)
                img = Image(im, id=self.i, colororder="BGR")

            self.i += 1
            return img

    def grab(self) -> Image:
        """
        Grab single frame from camera

        :return: next frame from the camera
        :rtype: :class:`Image`

        This is an alternative interface to the class iterator.
        """
        return next(self)

    def release(self) -> None:
        """
        Release the camera

        Disconnect from the local camera, and for cameras with a recording
        light, turn off that light.
        """
        self.cap.release()

    def __repr__(self) -> str:
        backend = self.cap.getBackendName()
        return f"VideoCamera({self.id}) {self.width} x {self.height} @ {self.framerate}fps using {backend}"

    # see https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    properties: dict[str, int] = {
        "brightness": cv.CAP_PROP_BRIGHTNESS,
        "contrast": cv.CAP_PROP_CONTRAST,
        "saturation": cv.CAP_PROP_SATURATION,
        "hue": cv.CAP_PROP_HUE,
        "gain": cv.CAP_PROP_GAIN,
        "exposure": cv.CAP_PROP_EXPOSURE,
        "auto-exposure": cv.CAP_PROP_AUTO_EXPOSURE,
        "gamma": cv.CAP_PROP_GAMMA,
        "temperature": cv.CAP_PROP_TEMPERATURE,
        "auto-whitebalance": cv.CAP_PROP_AUTO_WB,
        "whitebalance-temperature": cv.CAP_PROP_WB_TEMPERATURE,
        "ios:exposure": cv.CAP_PROP_IOS_DEVICE_EXPOSURE,
        "ios:whitebalance": cv.CAP_PROP_IOS_DEVICE_WHITEBALANCE,
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
        return int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """
        Height of video frame

        :return: height of video frame in pixels
        :rtype: int

        :seealso: :meth:`width` :meth:`shape`
        """
        return int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    @property
    def framerate(self) -> int:
        """
        Camera frame rate

        :return: camera frame rate in frames per second
        :rtype: int

        .. note:: If frame rate cannot be determined return -1
        """
        try:
            fps = int(self.cap.get(cv.CAP_PROP_FPS))
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

    def __enter__(self) -> VideoCamera:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class ImageCollection(ImageSource):
    """
    Iterate images from a collection of files

    :param filename: wildcard path to image files
    :type filename: str
    :param loop: Endlessly loop over the files, defaults to False
    :type loop: bool, optional
    :param kwargs: options applied to image frames, see :func:`~machinevisiontoolbox.base.imageio.convert`

    The resulting object is an iterator over the image files that match the
    wildcard description. The iterator returns :class:`Image` objects where
    the ``name`` attribute is the name of the image file

    If the path is not absolute, the video file is first searched for
    relative to the current directory, and if not found, it is searched for
    in the ``images`` folder of the ``mvtb-data`` package, installed as a
    Toolbox dependency.

    Example::

        >>> from machinevisiontoolbox import FileColletion
        >>> images = FileCollection('campus/*.png')
        >>> len(images)
        >>> for image in images:  # iterate over images
        >>>   # process image

    alternatively::

        >>> img = files[i]  # load i'th file from the collection

    or using a context manager::

        >>> with ImageCollection('campus/*.png') as images:
        ...     for image in images:
        ...         # process image

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

    def __init__(
        self, filename: str | None = None, loop: bool = False, **kwargs: Any
    ) -> None:

        if filename is not None:
            self.images, self.names = iread(filename, rgb=True)
        self.args = kwargs
        self.loop = loop
        self.i = 0

    def __getitem__(self, i: int | slice) -> ImageCollection | Image:

        if isinstance(i, slice):
            # slice of a collection -> ImageCollection
            new = self.__class__()
            new.images = self.images[i]
            new.names = self.names[i]
            new.args = self.args
            return new
        else:
            # element of a collection -> Image
            data = self.images[i]
            im = convert(data, **self.args)
            if im.ndim == 3:
                return Image(im, name=self.names[i], id=i, colororder="RGB")
            else:
                return Image(im, id=i, name=self.names[i])

    def __iter__(self) -> ImageCollection:
        self.i = 0
        return self

    def __str__(self) -> str:
        return "\n".join([str(f) for f in self.names])

    def __repr__(self) -> str:
        return str(self)

    def __next__(self) -> Image:
        if self.i >= len(self.names):
            if self.loop:
                self.i = 0
            else:
                raise StopIteration
        data = self.images[self.i]
        im = convert(data, **self.args)
        if im.ndim == 3:
            im = Image(im, id=self.i, name=self.names[self.i], colororder="RGB")
        else:
            im = Image(im, id=self.i, name=self.names[self.i])
        self.i += 1
        return im

    def __len__(self) -> int:
        return len(self.images)

    def __enter__(self) -> ImageCollection:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass


class ImageSequence(ImageSource):
    """
    An in-memory sequence of :class:`Image` objects with interactive display.

    :param images: sequence of images
    :type images: iterable of :class:`Image`

    The sequence is materialised into a list on construction.  Items are
    expected to be :class:`Image` instances; a ``timestamp`` attribute (ROS
    nanosecond epoch) and ``topic`` attribute are used for the display overlay
    when present.

    Example::

        >>> from machinevisiontoolbox import RosBag, ImageSequence
        >>> bag = RosBag("mybag.bag", msgfilter="Image")
        >>> seq = ImageSequence(bag)
        >>> seq.disp()                        # step through one frame at a time
        >>> seq.disp(animate=True, fps=5)     # timed playback

    :seealso: :class:`PointCloudSequence`
    """

    _frames: list

    def __init__(self, images) -> None:
        self._frames = list(images)

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, i: int) -> Image:
        return self._frames[i]

    def __iter__(self):
        return iter(self._frames)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def __repr__(self) -> str:
        return f"ImageSequence({len(self._frames)} frames)"


class ZipArchive(ImageSource):
    """
    Iterate images from a zip archive

    :param filename: path to zipfile
    :type filename: str
    :param filter: a Unix shell-style wildcard that specified which files
        to include when iterating over the archive
    :type filter: str
    :param kwargs: options applied to image frames, see :func:`~machinevisiontoolbox.base.imageio.convert`

    The resulting object is an iterator over the files within the zip  archive. The
    iterator returns the file as a :class:`Image` instance if it is an image (and the
    name of the file, within the archive, is given by its ``name`` attribute), else a
    bytes object containing the file contents.

    If the path is not absolute, the zip file is first searched for
    relative to the current directory, and if not found, it is searched for
    in the ``images`` folder of the ``mvtb-data`` package, installed as a
    Toolbox dependency.

    To read just the image files within the archive, use a ``filter`` such as
    ``"*.png"`` or ``"*.pgm"``.  Note that ``filter`` is a Unix shell style wildcard
    expression, not a Python regexp.

    Example::

        >>> from machinevisiontoolbox import ZipArchive
        >>> images = ZipArchive('bridge-l.zip')
        >>> len(images)
        >>> for image in images:  # iterate over files
        >>>   # process image

    alternatively::

        >>> image = images[i]  # load i'th file from the archive

    or using a context manager to ensure the archive is closed::

        >>> with ZipArchive('bridge-l.zip') as images:
        ...     for image in images:
        ...         # process image

    :references:
        - |RVC3|, Section 11.1.2.

    :seealso: :meth:`open` :func:`~machinevisiontoolbox.base.imageio.convert`
        `cv2.imread <https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56>`_
    """

    zipfile: zipfile.ZipFile
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

        filename = mvtb_path_to_datafile("images", filename)
        self.zipfile = zipfile.ZipFile(filename, "r")
        if filter is None:
            files = [f for f in self.zipfile.namelist() if not f.endswith("/")]
        else:
            files = fnmatch.filter(self.zipfile.namelist(), filter)
        self.files = sorted(files)
        self.args = kwargs
        self.loop = loop
        self.i = 0

    def open(self, name: str):
        """
        Open a file from the archive

        :param name: file name
        :type name: str
        :return: read-only handle to the named file
        :rtype: file object

        Opens the specified file within the archive.  Typically the
        ``ZipArchive`` instance is used as an iterator over the image files
        within, but this method can be used to access non-image data such as
        camera calibration data etc. that might also be contained within the
        archive and is excluded by the ``filter``.
        """
        return self.zipfile.open(name)

    def ls(self) -> None:
        """
        List all files within the archive to stdout.
        """
        for name in self.zipfile.namelist():
            print(name)

    def __getitem__(self, i: int) -> Image | bytes:
        im = self._read(i)
        if isinstance(im, np.ndarray):
            if im.ndim == 3:
                return Image(im, name=self.files[i], id=i, colororder="BGR")
            else:
                return Image(im, id=i, name=self.files[i])
        else:
            # not an image file, just return the contents
            return im

    def __iter__(self) -> ZipArchive:
        self.i = 0
        return self

    def __repr__(self) -> str:
        return "\n".join(self.files)

    def __next__(self) -> Image | bytes:
        if self.i >= len(self.files):
            if self.loop:
                self.i = 0
            else:
                raise StopIteration

        im = self._read(self.i)
        if isinstance(im, np.ndarray):
            if im.ndim == 3:
                im = Image(im, id=self.i, name=self.files[self.i], colororder="BGR")
            else:
                im = Image(im, id=self.i, name=self.files[self.i])
        self.i += 1
        return im

    def __len__(self) -> int:
        return len(self.files)

    def _read(self, i: int) -> np.ndarray | bytes:
        data = self.zipfile.read(self.files[i])
        img = cv.imdecode(
            np.frombuffer(data, np.uint8), cv.IMREAD_ANYDEPTH | cv.IMREAD_UNCHANGED
        )
        if img is None:
            # not an image file, just return the contents
            return data
        else:
            return convert(img, **self.args)

    def __enter__(self) -> ZipArchive:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.zipfile.close()


class WebCam(ImageSource):
    """
    Iterate images from an internet web camera

    :param url: URL of the camera
    :type url: str
    :param kwargs: options applied to image frames, see :func:`~machinevisiontoolbox.base.imageio.convert`

    The resulting object is an iterator over the frames returned from the
    remote camera. The iterator returns :class:`Image` objects.

    Example::

        >>> from machinevisiontoolbox import WebCam
        >>> webcam = WebCam('https://webcam.dartmouth.edu/webcam/image.jpg')
        >>> for image in webcam:  # iterate over frames
        >>>   # process image

    alternatively::

        >>> img = webcam.grab()  # grab next frame

    or using a context manager to ensure the connection is released::

        >>> with WebCam('https://webcam.dartmouth.edu/webcam/image.jpg') as webcam:
        ...     for image in webcam:
        ...         # process image

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
    cap: cv.VideoCapture | None

    def __init__(self, url: str, **kwargs: Any) -> None:

        self.url = url
        self.args = kwargs
        self.cap = None

    def __iter__(self) -> WebCam:
        if self.cap is not None:
            self.cap.release()
        self.cap = cv.VideoCapture(self.url)
        return self

    def __next__(self) -> Image:
        assert self.cap is not None
        ret, frame = self.cap.read()
        if ret is False:
            self.cap.release()
            raise StopIteration
        else:
            im = convert(frame, **self.args)
            if im.ndim == 3:
                return Image(im, colororder="RGB")
            else:
                return Image(im)

    def grab(self) -> Image:
        """
        Grab frame from web camera

        :return: next frame from the web camera
        :rtype: :class:`Image`

        This is an alternative interface to the class iterator.
        """
        if self.cap is None:
            self.cap = cv.VideoCapture(self.url)
        return next(self)

    def __enter__(self) -> WebCam:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


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

    The resulting object has a ``grab`` method that returns :class:`Image`
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

    Example::

        >>> from machinevisiontoolbox import EarthView
        >>> earth = EarthView()  # create an Earth viewer
        >>> image = earth(-27.475722, 153.0285, zoom=17 # make a view
        >>> # process image

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
        :rtype: :class:`Image`

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
        data = iread(url)

        if data[0].shape[2] == 4:
            colororder = "RGBA"
        elif data[0].shape[2] == 3:
            colororder = "RGB"
        else:
            colororder = None
        im = convert(data[0], **self.args)
        return Image(im, colororder=colororder)


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
            "Install it with: pip install rosbags"
        )
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
class RosMessage:
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


class RosStream(ImageSource):
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
    :param output: output mode, ``"image"`` yields :class:`Image` and
        ``"message"`` yields :class:`RosMessage`, defaults to ``"image"``
    :type output: str, optional
    :param blocking: if ``True`` (default) :meth:`__next__` blocks until a new
        frame arrives; if ``False`` it returns the most recently received frame
        immediately (or blocks until the very first frame is available)
    :type blocking: bool, optional
    :param rgb: if ``True`` (default) return RGB images; if ``False`` return BGR
    :type rgb: bool, optional
    :raises ImportError: if the ``roslibpy`` package is not installed

    In subscribe mode, the object is an iterator that yields :class:`Image`
    instances (``output="image"``) or :class:`RosMessage` instances
    (``output="message"``) as they arrive from the topic.  Use it as a context
    manager to ensure the connection is always closed::

        >>> with RosStream("/camera/image/compressed", host="192.168.1.10") as stream:
        ...     for img in stream:
        ...         img.disp()

    alternatively grab a single frame::

        >>> stream = RosStream("/camera/image/compressed")
        >>> img = stream.grab()
        >>> stream.release()

    For publish-only use, disable subscription setup and call :meth:`publish`::

        >>> pub = RosStream("/cmd_topic", message="std_msgs/String", subscribe=False)
        >>> pub.publish({"data": "hello"})
        >>> pub.release()

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
    ) -> None:

        if not _roslibpy_available:
            raise ImportError(
                "roslibpy is required for ROS streaming support. "
                "Install it with: pip install roslibpy"
            )

        self.host = host
        self.topic = topic
        self.message = message
        self.port = port
        self._subscribe = subscribe
        self._output = output
        self._blocking = blocking
        self._rgb = rgb
        self._compressed = "Compressed" in message
        if output not in {"image", "message"}:
            raise ValueError("output must be 'image' or 'message'")
        if output == "image" and not self._is_image_message_type(message):
            raise ValueError(
                "output='image' requires sensor_msgs/Image or "
                "sensor_msgs/CompressedImage"
            )
        self._latest_frame: np.ndarray | None = None
        self._latest_message: RosMessage | None = None
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
            self._latest_message = RosMessage(
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
                frame = cv.cvtColor(
                    frame, cv.COLOR_RGB2BGR if channels == 3 else cv.COLOR_RGBA2BGR
                )
            elif encoding == "mono8":
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
            elif encoding == "mono16":
                # Preserve full 16-bit depth precision.
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        else:
            # Compressed sensor_msgs/CompressedImage: JPEG or PNG bytes
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)  # BGR ndarray
        if frame is None:
            return
        self.i += 1
        self._latest_frame = frame
        if self._frame_event is not None:
            self._frame_event.set()

    def __iter__(self) -> RosStream:
        if not self._subscribe:
            raise TypeError("RosStream is publish-only (subscribe=False)")
        self.i = 0
        return self

    def __next__(self) -> Image | RosMessage:
        if not self._subscribe or self._frame_event is None:
            raise TypeError("RosStream is publish-only (subscribe=False)")
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
        if self._rgb:
            im = convert(self._latest_frame, rgb=True, copy=True)
            img = Image(im, id=self.i, colororder="RGB")
        else:
            im = convert(self._latest_frame, rgb=False, copy=True)
            img = Image(im, id=self.i, colororder="BGR")
        if self._latest_timestamp is None:
            raise StopIteration
        img.timestamp = int(self._latest_timestamp)
        img.topic = self.topic
        return img

    def publish(self, msg: dict) -> None:
        """
        Publish a message on the configured ROS topic.

        :param msg: serialisable ROS message dictionary
        :type msg: dict

        The topic is advertised lazily on first publish.
        """
        if not self._advertised:
            self._topic.advertise()
            self._advertised = True
        self._topic.publish(msg)

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

    def grab(self) -> Image | RosMessage:
        """
        Grab a single frame from the ROS topic.

        :return: next frame from the topic
        :rtype: :class:`Image` or :class:`RosMessage`

        This is an alternative to using the iterator interface.
        """
        if not self._subscribe:
            raise TypeError("RosStream is publish-only (subscribe=False)")
        return next(self)

    def __enter__(self) -> RosStream:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __repr__(self) -> str:
        if self._subscribe:
            mode = "blocking" if self._blocking else "latest"
        else:
            mode = "publish"
        return (
            f"RosStream({self.topic!r}, host={self.host!r}, port={self.port}, {mode})"
        )

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


class SyncRosStreams:
    """
    Synchronise multiple :class:`RosStream` objects by timestamp.

    :param streams: two or more ROS streams to synchronise
    :type streams: list of :class:`RosStream`
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

    Example::

        >>> rgb = RosStream("/camera/color/image_raw/compressed")
        >>> depth = RosStream("/camera/depth/image_rect_raw/compressed")
        >>> with SyncRosStreams([rgb, depth], tolerance=0.03) as sync:
        ...     for rgb_im, depth_im in sync:
        ...         # process aligned pair
        ...         pass

    **Interaction with ``RosStream`` blocking mode**

    For time-step synchronisation, set each input stream to ``blocking=True``.
    This gives one newly arrived frame per stream pull and avoids repeated
    reuse of stale frames.

    If any stream uses ``blocking=False``, repeated calls can return the same
    latest frame multiple times, which may lead to duplicate tuples, tighter
    polling loops, and weaker one-sample-per-time-step behaviour.  It remains
    useful for low-latency latest-state fusion, but is less suitable for
    strict frame-by-frame synchronisation.
    """

    streams: list[RosStream]
    tolerance: float
    _tol_ns: int

    def __init__(self, streams: list[RosStream], tolerance: float = 0.02) -> None:
        if len(streams) < 2:
            raise ValueError("SyncRosStreams requires at least two streams")
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

    def __iter__(self) -> SyncRosStreams:
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

    def __enter__(self) -> SyncRosStreams:
        for stream in self.streams:
            stream.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for stream in self.streams:
            stream.__exit__(exc_type, exc, tb)

    def __str__(self) -> str:
        topics = ", ".join(getattr(stream, "topic", "<unknown>") for stream in self.streams)
        return f"SyncRosStreams([{topics}])"

    def __repr__(self) -> str:
        return (
            f"SyncRosStreams(nstreams={len(self.streams)}, "
            f"tolerance={self.tolerance:.6f}s)"
        )


class RosBag(ImageSource):
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
    :param colororder: override the colour-plane order for all image, or by topic
        ``{topic: colororder}``
    :type colororder: str or dict or None
    :raises ImportError: if the ``rosbags`` package is not installed

    The resulting object is an iterator that yields:

    - :class:`Image` for messages whose type ends in ``Image``, which includes ``CompressedImage``.
    - :class:`PointCloud` for ``PointCloud2`` messages (requires ``open3d``)
    - the raw deserialised message object for all other types

    Each yielded object carries a ``timestamp`` attribute (ROS nanosecond
    epoch from the message header) and a ``topic`` attribute (the topic on which it was published).

    **Usage modes**

    *Implicit* — iterating directly over the object opens and closes the bag
    file automatically around the loop::

        >>> from machinevisiontoolbox import RosBag
        >>> for img in RosBag("mybag.bag", release="noetic"):
        ...     img.disp()

    *Explicit context manager* — use a ``with`` statement when you need to
    make multiple passes over the bag, call helper methods such as
    :meth:`topics` or :meth:`print`, or simply want a guaranteed close even
    if an exception is raised::

        >>> bag = RosBag("mybag.bag", release="noetic", msgfilter=None)
        >>> with bag:
        ...     bag.print()                     # inspect topics
        ...     for msg in bag:                 # iterate messages
        ...         print(msg.topic, msg.timestamp)

    .. note::
        ``filename`` may be an ``http://`` or ``https://`` URL, in which case
        the bag file is downloaded to a temporary file on first use and that
        file is reused for the lifetime of the ``RosBag`` object.  The
        temporary file is deleted automatically when the object is garbage
        collected or when the script exits.

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
    dtype: np.dtype | str | dict
    colororder: str | dict | None
    verbose: bool

    def __init__(
        self,
        filename: str,
        release: str = "ROS1_NOETIC",
        topicfilter: str | list[str] | None = None,
        msgfilter: str | list[str] | None = "Image",
        dtype: np.dtype | str | dict | None = None,
        colororder: str | dict | None = None,
        verbose: bool = False,
    ) -> None:
        if not _rosbags_available:
            raise ImportError(
                "rosbags is required for ROS bag support. "
                "Install it with: pip install rosbags"
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
        self._tmpfile: str | None = None
        self._is_ros2: bool = False

    def __repr__(self) -> str:
        return (
            f"RosBag({str(self.filename)!r}, release={self.release!r}, "
            f"topicfilter={self._topic_filter!r}, msgfilter={self._msgfilter!r})"
        )

    def __str__(self) -> str:
        return f"RosBag({str(self.filename)!r})"

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
        if self._tmpfile is not None:
            from pathlib import Path as _Path

            _Path(self._tmpfile).unlink(missing_ok=True)
            self._tmpfile = None

    def _close_reader(self) -> None:
        if self.reader is not None:
            self.reader.close()
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

        self._is_ros2 = _BagPath(path).is_dir()
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

    def __enter__(self) -> RosBag:
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

    def __iter__(self) -> RosBag:
        self._open_reader()
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
                    msg = _deser(rawdata, connection.msgtype)
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
                    # 1. Open the byte stream
                    # Pillow identifies the container (JPEG/PNG/etc.) automatically
                    try:
                        with PILImage.open(io.BytesIO(msg.data)) as pil_img:

                            # 2. Map PIL mode to color order
                            # 'L' is 8-bit mono, 'I' is 32-bit int, 'F' is 32-bit float
                            mode_to_order = {
                                "RGB": "rgb",
                                "RGBA": "rgba",
                                "BGR": "bgr",
                                "L": None,
                                "I": None,
                                "F": None,
                            }

                            # Default to 'rgb' if mode is unusual, but try to be specific
                            colororder = mode_to_order.get(pil_img.mode, "rgb")

                            # 3. Handle the "BGR-in-JPEG" legacy quirk
                            # In 2011-era ROS, BGR images were often shoved into JPEGs.
                            # PIL will decode them as RGB, but the channels remain swapped.
                            fmt_str = getattr(msg, "format", "").lower()
                            if "bgr" in fmt_str and colororder == "rgb":
                                colororder = "bgr"

                            # 4. Convert to numpy array
                            # This handles bit-depth automatically (uint8 for RGB, float32 for F, etc.)
                            img_array = np.array(pil_img)

                    except Exception as e:
                        # Re-raise with context for robotics debugging
                        raise RuntimeError(
                            f"Failed to decode compressed image: {e}"
                        ) from e

                    # check for a per-topic override
                    if isinstance(self.colororder, dict):
                        colororder = self.colororder.get(connection.topic, None)
                    else:
                        colororder = self.colororder

                    img = Image(img_array, colororder=colororder)
                    img.timestamp = self._stamp_to_ns(msg.header.stamp)
                    img.topic = connection.topic
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

                    img = Image(img_array, colororder=colororder)
                    img.timestamp = self._stamp_to_ns(msg.header.stamp)
                    img.topic = connection.topic
                    yield img

                elif connection.msgtype.endswith("PointCloud2"):
                    # Converts a sensor_msgs/PointCloud2 message to an Open3D PointCloud.
                    # Handles both colored and uncolored clouds.
                    if not _open3d_available:
                        raise ImportError(
                            "open3d is required to read PointCloud2 messages. "
                            "Install it with: pip install open3d"
                        )

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
                    pc.timestamp = self._stamp_to_ns(msg.header.stamp)
                    pc.topic = connection.topic
                    yield pc

                else:
                    # any other sort of message, just yield the deserialized object with timestamp and topic attributes
                    msg.timestamp = timestamp
                    msg.topic = connection.topic
                    yield msg
        finally:
            self._close_reader()


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

    Example::

        >>> from machinevisiontoolbox import RosBag, PointCloudSequence
        >>> bag = RosBag("mybag.bag", msgfilter="PointCloud2")
        >>> seq = PointCloudSequence(bag)
        >>> seq.disp()                       # step through one cloud at a time
        >>> seq.disp(animate=True, fps=10)   # timed playback

    :seealso: :class:`ImageSequence`
    """

    _clouds: list

    def __init__(self, clouds) -> None:
        self._clouds = list(clouds)

    def __len__(self) -> int:
        return len(self._clouds)

    def __getitem__(self, i: int) -> PointCloud:
        return self._clouds[i]

    def __iter__(self):
        return iter(self._clouds)

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
        import open3d.visualization.gui as gui
        import open3d.visualization.rendering as rendering

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
    :param colororder: colour plane order for multi-channel tensors,
        e.g. ``"RGB"`` or ``"BGR"``, defaults to None
    :type colororder: str, optional
    :param logits: if True, take argmax over the channel dimension to
        convert per-class logits to a class label image, defaults to False
    :type logits: bool, optional
    :param dtype: data type for output image arrays, e.g. ``np.uint8`` or
        ``np.float32``; if None, dtype is inferred from the tensor data,
        defaults to None
    :type dtype: numpy dtype or None, optional

    Example::

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
    ):
        """
        Initialize TensorStack from a batch tensor.

        :param tensor: batch tensor of shape ``(B, C, H, W)`` or ``(B, H, W)``
        :param colororder: colour plane order for display/export
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
        self._batch_size = self._array.shape[0]

    def __len__(self) -> int:
        """Return number of images in the batch."""
        return self._batch_size

    def __iter__(self):
        """Iterate over images as views into the batch."""
        for i in range(self._batch_size):
            yield self[i]

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

        return Image(frame, colororder=self._colororder, dtype=self._dtype)

    def __repr__(self) -> str:
        """Return string representation."""
        shape = self._array.shape
        dtype = self._array.dtype
        return (
            f"TensorStack({self._batch_size} frames, shape {shape[1:]}, dtype {dtype})"
        )


class LabelMe:
    """
    Read annotations from a LabelMe JSON file.

    :param filename: path to LabelMe JSON file
    :type filename: str

    The reader returns three values:

    - an :class:`Image`
    - a list of :class:`Polygon2` instances for all shapes
    - file-level ``flags`` as a dictionary

    For each returned polygon, additional attributes are attached:

    - ``group_id`` from the shape entry
    - ``flags`` from the shape entry as a dictionary

    Rectangle shapes are converted to 4-corner polygons.

    Example::

        >>> from machinevisiontoolbox import LabelMe
        >>> image, polygons, flags = LabelMe("scene.json").read()
        >>> len(polygons)
    """

    filename: str

    def __init__(self, filename: str) -> None:
        self.filename = filename

    def __repr__(self) -> str:
        """Return a concise summary with filename and number of shapes."""
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            shapes = data.get("shapes", [])
            if not isinstance(shapes, list):
                nshapes = 0
            else:
                nshapes = len(shapes)
            count = str(nshapes)
        except (OSError, json.JSONDecodeError, TypeError):
            count = "?"

        return f"LabelMe(filename={self.filename!r}, nshapes={count})"

    @staticmethod
    def _rectangle_points(points: list[list[float]]) -> list[tuple[float, float]]:
        """Convert LabelMe rectangle (2 corner points) to 4 polygon points."""
        if len(points) != 2:
            raise ValueError("Rectangle shape must have exactly 2 points")

        (x1, y1), (x2, y2) = points
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def read(self) -> tuple[Image, list[Polygon2], dict]:
        """
        Read LabelMe JSON and return image, polygons and file flags.

        :raises ImportError: if the ``labelme`` package is not installed
        :raises ValueError: if required LabelMe image metadata is missing
        :return: ``(image, polygons, flags)``
        :rtype: tuple
        """
        try:
            from labelme import utils as _labelme_utils
            from labelme.label_file import LabelFile
        except ImportError:
            raise ImportError(
                "labelme is required for LabelMe support. "
                "Install it with: pip install labelme"
            )

        label = LabelFile(filename=self.filename)

        if label.imageData is not None:
            array = _labelme_utils.img_data_to_arr(label.imageData)
        else:
            if label.imagePath is None:
                raise ValueError("LabelMe JSON must include imageData or imagePath")
            image_path = os.path.join(os.path.dirname(self.filename), label.imagePath)
            image_data = LabelFile.load_image_file(image_path)
            array = _labelme_utils.img_data_to_arr(image_data)

        image = Image(array, colororder="RGB")

        polygons: list[Polygon2] = []
        for shape in label.shapes:
            points = shape.get("points", [])
            shape_type = shape.get("shape_type", "polygon")

            if shape_type == "rectangle":
                polygon_points = self._rectangle_points(points)
            else:
                if len(points) < 2:
                    continue
                polygon_points = [tuple(p) for p in points]

            polygon = Polygon2(polygon_points, close=True)
            polygon.group_id = shape.get("group_id")
            polygon.flags = dict(shape.get("flags", {}))
            polygons.append(polygon)

        flags = dict(getattr(label, "flags", {}) or {})
        return image, polygons, flags


if __name__ == "__main__":
    from pathlib import Path

    import pytest

    pytest.main(
        [str(Path(__file__).parent.parent.parent / "tests" / "test_sources.py"), "-v"]
    )
