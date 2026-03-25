"""
Video, camera, image-collection, and ZIP-archive sources for streaming images.
"""

from __future__ import annotations

import fnmatch
import os
import tempfile
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from typing import Any

# from numpy.lib.arraysetops import isin
from abc import ABC, abstractmethod

import cv2 as cv
import numpy as np
from ansitable import ANSITable, Column

from machinevisiontoolbox import Image
from machinevisiontoolbox.base import convert, iread, mvtb_path_to_datafile

try:
    from rosbags.rosbag1 import Reader as _RosBagReader
    from rosbags.typesys import Stores as _Stores, get_typestore as _get_typestore

    _rosbags_available = True
except ImportError:
    _rosbags_available = False

try:
    import open3d as _o3d

    _open3d_available = True
except ImportError:
    _open3d_available = False


class ImageSource(ABC):
    @abstractmethod
    def __init__():
        pass


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

    .. note:: The value of ``id`` is system specific but generally 0 is the
        first attached video camera.


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
            print("camera read fail, camera is released")
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
        stream = iter(self)
        return next(stream)

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
        stream = iter(self)
        return next(stream)


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


class RosBag(ImageSource):
    """
    Iterate images and point clouds from a ROS 1 bag file.

    :param filename: path to a ``.bag`` file, or an ``http(s)://`` URL
    :type filename: str or Path
    :param release: ROS release name (or unique substring), e.g. ``"noetic"``
    :type release: str
    :param topicfilter: only yield messages from this topic or list of topics;
        ``None`` accepts all topics
    :type topicfilter: str or list of str or None
    :param msgfilter: only yield messages whose type contains this substring
        or matches any entry in the list; ``None`` accepts all types,
        defaults to ``"Image"``
    :type msgfilter: str or list of str or None
    :param dtype: numpy dtype for image pixel data, or a ``{topic: dtype}``
        mapping for per-topic overrides, defaults to ``"uint8"``
    :type dtype: str or dict
    :param colororder: colour-plane order for image data, or a
        ``{topic: colororder}`` mapping for per-topic overrides
    :type colororder: str or dict or None
    :raises ImportError: if the ``rosbags`` package is not installed

    The resulting object is an iterator that yields:

    - :class:`Image` for messages whose type ends in ``Image``
    - ``open3d.geometry.PointCloud`` for ``PointCloud2`` messages
      (requires ``open3d``)
    - the raw deserialised message object for all other types

    Each yielded object carries a ``timestamp`` attribute (ROS nanosecond
    epoch) and a ``topic`` attribute.

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
    """

    filename: str
    release: str
    dtype: np.dtype | str | dict
    colororder: str | dict | None

    def __init__(
        self,
        filename: str,
        release: str = "ROS1_NOETIC",
        topicfilter: str | list[str] | None = None,
        msgfilter: str | list[str] | None = "Image",
        dtype: np.dtype | str | dict = "uint8",
        colororder: str | dict | None = None,
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
        self.reader = None
        self.connections = []
        self.typestore = _resolve_typestore(release)
        self.dtype = dtype
        self.colororder = colororder
        self._tmpfile: str | None = None

    def __repr__(self) -> str:
        return (
            f"RosBag({str(self.filename)!r}, release={self.release!r}, "
            f"topicfilter={self._topic_filter!r}, msgfilter={self._msgfilter!r})"
        )

    def __str__(self) -> str:
        return f"RosBag({str(self.filename)!r})"

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
        """Topic filter — ``None`` accepts all topics."""
        return self._topic_filter

    @topicfilter.setter
    def topicfilter(self, topicfilter: str | list[str] | None) -> None:
        self._topic_filter = topicfilter

    @property
    def msgfilter(self) -> str | list[str] | None:
        """Message-type filter — ``None`` accepts all message types."""
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
            or (isinstance(self._topic_filter, list) and x.topic in self._topic_filter)
            or (isinstance(self._topic_filter, str) and x.topic == self._topic_filter)
        )
        return msg_ok and topic_ok

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

    def _open_reader(self) -> _RosBagReader:
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

        self.reader = _RosBagReader(path)
        self.reader.open()
        self.connections = [x for x in self.reader.connections if self._allowed(x)]
        return self.reader

    def topics(self) -> dict[str, str]:
        """All topics found in the ROS bag.

        :return: mapping of topic name to message type
        :rtype: dict
        """
        reader = self._open_reader()
        topicdict = {conn.topic: conn.msgtype for conn in reader.connections}
        self._close_reader()
        return topicdict

    def traffic(self) -> Counter:
        """Message counts by type across the entire bag.

        :return: mapping of message type to count
        :rtype: :class:`~collections.Counter`
        """
        reader = self._open_reader()
        counts = Counter(conn.msgtype for conn, _ts, _raw in reader.messages())
        self._close_reader()
        return counts

    def print(self) -> None:
        """Print a summary table of topics in the ROS bag.

        For each topic, shows the message type, total message count, and
        whether it passes the current topic and message filters.
        """
        table = ANSITable(
            Column("topic", colalign="<", headalign="^"),
            Column("msgtype", colalign="<", headalign="^"),
            Column("count", colalign=">", headalign="^"),
            Column("allowed", colalign="^", headalign="^"),
            border="thin",
        )
        counts = self.traffic()
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
        print(table)

    def __enter__(self) -> RosBag:
        self._open_reader()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._close_reader()

    def __iter__(self) -> RosBag:
        self._open_reader()
        try:
            for connection, timestamp, rawdata in self.reader.messages(
                connections=self.connections
            ):
                msg = self.typestore.deserialize_ros1(rawdata, connection.msgtype)

                if connection.msgtype.endswith("Image"):
                    if isinstance(self.dtype, dict):
                        dtype = self.dtype.get(connection.topic, "uint8")
                    else:
                        dtype = self.dtype or "uint8"

                    arr = np.frombuffer(msg.data, dtype=dtype).reshape(
                        msg.height, msg.width, -1
                    )

                    if isinstance(self.colororder, dict):
                        colororder = self.colororder.get(connection.topic, None)
                    else:
                        colororder = self.colororder

                    if colororder is None and arr.shape[2] == 3:
                        colororder = "BGR"

                    img = Image(arr, colororder=colororder)
                    img.timestamp = timestamp
                    img.topic = connection.topic
                    yield img

                elif connection.msgtype.endswith("PointCloud2"):
                    if not _open3d_available:
                        raise ImportError(
                            "open3d is required to read PointCloud2 messages. "
                            "Install it with: pip install open3d"
                        )

                    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                        -1, msg.point_step
                    )

                    def extract(field):
                        dtype, size = _PC2_DTYPE[field.datatype]
                        col = raw[:, field.offset : field.offset + size * field.count]
                        return col.view(dtype).flatten().astype(np.float64)

                    pc2_fields = {f.name: f for f in msg.fields}
                    x = extract(pc2_fields["x"])
                    y = extract(pc2_fields["y"])
                    z = extract(pc2_fields["z"])

                    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                    points = np.column_stack([x[mask], y[mask], z[mask]])

                    pcd = _o3d.geometry.PointCloud()
                    pcd.points = _o3d.utility.Vector3dVector(points)
                    pcd.timestamp = timestamp
                    pcd.topic = connection.topic
                    yield pcd

                else:
                    msg.topic = connection.topic
                    yield msg
        finally:
            self._close_reader()


if __name__ == "__main__":
    from pathlib import Path

    import pytest

    pytest.main(
        [str(Path(__file__).parent.parent.parent / "tests" / "test_sources.py"), "-v"]
    )
