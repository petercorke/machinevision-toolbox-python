import os
import cv2 as cv
import zipfile
import numpy as np
import fnmatch
from numpy.core.numeric import _rollaxis_dispatcher
# from machinevisiontoolbox.ImageCore import ImageCoreMixin
# from machinevisiontoolbox.ImageIO import ImageIOMixin
# from machinevisiontoolbox.ImageConstants import ImageConstantsMixin
# from machinevisiontoolbox.ImageProcessing import ImageProcessingMixin
# from machinevisiontoolbox.ImageMorph import ImageMorphMixin
# from machinevisiontoolbox.ImageSpatial import ImageSpatialMixin
# from machinevisiontoolbox.ImageColor import ImageColorMixin
# from machinevisiontoolbox.ImageReshape import ImageReshapeMixin
# from machinevisiontoolbox.ImageFeatures import ImageFeaturesMixin
# from machinevisiontoolbox.ImageBlobs import ImageBlobsMixin
# from machinevisiontoolbox.ImageLineFeatures import ImageLineFeaturesMixin
# from machinevisiontoolbox.ImagePointFeatures import ImagePointFeaturesMixin

from machinevisiontoolbox.base import mvtb_path_to_datafile, iread, convert
from machinevisiontoolbox import Image
from numpy.lib.arraysetops import isin
from abc import ABC, abstractmethod
# class Image(
#             ImageCoreMixin,
#             ImageIOMixin,
#             ImageConstantsMixin,
#             ImageProcessingMixin,
#             ImageMorphMixin,
#             ImageSpatialMixin,
#             ImageColorMixin,
#             ImageReshapeMixin,
#             ImageBlobsMixin,
#             ImageFeaturesMixin,
#             ImageLineFeaturesMixin,
#             ImagePointFeaturesMixin
#             ):
#     pass

class ImageSource(ABC):

    @abstractmethod
    def __init__():
        pass

class VideoFile(ImageSource):
    def __init__(self, filename, **kwargs):
        """
        Image source from video file

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
            - Robotics, Vision & Control for Python, Section 11.1.4, P. Corke, Springer 2023.

        :seealso: :func:`~machinevisiontoolbox.base.imageio.convert`
            `opencv.VideoCapture <https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80>`_
        """
        self.filename = str(mvtb_path_to_datafile('images', filename))

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

    def __iter__(self):
        self.i = 0
        if self.cap is not None:
            self.cap.release()
        self.cap = cv.VideoCapture(self.filename)
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if ret is False:
            self.cap.release()
            raise StopIteration
        else:
            im = convert(frame, **self.args)
            if im.ndim == 3:
                im = Image(im, id=self.i, name=self.filename, colororder='RGB')
            else:
                im = Image(im, id=self.i, name=self.filename)
            self.i += 1
            return im

    def __len__(self):
        return self.nframes

    def __repr__(self):
        return f"VideoFile({os.path.basename(self.filename)}) {self.shape[1]} x {self.shape[0]}, {self.nframes} frames @ {self.fps}fps"

class VideoCamera(ImageSource):

    def __init__(self, id=0, **kwargs):
        """
        Image source from a local video camera

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
            - Robotics, Vision & Control for Python, Section 11.1.3, P. Corke, Springer 2023.

        :seealso: :func:`~machinevisiontoolbox.base.imageio.convert`
            `cv2.VideoCapture <https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80>`_,

        """
        self.id = id
        self.cap = None
        self.args = kwargs
        self.cap = cv.VideoCapture(self.id)

    def __iter__(self):
        self.i = 0
        if self.cap is not None:
            self.cap.release()
        self.cap = cv.VideoCapture(self.id)
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if ret is False:
            self.cap.release()
            raise StopIteration
        else:
            im = convert(frame, **self.args)
            if im.ndim == 3:
                return Image(im, id=self.i, colororder='RGB')
            else:
                return Image(im, id=self.i)
            self.i += 1

    def grab(self):
        """
        Grab single frame from camera

        :return: next frame from the camera
        :rtype: :class:`Image`

        This is an alternative interface to the class iterator.
        """
        stream = iter(self)
        return next(stream)

    def release(self):
        """
        Release the camera

        Disconnect from the local camera, and for cameras with a recording 
        light, turn off that light.
        """
        self.cap.release()

    def __repr__(self):
        return f"VideoCamera({self.id}) {self.width} x {self.height} @ {self.framerate}fps"

    @property
    def width(self):
        """
        Width of video frame

        :return: width of video frame in pixels
        :rtype: int

        :seealso: :meth:`height` :meth:`shape`
        """
        return int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        """
        Height of video frame

        :return: height of video frame in pixels
        :rtype: int
        
        :seealso: :meth:`width` :meth:`shape`
        """
        return int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    @property
    def framerate(self):
        """
        Camera frame rate

        :return: camera frame rate in frames per second
        :rtype: int
        """
        return int(self.cap.get(cv.CAP_PROP_FPS))

    @property
    def shape(self):
        """
        Shape of video frame

        :return: height and width of video frame in pixels
        :rtype: int, int

        :seealso: :meth:`height` :meth:`width`
        """
        return (self.height, self.width)
        
class ImageCollection(ImageSource):

    def __init__(self, filename = None, loop=False, **kwargs):
        """ 
        Image source from a collection of image files

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
            - Robotics, Vision & Control for Python, Section 11.1.2, P. Corke, Springer 2023.

        :seealso: :func:`~machinevisiontoolbox.base.imageio.convert`
            `cv2.imread <https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56>`_
        """
        if filename is not None:
            self.images, self.names = iread(filename)
        self.args = kwargs
        self.loop = loop

    def __getitem__(self, i):

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
                    return Image(im, name=self.names[i], id=i, colororder='RGB')
                else:
                    return Image(im, id=i, name=self.names[i])

    def __iter__(self):
        self.i = 0
        return self

    def __str__(self):
        return '\n'.join([str(f) for f in self.names])

    def __repr__(self):
        return str(self)

    def __next__(self):
        if self.i >= len(self.names):
            if self.loop:
                self.i = 0
            else:
                raise StopIteration
        data = self.images[self.i]
        im = convert(data, **self.args)
        if im.ndim == 3:
            im = Image(im, id=self.i, name=self.names[self.i], colororder='BGR')
        else:
            im = Image(im, id=self.i, name=self.names[self.i])
        self.i += 1
        return im

    def __len__(self):
        return len(self.images)


class ZipArchive(ImageSource):

    def __init__(self, filename, filter=None, loop=False, **kwargs):
        """
        Image source from a zip archive of image files

        :param filename: path to zipfile
        :type filename: str
        :param filter: a Unix shell-style wildcard that specified which files
            to include when iterating over the archive
        :type filter: str
        :param kwargs: options applied to image frames, see :func:`~machinevisiontoolbox.base.imageio.convert`

        The resulting object is an iterator over the image files within the
        zip  archive. The iterator returns :class:`Image` objects and the name of
        the file, within the archive, is given by its ``name`` attribute.

        If the path is not absolute, the video file is first searched for
        relative to the current directory, and if not found, it is searched for
        in the ``images`` folder of the ``mvtb-data`` package, installed as a
        Toolbox dependency.

        Example::

            >>> from machinevisiontoolbox import ZipArchive
            >>> images = ZipArchive('bridge-l.zip')
            >>> len(images)
            >>> for image in images:  # iterate over files
            >>>   # process image

        alternatively::

            >>> image = images[i]  # load i'th file from the archive


        :references: 
            - Robotics, Vision & Control for Python, Section 11.1.2, P. Corke, Springer 2023.

        .. note::  ``filter`` is a Unix style wildcard expression, not a Python
            regexp, so expressions like ``*.png`` would select all PNG files in
            the archive for iteration.

        :seealso: :meth:`open` :func:`~machinevisiontoolbox.base.imageio.convert`
            `cv2.imread <https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56>`_
        """
        filename = mvtb_path_to_datafile('images', filename)
        self.zipfile = zipfile.ZipFile(filename, 'r')
        if filter is None:
            files = self.zipfile.namelist()
        else:
            files = fnmatch.filter(self.zipfile.namelist(), filter)
        self.files = sorted(files)
        self.args = kwargs
        self.loop = loop

    def open(self, name):
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

    def ls(self):
        """
        List all files within the archive to stdout.
        """
        for name in self.zipfile.namelist():
            print(name)

    def __getitem__(self, i):
            im = self._read(i)
            if im.ndim == 3:
                return Image(im, name=self.files[i], id=i, colororder='BGR')
            else:
                return Image(im, id=i, name=self.files[i])

    def __iter__(self):
        self.i = 0
        return self

    def __repr__(self):
        return '\n'.join(self.files)

    def __next__(self):
        if self.i >= len(self.files):
            if self.loop:
                self.i = 0
            else:
                raise StopIteration

        im = self._read(self.i)
        if im.ndim == 3:
            im = Image(im, id=self.i, name=self.files[self.i], colororder='BGR')
        else:
            im = Image(im, id=self.i, name=self.files[self.i])
        self.i += 1
        return im

    def __len__(self):
        return len(self.files)
        
    def _read(self, i):
        data = self.zipfile.read(self.files[i])
        img = cv.imdecode(np.frombuffer(data, np.uint8),  cv.IMREAD_ANYDEPTH | cv.IMREAD_UNCHANGED) 
        return convert(img, **self.args)

class WebCam(ImageSource):
    def __init__(self, url, **kwargs):
        """
        Image source from an internet web camera

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
            - Robotics, Vision & Control for Python, Section 11.1.5, P. Corke, Springer 2023.

        :seealso: :func:`~machinevisiontoolbox.base.imageio.convert`
            `cv2.VideoCapture <https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80>`_
        """
        self.url = url
        self.args = kwargs
        self.cap = None

    def __iter__(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv.VideoCapture(self.url)
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if ret is False:
            self.cap.release()
            raise StopIteration
        else:
            im = convert(frame, **self.args)
            if im.ndim == 3:
                return Image(im, colororder='RGB')
            else:
                return Image(im)

    def grab(self):
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
    def __init__(self, key=None, type='satellite', zoom=18, scale=1, shape=(500, 500), **kwargs):
        """
        Image source from GoogleEarth

        :param key: Google API key, defaults to None
        :type key: str
        :param type: type of map (API ``maptype``): 'satellite' [default], 'roadmap', 'hybrid', and 'terrain'.
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
            - Robotics, Vision & Control for Python, Section 11.1.6, P. Corke, Springer 2023.

        :seealso: :meth:`grab` :func:`~machinevisiontoolbox.base.imageio.convert`
        """


        if key is None:
            self.key = os.getenv('GOOGLE_KEY')
        else:
            self.key = key

        self.type = type
        self.scale = scale
        self.zoom = zoom
        self.shape = shape
        self.args = kwargs

    def grab(self, lat, lon, zoom=None, type=None, scale=None, shape=None, roadnames=False, placenames=False):
        """
        Google map view as an image

        :param lat: lattitude (degrees)
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

        # type: satellite map hybrid terrain roadmap roads
        if type == 'roadmap':
            type = 'roads'
            onlyroads = True
        else:
            onlyroads = False

        # https://developers.google.com/maps/documentation/maps-static/start#URL_Parameters

        # now read the map
        url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size={shape[0]}x{shape[1]}&scale={scale}&format=png&maptype={type}&key={self.key}&sensor=false"

        opturl = []
        
        if roadnames:
            opturl.append('style=feature:road|element:labels|visibility:off')
        if placenames:
            opturl.append('style=feature:administrative|element:labels.text|visibility:off&style=feature:poi|visibility:off')
        
        if onlyroads:
            opturl.extend([
                'style=feature:landscape|element:geometry.fill|color:0x000000|visibility:on',
                'style=feature:landscape|element:labels|visibility:off',
                'style=feature:administrative|visibility:off',
                'style=feature:road|element:geometry|color:0xffffff|visibility:on',
                'style=feature:road|element:labels|visibility:off',
                'style=feature:poi|element:all|visibility:off',
                'style=feature:transit|element:all|visibility:off',
                'style=feature:water|element:all|visibility:off',
                ])

        if len(opturl) > 0:
            url += '&' + '&'.join(opturl)
        data = iread(url)

        if data[0].shape[2] == 4:
            colororder = 'RGBA'
        elif data[0].shape[2] == 3:
            colororder = 'RGB'
        else:
            colororder = None
        im = convert(data[0], **self.args)
        return Image(im, colororder=colororder)

# if __name__ == "__main__":

#     import machinevisiontoolbox as mvtb
#     campus = ImageCollection("campus/*.png")

#     a  = campus[3]
#     print(a)
#     # campus/*.png
#     # traffic_sequence.mpg

#     # v = VideoFile("traffic_sequence.mpg")
    
#     # f = FileCollection("campus/*.png")
#     # print(f)

#     zf = ZipArchive('bridge-l.zip', filter='*02*')
#     print(zf)
#     print(len(zf))
#     # print(zf)
#     print(zf[12])
#     for im in zf:
#         print(im, im.max)
#     pass