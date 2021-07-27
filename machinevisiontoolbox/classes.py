import os
import cv2 as cv
import zipfile
import numpy as np
import fnmatch
from numpy.core.numeric import _rollaxis_dispatcher
from machinevisiontoolbox.ImageCore import ImageCoreMixin
from machinevisiontoolbox.ImageIO import ImageIOMixin
from machinevisiontoolbox.ImageConstants import ImageConstantsMixin
from machinevisiontoolbox.ImageProcessing import ImageProcessingMixin
from machinevisiontoolbox.ImageMorph import ImageMorphMixin
from machinevisiontoolbox.ImageSpatial import ImageSpatialMixin
from machinevisiontoolbox.ImageColor import ImageColorMixin
from machinevisiontoolbox.ImageReshape import ImageReshapeMixin
from machinevisiontoolbox.ImageFeatures import ImageFeaturesMixin
from machinevisiontoolbox.ImageBlobs import ImageBlobsMixin
from machinevisiontoolbox.ImageLineFeatures import ImageLineFeaturesMixin
from machinevisiontoolbox.ImagePointFeatures import ImagePointFeaturesMixin

from machinevisiontoolbox.base import mvtb_path_to_datafile, iread, convert

class Image(
            ImageCoreMixin,
            ImageIOMixin,
            ImageConstantsMixin,
            ImageProcessingMixin,
            ImageMorphMixin,
            ImageSpatialMixin,
            ImageColorMixin,
            ImageReshapeMixin,
            ImageBlobsMixin,
            ImageFeaturesMixin,
            ImageLineFeaturesMixin,
            ImagePointFeaturesMixin
            ):
    pass

class VideoFile:
    def __init__(self, filename, **kwargs):
        """
        Image source from video file

        :param filename: Path to video file
        :type filename: str
        :param kwargs: options applied to image frames, see :func:`~machinevisiontoolbox.base.imageio.convert`

        The resulting object is an iterator over the frames of the video file.
        The iterator returns ``Image`` objects.

        If the path is not absolute the video file is first searched for
        relative to the current directory, and if not found, it is searched for
        in the ``images`` folder of the Toolbox installation.

        Example::

            >>> from machinevisiontoolbox import VideoFile
            >>> video = VideoFile("traffic_sequence.mpg")
            >>> for im in video:
            >>>   # process image

        :seealso: `cv2.VideoCapture <https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80>`_, :func:`~machinevisiontoolbox.base.imageio.convert`
        """
        self.filename = str(mvtb_path_to_datafile(filename, folder='images'))

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
                return Image(im, id=self.i, name=self.filename, colororder='RGB')
            else:
                return Image(im, id=self.i, name=self.filename)
            self.i += 1

    def __len__(self):
        return self.nframes

    def __repr__(self):
        return f"VideoFile({os.path.basename(self.filename)}) {self.shape[1]} x {self.shape[0]}, {self.nframes} frames @ {self.fps}fps"

class VideoCamera:

    def __init__(self, id=0, **kwargs):
        """
        Image source from a local video camera

        :param id: Identity of local camera
        :type id: int
        :param kwargs: options applied to image frames, see :func:`~machinevisiontoolbox.base.imageio.convert`

        The resulting object is an iterator over the frames from the video
        camera. The iterator returns ``Image`` objects.

        Example::

            >>> from machinevisiontoolbox import VideoCamera
            >>> video = VideoCamera(0)
            >>> for im in video:
            >>>   # process image

        alternatively::

            >>> im = video.grab()

        .. note:: The value of ``id`` is system specific but generally 0 is the
            first attached video camera.

        :seealso: `cv2.VideoCapture <https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80>`_, :func:`~machinevisiontoolbox.base.imageio.convert`

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
        Grab frame from camera

        :return: next frame from the camera
        :rtype: Image instance

        This is an alternative interface to the class iterator.
        """
        stream = iter(self)
        return next(stream)

    def release(self):
        self.cap.release()

    def __repr__(self):
        width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv.CAP_PROP_FPS))
        return f"VideoCamera({self.id}) {width} x {height} @ {fps}fps"

class FileCollection:

    def __init__(self, filename, **kwargs):
        """
        Image source from a collection of image files

        :param filename: Path image files, with wildcard
        :type filename: str
        :param kwargs: options applied to image frames, see :func:`~machinevisiontoolbox.base.imageio.convert`

        The resulting object is an iterator over the frames that match the 
        wildcard description. The iterator returns ``Image`` objects.

        If the path is not absolute the files are first searched for
        relative to the current directory, and if not found, it is searched for
        in the ``images`` folder of the Toolbox installation.

        Example::

            >>> from machinevisiontoolbox import FileColletion
            >>> files = FileCollection('campus/*.png')
            >>> for im in files:  # iterate over files
            >>>   # process image

        alternatively::

            >>> im = files[i]  # load i'th file from the collection


        :seealso: `cv2.imread <https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56>`_, :func:`~machinevisiontoolbox.base.imageio.convert`
        """
        self.files = iread(filename)
        self.args = kwargs

    def __getitem__(self, i):
            data = self.files[i]
            im = convert(data[0], **self.args)
            if im.ndim == 3:
                return Image(im, name=data[1], id=i, colororder='BGR')
            else:
                return Image(im, id=i, name=data[1])

    def __iter__(self):
        self.i = 0
        return self

    def __repr__(self):
        return '\n'.join([f[1] for f in self.files])

    def __next__(self):
        if self.i >= len(self.files):
            raise StopIteration
        else:
            data = self.files[self.i]
            im = convert(data[0], **self.args)
            if im.ndim == 3:
                im = Image(im, id=self.i, name=data[1], colororder='BGR')
            else:
                im = Image(im, id=self.i, name=data[1])
            self.i += 1
            return im

    def __len__(self):
        return len(self.files)


class ZipArchive:

    def __init__(self, filename, pattern=None, **kwargs):
        """
        Image source from a collection within a single zip archive

        :param filename: Path to zipfile
        :type filename: str
        :param pattern: a Unix shell-style wildcard that specified which files
        to include from the archive
        :type pattern: str
        :param kwargs: options applied to image frames, see :func:`~machinevisiontoolbox.base.imageio.convert`

        The resulting object is an iterator over the image files within the
        zip  archive. The iterator returns ``Image`` objects.

        If the path is not absolute the zip archive is first searched for
        relative to the current directory, and if not found, it is searched for
        in the ``images`` folder of the Toolbox installation.

        Example::

            >>> from machinevisiontoolbox import ZipArchive
            >>> files = ZipArchive('bridge-l.zip')
            >>> for im in files:  # iterate over files
            >>>   # process image

        alternatively::

            >>> im = files[i]  # load i'th file from the archive

        .. note::  ``pattern`` is a Unix style wildcard expression, not a Python
            regexp, so expressions like ``*.png`` would select all PNG files in
            the archive.

        :seealso: `cv2.imread <https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56>`_, :func:`~machinevisiontoolbox.base.imageio.convert`
        """
        filename = mvtb_path_to_datafile(filename, folder='images')
        self.zipfile = zipfile.ZipFile(filename, 'r')
        if pattern is None:
            files = self.zipfile.namelist()
        else:
            files = fnmatch.filter(self.zipfile.namelist(), pattern)
        self.files = sorted(files)
        self.args = kwargs

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
            raise StopIteration
        else:
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

class WebCam:
    def __init__(self, url, **kwargs):
        """
        Image source from an internet web camera

        :param url: URL of the camera
        :type url: str
        :param kwargs: options applied to image frames, see :func:`~machinevisiontoolbox.base.imageio.convert`

        The resulting object is an iterator over the frames returned from the
        remove camera. The iterator returns ``Image`` objects.

        Example::

            >>> from machinevisiontoolbox import WebCam
            >>> webcam = WebCam('https://webcam.dartmouth.edu/webcam/image.jpg')
            >>> for im in webcam:  # iterate over frames
            >>>   # process image

        alternatively::

            >>> im = webcam.grab()  # grab next frame

        :seealso: `cv2.VideoCapture <https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a188e80>`_, :func:`~machinevisiontoolbox.base.imageio.convert`
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
        :rtype: Image instance

        This is an alternative interface to the class iterator.
        """
        stream = iter(self)
        return next(stream)

# dartmouth = WebCam('https://webcam.dartmouth.edu/webcam/image.jpg')

class EarthView:
    def __init__(self, key=None, type='satellite', zoom=18, scale=1, shape=(500, 500), **kwargs):
        """
        Image source from GoogleEarth

        :param filename: Path image files, with wildcard
        :type filename: str
        :param kwargs: options applied to image frames, see :func:`~machinevisiontoolbox.base.imageio.convert`

        The resulting object has a ``grab`` method that returns ``Image`` objects for specified
        position on the planet.

        Example::

            >>> from machinevisiontoolbox import EarthView
            >>> earth = EarthView()  # create an Earth viewer
            >>> im = earth(-27.475722, 153.0285, zoom=17 # make a view
            >>> # process image

        .. warning:: You must have a Google account and a valid key to access
            this service.
        
        .. note:: If the key is not passed in, a value is sought from the 
            environment variable ``GOOGLE_KEY``

        :seealso: :func:`~machinevisiontoolbox.base.imageio.convert`
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

    def grab(self, lat, lon, type=None, zoom=None, scale=None, shape=None, roadnames=False, placenames=False):
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
        :param scale: pixel zoom, defaults to None
        :type scale: int, 1 or 2, optional
        :param shape: image shape, defaults to None
        :type shape: tuple (width, height), optional
        :param roadnames: show roadnames, defaults to False
        :type roadnames: bool, optional
        :param placenames: show place names, defaults to False
        :type placenames: bool, optional
        :return: Google map view
        :rtype: Image instance

        If parameters are not given the values provided to the constructor
        are taken as defaults.

        .. note:  ``zoom`` varies from 1 (whole world) to a maximum of 18.

        .. note:: The image may have an alpha plane.
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

if __name__ == "__main__":

    # campus/*.png
    # traffic_sequence.mpg

    # v = VideoFile("traffic_sequence.mpg")
    
    # f = FileCollection("campus/*.png")
    # print(f)

    zf = ZipArchive('/Users/corkep/Dropbox/code/machinevision-toolbox-python/machinevisiontoolbox/images/bridge-l.zip', pattern='*02*')
    print(zf)
    print(len(zf))
    # print(zf)
    print(zf[12])
    for im in zf:
        print(im, im.max)
    pass