#!/usr/bin/env python
"""
Images class
@author: Dorian Tsai
@author: Peter Corke
"""

from pathlib import Path
import os.path
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib as mpl
from spatialmath.base import isscalar, islistof
# import spatialmath.base.argcheck as argcheck

from machinevisiontoolbox.IImage import IImage
from machinevisiontoolbox.ImageProcessingBase import ImageProcessingBaseMixin
from machinevisiontoolbox.ImageProcessingMorph import ImageProcessingMorphMixin
from machinevisiontoolbox.ImageProcessingKernel import \
    ImageProcessingKernelMixin
from machinevisiontoolbox.ImageProcessingColor import ImageProcessingColorMixin
from machinevisiontoolbox.blobs import BlobFeaturesMixin
from machinevisiontoolbox.image_feature import FeaturesMixin
from machinevisiontoolbox.features2d import Features2DMixin
from machinevisiontoolbox.reshape import ReshapeMixin
from machinevisiontoolbox.base.imageio import idisp, iread, iwrite, convert
import urllib
import xml.etree.ElementTree as ET

class Video:
    def __init__(self, filename, **kwargs):
        self.filename = filename

        # get the number of frames in the video
        #  not sure it's always correct
        cap = cv.VideoCapture(filename)
        ret, frame = cap.read()
        self.nframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.fps = int(cap.get(cv.CAP_PROP_FPS))
        self.args = kwargs
        cap.release()
        self.cap = None


    def __iter__(self):
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
                return Image(im, colororder='RGB')
            else:
                return Image(im)

    def __len__(self):
        return self.nframes

class Camera:
    def __init__(self, id, **kwargs):
        self.id = id
        self.cap = None
        self.args = kwargs


    def __iter__(self):
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
                return Image(im, colororder='RGB')
            else:
                return Image(im)

class Files:
    def __init__(self, filename, **kwargs):
        self.files = iread(filename)
        self.args = kwargs

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.files):
            raise StopIteration
        else:
            data = self.files[self.i]
            self.i += 1
            im = convert(data[0], **self.args)
            if im.ndim == 3:
                return Image(im, filenames=data[1], colororder='BGR')
            else:
                return Image(im, filenames=data[1])

    def __len__(self):
        return len(self.files)

class WebCam:
    def __init__(self, url, **kwargs):
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
        stream = iter(self)
        return next(stream)

`   # dartmouth = WebCam('https://webcam.dartmouth.edu/webcam/image.jpg')

class EarthView:
    def __init__(self, key=None, type='satellite', zoom=18, scale=1, shape=(500, 500), **kwargs):
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

class ImageCoreMixin:

    def __init__(self,
                 arg=None,
                 colororder=None,
                 checksize=True,
                 checktype=True,
                 logical=False,
                 copy=False,
                 shape=None,
                 filenames=None,
                 **kwargs):
        """
        An image class for MVT

            :param arg: image
            :type arg: Image, list of Images, numpy array, list of numpy arrays,
            filename string, list of filename strings
            :param colororder: order of color channels ('BGR' or 'RGB')
            :type colororder: string or None
            :param checksize: if reading a sequence, check all are the same size
            :type checksize: bool
            :param iscolor: True if input images are color
            :type iscolor: bool

        :seealso: :func:`~machinevisiontoolbox.base.iread`
        """

        # TODO:
        #  - docstring
        #  - copy option
        #  - optimize constructor logic, is it all needed
        
        if isinstance(arg, str):
            raise DeprecationWarning('Use Image.Read()')
        
        self._imlist = []
        self._filenamelist = filenames
        self._colordict = None
        
        if arg is None:
            return

        colordict = None
        if colororder is not None:
            if not isinstance(colororder, str):
                raise ValueError('color order must be a string')
            if ':' in colororder:
                colororder = colororder.split(':')
            colordict = {}
            for i, color in enumerate(colororder):
                colordict[color] = i
        
        if isinstance(arg, np.ndarray):
            # is an actual image or sequence of images compounded into
            # single ndarray
            # make this into a list of images
            arg = Image.getimage(arg)  # convert to OpenCV friendly type

            if shape is not None:
                arg = arg.reshape(shape)

            if arg.ndim == 2:
                # single (W,H)
                if colororder is not None:
                    raise ValueError('cannot specify color order for 2D image')
                self._imlist = [arg]

            elif arg.ndim == 3:
                # could be single (W,H,3) -> 1 colour image
                # or (W,H,N) -> N grayscale images
                if colororder is not None:
                    # color image, not a sequence
                    self._imlist = [arg]
                else:
                    # greyscale image sequence
                    self._imlist = [arg[..., i] for i in range(arg.shape[2])]

            elif arg.ndim == 4:
                # color image sequence
                if colororder is None:
                    raise ValueError('must specify color order for 4D image')
                self._imlist = [arg[..., i] for i in range(arg.shape[3])]

            if filenames is not None:
                if self.nimages == 1:
                    # single image
                    if isinstance(filenames, str):
                        filenames = [filenames]
                    elif len(filenames) == 1:
                        pass
                    else:
                        raise ValueError('filename must be a string or list of length 1')
            
                else:
                    # image sequence
                    if isinstance(filenames, str):
                        filenames = [f"{filenames}[{i}]" for i in range(self.nimages)]
                    elif len(filenames) == self.nimages:
                        pass
                    else:
                        raise ValueError('filename must be string or list')

        elif islistof(arg, np.ndarray):
            # list of images, with each item being a numpy array
            # imlist = TODO deal with iscolor=False case

            for image in arg:
                image = Image.getimage(image)  # convert to OpenCV friendly type

                if image.ndim == 2:
                    # single (W,H)
                    if colororder is not None:
                        raise ValueError('cannot specify color order for 2D image')
                    self._imlist.append(image)

                elif image.ndim == 3:
                    # assume colour image
                    if colororder is None:
                        raise ValueError('must specify color order for 3D image')
                    else:
                        self._imlist.append(image)

                elif arg.ndim == 4:
                    # color image sequence
                    raise ValueError('cannot have 4D image in the list')


            if filenames is not None:
                if self.nimages == 1:
                    # single image
                    if not isinstance(filenames, str):
                        raise ValueError('filename must be a string')
                    else:
                        # image sequence
                        if isinstance(filenames, str):
                            filenames = [f"{filenames}[{i}]" for i in range(self.nimages)]
                        elif len(filenames) == self.nimages:
                            pass
                        else:
                            raise ValueError('filename must be string or list')

        elif isinstance(arg, Image):
            # Image instance
            self._imlist = arg._imlist
            self._filenamelist = arg._filenamelist
            self._colororder = arg._colororder

        elif islistof(arg, Image):
            # list of Image instances
            # assuming Images are all of the same size

            shape = [im.shape for im in arg]
            if any(sh != shape[0] for sh in shape):
                raise ValueError(arg, 'input list of Image objects must \
                    be of the same shape')

            # TODO replace with list comprehension or itertools/chain method
            self._imlist = []
            self._filenamelist = []
            for imobj in arg:
                for im in imobj:
                    self._imlist.append(im.image)
                    self._filenamelist.append(im.filename)

        else:
            raise ValueError('bad argument to Image constructor')

        # check list of images for size consistency

        # VERY IMPORTANT!
        # We assume that the image stack has the same size image for the
        # entire list. TODO maybe in the future, we remove this assumption,
        # which can cause errors if not adhered to,
        # but for now we simply check the shape of each image in the list
        
        # TODO shape = [img.shape for img in self._imlist[]]
        # if any(shape[i] != list):
        #   raise
        if checksize and self.nimages > 1:
            shapes = [im.shape for im in self._imlist]
            if np.any([shape != shapes[0] for shape in shapes[1:]]):
                raise ValueError(arg, 'inconsistent input image shape')

        if copy:
            for i in self.nimages:
                self._imlist[i] = self._imlist[i].copy()

        if filenames is not None and len(filenames) != self.nimages:
            raise ValueError('filename list length must match number of images')

        self._filenamelist = filenames
        self._colordict = colordict

        # ability for user to specify iscolor manually to remove ambiguity
        # if iscolor is None:
        #     # our best guess
        #     shape = self._imlist[0].shape
        #     self._iscolor = (len(shape) == 3) and (shape[2] == 3)
        # else:
        #     self._iscolor = iscolor

        # self.nimages = len(self._imlist)

        # if self._imlist[0].ndim == 3:
        #     self._numimagechannels = self._imlist[0].shape[2]
        # elif self._imlist[0].ndim == 2:
        #     self._numimagechannels = 1
        # else:
        #     raise ValueError(self._numimagechannels, 'unknown number of \
        #                         image channels')

        
        if colordict is None:
            if self.nplanes > 1:
                raise ValueError('color order must be specified')
        elif len(colordict) != self.nplanes:
            raise ValueError('colororder length does not match number of planes')

        # check uniform type
        dtype = [im.dtype for im in self._imlist]
        if checktype:
            if np.any([dtype[i] != dtype[0] for i in range(len(dtype))]):
                raise ValueError(arg, 'inconsistent input image dtype')


    # ------------------------- properties ------------------------------ #

    # ---- image type ---- #
    @property
    def isfloat(self):
        """
        Image has floating point values

        :return: True if image has floating point values
        :rtype: bool
        """
        return np.issubdtype(self.dtype, np.floating)

    @property
    def isint(self):
        """
        Image has integer values

        :return: True if image has integer values
        :rtype: bool
        """
        return np.issubdtype(self.dtype, np.integer)

    @property
    def dtype(self):
        """
        Datatype of image

        :return: NumPy datatype of image
        :rtype: numpy.dtype
        """
        return self._imlist[0].dtype

    @property
    def min(self):
        """
        Minimum value of image

        :return: minimum value
        :rtype: int or float
        """
        return np.min(self._imlist[0])

    @property
    def max(self):
        """
        Maximum value of image

        :return: maximum value
        :rtype: int or float
        """
        return np.max(self._imlist[0])

    # ---- image dimension ---- #

    @property
    def width(self):
        """
        Image width

        :return: Width of image
        :rtype: int
        """
        return self._imlist[0].shape[1]

    @property
    def height(self):
        """
        Image height

        :return: Height of image
        :rtype: int
        """
        return self._imlist[0].shape[0]

    @property
    def size(self):
        """
        Image size

        :return: Size of image
        :rtype: (height, width)
        """
        return self._imlist[0].shape[:2]

    @property
    def npixels(self):
        return self._imlist[0].shape[0] * self._imlist[0].shape[1]

    @property
    def shape(self):
        """
        Image shape

        :return: Shape of internal NumPy array
        :rtype: 2-tuple or 3-tuple if color
        """
        return self._imlist[0].shape

    @property
    def ndim(self):
        return self._imlist[0].ndim

    # ---- color related ---- #
    @property
    def iscolor(self):
        """
        Image has color pixels

        :return: Image is color
        :rtype: bool
        """
        return self._imlist[0].ndim > 2

    @property
    def nplanes(self):
        """
        Number of color planes

        :return: Number of color channels
        :rtype: int
        """
        if self._imlist[0].ndim > 2:
            return self._imlist[0].shape[2]
        else:
            return 1

    @property
    def colororder(self):
        """
        Image color order

        :return: Color order
        :rtype: 'RGB' or 'BGR' or None

        .. note:: Is None if image is not color.
        """
        return self._colororder

    @property
    def isbgr(self):
        """
        Image has BGR color order

        :return: Image has BGR color order
        :rtype: bool

        .. note:: Is False if image is not color.
        """
        return self.colororder == 'B:G:R'

    @property
    def isrgb(self):
        """
        Image has RGB color order

        :return: Image has RGB color order
        :rtype: bool

        .. note:: Is False if image is not color.
        """
        return self.colororder == 'R:G:B'

    # NOTE, is this actually used?? Compared to im.shape[2], more readable

    @property
    def colororder(self):
        if self._colordict is not None:
            return ':'.join(self._colordict.keys())

    # ---- sequence related ---- #

    @property
    def issequence(self):
        """
        Image contains a sequence

        :return: Image contains a sequence of images
        :rtype: bool
        """
        return self.nimages > 1

    @property
    def issingleton(self):
        """
        Image contains a single image

        :return: Image contains a single images
        :rtype: bool
        """
        return len(self._imlist) == 1

    @property
    def nimages(self):
        """
        Number of images in sequence

        :return: Length of image sequence
        :rtype: int

        :seealso: len
        """
        return len(self._imlist)



    @property
    def filename(self):
        """
        File from which image was read

        :return: Name of image file
        :rtype: str or None

        .. note:: This is only set for ``Image``s read from a file.
        """
        if self._filenamelist is not None:
            return self._filenamelist[0]

    # ---- NumPy array access ---- #

    @property
    def image(self):
        """
        Image as NumPy array

        :return: image as a NumPy array
        :rtype: ndarray(h,w) or ndarray(h,w,3)

        .. note:: If the image is color the color order might be RGB or BGR.
        """
        return self._imlist[0]

    @property
    def rgb(self):
        """
        Image as NumPy array in RGB color order

        :raises ValueError: image is greyscale
        :return: image as a NumPy array in RGB color order
        :rtype: ndarray(h,w,3)
        """
        if not self.iscolor:
            raise ValueError('greyscale image has no rgb property')
        if self.isrgb:
            return self[0].image
        elif self.isbgr:
            return self[0].image[:, :, ::-1]

    @property
    def bgr(self):
        """
        Image as NumPy array in BGR color order

        :raises ValueError: image is greyscale
        :return: image as a NumPy array in BGR color order
        :rtype: ndarray(h,w,3)
        """
        if not self.iscolor:
            raise ValueError('greyscale image has no bgr property')
        if self.isbgr:
            return self[0].image
        elif self.isrgb:
            return self[0].image[:, :, ::-1]

    @classmethod
    def Read(cls, arg, alpha=False, **kwargs):
        if not isinstance(arg, (str, Path)) or islistof(arg, str):
            raise ValueError('expecting a string or path')

        # string, name of an image file to read in
        images = iread(arg, **kwargs)

        # result is a tuple(image, filename) or a list of tuples

        # TODO once iread, then filter through imlist and arrange into
        # proper.nimages and numchannels, based on user inputs, though
        # default to single list

        # NOTE stylistic change to line below
        # if (iscolor is False) and (imlist[0].ndim == 3):

        colororder = None
        if isinstance(images, tuple):
            # singleton image, make it a list
            image = images[0]
            if not alpha and image.ndim == 3 and image.shape[2] == 4:
                image = image[:, :, :3]
            if image.ndim > 2:
                colororder = 'BGR'
            image = cls(image, filenames=images[1], colororder=colororder)  # OpenCV file read order)

        elif isinstance(images, list):
            # image wildcard read is a tuple of lists, make a sequence
            imlist, filenamelist = zip(*images)
            if not alpha:
                for image in imlist:
                    if image.ndim == 3 and image.shape[2] == 4:
                        image = image[:, :, :3]
            coloroder = None
            for image in imlist:
                if image.ndim > 2:
                    colororder = 'BGR'
            image = cls(imlist, filenames=filenamelist,
                colororder=colororder)  # OpenCV file read order

        return image


    def __len__(self):
        return len(self._imlist)

    def __getitem__(self, ind):
        # try to return the ind'th image in an image sequence if it exists
        new = Image()

        new._colordict = self._colordict
        new._imlist = self.listimages(ind)
        if self._filenamelist is not None:
            new._filenamelist = self._filenamelist[ind]

        return new

    def __repr__(self):
        s = f"{self.width} x {self.height} ({self.dtype})"
        if self.nimages > 1:
            s += f" x {self.nimages}"
        if self.iscolor:
            s += ", " + self.colororder
        if self._filenamelist == []:
            s += ": " + self._filenamelist[0]
        return s

    def stats(self):
        def printstats(plane):
            print(f"range={plane.min()} - {plane.max()}, \
                mean={plane.mean()}, \
                sdev={plane.std()}")

        if self.iscolor:
            im = self.rgb
            print("red:   ", end="")
            printstats(im[:, :, 0])
            print("green: ", end="")
            printstats(im[:, :, 1])
            print("blue:  ", end="")
            printstats(im[:, :, 2])
        else:
            printstats(self.image)

    # ------------------------- color plane access -------------------------- #

    def plane(self, plane):
        """
        Extract the i'th plane of a color image

        :raises ValueError: if image is not color
        :return out: greyscale image representing the blue image plane
        :rtype: Image instance
        """
        if not self.iscolor:
            raise ValueError('cannot extract color plane from greyscale image')

        if isinstance(plane, int):
            if plane < 0 or plane >= self.nplanes:
                raise ValueError('plane index out of range')
        elif isinstance(plane, str):
            try:
                plane = self._colordict[plane]
            except KeyError:
                raise ValueError('bad plane name specified')
        else:
            raise ValueError('bad plane specified')

        return self.__class__([im[:, :, plane] for im in self._imlist])

    def _plane(self, plane):
        if not self.iscolor:
            raise ValueError('cannot extract color plane from greyscale image')

        if isinstance(plane, int):
            if plane < 0 or plane >= self.nplanes:
                raise ValueError('plane index out of range')
        elif isinstance(plane, str):
            try:
                plane = self._colordict[plane]
            except KeyError:
                raise ValueError('bad plane name specified')
        else:
            raise ValueError('bad plane specified')

        return self.image[:, :, plane]

    def red(self):
        """
        Extract the red plane of a color image

        :raises ValueError: if image is not color
        :return out: greyscale image representing the red image plane
        :rtype: Image instance
        """
        return self.plane('R')

    def green(self):
        """
        Extract the green plane of a color image

        :raises ValueError: if image is not color
        :return out: greyscale image representing the green image plane
        :rtype: Image instance
        """
        return self.plane('G')

    def blue(self):
        """
        Extract the blue plane of a color image

        :raises ValueError: if image is not color
        :return out: greyscale image representing the blue image plane
        :rtype: Image instance
        """
        return self.plane('B')

    # ------------------------- operators ------------------------------ #

    def stack(self):
        # convert list to 3 or 4D stak
        # option as to which dimension is first
        # first='color', 'sequence'
        pass

    def column(self):
        image = self.image
        if image.ndim == 2:
            return image.reshape((-1,))
        elif image.ndim == 3:
            return image.reshape((-1, self.nplanes))

    # ------------------------- operators ------------------------------ #

    # arithmetic
    def __mul__(self, other):
        """
        Overloaded * operator

        :return: elementwise product of images
        :rtype: Image

        Supports:

        * image * image
        * scalar * image
        * image * scalar
        """
        return Image._binop(self, other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        """
        Overloaded ** operator

        :return: elementwise exponent of images
        :rtype: Image
        """
        if not isscalar(other):
            raise ValueError('exponent must be a scalar')
        return Image._binop(self, other, lambda x, y: x ** y)

    def __add__(self, other):
        """
        Overloaded + operator

        :return: elementwise addition of images
        :rtype: Image
        """
        return Image._binop(self, other, lambda x, y: x + y)

    def __radd__(self, other):
        return other.__add__(self)

    def __sub__(self, other):
        """
        Overloaded - operator

        :return: elementwise subtraction of images
        :rtype: Image
        """
        return Image._binop(self, other, lambda x, y: x - y)

    def __rsub__(self, other):
        return Image._binop(self, other, lambda x, y: y - x)

    def __truediv__(self, other):
        """
        Overloaded / operator

        :return: elementwise division of images
        :rtype: Image
        """
        return Image._binop(self, other, lambda x, y: x / y)

    def __floordiv__(self, other):
        """
        Overloaded // operator

        :return: elementwise floored division of images
        :rtype: Image
        """
        return Image._binop(self, other, lambda x, y: x // y)

    def __minus__(self):
        """
        Overloaded unary - operator

        :return: elementwise negation of image
        :rtype: Image
        """
        return Image._unop(self, lambda x: -x)

    # bitwise
    def __and__(self, other):
        """
        Overloaded & operator

        :return: elementwise binary and of images
        :rtype: Image
        """
        return Image._binop(self, other, lambda x, y: x & y)

    def __or__(self, other):
        """
        Overloaded | operator

        :return: elementwise binary or of images
        :rtype: Image
        """
        return Image._binop(self, other, lambda x, y: x | y)

    def __inv__(self):
        """
        Overloaded ~ operator

        :return: elementwise bitwise inverse of image
        :rtype: Image
        """
        return Image._unop(self, lambda x: ~x)

    # relational
    def __eq__(self, other):
        """
        Overloaded == operator

        :return: elementwise comparison of images
        :rtype: Image

        True is 1 and False is 0.
        """
        return Image._binop(self, other, lambda x, y: x == y, logical=True)

    def __ne__(self, other):
        """
        Overloaded != operator

        :return: elementwise comparison of images
        :rtype: Image

        True is 1 and False is 0.
        """
        return Image._binop(self, other, lambda x, y: x != y, logical=True)

    def __gt__(self, other):
        """
        Overloaded > operator

        :return: elementwise comparison of images
        :rtype: Image

        True is 1 and False is 0.
        """
        return Image._binop(self, other, lambda x, y: x > y, logical=True)

    def __ge__(self, other):
        """
        Overloaded >= operator

        :return: elementwise comparison of images
        :rtype: Image

        True is 1 and False is 0.
        """
        return Image._binop(self, other, lambda x, y: x >= y, logical=True)

    def __lt__(self, other):
        """
        Overloaded < operator

        :return: elementwise comparison of images
        :rtype: Image

        True is 1 and False is 0.
        """
        return Image._binop(self, other, lambda x, y: x < y, logical=True)

    def __le__(self, other):
        """
        Overloaded <= operator

        :return: elementwise comparison of images
        :rtype: Image

        True is 1 and False is 0.
        """
        return Image._binop(self, other, lambda x, y: x <= y, logical=True)

    def __not__(self):
        """
        Overloaded not operator

        :return: elementwise comparison of images
        :rtype: Image

        Returns logical not operation interpretting the images as True is 1 and False is 0. 
        """

        return Image._unop(self, lambda x: not x, logical=True)

    # functions
    def abs(self):
        """
        Absolute value of image

        :return: elementwise absolute value of images
        :rtype: Image
        """
        return Image._unop(self, np.abs)

    def sqrt(self):
        """
        Square root of image

        :return: elementwise square root of images
        :rtype: Image
        """
        return Image._unop(self, np.sqrt)

    @staticmethod
    def _binop(left, right, op, logical=False):
        out = []
        if isinstance(right, Image):
            # Image OP Image
            if left.nimages == right.nimages:
                # two sequences of equal length
                for x, y in zip(left._imlist, right._imlist):
                    out.append(op(x, y))
            elif left.nimages == 1:
                # singleton OP sequence
                for y in right._imlist:
                    out.append(op(left.image, y))
            elif right.nimages == 1:
                # sequence OP singleton
                for x in left._imlist:
                    out.append(op(x, right.image))
            else:
                raise ValueError('cannot perform binary operation \
                    on sequences of unequal length')
        elif isscalar(right):
            # Image OP scalar
            for x in left._imlist:
                out.append(op(x, right))
        else:
            raise ValueError('right operand can only be scalar or Image')

        if logical:
            if np.issubdtype(left.dtype, np.floating):
                out = [x.astype('float32') for x in out]
            else:
                out = [x.astype('uint8') for x in out]

        return Image(out, colororder=left.colororder)

    @staticmethod
    def _unop(left, op):
        return Image([op(im) for im in left._imlist])


    def map(self, func):

        vfunc = np.vectorize(func)
        return Image([vfunc(im) for im in left._imlist])

    # ---- class functions? ---- #

    def disp(self, title=None, **kwargs):
        """
        Display first image in imlist

        :seealso: :func:`~machinevisiontoolbox.base.idisp`
        """
        if len(self) != 1:
            raise ValueError('bad length: must be 1 (not a sequence or empty)')
        if title is False:
            title = None
        elif title is None and self._filenamelist is not None:
            _, title = os.path.split(self._filenamelist[0])

        im = self[0].image
        # if self.iscolor and self.colororder != 'B:G:R':
        #     # for all other color orders ensure that the first plane is red, etc.
        #     im = im[:, :, ::-1]

        return idisp(im,
                title=title,
                bgr=self.isbgr,
                **kwargs)

    def showpixels(self, text=True, textcolors=['yellow', 'blue'], fmt="{:d}", ax=None, window=False, windowcolor=None, windowalpha=0.6, **kwargs):

        if ax is None:
            ax = plt.gca()

        for v in range(a.shape[0]):
            for u in range(a.shape[1]):
                if isinstance(textcolors, (list, tuple)):
                    if a[v,u] < 0.5:
                        color = textcolors[0]
                    else:
                        color = textcolors[1]
                elif textcolors == 'grey':
                    if a[v,u] < 0.5:
                        color = a[v,u] + 0.4 * np.r_[1,1,1]
                    else:
                        color = a[v,u] - 0.4 * np.r_[1,1,1]

                ax.text(u, v, fmt.format(a[v,u]), horizontalalignment='center', 
                    verticalalignment='center', color=color, **kwargs)

        ax.imshow(a, cmap='gray')
        ax.set_xlabel('u (pixels)')
        ax.set_ylabel('v (pixels)')

        plt.draw()

        class Window:
            def __init__(self, h, color='red', alpha=0.6, ax=None):
                self.h = h
                self.color = color
                w = 2 * h + 1
                patch = plt.Rectangle((0, 0), w, w, color='none', alpha=alpha)
                if ax is None:
                    ax = plt.gca()

                ax.add_patch(patch)
                self.patch = patch

            def move(self, u, v, color=None):
                if color is None:
                    color = self.color
                self.patch.set_color(color)
                self.patch.set_x(u - self.h - 0.5)
                self.patch.set_y(v - self.h - 0.5)

        if windowcolor is not None:
            return Window(color=windowcolor, alpha=windowalpha)

    def listimages(self, ind=None):

        if ind is None:
            if self.nimages == 1:
                ind = 0
            else:
                ind = np.arange(0, self.nimages)

        if isinstance(ind, int) and (ind >= -1) and (ind <= len(self._imlist)):
            return [self._imlist[ind]]

        elif isinstance(ind, slice):
            islice = np.arange(ind.start, ind.stop, ind.step)
            return [self._imlist[i] for i in islice]

        # elif isinstance(ind, tuple) and (len(ind) == 3):
        # slice object from numpy as a 3-tuple -> but how can we
        # differentiate between a normal 3-tuple eg (0,1,2) vs a numpy slice
        # (0, 2, 1)? TODO ruminate for later
        #     islice = np.arange()

        elif (len(ind) > 1) and (np.min(ind) >= -1) and \
             (np.max(ind) <= len(self._imlist)):
            return [self._imlist[i] for i in ind]

    def listimagefilenames(self, ind=None):

        if ind is None:
            if self.nimages == 1:
                ind = 0
            else:
                ind = np.arange(0, self.nimages)

        if isinstance(ind, int) and (ind >= -1) and \
           (ind <= len(self._filenamelist)):
            return [self._filenamelist[ind]]

        elif isinstance(ind, slice):
            islice = np.arange(ind.start, ind.stop, ind.step)
            return [self._filenamelist[i] for i in islice]

        elif (len(ind) > 1) and (np.min(ind) >= -1) and \
             (np.max(ind) <= len(self._filenamelist)):
            return [self._filenamelist[i] for i in ind]

    def pickpoints(self, n=None, matplotlib=True):
        """
        Pick points on image

        :param n: number of points to input, defaults to infinite number
        :type n: int, optional
        :return: Picked points, one per column
        :rtype: ndarray(2,n)

        Allow the user to select points on the displayed image.  A marker is
        displayed at each point selected with a left-click.  Points can be removed
        by a right-click, like an undo function.  middle-click or Enter-key
        will terminate the entry process.  If ``n`` is
        given the entry process terminates after ``n`` points are entered, but
        can terminated prematurely as above.

        .. note:: Picked coordinates have floating point values.

        :seealso: :func:`disp`
        """

        if matplotlib:
            points = plt.ginput(n)
            return np.c_[points].T
        else:

            def click_event(event, x, y, flags, params): 
  
                # checking for left mouse clicks 
                if event == cv2.EVENT_LBUTTONDOWN: 
            
                    # displaying the coordinates 
                    # on the Shell 
                    print(x, ' ', y) 

            cv.setMouseCallback('image', click_event) 
        
            # wait for a key to be pressed to exit 
            cv.waitKey(0) 

    # ------------------------- class methods ------------------------------ #

    @classmethod
    def isimage(cls, imarray):
        """
        Test if input is an image

        :param im: input image
        :type im: numpy array, shape (N,M), (N,M,3) or (N, M,3, P)
        :return: out
        :rtype: boolean True or False

        .. note::

            - ``isimage(im)`` returns False if.
            - ('im is not of type int or float').
            - ('im has ndims < 2').
            - ('im is (H,W), but does not have enough columns or rows to be an
              image').
            - ('im (H,W,N), but does not have enough N to be either a color
              image (N=3).
            - or a sequence of monochrome images (N > 1)').
            - ('im (H,W,M,N) should be a sequence of color images, but M is
              not equal to 3').
        """
        # return a consistent data type/format?
        # Can we convert any format to BGR 32f?
        # How would we know format in is RGB vs BGR?

        # convert im to nd.array
        imarray = np.array(imarray)

        # TODO consider complex floats?
        # check if image is int or floats
        # TODO shouldn't np.integer and np.float be the catch-all types?
        imtypes = [np.bool_,
                   np.uint8,
                   np.uint16,
                   np.uint32,
                   np.uint64,
                   np.int8,
                   np.int16,
                   np.int32,
                   np.int64,
                   np.float32,
                   np.float64,
                   np.integer,
                   np.float]

        if imarray.dtype not in imtypes:
            return False

        # check im.ndims > 1
        if imarray.ndim < 2:
            return False

        # check if im.ndims == 2, then im.shape (W,H), W >= 1, H >= 1
        if (imarray.ndim == 2) and ((imarray.shape[0] >= 1) and
                                    (imarray.shape[1] >= 1)):
            return True

        # check if im.ndims == 3, then im.shape(W,H,N), N >= 1
        if (imarray.ndim == 3) and (imarray.shape[2] >= 1):
            return True

        # check if im.ndims == 4, then im.shape(W,H,N,M), then N == 3
        if (imarray.ndim == 4) and (imarray.shape[2] == 3):
            return True

        # return consistent image format
        return False

    @classmethod
    def getimage(cls, imarray):
        """
        converts ``im`` to image compatible with OpenCV

        :param im: image
        :type im: numpy array (N,H,3) or (N,H) or TODO Image object?
        :return out: image of type np.(uint8, uint16, int16, float32, float64)
        :rtype: numpy array of the size of im

        - ``getimage(im)`` converts ``im`` into a compatible datatype with
          OpenCV: CV_8U, CV_16U, CV_16S, CV_32F or CV_64F. By default, if int
          then CV_8U, and if float then CV_64F. Boolean images are converted to
          0's and 1's int
        """
        # if isinstance(im, Image):
        #    imlist = im.imlist
        if not Image.isimage(imarray):
            raise ValueError(imarray, 'im is not a valid image')

        imarray = np.array(imarray)

        validTypes = [np.uint8, np.uint16, np.int16, np.float32, np.float64]
        # if im.dtype is not one of the valid image types,
        # convert to the nearest type or default to float64
        # TODO: what about image scaling?
        if imarray.dtype not in validTypes:
            # if float, just convert to CV_64F
            if np.issubdtype(imarray.dtype, np.float):
                imarray = np.float64(imarray)
            elif np.issubdtype(imarray.dtype, np.integer):
                if imarray.min() < 0:
                    imarray = np.int16(imarray)
                elif imarray.max() < np.iinfo(np.uint8).max:
                    imarray = np.uint8(imarray)
                elif imarray.max() < np.iinfo(np.uint16).max:
                    imarray = np.uint16(imarray)
                else:
                    raise ValueError(
                        imarray, 'max value of im exceeds np.uint16')
            elif np.issubdtype(imarray.dtype, np.bool_):
                imarray = np.uint8(imarray)

        return imarray

    def write(self, filename, dtype='uint8', **kwargs):
        """
        Write first image in imlist to filename

        :param filename: filename to write to
        :type filename: string

        :seealso: :func:`~machinevisiontoolbox.base.iwrite`
        """

        # cv.imwrite can only save 8-bit single channel or 3-channel BGR images
        # with several specific exceptions
        # https://docs.opencv.org/4.4.0/d4/da8/group__imgcodecs.html
        # #gabbc7ef1aa2edfaa87772f1202d67e0ce
        # TODO imwrite has many quality/type flags

        # TODO how do we handle sequence?
        # TODO how do we handle image file format?

        ret = iwrite(self.image.astype(dtype), filename, **kwargs)

        return ret

class Image(IImage,
            ImageCoreMixin,
            ImageProcessingBaseMixin,
            ImageProcessingMorphMixin,
            ImageProcessingKernelMixin,
            ImageProcessingColorMixin,
            # BlobFeaturesMixin,
            FeaturesMixin,
            # PointFeatureMixin,
            ReshapeMixin
            ):
    pass

def col2im(col, im):
    """
    Convert pixel vector to image

    :param col: set of pixel values
    :type col: numpy array, shape (N, P)
    :param im: image
    :type im: numpy array, shape (N, M, P), or a 2-vector (N, M)
    indicating image size
    :return: image of specified shape
    :rtype: numpy array

    - ``col2im(col, imsize)`` is an image (H, W, P) comprising the pixel
        values in col (N,P) with one row per pixel where N=HxW. ``imsize`` is
        a 2-vector (N,M).

    - ``col2im(col, im)`` as above but the dimensions of the return are the
        same as ``im``.

    .. note::

        - The number of rows in ``col`` must match the product of the
            elements of ``imsize``.

    :references:

        - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
    """

    # col = argcheck.getvector(col)
    col = np.array(col)
    if col.ndim == 1:
        nc = 1
    elif col.ndim == 2:
        nc = col.shape[1]
    else:
        raise ValueError(col, 'col does not have valid shape')

    # second input can be either a 2-tuple/2-array, or a full image
    im = np.array(im)  # ensure we can use ndim and shape
    if im.ndim == 1:
        # input is a tuple/1D array
        sz = im
    elif im.ndim == 2:
        im = Image.getimage(im)
        sz = im.shape
    elif im.ndim == 3:
        im = Image.getimage(im)
        sz = np.array([im.shape[0], im.shape[1]])  # ignore 3rd channel
    else:
        raise ValueError(im, 'im does not have valid shape')

    if nc > 1:
        sz = np.hstack((sz, nc))

    # reshape:
    # TODO need to test this
    return np.reshape(col, sz)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    # import machinevisiontoolbox as mvtb
    # from machinevisiontoolbox import Image

    # im = Image("flowers2.png")
    # print(im)
    # im = Image("machinevisiontoolbox/images/campus/*.png")
    # # im = Image("machinevisiontoolbox/images/flowers*.png")
    # print(im)

    # a = im[0]
    # print(type(a), len(a))
    # print(im[0])
    # # im[0].disp(block=True)

    # im = Image('campus/*.png')
    # print(im[0].filename)

    # # imcs = im.showcolorspace()
    # from machinevisiontoolbox import showcolorspace
    # imcs = showcolorspace('ab')
    # imcs.disp()

    # imc = im.colorise([1, 0, 0])
    # #  imc[0].disp(block=False)
    # for img in im:
    #     print(img.filename)

    # ims = im.smooth(2)
    # ims[0].disp(block=False)
    # ims[-1].disp(block=False)

    # grey = im[0].mono()
    # greysm = grey.smooth(1)
    # greysm.disp(block=False)

    # print(grey)
    # grey[0].disp(block=False)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.imshow(grey.image, cmap='gray', vmin=0, vmax=255)
    # plt.show(block=False)

    # mb = Image("multiblobs.png")
    # mb.disp()
    # # read im image:

    # # test for single colour image
    # imfile = 'images/test/longquechen-mars.png'
    # rawimage = iread(imfile)

    # # test for image string
    # rawimage = imfile

    # # test for multiple images, stack them first:
    # flowers = [str(('flowers' + str(i+1) + '.png')) for i in range(8)]
    # print(flowers)

    # # list of images
    # imlist = [iread(('images/' + i)) for i in flowers]

    # # plt.show()

    # im = Image(imlist)

    # print('im.image dtype =', im.image.dtype)
    # print('im.shape =', im.shape)
    # print('im.iscolor =', im.iscolor)
    # print('im.nimages =', im.nimages)
    # print('im.numchannels =', im.nchannels)

    # import code
    # code.interact(local=dict(globals(), **locals()))

    # im = Image('monalisa.png', grey=True)

    # i2 = im + im
    # i2 = im ** 2
    # i2 = im * 3
    # i2 = 3 * im

    im = Image.Read('flowers4.png')
    print(im)
    red = im.red()
    r = red.image
    r *= 0
    im.disp(block=True)

    # im = Image('campus/*.png')

    # print(im)
    # im.red().disp(block=True)

    # hsv = im.colorspace('HSV')
    # print(hsv)
    # hsv.plane('H').disp(block=True)
