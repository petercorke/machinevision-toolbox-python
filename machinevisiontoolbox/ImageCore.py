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
from numpy.lib.arraysetops import isin
from machinevisiontoolbox.base import int_image, float_image
from machinevisiontoolbox.ImageSpatial import Kernel
from spatialmath.base import isscalar, islistof
import warnings
# import spatialmath.base.argcheck as argcheck


from machinevisiontoolbox.base.imageio import idisp, iread, iwrite, convert
import urllib
import xml.etree.ElementTree as ET


class ImageCoreMixin:

    def __init__(self,
                 image=None,
                 colororder=None,
                 copy=False,
                 shape=None,
                 dtype=None,
                 name=None,
                 id=None,
                 **kwargs):
        """
        Create an Image instance

        :param image: image data
        :type arg: NumPy array or Image instance
        :param colororder: order of color channels ('BGR' or 'RGB')
        :type colororder: str, dict
        :param copy: copy the image, defaults to False
        :type copy: bool, optional
        :param shape: new shape for the image, defaults to None
        :type shape: tuple, optional
        :param dtype: data type for image, defaults to same type as ``image``
        :type dtype: str or NumPy dtype, optional
        :param name: name of image, defaults to None
        :type name: str, optional
        :raises TypeError: unknown type passed to constructor

        If no ``dtype`` is given it is inferred from the type and value of 
        ``image``:

        - if ``image`` is a float of any sort it is cast to float32
        - if ``image`` is an int of any sort the type is chosen as the smallest
          unsigned int that can represent the maximum value

        If ``image`` is bool it is converted to a numeric value of type
        ``dtype``.  If ``dtype`` is not given then it defaults to uint8.  
        
        - False is represented by 0
        - True is represented by 1.0 for a float type and the maximum integer
          value for an integer type
        """

        self._A = None
        self._name = None
        self._colororder = None
        self.id = id

        if isinstance(image, np.ndarray):
            self.name = name

        elif isinstance(image, self.__class__):
            # Image instance
            name = image.name
            colororder = image.colororder
            image = image.A

        else:
            raise TypeError('bad argument to Image constructor')

        if shape is not None:
            if image.ndim <= 2:
                # 2D image
                image = image.reshape(shape)
            elif image.ndim == 3:
                # 3D image
                image = image.reshape(shape + (-1,))

        if dtype is None:
            if image.dtype == np.float64:
                # this the default format created by NumPy if there is a float
                # in the value list
                dtype = np.float32

            elif image.dtype == np.int64:
                # this the default format created by NumPy if the value list is
                # all ints
                for type in ['uint8', 'uint16', 'uint32']:
                    if image.max() < np.iinfo(type).max:
                        dtype = np.dtype(type)
                        break
            
        if image.dtype == np.bool:
            if dtype is None:
                dtype = np.uint8
            false = 0
            if np.issubdtype(dtype, np.floating):
                true = 1
            elif np.issubdtype(dtype, np.integer):
                true = np.iinfo(dtype).max
            false = np.dtype(dtype).type(false)
            true = np.dtype(dtype).type(true)
            image = np.where(image, true, false)
        elif dtype is not None:
            image = image.astype(dtype)

        if copy:
            self._A = image.copy()
        else:
            self._A = image

        if self.nplanes > 1 and colororder is None:
            colororder = 'RGB'
            warnings.warn('defaulting color to RGB')
            
        if colororder is not None:
            self.colororder = colororder

        self.name = name

    def __str__(self):
        """
        Single line summary of image parameters

        :return: single line summary of image
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> print(im)
        """
        s = f"Image: {self.width} x {self.height} ({self.dtype})"

        if self.iscolor:
            s += ", " + self.colororder_str
        if self.id is not None:
            s += f", id={self.id}"
        if self.name is not None:
            name = self.name
            if len(name) > 20:
                k = [i for i, c in enumerate(name) if c == '/']
                if len(k) >= 2:
                    name = name[k[-2]:]
                else:
                    name = name[-20:]
                name = "..." + name
            s += f" [{name}]"
        return s

    def __repr__(self):
        """
        Single line summary of image parameters

        :return: single line summary of image
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im
        """
        return str(self)

    def copy(self):
        return self.__class__(self, copy=True)

    # ------------------------- properties ------------------------------ #

    @property
    def colororder(self):
        """
        Image color order as a dict

        :return: Image color plane order
        :rtype: dict

        Color order is a dict where the key is the color plane name, eg. 'R'
        and the value is the NumPy index value.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.colororder

        """
        return self._colororder

    @property
    def colororder_str(self):
        """
        Image color order as a string

        :return: Image color plane order as a colon separated string.
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.colororder_str
        """
        if self.colororder is not None:
            s = sorted(self.colororder.items(), key=lambda x: x[1])
            return ':'.join([x[0] for x in s])
        else:
            return ""

    @colororder.setter
    def colororder(self, colororder):
        """
        Set image color order

        :param colororder: color order
        :type colororder: str or dict

        Color order can be any of:

        * simple string, one plane per character, eg. ``"RGB"``
        * colon separated string, eg. ``"R:G:B"``, ``"L*:a*:b*"``
        * dict, eg. ``dict(R=0, G=1, B=2)``

        For the first two cases the color plane indices are implicit in the
        order in the string.
        """
        if isinstance(colororder, dict):
            cdict = colororder
        elif isinstance(colororder, str):
            if ':' in colororder:
                colororder = colororder.split(':')
            else:
                colororder = list(colororder)
                
            cdict = {}
            for i, color in enumerate(colororder):
                cdict[color] = i
        else:
            raise ValueError('color order must be a dict or string')

        if len(cdict) != self.nplanes:
            raise ValueError('colororder length does not match number of planes')
        self._colororder = cdict

    @property
    def name(self):
        """
        Image name

        :return: image name, optional
        :rtype: str

        :type name: str

        This is shown by the Image repr and when images are displayed
        graphically.
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Set image name

        :param name: Image name
        :type name: str

        This is shown by the Image repr and when images are displayed
        graphically.
        """
        self._name = name

    # ---- image type ---- #
    @property
    def isfloat(self):
        """
        Image has floating point pixel values

        :return: True if image has floating point values
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.isfloat
            >>> im = Image.Read('flowers1.png', dtype='float32')
            >>> im.isfloat

        """
        return np.issubdtype(self.dtype, np.floating)

    @property
    def isint(self):
        """
        Image has integer values

        :return: True if image has integer pixel values
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.isfloat
            >>> im = Image.Read('flowers1.png', dtype='float32')
            >>> im.isfloat
        """
        return np.issubdtype(self.dtype, np.integer)

    @property
    def dtype(self):
        """
        Datatype of image

        :return: NumPy datatype of image
        :rtype: numpy.dtype

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.dtype
            >>> im = Image.Read('flowers1.png', dtype='float32')
            >>> im.dtype
        """
        return self.A.dtype

    @property
    def min(self):
        """
        Minimum value of all pixels

        :return: minimum value
        :rtype: int or float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.min
            >>> im = Image.Read('flowers1.png', dtype='float32')
            >>> im.min
        """
        return np.min(self.A)

    @property
    def max(self):
        """
        Maximum value of all pixels

        :return: maximum value
        :rtype: int or float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.max
            >>> im = Image.Read('flowers1.png', dtype='float32')
            >>> im.max
        """
        return np.max(self.A)

    @property
    def mean(self):
        """
        Mean value of all pixels

        :return: mean value
        :rtype: int or float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.mean
            >>> im = Image.Read('flowers1.png', dtype='float32')
            >>> im.mean
        """
        return np.mean(self.A)

    @property
    def std(self):
        """
        Standard deviation of all pixels

        :return: standard deviation value
        :rtype: int or float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.std
            >>> im = Image.Read('flowers1.png', dtype='float32')
            >>> im.std
        """
        return np.std(self.A)

    @property
    def median(self):
        """
        Median value of all pixels

        :return: median value
        :rtype: int or float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.median
            >>> im = Image.Read('flowers1.png', dtype='float32')
            >>> im.median
        """
        return np.median(self.A)

    def stats(self):
        """
        Display pixel value statistics

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.stats()
        """
        def printstats(plane):
            print(f"range={plane.min()} - {plane.max()}, "
                f"mean={plane.mean():.3f}, "
                f"sdev={plane.std():.3f}")

        if self.iscolor:
            for k, v in sorted(self.colororder.items(), key=lambda x: x[1]):
                print(f"{k:s}: ", end="")
                printstats(self.A[..., v])
        else:
            printstats(self.A)

    # ---- image dimension ---- #

    @property
    def width(self):
        """
        Image width

        :return: Width of image in pixels
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.width
        """
        return self.A.shape[1]

    @property
    def height(self):
        """
        Image height

        :return: Height of image in pixels
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.height
        """
        return self.A.shape[0]

    def uspan(self, step=1):
        if self.domain is None:
            return np.arange(0, self.width, step)
        else:
            return self.domain[0]

    def vspan(self, step=1):
        if self.domain is None:
            return np.arange(0, self.height, step)
        else:
            return self.domain[1]

    @property
    def size(self):
        """
        Image size

        :return: Size of image as a tuple
        :rtype: (width, height)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.size
        """
        return (self.A.shape[1], self.A.shape[0])

    @property
    def centre(self):
        return (self.A.shape[1] // 2, self.A.shape[0] // 2)

    @property
    def npixels(self):
        """
        Number of pixels in image

        :return: Number of pixels in image: width x height
        :rtype: int

        .. note:: Number of planes is not considered.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.npixels
        """
        return self.A.shape[0] * self.A.shape[1]

    @property
    def shape(self):
        """
        Image shape

        :return: Shape of internal NumPy array
        :rtype: 2-tuple or 3-tuple if color

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.shape

    :seealso: :meth:`.nplanes` :meth:`.ndim`
        """
        return self.A.shape

    @property
    def ndim(self):
        """
        Number of image array dimensions

        :return: number of image dimensions
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.ndim
        
        :seealso: :meth:`.nplanes` :meth:`.shape`
        """
        return self.A.ndim

    # ---- color related ---- #
    @property
    def iscolor(self):
        """
        Image has color pixels

        :return: Image is color
        :rtype: bool

        :return: number of image dimensions
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.iscolor

        :seealso: :meth:`.isrgb` :meth:`.nplanes`
        """
        return self.A.ndim > 2

    @property
    def isbgr(self):
        """
        Image has BGR color order

        :return: Image has BGR color order
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.isbgr

        .. note:: Is False if image is not color.

        """
        return self.colororder_str == 'B:G:R'

    @property
    def isrgb(self):
        """
        Image has RGB color order

        :return: Image has RGB color order
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.isrgb

        .. note:: Is False if image is not color.
        """
        return self.colororder == 'R:G:B'


    # ---- NumPy array access ---- #

    @property
    def image(self):
        """
        Image as NumPy array

        :return: image as a NumPy array
        :rtype: ndarray(h,w) or ndarray(h,w,3)

        .. note:: If the image is color the color order might be RGB or BGR.
        """
        warnings.warn(
        "this property is deprecated, use .A instead",
            DeprecationWarning
        )

        return self.A

    @property
    def A(self):
        """
        Image as NumPy array

        :return: image as a NumPy array
        :rtype: ndarray(h,w) or ndarray(h,w,3)

        This is the underlying image contained by the ``Image`` object.  It is
        a NumPy ``ndarray`` with two (greyscale) or three (color) dimensions.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> type(im)
            >>> type(im.A)
        """
        return self._A

    @A.setter
    def A(self, A):
        self._A = A

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
            return self.A
        elif self.isbgr:
            return self.A[:, :, ::-1]

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
            return self.A
        elif self.isrgb:
            return self.A[:, :, ::-1]

    def to_int(self, intclass='uint8'):
        """
        Convert image to integer NumPy array

        :param intclass: name of NumPy supported integer class
        :type intclass: str
        :return: NumPy array with integer values
        :rtype: ndarray

        - ``IM.int()`` is a copy of image with pixels converted to unsigned
          8-bit integer (uint8) elements in the range 0 to 255.

        - ``IM.int(intclass)`` as above but the output pixels are converted to
          the integer class ``intclass``.

        Floating point images, with pixels in the interval [0, 1] are rescaled
        to the interval [0, intmax] where intmax is the maximum integer value
        in the class.

        Integer image are simply cast.  
        
        .. warning:: For integer to interer conversion, integer values  greater
            than the maximum value of the destination class are wrapped not
            clipped.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png', dtype='float64')
            >>> im
            >>> im_int = im.asint()
            >>> type(im_int)

        :seealso: :meth:`.asfloat`
        """
        return int_image(self.image, intclass)

    def to_float(self, floatclass='float32'):
        """
        Convert image to float NumPy array

        :param floatclass: 'single', 'double', 'float32' [default], 'float64'
        :type floatclass: str
        :return: Image with floating point pixel types
        :rtype: Image instance

        - ``IM.float()`` is a copy of image with pixels converted to
          ``float32`` floating point values spanning the range 0 to 1. The
          input integer pixels are assumed to span the range 0 to the maximum
          value of their integer class.

        - ``IM.float(im, floatclass)`` as above but with floating-point pixel
          values belonging to the class ``floatclass``.

        Floating point images, with pixels in the interval [0, 1] are simply
        cast.

        Integer image are rescaled from the interval [0, intmax] to the 
        interval [0, intmax] where intmax is the maximum integer value
        in the class.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im
            >>> im_float = im.asfloat()
            >>> type(im_float)

        :seealso: :meth:`.asint`
        """
        return float_image(self.image, floatclass)

    def to(self, dtype):
        dtype = np.dtype(dtype)  # convert to dtype if it's a string

        if np.issubdtype(dtype, np.integer):
            out = self.to_int(dtype)
        elif np.issubdtype(dtype, np.floating):
            out = self.to_float(dtype)
        return self.__class__(out, dtype=dtype)

    def cast(self, value):
        """
        Cast NumPy values to same type as image

        :param value: NumPy array or scalar
        :type value: any NumPy type
        :return: NumPy array or scalar
        :rtype: same type as image

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> x = im.cast(12.5)
            >>> x
            >>> type(x)

        :seealso: :meth:`.like`
        """
        return self.A.dtype.type(value)

    def like(self, value, max=None):
        """
        Convert NumPy value to same type as image

        :param value: NumPy array or scalar
        :type value: any NumPy type
        :param max: maximum value of ``value``'s type, defaults to None
        :type max: int, optional
        :raises ValueError: [description]
        :return: converted value
        :rtype: NumPy type

        Values are optionally rescaled and cast.

        * Float to float: values are cast
        * Float to int: values in the interval [0, 1] are scaled to the interval
          [0, intmax] and then cast, where intmax is the maximum integer value
          in the class.
        * Int to float: values in the interval [0, intmax] are scaled to the
          interval [0, 1] and then cast, where intmax is the maximum integer
          value in the class.
        * Int to int: values are cast.

        .. warning:: For integer to interer conversion, integer values  greater
            than the maximum value of the destination class are wrapped not
            clipped.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.like(0.5)

        :seealso: :meth:`.like`

        """
        if self.isint:
            # fitting an integer image
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.integer) or isinstance(value, (int, np.integer)):
                # already an integer, cast it to right sort
                return self.cast(value)
            else:
                # it's a float, rescale it then cast
                return self.cast(value * self.maxval)
        else:
            # fitting a float image
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating) or isinstance(value, (float, np.floating)):
                # already a float of some sort, cast it to the right sort
                return self.cast(value)
            else:
                # it's an int.  We use hints to determine the size, otherwise
                # get it from the type
                if max is None:
                    max = np.iinfo(value.dtype).max
                elif isinstance(max, int):
                    pass
                elif isinstance(max, str) or isinstance(max, np.dtype):
                    max = np.iinfo(max).max
                else:
                    raise ValueError('bad max value specified')
                return self.cast(value / max)


    @property
    def minval(self):
        """
        Minimum value of image datatype

        :return: Minimum value
        :rtype: int or float

        For the datatype of the image, return its minimum possible value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image(np.ones((20, 20)), dtype='float32')
            >>> print(im.minval)
            >>> im = Image(np.ones((20, 20)), dtype='uint8')
            >>> print(im.minval)
        
        :seealso: :meth:`.maxval`
        """
        if self.isint:
            return np.iinfo(self.dtype).min
        else:
            return np.finfo(self.dtype).min

    @property
    def maxval(self):
        """
        Maximum value of image datatype

        :return: Maximum value
        :rtype: int or float

        For the datatype of the image, return its maximum possible value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image(np.ones((20, 20)), dtype='float32')
            >>> print(im.maxval)
            >>> im = Image(np.ones((20, 20)), dtype='uint8')
            >>> print(im.maxval)
        
        :seealso: :meth:`.minval`
        """
        if self.isint:
            return np.iinfo(self.dtype).max
        else:
            return np.finfo(self.dtype).max

    @property
    def true(self):
        """
        True value for logical image

        :return: Value used as true
        :rtype: int or float

        The true value is 1.0 for a floating point image and maximum integer
        value for an integer value.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image(np.ones((20, 20)), dtype='float32')
            >>> print(im.true)
            >>> im = Image(np.ones((20, 20)), dtype='uint8')
            >>> print(im.true)
        
        :seealso: :meth:`.false` :meth:`.maxval`
        """
        if self.isint:
            return self.maxval
        else:
            return 1.0
    
    @property
    def false(self):
        """
        False value for logical image

        :return: Value used as true
        :rtype: int or float

        The false value is 0.0 for a floating point image and 0
        value for an integer value.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image(np.ones((20, 20)), dtype='float32')
            >>> print(im.false)
            >>> im = Image(np.ones((20, 20)), dtype='uint8')
            >>> print(im.false)
        
        :seealso: :meth:`.true`
        """
        return 0

    def astype(self, type):
        return self.__class__(self.A, dtype=type)

    # ------------------------- color plane access -------------------------- #

    @property
    def nplanes(self):
        """
        Number of color planes

        :return: Number of color planes
        :rtype: int

        For a 2D or greyscale image this is one, otherwise it is the third
        dimension of the image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('street.png')
            >>> im.nplanes
            >>> im = Image.Read('flowers1.png')
            >>> im.nplanes

        :seealso: :meth:`.shape` :meth:`.ndim`
        """
        if self.A.ndim == 2:
            return 1
        else:
            return self.A.shape[2]

    def plane(self, planes):
        """
        Extract the i'th plane of a color image

        :param planes: planes to extract
        :type planes: int, list, str
        :raises ValueError: if image is not color
        :return out: greyscale image representing the blue image plane
        :rtype: Image instance

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read("flowers4.png") # in BGR order
            >>> print(im.nplanes)
            >>> red = im.plane(0) # red plane
            >>> red.nplanes
            >>> green_blue = im.plane('GB') # green and blue planes
            >>> green_blue
            >>> red_blue = im.plane([0, 2]) # blue and red planes
            >>> red_blue

        .. note:: For the last case the resulting color order is BR since the
            new image is built from plane 0 of the original (B), then plane 2
            of the original (R).

        :seealso: :meth:`.red` :meth:`.green` :meth:`.blue`
        """
        if not self.iscolor:
            raise ValueError('cannot extract color plane from greyscale image')

        if isinstance(planes, int):
            if planes < 0 or planes >= self.nplanes:
                raise ValueError('plane index out of range')
            iplanes = planes
            colororder = None
        elif isinstance(planes, str):
            iplanes = []
            colororder = {}
            if ':' in planes:
                planes = planes.split(':')
            for plane in planes:
                try:
                    i = self.colororder[plane]
                    iplanes.append(i)
                    colororder[plane] = len(colororder)  # copy to new dict
                except KeyError:
                    raise ValueError('bad plane name specified')
        elif isinstance(planes, list):
            colororder = {}
            for plane in planes:
                if plane < 0 or plane >= self.nplanes:
                    raise ValueError('plane index out of range')
                colorname = [k for k, v in self.colororder.items() if v == plane][0]
                colororder[colorname] = plane
            iplanes = planes
        else:
            raise ValueError('bad plane specified')

        if isinstance(iplanes, list) and len(iplanes) == 1:
                iplanes = iplanes[0]
                colororder = None
        return self.__class__(self.A[:, :, iplanes], colororder=colororder)

    def red(self):
        """
        Extract the red plane of a color image

        :raises ValueError: if image is not color
        :return out: greyscale image representing the red image plane
        :rtype: Image instance

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read("flowers4.png")
            >>> red = im.red() # red plane
            >>> red

        :seealso: :meth:`.plane`
        """
        return self.plane('R')

    def green(self):
        """
        Extract the green plane of a color image

        :raises ValueError: if image is not color
        :return out: greyscale image representing the green image plane
        :rtype: Image instance

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read("flowers4.png")
            >>> green = im.green() # green plane
            >>> green

        :seealso: :meth:`.plane`
        """
        return self.plane('G')

    def blue(self):
        """
        Extract the blue plane of a color image

        :raises ValueError: if image is not color
        :return out: greyscale image representing the blue image plane
        :rtype: Image instance

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read("flowers4.png")
            >>> blue = im.blue() # blue plane
            >>> blue

        :seealso: :meth:`.plane`
        """
        return self.plane('B')



    # ------------------------- operators ------------------------------ #

    # arithmetic
    def __mul__(self, other):
        """
        Overloaded * operator

        :return: elementwise product of images
        :rtype: Image

        Supports:

        * image ``*`` image
        * scalar ``*`` image
        * image ``*`` scalar
        """
        return self._binop(self, other, lambda x, y: x * y)

    def __rmul__(self, other):

        return self._binop(self, other, lambda x, y: y * x)

    def __pow__(self, other):
        """
        Overloaded ** operator

        :return: elementwise exponent of images
        :rtype: Image
        """
        if not isscalar(other):
            raise ValueError('exponent must be a scalar')
        return self._binop(self, other, lambda x, y: x ** y)

    def __add__(self, other):
        """
        Overloaded + operator

        :return: elementwise addition of images
        :rtype: Image

        Supports:

        * image ``+`` image
        * scalar ``+`` image
        * image ``+`` scalar
        """
        return self._binop(self, other, lambda x, y: x + y)

    def __radd__(self, other):
        return self._binop(self, other, lambda x, y: y + x)

    def __sub__(self, other):
        """
        Overloaded - operator

        :return: elementwise subtraction of images
        :rtype: Image

        Supports:

        * image ``-`` image
        * scalar ``-`` image
        * image ``-`` scalar
        """
        return self._binop(self, other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self._binop(self, other, lambda x, y: y - x)

    def __truediv__(self, other):
        """
        Overloaded / operator

        :return: elementwise division of images
        :rtype: Image

        Supports:

        * image ``/`` image
        * scalar ``/`` image
        * image ``/`` scalar
        """
        return self._binop(self, other, lambda x, y: x / y)

    def __rtruediv__(self, other):
        """
        Overloaded / operator

        :return: elementwise division of images
        :rtype: Image
        """
        return self._binop(self, other, lambda x, y: y / x)

    def __floordiv__(self, other):
        """
        Overloaded // operator

        :return: elementwise floored division of images
        :rtype: Image

        Supports:

        * image ``//`` image
        * scalar ``//`` image
        * image ``//`` scalar
        """
        return self._binop(self, other, lambda x, y: x // y)

    def __rfloordiv__(self, other):
        """
        Overloaded // operator

        :return: elementwise floored division of images
        :rtype: Image

        """
        return self._binop(self, other, lambda x, y: y // x)

    def __minus__(self):
        """
        Overloaded unary - operator

        :return: elementwise negation of image
        :rtype: Image
        """
        return self._unop(self, lambda x: -x)

    # bitwise
    def __and__(self, other):
        """
        Overloaded & operator

        :return: elementwise binary and of images
        :rtype: Image
        """
        return self._binop(self, other, lambda x, y: x & y)

    def __or__(self, other):
        """
        Overloaded | operator

        :return: elementwise binary or of images
        :rtype: Image
        """
        return self._binop(self, other, lambda x, y: x | y)

    # relational
    def __eq__(self, other):
        """
        Overloaded == operator

        :return: elementwise comparison of images
        :rtype: Image

        For a:

            * floating image: True is 1.0, False is 0.0
            * integer image: True is maximum value, False is 0

        Supports:

            * image ``==`` image
            * scalar ``==`` image
            * image ``==`` scalar

        :seealso: :meth:`.true` :meth:`.false`
        """
        return self._logicalop(self, other, lambda x, y: x == y)

    def __ne__(self, other):
        """
        Overloaded != operator

        :return: elementwise comparison of images
        :rtype: Image

        For a:

            * floating image: True is 1.0, False is 0.0
            * integer image: True is maximum value, False is 0

        Supports:

            * image ``!=`` image
            * scalar ``!=`` image
            * image ``!=`` scalar

        :seealso: :meth:`.true` :meth:`.false`
        """
        return self._logicalop(self, other, lambda x, y: x != y)

    def __gt__(self, other):
        """
        Overloaded > operator

        :return: elementwise comparison of images
        :rtype: Image

        For a:

            * floating image: True is 1.0, False is 0.0
            * integer image: True is maximum value, False is 0

        Supports:
        
            * image ``>`` image
            * scalar ``>`` image
            * image ``>`` scalar

        :seealso: :meth:`.true` :meth:`.false`
        """
        return self._logicalop(self, other, lambda x, y: x > y)

    def __ge__(self, other):
        """
        Overloaded >= operator

        :return: elementwise comparison of images
        :rtype: Image

        For a:

            * floating image: True is 1.0, False is 0.0
            * integer image: True is maximum value, False is 0

        Supports:

            * image ``>=`` image
            * scalar ``>=`` image
            * image ``>=`` scalar

        :seealso: :meth:`.true` :meth:`.false`
        """
        return self._logicalop(self, other, lambda x, y: x >= y)

    def __lt__(self, other):
        """
        Overloaded < operator

        :return: elementwise comparison of images
        :rtype: Image

        For a:

            * floating image: True is 1.0, False is 0.0
            * integer image: True is maximum value, False is 0

        Supports:

            * image ``<`` image
            * scalar ``<`` image
            * image ``<`` scalar

        :seealso: :meth:`.true` :meth:`.false`
        """
        return self._logicalop(self, other, lambda x, y: x < y)

    def __le__(self, other):
        """
        Overloaded <= operator

        :return: elementwise comparison of images
        :rtype: Image

        For a:

            * floating image: True is 1.0, False is 0.0
            * integer image: True is maximum value, False is 0

        Supports:

            * image ``<=`` image
            * scalar ``<=`` image
            * image ``<=`` scalar

        :seealso: :meth:`.true` :meth:`.false`
        """
        return self._logicalop(self, other, lambda x, y: x <= y)

    def __invert__(self):
        """
        Overloaded ~ operator

        :return: elementwise inversion of logical images
        :rtype: Image

        Returns logical not operation interpretting the images as:

            * floating image: True is 1.0, False is 0.0
            * integer image: True is maximum value, False is 0 True is 1 and False is 0. 
        """

        true = self.cast(self.true)
        false = self.cast(self.false)

        out = np.where(self.image > 0, false, true)

        return self.__class__(out, colororder=self.colororder)

    # functions
    def abs(self):
        """
        Absolute value of image

        :return: elementwise absolute value of images
        :rtype: Image
        """
        return self._unop(self, np.abs)

    def sqrt(self):
        """
        Square root of image

        :return: elementwise square root of images
        :rtype: Image
        """
        return self._unop(self, np.sqrt)

    @staticmethod
    def _binop(left, right, op, logical=False):
        if isinstance(right, left.__class__):
            return left.__class__(op(left.A, right.A), colororder=left.colororder)
        else:
            # right is a scalar or numpy array
            return left.__class__(op(left.A, right), colororder=left.colororder)

    @staticmethod
    def _logicalop(left, right, op):
        true = left.cast(left.true)
        false = left.cast(left.false)

        if isinstance(right, left.__class__):
            # image OP image
            out = np.where(op(left.A, right.A), true, false)
        else:
            out = np.where(op(left.image, right), true, false)

        return left.__class__(out, colororder=left.colororder)

    @staticmethod
    def _unop(left, op):
        return left.__class__(op(left.A), colororder=left.colororder)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    import pathlib
    import os.path
    from machinevisiontoolbox import Image
    
    exec(open(pathlib.Path(__file__).parent.parent.absolute() / "tests" / "test_core.py").read())  # pylint: disable=exec-used