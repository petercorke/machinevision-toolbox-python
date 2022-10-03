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
from machinevisiontoolbox.base import int_image, float_image, draw_line, draw_circle, draw_box
from machinevisiontoolbox.ImageSpatial import Kernel
from spatialmath.base import isscalar, islistof
import warnings
# import spatialmath.base.argcheck as argcheck

from machinevisiontoolbox.base.imageio import idisp, iread, iwrite, convert
import urllib

from machinevisiontoolbox.ImageIO import ImageIOMixin
from machinevisiontoolbox.ImageConstants import ImageConstantsMixin
from machinevisiontoolbox.ImageProcessing import ImageProcessingMixin
from machinevisiontoolbox.ImageMorph import ImageMorphMixin
from machinevisiontoolbox.ImageSpatial import ImageSpatialMixin
from machinevisiontoolbox.ImageColor import ImageColorMixin
from machinevisiontoolbox.ImageReshape import ImageReshapeMixin
from machinevisiontoolbox.ImageWholeFeatures import ImageWholeFeaturesMixin
from machinevisiontoolbox.ImageBlobs import ImageBlobsMixin
from machinevisiontoolbox.ImageRegionFeatures import ImageRegionFeaturesMixin
from machinevisiontoolbox.ImageLineFeatures import ImageLineFeaturesMixin
from machinevisiontoolbox.ImagePointFeatures import ImagePointFeaturesMixin
from machinevisiontoolbox.ImageMultiview import ImageMultiviewMixin

"""
This class encapsulates a Numpy array containing the pixel values.  The object
supports arithmetic using overloaded operators, as well a large number of methods.
"""
class Image(
            ImageIOMixin,
            ImageConstantsMixin,
            ImageProcessingMixin,
            ImageMorphMixin,
            ImageSpatialMixin,
            ImageColorMixin,
            ImageReshapeMixin,
            ImageBlobsMixin,
            ImageWholeFeaturesMixin,
            ImageRegionFeaturesMixin,
            ImageLineFeaturesMixin,
            ImagePointFeaturesMixin,
            ImageMultiviewMixin
            ):

    def __init__(self,
                 image=None,
                 colororder=None,
                 copy=False,
                 shape=None,
                 dtype=None,
                 name=None,
                 id=None,
                 domain=None,
                 **kwargs):
        """
        Create an Image instance

        :param image: image data
        :type image: array_like(H,W), :class:`Image`
        :param colororder: order of color channels
        :type colororder: str, dict
        :param copy: copy the image data, defaults to False
        :type copy: bool, optional
        :param shape: new shape for the image, defaults to None
        :type shape: tuple, optional
        :param dtype: data type for image, defaults to same type as ``image``
        :type dtype: str or NumPy dtype, optional
        :param name: name of image, defaults to None
        :type name: str, optional
        :param id: numeric id of image, typically a sequence number, defaults to None
        :type id: int, optional
        :param domain: domain of image, defaults to None
        :type domain: array_like(W), array_like(H), optional
        :raises TypeError: unknown type passed to constructor

        Create a new image instance which contains pixel values as well as 
        information about image size, datatype, color planes and domain.

        The ``image`` can be specified in several ways:

        - as an :class:`Image` instance and its pixel array will be *referenced*
          by the new :class:`Image`.
        - an a NumPy 2D or 3D array for a greyscale or color image respectively.
        - a lists of lists of pixel values, each inner list must have the same
          number of elements (columns).

        **Pixel datatype**

        The ``dtype`` of an image comes from the internal NumPy pixel array.
        If ``image`` is an :class:`Image` instance or NumPy ndarray then its dtype is
        determined by the NumPy ndarray.

        If ``image`` is given as a list of lists and ``dtype`` is not given,
        then the :class:`Image` type is:
        
        - float32 if the list contains any floating point values, otherwise
        - the smallest signed or unsigned int that can represent its
          value span.

        An ``image`` can have bool values.  When used in a numerical expression
        its values will bs cast to numeric values of 0 or 1 representing
        False and True respectively. 
        
        **Color planes**

        Images can have multiple planes, typically three (representing the 
        primary colors red, green and blue) but any number is possible. In
        the underlying Numpy array these planes are identified by an integer
        plane index (the last dimension of the 3D array).  The :class:`Image`
        contains a dictionary that maps the name of a color plane to its
        index value.  The color plane order can be specified as a dict or
        a string.

        **Image domain**

        An :class:`Image` has a width and height in units of pixesl, but for
        some applications it is useful to specify the pixel coordinates in other
        units, perhaps metres, or latitude/longitude, or for a spherical image
        as azimuth and colatitude.  The domain is specified by two 1D arrays
        that map the pixel coordinate to the domain variable.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> Image([[1, 2], [3, 4]])
            >>> Image(np.array([[1, 2], [3, 4]]))
            >>> Image([[1, 2], [3, 1000]])
            >>> Image([[0.1, 0.2], [0.3, 0.4]])
            >>> Image([[True, False], [False, True]])

        .. note:: By default the encapsulated pixel data is a reference to
            the passed image data.

        :seealso: :meth:`colororder` :meth:`colororder_str`
        """

        self._A = None
        self._name = None
        self._colororder = None
        self.id = id
        self.domain = domain

        if isinstance(name, Path):
            name = str(name)

        if isinstance(image, np.ndarray):
            pass

        elif isinstance(image, self.__class__):
            # Image instance
            name = image.name
            colororder = image.colororder
            image = image.A

        elif isinstance(image, list):
            # list of lists

            # attempt to convert it to an ndarray
            try:
                image = np.array(image)
            except VisibleDeprecationWarning:
                raise ValueError('bad argument passed to Image constructor')

            if dtype is None:
                # no type given, automatically choose it
                if np.issubdtype(image.dtype, np.floating):
                    # list contained a float
                    dtype = np.float32
                
                elif np.issubdtype(image.dtype, np.integer):
                    # list contained only ints, convert to int/uint8 of smallest
                    # size to contain all values
                    if image.min() < 0:
                        # value is signed
                        for type in ['int8', 'int16', 'int32']:
                            if (image.max() <= np.iinfo(type).max) and (image.min() >= np.iinfo(type).min):
                                dtype = np.dtype(type)
                                break
                    else:
                        # value is unsigned
                        for type in ['uint8', 'uint16', 'uint32']:
                            if image.max() <= np.iinfo(type).max:
                                dtype = np.dtype(type)
                                break
        else:
            raise ValueError('bad argument passed to Image constructor')

        # change type of array if dtype was specified
        if dtype is not None:
            image = image.astype(dtype)

        self.name = name

        if shape is not None:
            if image.ndim <= 2:
                # 2D image
                image = image.reshape(shape)
            elif image.ndim == 3:
                # 3D image
                image = image.reshape(shape + (-1,))

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
            >>> img = Image.Read('flowers1.png')
            >>> str(img)
        """
        s = f"Image: {self.width} x {self.height} ({self.dtype})"

        if self.iscolor:
            s += ", " + self.colororder_str
        if self.id is not None:
            s += f", id={self.id}"
        if self.name is not None:
            name = self.name
            # if it's a long name, take from rightmost / and add ellipsis
            if len(name) > 20:
                k = [i for i, c in enumerate(name) if c == '/']
                if len(k) >= 2:
                    name = name[k[-2]:]
                else:
                    name = name[-20:]
                name = "..." + name
            s += f" [{name}]"
        return s

    def print(self, fmt=None, seperator=' ', precision=2):
        """
        Print image pixels in compact format

        :param fmt: format string, defaults to None
        :type fmt: str, optional
        :param separator: value separator, defaults to single space
        :type separator: str, optional
        :param precision: precision for floating point pixel values, defaults to 2
        :type precision: int, optional

        Very compact display of pixel numerical values in grid layout.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Squares(1, 10)
            >>> img.print()
            >>> img = Image.Squares(1, 10, dtype='float')
            >>> img.print()
        
        .. note::
            - For a boolean image True and False are displayed as 1 and 0
              respectively.
            - For a multiplane images the planes are printed sequentially.

        :seealso: :meth:`showpixels`
        """
        if fmt is None:
            if self.isint:
                width = max(
                    len(str(self.max())),
                    len(str(self.min()))
                )
                fmt = f"{seperator}{{:{width}d}}"
            elif self.isbool:
                width = 1
                fmt = f"{seperator}{{:{width}d}}"
            elif self.isfloat:
                ff = f"{{:.{precision}f}}"
                width = max(
                    len(ff.format(self.max())),
                    len(ff.format(self.min()))
                    )
                fmt = f"{seperator}{{:{width}.{precision}f}}"                
        
        if self.iscolor:
            plane_names = self.colororder_str.split(':')
            for plane in range(self.nplanes):
                print(f"plane {plane_names[plane]}:")
                self.plane(plane).print()
        else:
            for v in self.vspan():
                row = ""
                for u in self.uspan():
                    row += fmt.format(self.image[v,u])
                print(row)

    def __repr__(self):
        """
        Single line summary of image parameters

        :return: single line summary of image
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> im
        """
        return str(self)

    def copy(self, copy=True):
        """
        Create image copy

        :param copy: copy the image data
        :type copy: bool
        :return: copy of image
        :rtype: :class:`Image`

        Create a new :class:`Image` instance which contains a copy of the
        original image data. If ``copy`` is False the new :class:`Image`
        instance contains a reference to the original image data.
        """
        return self.__class__(self, copy=copy)

    # ------------------------- properties ------------------------------ #

    @property
    def colororder(self):
        """
        Set/get color order of image

        The color order is a dict where the key is the color plane name, eg. 'R'
        and the value is the index of the plane in the 3D pixel value array.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.colororder
            >>> img.colororder = "BGR"
            >>> img.colororder

        When setting the color order the value can be any of:

        * simple string, one plane per character, eg. ``"RGB"``
        * colon separated string, eg. ``"R:G:B"``, ``"L*:a*:b*"``
        * dictionary, eg. ``dict(R=0, G=1, B=2)``

        For the first two cases the color plane indices are implicit in the
        order in the string.

        .. note:: Changing the color order does not change the order of the planes
            in the image array, it simply changes their label.

        :seealso: :meth:`colororder_str` :meth:`colordict` :meth:`plane` :meth:`red` :meth:`green` :meth:`blue` 
        """
        return self._colororder

    @colororder.setter
    def colororder(self, colororder):

        cdict = Image.colordict(colororder)

        if len(cdict) != self.nplanes:
            raise ValueError('colororder length does not match number of planes')
        self._colororder = cdict

    @staticmethod
    def colordict(colororder):
        """
        Parse a color order specification

        :param colororder: order of color channels
        :type colororder: str, dict
        :raises ValueError: ``colororder`` not a string or dict
        :return: dictionary mapping color names to plane indices
        :rtype: dict

        The color order the value can be given in a variety of forms:

        * simple string, one plane per character, eg. ``"RGB"``
        * colon separated string, eg. ``"R:G:B"``, ``"L*:a*:b*"``
        * dictionary, eg. ``dict(R=0, G=1, B=2)``

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> Image.colordict('RGB')
            >>> Image.colordict('red:green:blue')
            >>> Image.colordict({'L*': 0, 'U*': 1, 'V*': 2})
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
        return cdict

    @property
    def colororder_str(self):
        """
        Image color order as a string

        :return: Image color plane order as a colon separated string.
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.colororder_str
        
        :seealso: :meth:`colororder`
        """
        if self.colororder is not None:
            s = sorted(self.colororder.items(), key=lambda x: x[1])
            return ':'.join([x[0] for x in s])
        else:
            return ""

    @property
    def name(self):
        """
        Set/get image name

        An image has a string-valued name that can be read and written.
        The name is shown by the Image repr and when images are displayed
        graphically.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.name[-70:]
            >>> img.name = 'my image'
            >>> img.name

        .. note:: Images loaded from a file have their name initially set to
            the full file pathname.
        
        :seealso: :meth:`Read`
        """
        return self._name

    @name.setter
    def name(self, name):
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
            >>> img = Image.Read('flowers1.png')
            >>> img.isfloat
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.isfloat

        :seealso: :meth:`isint` :meth:`isbool`
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
            >>> img = Image.Read('flowers1.png')
            >>> img.isint
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.isint

        :seealso: :meth:`isfloat` :meth:`isbool`
        """
        return np.issubdtype(self.dtype, np.integer)

    @property
    def isbool(self):
        """
        Image has bolean values

        :return: True if image has boolean pixel values
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png') > 200
            >>> img.isint
            >>> img.isbool

        :seealso: :meth:`isint` :meth:`isfloat`
        """
        return np.issubdtype(self.dtype, np.bool_)

    @property
    def dtype(self):
        """
        Datatype of image

        :return: NumPy datatype of image
        :rtype: numpy.dtype

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.dtype
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.dtype

        :seealso: :meth:`to` :meth:`to_int` :meth:`to_float` :meth:`like` :meth:`cast` :func:`numpy.dtype`
        """
        return self.A.dtype

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
            >>> img = Image.Read('flowers1.png')
            >>> img.width
        
        :seealso: :meth:`height` :meth:`size` :meth:`umax`
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
            >>> img = Image.Read('flowers1.png')
            >>> img.height

        :seealso: :meth:`width` :meth:`size` :meth:`vmax`
        """
        return self.A.shape[0]

    @property
    def umax(self):
        """
        Image maximum u-coordinate

        :return: Maximum u-coordinate in image in pixels
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.umax

        :seealso: :meth:`width`
        """
        return self.A.shape[1] - 1

    @property
    def vmax(self):
        """
        Image maximum v-coordinate

        :return: Maximum v-coordinate in image in pixels
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.vmax

        :seealso: :meth:`height`
        """
        return self.A.shape[0] - 1


    def uspan(self, step=1):
        """
        Linear span of image horizontally

        :param step: step size, defaults to 1
        :type step: int, optional
        :return: 1D array of values [0 ... width-1]
        :rtype: ndarray(W)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> img = Image.Read('flowers1.png')
            >>> with np.printoptions(threshold=10):
            >>>     img.uspan()

        .. note:: If the image has a ``domain`` specified the horizontal
            component of this is returned instead.

        .. warning:: Computed using :meth:`numpy.arange` and for ``step>1`` the
            maximum coordinate may not be returned.
            
        :seealso: :meth:`umax` :meth:`vspan`
        """
        if self.domain is None:
            return np.arange(0, self.width, step)
        else:
            return self.domain[0]

    def vspan(self, step=1):
        """
        Linear span of image vertically

        :param step: step size, defaults to 1
        :type step: int, optional
        :return: 1D array of values [0 ... height-1]
        :rtype: ndarray(H)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> img = Image.Read('flowers1.png')
            >>> with np.printoptions(threshold=10):
            >>>     img.vspan()

        .. note:: If the image has a ``domain`` specified the vertical
            component of this is returned instead.

        .. warning:: Computed using :meth:`numpy.arange` and for ``step>1`` the
            maximum coordinate may not be returned.

        :seealso: :meth:`vmax` :meth:`uspan`
        """
        if self.domain is None:
            return np.arange(0, self.height, step)
        else:
            return self.domain[1]

    @property
    def size(self):
        """
        Image size

        :return: Size of image (width, height)
        :rtype: tuple

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.size

        :seealso: :meth:`width` :meth:`height`
        """
        return (self.A.shape[1], self.A.shape[0])

    @property
    def centre(self):
        """
        Coordinate of centre pixel

        :return: Coordinate (u,v) of the centre pixel
        :rtype: tuple

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Zeros((50,50))
            >>> img.centre
            >>> img = Image.Zeros((51,51))
            >>> img.centre

        .. note:: If the image has an even dimension the centre will lie
            between pixels.

        :seealso: :meth:`center` :meth:`centre_int`
        """
        return ((self.A.shape[1] - 1) / 2, (self.A.shape[0] - 1) / 2)

    @property
    def center(self):
        """
        Coordinate of center pixel

        :return: Coordinate (u,v) of the centre pixel
        :rtype: tuple

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Zeros((50,50))
            >>> img.center
            >>> img = Image.Zeros((51,51))
            >>> img.center

        .. note::
            - If the image has an even dimension the centre will lie
              between pixels.
            - Same as ``centre``, just US spelling

        :seealso: :meth:`center_int`
        """
        return self.centre

    @property
    def centre_int(self):
        """
        Coordinate of centre pixel as integer

        :return: Coordinate (u,v) of the centre pixel
        :rtype: tuple

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Zeros((50,50))
            >>> img.centre_int
            >>> img = Image.Zeros((51,51))
            >>> img.centre_int

        .. note:: If the image has an even dimension the centre coordinate will
              be truncated toward zero.

        :seealso: :meth:`centre`
        """
        return ((self.A.shape[1] - 1) // 2, (self.A.shape[0] - 1) // 2)

    @property
    def center_int(self):
        """
        Coordinate of centre pixel as integer

        :return: Coordinate (u,v) of the centre pixel
        :rtype: tuple

        Integer coordinate of centre pixel.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Zeros((50,50))
            >>> img.center_int
            >>> img = Image.Zeros((51,51))
            >>> img.center_int

        .. note::
            - If the image has an even dimension the centre coordinate will
              be truncated toward zero.
            - Same as ``centre_int``, just US spelling

        :seealso: :meth:`center`
        """
        return self.centre_int

    @property
    def npixels(self):
        """
        Number of pixels in image plane

        :return: Number of pixels in image plane: width x height
        :rtype: int

        .. note:: Number of planes is not considered.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.npixels
            >>> img.width * img.height

        :seealso: :meth:`size`
        """
        return self.A.shape[0] * self.A.shape[1]

    @property
    def shape(self):
        """
        Image shape

        :return: Shape of internal NumPy array
        :rtype: 2-tuple, or 3-tuple if color

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.shape
            >>> img = Image.Read('street.png')
            >>> img.shape

        :seealso: :meth:`nplanes` :meth:`ndim` :meth:`iscolor`
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
            >>> img = Image.Read('flowers1.png')
            >>> img.ndim
            >>> img = Image.Read('street.png')
            >>> img.ndim
        
        :seealso: :meth:`nplanes` :meth:`shape`
        """
        return self.A.ndim


    def contains(self, p):
        """
        Test if coordinate lies within image

        :param p: pixel coordinate
        :type p: array_like(2), ndarray(2,N)
        :return: whether pixel coordinate lies within image bounds
        :rtype: bool, ndarray(N)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> img = Image.Zeros(10)
            >>> img.contains((4,6))
            >>> img.contains((-1, 7))
            >>> img.contains(np.array([[4, 6], [-1, 7], [10, 10]]).T)
        """
        if isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[0] == 2:
            u = p[0, :]
            v = p[1, :]
        else:
            u = p[0]
            v = p[1]
        
        return np.logical_and.reduce((u >= 0, v >= 0, u < self.width, v < self.height))


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
            >>> img = Image.Read('flowers1.png')
            >>> img.iscolor
            >>> img = Image.Read('street.png')
            >>> img.iscolor

        :seealso: :meth:`isrgb` :meth:`nplanes`
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
            >>> img = Image.Read('flowers1.png')
            >>> img.isbgr

        .. note:: Is False if image is not color.

        :seealso: :meth:`colororder`
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
            >>> img = Image.Read('flowers1.png')
            >>> img.isrgb

        .. note:: Is False if image is not color.

        :seealso: :meth:`colororder`
        """
        return self.colororder_str == 'R:G:B'

    def to(self, dtype):
        """
        Convert image datatype

        :param dtype: Numpy data type
        :type dtype: str
        :return: image
        :rtype: :class:`Image`

        Create a new image, same size as input image, with pixels of a different
        datatype.  Integer values are scaled according to the maximum value
        of the datatype, floating values are in the range 0.0 to 1.0.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Random(3)
            >>> img.image
            >>> img.to('float').image
            >>> img = Image.Random(3, dtype='float')
            >>> img.image
            >>> img.to('uint8').image

        :seealso: :meth:`astype` :meth:`to_int` :meth:`to_float`
        """
        # convert image to different type, does rescaling
        # as just changes type
        dtype = np.dtype(dtype)  # convert to dtype if it's a string

        if np.issubdtype(dtype, np.integer):
            out = self.to_int(dtype)
        elif np.issubdtype(dtype, np.floating):
            out = self.to_float(dtype)
        return self.__class__(out, dtype=dtype)

    def astype(self, dtype):
        """
        Cast image datatype

        :param dtype: Numpy data type
        :type dtype: str
        :return: image
        :rtype: :class:`Image`

        Create a new image, same size as input image, with pixels of a different
        datatype.  Values are retained, only the datatype is changed.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Random(3)
            >>> img.image
            >>> img.astype('float').image
            >>> img = Image.Random(3, dtype='float')
            >>> img.image
            >>> img.astype('uint8').image

        :seealso: :meth:`to`
        """
        return self.__class__(self.A.astype(dtype), dtype=dtype)
    # ---- NumPy array access ---- #

    @property
    def image(self):
        """
        Image as NumPy array

        :return: image as a NumPy array
        :rtype: ndarray(H,W) or ndarray(H,W,3)

        Return a reference to the encapsulated NumPy array that holds the pixel
        values.
        
        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img
            >>> type(img)
            >>> type(img.image)

        .. note:: For a color image the color plane order is given by the
            colororder dictionary.

        :seealso: :meth:`A` :meth:`colororder`
        """
        return self._A

    @property
    def A(self):
        """
        Set/get the NumPy array containing pixel values

        **Getting**

        :return: image as a NumPy array
        :rtype: ndarray(H,W) or ndarray(H,W,3)

        Return a reference to the encapsulated NumPy array that holds the pixel
        values.

        **Setting**

        Replace the encapsulated NumPy array with another.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> img = Image.Read('flowers1.png')
            >>> img
            >>> type(img)
            >>> type(img.A)
            >>> img.A = np.zeros((50,50))
            >>> img

        :seealso: :meth:`image`
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
        :rtype: ndarray(H,W,3)

        The image is guaranteed to be in RGB order irrespective of current color order.

        :seealso: :meth:`image` :meth:`bgr` :meth:`colororder`
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
        :rtype: ndarray(H,W,3)

        The image is guaranteed to be in BGR (OpenCV standard) order irrespective of current color order.

        :seealso: :meth:`image` :meth:`rgb` :meth:`colororder`
        """
        if not self.iscolor:
            raise ValueError('greyscale image has no bgr property')
        if self.isbgr:
            return self.A
        elif self.isrgb:
            return self.A[:, :, ::-1]

    # ------------------------- datatype operations ----------------------- #

    def to_int(self, intclass='uint8'):
        """
        Convert image to integer NumPy array

        :param intclass: name of NumPy supported integer class, default is 'uint8'
        :type intclass: str, optional
        :return: NumPy array with integer values
        :rtype: ndarray(H,W) or ndarray(H,W,P)

        Return a NumPy array with pixels converted to the integer class 
        ``intclass``.  For the case where the input image is:

        * a floating point class, the pixel values are scaled from an
          input range of [0,1] to a range spanning zero to the maximum positive
          value of the output integer class.
        * an integer class, then the pixels are scaled and cast to ``intclass``.
          The scale factor is the ratio of the maximum value of the input and
          output integer classes.
        * boolean class, False is mapped to zero and True is mapped to the
          maximum positive value.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[5_000, 10_000], [30_000, 60_000]])
            >>> img
            >>> img.to_int('uint8')
            >>> img.to_int('uint32')
            >>> img = Image([[0.0, 0.3], [0.5, 1.0]])
            >>> img
            >>> img.to_int('uint8')
            >>> img = Image([[False, True], [True, False]])
            >>> img
            >>> img.to_int('uint8')
            >>> img.to_int('uint16')

        .. note:: Works for greyscale or color (arbitrary number of planes) image

        :seealso: :func:`to_float` :meth:`cast` :meth:`like` 
        """
        return int_image(self.image, intclass)

    def to_float(self, floatclass='float32'):
        """
        Convert image to float NumPy array

        :param floatclass: 'single', 'double', 'float32' [default], 'float64'
        :type floatclass: str
        :return: Image with floating point pixel types
        :rtype: :class:`Image`

        Return a NumPy array with pixels converted to the floating point class ``floatclass``
        and the values span the range 0 to 1. For the case where the input image
        is:
        
        * an integer class, the pixel values are scaled from an input range
          spanning zero to the maximum positive value of the integer
          class to [0.0, 1.0]
        * a floating class, the pixels are cast to change type but not their
          value.
        * boolean class, False is mapped to 0.0 and True is mapped to 1.0.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[50, 100], [150, 200]])
            >>> img
            >>> img.to_float()
            >>> img = Image([[0.0, 0.3], [0.5, 1.0]])
            >>> img
            >>> img.to_float('float64')
            >>> img = Image([[False, True], [True, False]])
            >>> img.to_float()    

        .. note:: Works for greyscale or color (arbitrary number of planes) image

        :seealso: :meth:`to_int` :meth:`cast` :meth:`like`
        """
        return float_image(self.image, floatclass)


    def cast(self, value):
        """
        Cast value to same type as image

        :param value: value to cast
        :type value: scalar, ndarray
        :return: value cast to same type as image
        :rtype: numpy type, ndarray

        The value, scalar or integer, is **cast** to the same type as the image.  
        The result has the same numeric value, but the type is changed.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> x = img.cast(12.5)
            >>> x
            >>> type(x)

        .. note:: Scalars are cast to NumPy types not native Python types.

        :seealso: :meth:`like`
        """
        return self.A.dtype.type(value)

    def like(self, value, maxint=None):
        """
        Convert value to the same type as image

        :param value: scalar or NumPy array
        :type value: scalar, ndarray
        :param maxint: maximum integer value for an integer image, defaults to
            maximum positive value of the class
        :type maxint: int, optional
        :raises ValueError: [description]
        :return: converted value
        :rtype: NumPy type

        The value, scalar or integer, is **converted** to the same type as the
        image. The result is optionally rescaled and cast:

        * Float to float: values are cast
        * Float to int: values in the interval [0, 1] are scaled to the interval
          [0, ``maxint``] and then cast
        * Int to float: values in the interval [0, ``maxint``] are scaled to the
          interval [0, 1] and then cast.
        * Int to int: values are scaled and cast.

        .. warning:: For integer to integer conversion, integer values  greater
            than the maximum value of the destination class are wrapped not
            clipped.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.like(0.5)

        :seealso: :meth:`cast` :meth:`to_int` :meth:`to_float`

        """
        if self.isint:
            # matching to an integer image
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.integer) or isinstance(value, (int, np.integer)):
                # already an integer, cast it to right sort
                return self.cast(value)
            else:
                # it's a float, rescale it then cast
                return self.cast(value * self.maxval)
        else:
            # matching to a float image
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating) or isinstance(value, (float, np.floating)):
                # already a float of some sort, cast it to the right sort
                return self.cast(value)
            else:
                # it's an int.  We use hints to determine the size, otherwise
                # get it from the type
                if maxint is None:
                    maxint = np.iinfo(value.dtype).max
                elif isinstance(maxint, int):
                    pass
                elif isinstance(maxint, str) or isinstance(maxint, np.dtype):
                    maxint = np.iinfo(maxint).max
                else:
                    raise ValueError('bad max value specified')
                return self.cast(value / maxint)


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
            >>> img = Image.Zeros(20, dtype='float32')
            >>> img.minval
            >>> img = Image.Zeros(20, dtype='uint8')
            >>> img.minval
        
        :seealso: :meth:`maxval`
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
            >>> img = Image.Zeros(20, dtype='float32')
            >>> img.maxval
            >>> img = Image.Zeros(20, dtype='uint8')
            >>> img.maxval
        
        :seealso: :meth:`minval`
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
            >>> img = Image.Zeros(20, dtype='float32')
            >>> img.true
            >>> img = Image.Zeros(20, dtype='uint8')
            >>> img.true
        
        :seealso: :meth:`false` :meth:`maxval`
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
            >>> img = Image.Zeros(20, dtype='float32')
            >>> img.false
            >>> img = Image.Zeros(20, dtype='uint8')
            >>> img.false
        
        :seealso: :meth:`true`
        """
        return 0
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
            >>> img = Image.Read('street.png')
            >>> img.nplanes
            >>> img = Image.Read('flowers1.png')
            >>> img.nplanes

        .. note:: A greyscale image is stored internally as a 2D NumPy array
            which has zero planes, but ``nplanes`` will return ` in that case.

        :seealso: :meth:`shape` :meth:`ndim`
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
        :return out: image containing only the selected planes
        :rtype: :class:`Image`

        Create a new image from the selected planes of the image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("flowers4.png") # in BGR order
            >>> img.colororder_str
            >>> img.nplanes
            >>> red = img.plane(0) # red plane
            >>> red
            >>> red.iscolor
            >>> red.nplanes
            >>> green_blue = img.plane('G:B') # green and blue planes
            >>> green_blue
            >>> green_blue.iscolor
            >>> green_blue.nplanes
            >>> red_blue = img.plane([0, 2]) # blue and red planes
            >>> red_blue

        .. note:: 
            - This can also be performed using the overloaded ``__getitem__``
              operator.
            - To select more than one plane, use either a sequence of integers or a string
              of colon separated plane names.

        :seealso: :meth:`red` :meth:`green` :meth:`blue` :meth:``__getitem__``
        """
        if not self.iscolor:
            raise ValueError('cannot extract color plane from greyscale image')

        if isinstance(planes, int):
            if planes < 0 or planes >= self.nplanes:
                raise ValueError('plane index out of range')
            iplanes = planes
            colororder = None
            planes = [planes]
        elif isinstance(planes, str):
            iplanes = []
            colororder = {}
            if ':' in planes:
                planes = planes.split(":")
                planes = [p for p in planes if p != '']
            for plane in planes:
                try:
                    i = self.colororder[plane]
                    iplanes.append(i)
                    colororder[plane] = len(colororder)  # copy to new dict
                except KeyError:
                    raise ValueError('bad plane name specified')
        elif isinstance(planes, (tuple, list)):
            colororder = {}
            for plane in planes:
                if not isinstance(plane, int) or plane < 0 or plane >= self.nplanes:
                    raise ValueError('plane index invalid or out of range')
                colorname = [k for k, v in self.colororder.items() if v == plane][0]
                colororder[colorname] = plane
            iplanes = planes
        else:
            raise ValueError('bad plane specified')

        if isinstance(iplanes, list) and len(iplanes) == 1:
                iplanes = iplanes[0]
                colororder = None
        return self.__class__(self.A[:, :, iplanes], colororder=colororder)

    def __getitem__(self, key):
        """
        Extract slice of image

        :param key: slice to extract
        :type planes: int, str, tuple of slice
        :return: slice of image
        :rtype: :class:`Image`

        Create a new image from the selected slice of the image, either a plane
        or a region of interest.  If ``key`` is:

        - an int, select this plane
        - a string, select this named plane or planes
        - a 2-tuple of slice objects, select this uv-region across all planes
        - a 3-tuple of slice objects, select this region of uv and planes

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("flowers4.png") # in RGB order
            >>> red = img[0] # red plane
            >>> red
            >>> green = img["G"]
            >>> green
            >>> roi = img[100:200, 300:400]
            >>> roi
            >>> roi = img[100:200, 300:400, 1:]
            >>> roi

        :seealso: :meth:`red` :meth:`green` :meth:`blue` :meth:`plane` :meth:`roi`
        """
        if isinstance(key, int):
            return self.__class__(self.image[...,key])
        elif isinstance(key, str):
            return self.plane(key)
        elif isinstance(key, (list, tuple)):
            if self.iscolor and len(key) == 2:
                key = (key[0], key[1], slice(None))
            out = self.image[key]
            colororder = None
            if out.ndim == 3:
                colororder = self.colororder_str.split(":")
                colororder = colororder[key[2]]
                colororder = ":".join(colororder)
            return self.__class__(out, colororder=colororder)

        else:
            raise ValueError('invalid slice')

    def red(self):
        """
        Extract the red plane of a color image

        :raises ValueError: if image is not color
        :return out: greyscale image representing the red image plane
        :rtype: :class:`Image`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("flowers4.png")
            >>> red = img.red() # red plane
            >>> red
            >>> red.iscolor

        :seealso: :meth:`plane` :meth:`green` :meth:`blue` 
        """
        return self.plane('R')

    def green(self):
        """
        Extract the green plane of a color image

        :raises ValueError: if image is not color
        :return out: greyscale image representing the green image plane
        :rtype: :class:`Image`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("flowers4.png")
            >>> green = img.green() # green plane
            >>> green
            >>> green.iscolor

        :seealso: :meth:`plane` :meth:`red` :meth:`blue` 
        """
        return self.plane('G')

    def blue(self):
        """
        Extract the blue plane of a color image

        :raises ValueError: if image is not color
        :return out: greyscale image representing the blue image plane
        :rtype: :class:`Image`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("flowers4.png")
            >>> blue = img.blue() # blue plane
            >>> blue
            >>> blue.iscolor

        :seealso: :meth:`plane` :meth:`red` :meth:`green` 
        """
        return self.plane('B')



    # I think these are not used anymore

    # def astype(self, type):
    #     return self.__class__(self.A, dtype=type)


    # ------------------------- operators ------------------------------ #

    # arithmetic
    def __mul__(self, other):
        """
        Overloaded ``*`` operator

        :return: elementwise product of images
        :rtype: :class:`Image`

        Compute the product of an Image with another image or a scalar.
        Supports:

        * image ``*`` image, elementwise
        * scalar ``*`` image
        * image ``*`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img * img
            >>> z.image
            >>> z = 2 * img
            >>> z.image

        ..warning:: Values will be wrapped not clipped to the range of the
            pixel datatype.
        """
        return self._binop(self, other, lambda x, y: x * y)

    def __rmul__(self, other):

        return self._binop(self, other, lambda x, y: y * x)

    def __pow__(self, other):
        """
        Overloaded ``**`` operator

        :return: elementwise exponent of image
        :rtype: :class:`Image`

        Compute the elementwise power of an Image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img**3
            >>> z.image

        ..warning:: Values will be wrapped not clipped to the range of the
            pixel datatype.
        """
        if not isscalar(other):
            raise ValueError('exponent must be a scalar')
        return self._binop(self, other, lambda x, y: x ** y)

    def __add__(self, other):
        """
        Overloaded ``+`` operator

        :return: elementwise addition of images
        :rtype: :class:`Image`

        Compute the sum of an Image with another image or a scalar.
        Supports:

        * image ``+`` image, elementwise
        * scalar ``+`` image
        * image ``+`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img + img
            >>> z.image
            >>> z = 10 + img
            >>> z.image

        ..warning:: Values will be wrapped not clipped to the range of the
            pixel datatype.
        """
        return self._binop(self, other, lambda x, y: x + y)

    def __radd__(self, other):
        return self._binop(self, other, lambda x, y: y + x)

    def __sub__(self, other):
        """
        Overloaded ``-`` operator

        :return: elementwise subtraction of images
        :rtype: :class:`Image`

        Compute the difference of an Image with another image or a scalar.
        Supports:

        * image ``-`` image, elementwise
        * scalar ``-`` image
        * image ``-`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img - img
            >>> z.image
            >>> z = img - 1
            >>> z.image

        ..warning:: Values will be wrapped not clipped to the range of the
            pixel datatype.
        """
        return self._binop(self, other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self._binop(self, other, lambda x, y: y - x)

    def __truediv__(self, other):
        """
        Overloaded ``/`` operator

        :return: elementwise division of images
        :rtype: :class:`Image`

        Compute the quotient of an Image with another image or a scalar.
        Supports:

        * image ``/`` image, elementwise
        * scalar ``/`` image
        * image ``/`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img / img
            >>> z.image
            >>> z = img / 2
            >>> z.image

        .. note:: The resulting values are floating point.
        """
        return self._binop(self, other, lambda x, y: x / y)

    def __rtruediv__(self, other):
        return self._binop(self, other, lambda x, y: y / x)

    def __floordiv__(self, other):
        """
        Overloaded ``//`` operator

        :return: elementwise floored division of images
        :rtype: :class:`Image`

        Compute the integer quotient of an Image with another image or a scalar.
        Supports:

        * image ``//`` image
        * scalar ``//`` image
        * image ``//`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img / 2
            >>> z.image
            >>> z = img // 2
            >>> z.image
        """
        return self._binop(self, other, lambda x, y: x // y)

    def __rfloordiv__(self, other):
        return self._binop(self, other, lambda x, y: y // x)

    def __minus__(self):
        """
        Overloaded unary ``-`` operator

        :return: elementwise negation of image
        :rtype: :class:`Image`

        
        Compute the elementwise negation of an Image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, -2], [-3, 4]], 'int8')
            >>> z = -img
            >>> z.image
        """
        return self._unop(self, lambda x: -x)

    # bitwise
    def __and__(self, other):
        """
        Overloaded ``&`` operator

        :return: elementwise binary-and of images
        :rtype: :class:`Image`

        Compute the binary-and of an Image with another image or a scalar.
        Supports:

        * image ``&`` image
        * scalar ``&`` image
        * image ``&`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img & Image([[2, 2], [2, 2]])
            >>> z.image
            >>> z = img & 1
            >>> z.image
        """
        return self._binop(self, other, lambda x, y: x & y)

    def __or__(self, other):
        """
        Overloaded ``|`` operator

        :return: elementwise binary-or of images
        :rtype: :class:`Image`

        Compute the binary-or of an Image with another image or a scalar.
        Supports:

        * image ``|`` image
        * scalar ``|`` image
        * image ``|`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img | Image([[2, 2], [2, 2]])
            >>> z.image
            >>> z = img | 1
            >>> z.image
        """
        return self._binop(self, other, lambda x, y: x | y)

    def __xor__(self, other):
        """
        Overloaded ``^`` operator

        :return: elementwise binary-xor of images
        :rtype: :class:`Image`

        Compute the binary-xor of an Image with another image or a scalar.
        Supports:

        * image ``^`` image
        * scalar ``^`` image
        * image ``^`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img ^ Image([[2, 2], [2, 2]])
            >>> z.image
            >>> z = img ^ 1
            >>> z.image
        """
        return self._binop(self, other, lambda x, y: x ^ y)

    def __lshift__(self, other):
        """
        Overloaded ``<<`` operator

        :return: elementwise binary-left-shift of images
        :rtype: :class:`Image`

        Left shift pixel values in an Image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img  << 1
            >>> z.image
        """
        if not isinstance(other, int):
            raise ValueError('left shift must be by integer amount')
        return self._binop(self, other, lambda x, y: x << y)

    def __rshift__(self, other):
        """
        Overloaded ``>>`` operator

        :return: elementwise binary-right-shift of images
        :rtype: :class:`Image`

        Right shift pixel values in an Image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img >> 2
            >>> z.image
        """
        if not isinstance(other, int):
            raise ValueError('left shift must be by integer amount')
        return self._binop(self, other, lambda x, y: x >> y)

    # relational
    def __eq__(self, other):
        """
        Overloaded ``==`` operator

        :return: elementwise comparison of pixels
        :rtype: bool :class:`Image`

        Compute the equality between an Image and another image or a scalar.
        Supports:

            * image ``==`` image
            * scalar ``==`` image
            * image ``==`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img == 2
            >>> z.image
            >>> z = img == Image([[0, 2], [3, 4]])
            >>> z.image

        :seealso: :meth:`true` :meth:`false`
        """
        return self._binop(self, other, lambda x, y: x == y)

    def __ne__(self, other):
        """
        Overloaded ``!=`` operator

        :return: elementwise comparison of pixels
        :rtype: bool :class:`Image`

        Compute the inequality between an Image and another image or a scalar.
        Supports:

            * image ``!=`` image
            * scalar ``!=`` image
            * image ``!=`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img != 2
            >>> z.image
            >>> z = img != Image([[0, 2], [3, 4]])
            >>> z.image

        :seealso: :meth:`true` :meth:`false`
        """
        return self._binop(self, other, lambda x, y: x != y)

    def __gt__(self, other):
        """
        Overloaded ``>`` operator

        :return: elementwise comparison of pixels
        :rtype: bool :class:`Image`

        Compute the inequality between an Image and another image or a scalar.
        Supports:
        
            * image ``>`` image
            * scalar ``>`` image
            * image ``>`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img > 2
            >>> z.image
            >>> z = img > Image([[0, 2], [3, 4]])
            >>> z.image

        :seealso: :meth:`true` :meth:`false`
        """
        return self._binop(self, other, lambda x, y: x > y)

    def __ge__(self, other):
        """
        Overloaded ``>=`` operator

        :return: elementwise comparison of pixels
        :rtype: bool :class:`Image`

        Compute the inequality between an Image and another image or a scalar.
        Supports:

            * image ``>=`` image
            * scalar ``>=`` image
            * image ``>=`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img >= 2
            >>> z.image
            >>> z = img >= Image([[0, 2], [3, 4]])
            >>> z.image

        :seealso: :meth:`true` :meth:`false`
        """
        return self._binop(self, other, lambda x, y: x >= y)

    def __lt__(self, other):
        """
        Overloaded ``<`` operator

        :return: elementwise comparison of images
        :rtype: bool :class:`Image`

        Compute the inequality between an Image and another image or a scalar.
        Supports:

            * image ``<`` image
            * scalar ``<`` image
            * image ``<`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img < 2
            >>> z.image
            >>> z = img < Image([[10, 2], [3, 4]])
            >>> z.image

        :seealso: :meth:`true` :meth:`false`
        """
        return self._binop(self, other, lambda x, y: x < y)

    def __le__(self, other):
        """
        Overloaded ``<=`` operator

        :return: elementwise comparison of images
        :rtype: bool :class:`Image`

        Compute the inequality between an Image and another image or a scalar.
        Supports:

            * image ``<=`` image
            * scalar ``<=`` image
            * image ``<=`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img <= 2
            >>> z.image
            >>> z = img <= Image([[0, 2], [3, 4]])
            >>> z.image

        :seealso: :meth:`true` :meth:`false`
        """
        return self._binop(self, other, lambda x, y: x <= y)

    def __invert__(self):
        """
        Overloaded ``~`` operator

        :return: elementwise inversion of logical values
        :rtype: boo, :class:`Image`

        Returns logical not operation where image values are interpretted as:

            * floating image: True is 1.0, False is 0.0
            * integer image: True is maximum value, False is 0 True is 1 and False is 0.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[True, False], [False, True]])
            >>> z = ~img
            >>> z.image
        """

        return self._unop(self, lambda x: ~x)

    @staticmethod
    def _binop(left, right, op, logical=False):
        if isinstance(right, left.__class__):
            # both images
            if left.nplanes == right.nplanes :
                return left.__class__(op(left.A, right.A), colororder=left.colororder)
            elif left.nplanes > 1 and right.nplanes == 1:
                # left image is multiplane, right is singleton
                out = []
                for i in range(left.nplanes):
                    out.append(op(left.A[:,:,i], right.A))
                return left.__class__(np.stack(out, axis=2), colororder=left.colororder)
            elif left.nplanes == 1 and right.nplanes > 1:
                # right image is multiplane, left is singleton
                out = []
                for i in range(right.nplanes):
                    out.append(op(left.A, right.A[:,:,i]))
                return right.__class__(np.stack(out, axis=2), colororder=right.colororder)
            else:
                raise ValueError('planes mismatch')
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
            out = np.where(op(left.A, right), true, false)

        return left.__class__(out, colororder=left.colororder)

    @staticmethod
    def _unop(left, op):
        return left.__class__(op(left.A), colororder=left.colororder)


    # ---------------------------- functions ---------------------------- #

    def abs(self):
        """
        Absolute value of image

        :return: elementwise absolute value of image
        :rtype: :class:`Image`

        Return elementwise absolute value of pixel values.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[-1, 2], [3, -4]], dtype='int8')
            >>> z = img.abs()
            >>> z.image            
        """
        return self._unop(self, np.abs)

    def sqrt(self):
        """
        Square root of image

        :return: elementwise square root of image
        :rtype: :class:`Image`

        Return elementwise square root of pixel values.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img.sqrt()
            >>> z.image
        """
        return self._unop(self, np.sqrt)

    def sum(self, *args, **kwargs):
        r"""
        Sum of all pixels

        :param args: additional positional arguments to :func:`numpy.sum`
        :param kwargs: additional keyword arguments to :func:`numpy.sum`
        :return: sum

        Computes the sum of pixels in the image:
        
        .. math::
        
            \sum_{uvc} I_{uvc}

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.sum()  # R+G+B
            >>> img.sum(axis=(0,1)) # sum(R), sum(G), sum(B)
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.sum(axis=2)


        .. note::
            - The return value type is the same as the image type.
            - By default the result is a scalar computed over all pixels,
              if the ``axis`` option is given the results is a 1D or 2D NumPy
              array.

        :seealso: :func:`numpy.sum` :meth:`~~machinevisiontoolbox.ImageWholeFeatures.ImageWholeFeaturesMixin.mpq` 
            :meth:`~machinevisiontoolbox.ImageWholeFeatures.ImageWholeFeaturesMixin.npq` 
            :meth:`~machinevisiontoolbox.ImageWholeFeatures.ImageWholeFeaturesMixin.upq`
        """
        return np.sum(self.A, *args, **kwargs)

    def min(self, *args, **kwargs):
        """
        Minimum value of all pixels

        :param args: additional positional arguments to :func:`numpy.min`
        :param kwargs: additional keyword arguments to :func:`numpy.min`
        :return: minimum value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.min()
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.min(axis=2)

        .. note::
            - The return value type is the same as the image type.
            - By default the result is a scalar computed over all pixels,
              if the ``axis`` option is given the results is a 1D or 2D NumPy
              array.

        :seealso: :func:`numpy.min`
        """
        return np.min(self.A, *args, **kwargs)

    def max(self, *args, **kwargs):
        """
        Maximum value of all pixels

        :param args: additional positional arguments to :func:`numpy.max`
        :param kwargs: additional keyword arguments to :func:`numpy.max`
        :return: maximum value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.max()
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.max(axis=2)

        .. note::
            - The return value type is the same as the image type.
            - By default the result is a scalar computed over all pixels,
              if the ``axis`` option is given the results is a 1D or 2D NumPy
              array.

        :seealso: :func:`numpy.max`
        """
        return np.max(self.A, *args, **kwargs)

    def mean(self, *args, **kwargs):
        """
        Mean value of all pixels

        :param args: additional positional arguments to :func:`numpy.mean`
        :param kwargs: additional keyword arguments to :func:`numpy.mean`
        :return: mean value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.mean()
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.mean(axis=2)

        .. note::
            - The return value type is the same as the image type.
            - By default the result is a scalar computed over all pixels,
              if the ``axis`` option is given the results is a 1D or 2D NumPy
              array.

        :seealso: :func:`numpy.mean`
        """
        return np.mean(self.A, *args, **kwargs)

    def std(self, *args, **kwargs):
        """
        Standard deviation of all pixels

        :param args: additional positional arguments to :func:`numpy.std`
        :param kwargs: additional keyword arguments to :func:`numpy.std`
        :return: standard deviation value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.std()
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.std()

        .. note::
            - The return value type is the same as the image type.
            - By default the result is a scalar computed over all pixels,
              if the ``axis`` option is given the results is a 1D or 2D NumPy
              array.

        :seealso: :func:`numpy.std`
        """
        return np.std(self.A, *args, **kwargs)

    def var(self, *args, **kwargs):
        """
        Variance of all pixels

        :param args: additional positional arguments to :func:`numpy.var`
        :param kwargs: additional keyword arguments to :func:`numpy.var`
        :return: variance value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.var()
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.var()

        .. note::
            - The return value type is the same as the image type.
            - By default the result is a scalar computed over all pixels,
              if the ``axis`` option is given the results is a 1D or 2D NumPy
              array.

        :seealso: :func:`numpy.var`
        """
        return np.std(self.A, *args, **kwargs)

    def median(self, *args, **kwargs):
        """
        Median value of all pixels

        :param args: additional positional arguments to :func:`numpy.median`
        :param kwargs: additional keyword arguments to :func:`numpy.median`
        :return: median value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.median()
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.median()

        .. note::
            - The return value type is the same as the image type.
            - By default the result is a scalar computed over all pixels,
              if the ``axis`` option is given the results is a 1D or 2D NumPy
              array.

        :seealso: :func:`numpy.median`
        """
        return np.median(self.A, *args, **kwargs)

    def stats(self):
        """
        Display pixel value statistics

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.stats()
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

    # ---------------------------- graphics ---------------------------- #

    def draw_line(self, start, end, **kwargs):
        """
        Draw line into image

        :param start: start coordinate (u,v)
        :type start: array_like(2)
        :param end: end coordinate (u,v)
        :type end: array_like(2)
        :param kwargs: parameters passed to :func:`~machinevisiontoolbox.base.graphics.draw_line`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Zeros(100)
            >>> img.draw_line((20,30), (60,70), color=200)
            >>> img.disp()
            >>> img = Image.Zeros(100, colororder="RGB")
            >>> img.draw_line((20,30), (60,70), color=[0, 200, 0]) # green line
            >>> img.disp()

        .. note:: If the image has N planes the color should have N elements.

        :seealso: :func:`~machinevisiontoolbox.base.graphics.draw_line`
        """
        draw_line(self.image, start, end, **kwargs)

    def draw_circle(self, centre, radius, **kwargs):
        """
        Draw circle into image

        :param centre: centre coordinate (u,v)
        :type centre: array_like(2)
        :param radius: circle radius in pixels
        :type radius: int
        :param kwargs: parameters passed to :func:`~machinevisiontoolbox.base.graphics.draw_circle`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Zeros(100)
            >>> img.draw_circle((20,30), 15, color=200)
            >>> img.disp()
            >>> img = Image.Zeros(100, colororder="RGB")
            >>> img.draw_circle((20,30), 15, color=[0, 200, 0], thickness=-1) # filled green circle
            >>> img.disp()

        .. note:: If the image has N planes the color should have N elements.

        :seealso: :func:`~machinevisiontoolbox.base.graphics.draw_circle` 
        """

        draw_circle(self.image, centre, radius, **kwargs)

    def draw_box(self, **kwargs):
        """
        Draw box into image

        :param kwargs: parameters passed to :func:`~machinevisiontoolbox.base.graphics.draw_box`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Zeros(100)
            >>> img.draw_box(lt=(20,70), rb=(60,30), color=200)
            >>> img.disp()
            >>> img = Image.Zeros(100, colororder="RGB")
            >>> img.draw_box(lt=(20,70), rb=(60,30), color=[0, 200, 0], thickness=-1)  # filled green box
            >>> img.disp()

        .. note:: If the image has N planes the color should have N elements.

        :seealso: :func:`~machinevisiontoolbox.base.graphics.draw_box`
        """
        draw_box(self.image, **kwargs)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    import pathlib
    import os.path
    from machinevisiontoolbox import Image

    # street = Image.Read("street.png")
    # subimage = street[100:200, 200:300]

    flowers = Image.Read("flowers8.png")

    flowers.stats()

    print(flowers[100:200, 100:200])
    print(flowers[100:200, 100:200, 1:])


    # Image.Constant(5, value='r').print()
    # img = Image.Squares(1, 20) > 0
    # img.print()

    # flowers = Image.Read("flowers8.png")
    # print(flowers)
    # z = flowers.plane("G:B:R")
    # print(z)

    # im = Image.Read("street.png")
    # print(im.image[10,20])
    # print(im[10,20])
    
    # exec(open(pathlib.Path(__file__).parent.parent.absolute() / "tests" / "test_core.py").read())  # pylint: disable=exec-used