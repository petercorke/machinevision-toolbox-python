"""
Factory class methods for constructing Image objects with common pixel patterns.
"""

from __future__ import annotations

import os
import os.path
import urllib
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from typing import Self
    from machinevisiontoolbox._image_typing import _ImageBase

import cv2
import numpy as np
from spatialmath import Polygon2
from spatialmath import base as smb
from spatialmath.base import islistof, isscalar

# from numpy.lib.arraysetops import isin
from machinevisiontoolbox.base import float_image, int_image, name2color
from machinevisiontoolbox.base.imageio import convert, idisp, iread, iwrite
from machinevisiontoolbox.Kernel import Kernel
from machinevisiontoolbox.mvtb_types import Dtype

# import spatialmath.base.argcheck as argcheck


# TODO
#  - get rid of w, h, just use size
#  - use colornames for fg and bg
#  - consistently use fg and bg for foreground and background values
#

def _getshape(
    cls,
    w: int | Sequence[int] | None,
    h: int | None,
    colororder: str | None,
    size: int | Sequence[int] | None,
):
    p = None

    if size is not None:
        if w is not None or h is not None:
            raise ValueError("specify either size= or w,h")
        if isinstance(size, Iterable):
            w, h = size[:2]
        else:
            w, h = size, size

    if h is None:
        if isinstance(w, (tuple, list)):
            h = w[1]
            w = w[0]
        elif isinstance(w, int):
            h = w

    if w is None or h is None:
        raise ValueError("dimensions not specified by size, w, h")
    shape = [h, w]

    if size is not None and isinstance(size, Iterable) and len(size) == 3:
        p = size[2]

    if colororder is not None:
        if p is not None:
            if len(cls.colordict(colororder)) != p:
                raise ValueError("colororder length does not match number of planes")
        p = len(cls.colordict(colororder))

    if p is not None:
        shape.append(p)

    return shape


def _resolve_pattern_options(
    cls,
    *,
    size: Any,
    dtype: Dtype | None,
    colororder: str | None,
    like: Any,
    default_size: Any,
    default_dtype: Dtype,
):
    if like is not None and not isinstance(like, cls):
        raise TypeError("like must be an Image")

    if size is None:
        if like is not None:
            size = like.size
        elif default_size is not None:
            size = default_size
        else:
            raise ValueError("size must be specified by size or like")
    elif isinstance(size, cls):
        size = size.size

    if dtype is None:
        if like is not None:
            dtype = like.dtype
        else:
            dtype = default_dtype

    if colororder is None and like is not None and like.iscolor:
        colororder = like.colororder_str.replace(":", "")

    return size, dtype, colororder


def _pattern_image(cls, image: np.ndarray, colororder: str | None):
    if colororder is not None:
        image = np.repeat(
            image[..., np.newaxis], len(cls.colordict(colororder)), axis=2
        )
    return cls(image, colororder=colororder)


class ImageConstantsMixin(_ImageBase if TYPE_CHECKING else object):
    # ======================= patterns ================================== #

    @classmethod
    def Zeros(
        cls,
        *,
        size: int | Sequence[int] | None = None,
        colororder: str | None = None,
        dtype: Dtype | None = None,
        like=None,
    ) -> Self:
        """
        Create image with zero value pixels

        :param size: image size, width x height, defaults to ``like.size`` if ``like`` is given
        :type size: int or 2-tuple, optional
        :param colororder: color plane names, defaults to None
        :type colororder: str
        :param dtype: NumPy datatype, defaults to 'uint8'
        :type dtype: str or NumPy dtype, optional
        :param like: template image supplying default ``size`` and ``colororder``
            when those are not given explicitly
        :type like: :class:`Image` or None, optional
        :return: image of zero values
        :rtype: :class:`Image`

        Create a greyscale image of zero-valued pixels. If only one dimension is
        given the image is square.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> Image.Zeros(size=20)
            >>> Image.Zeros(size=(10, 20))
            >>> Image.Zeros(size=(10,20))
            >>> Image.Zeros(size=(10,20,3), colororder='RGB')
            >>> Image.Zeros(size=20, dtype='float', colororder="RGB") # create color image, all black

        :seealso: :meth:`Constant`
        """

        size, dtype, colororder = _resolve_pattern_options(
            cls,
            size=size,
            dtype=dtype,
            colororder=colororder,
            like=like,
            default_size=None,
            default_dtype="uint8",
        )

        shape = _getshape(cls, None, None, colororder, size)
        return cls(np.zeros(shape, dtype=dtype), colororder=colororder)

    @classmethod
    def Constant(
        cls,
        value: Any = 0,
        *,
        colororder: str | None = None,
        dtype: Dtype | None = None,
        size: Sequence[int] | None = None,
        like=None,
    ) -> Self:
        """
        Create image with all pixels having same value

        :param value: value for all pixels, defaults to 0
        :type value: scalar, array_like, str
        :param size: image size, width x height, defaults to ``like.size`` if ``like`` is given
        :type size: int or 2-tuple, optional
        :param colororder: color plane names, defaults to None
        :type colororder: str
        :param dtype: NumPy datatype, defaults to 'uint8'
        :type dtype: str or NumPy dtype, optional
        :param like: template image supplying default ``size``, ``dtype`` and
            ``colororder`` when those are not given explicitly
        :type like: :class:`Image` or None, optional
        :return: image of constant values
        :rtype: :class:`Image`

        Creates a new image initialized to ``value``.  If ``value`` is iterable
        then the image has ``len(value)`` planes, each initialized to the
        corresponding element of ``value``.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> Image.Constant(17, size=10).print()
            >>> Image.Constant([100, 50, 200], size=(10, 20), colororder='RGB').print()
            >>> Image.Constant(range(6), size=10, colororder='ABCDEF').print()
            >>> Image.Constant('lightgreen', size=10).print()

        .. note:: If ``len(value) == 3`` and ``colororder`` is not specified
            then RGB is assumed.

        :seealso: :meth:`Zeros`
        """
        default_dtype = "float" if isinstance(value, float) else "uint8"

        size, dtype, colororder = _resolve_pattern_options(
            cls,
            size=size,
            dtype=dtype,
            colororder=colororder,
            like=like,
            default_size=None,
            default_dtype=default_dtype,
        )
        dtype = np.dtype(dtype)

        shape = _getshape(cls, None, None, colororder, size)

        if isinstance(value, str):
            # value given as a string, assume colorname
            value = name2color(value, dtype=dtype)

        if isinstance(value, Iterable):
            value = list(value)
            # iterable
            if len(shape) == 3 and len(value) != shape[2]:
                raise ValueError("length of value does not match number of planes")

            if len(value) == 3 and colororder is None:
                colororder = "RGB"

            planes = []
            for bg in value:
                planes.append(np.full(shape[:2], bg, dtype=dtype))
            return cls(np.stack(planes, axis=2), colororder=colororder)

        else:
            # scalar
            return cls(np.full(shape, value, dtype=dtype))

    @classmethod
    def String(
        cls,
        *planes: str,
        colororder: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Create a small image from text string

        :param planes: text strings, one per plane
        :type planes: str
        :param colororder: color plane names, defaults to None
        :type colororder: str
        :param kwargs: additional arguments passed to ``Image`` constructor
        :return: image
        :rtype: :class:`Image`

        Useful for creating simple images, particularly for unit tests.

        Creates a new image initialized to a compact representation given by a
        string.  Each string defines a single plane, multiple plane images
        can be created. Two string formats are supported:

        **Single line**

        Each pixel is a single character, and image
        rows are separated by a pipe.  There are no spaces.  All rows must be
        the same length.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.String('01234|56789|87654')
            >>> img.print()

        .. note:: Pixel values are determined by the unicode value of the
            character relative to unicode for '0', so other ASCII characters
            (apart from pipe) can be used to obtain pixel values greater than 9.
            'Z' is 90 and 'z' is 122.

        **Multiline**

        The string representation of the image is given in ASCII art format:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.String(r'''
                    ..........
                    .########.
                    .########.
                    .########.
                    .########.
                    ..........
                    ''', binary=True)
            >>> img.print()

        where the characters represent pixel values: "." is zero, otherwise the
        character's ordinal value is used.  Use the ``binary`` option to turn 0 and
        ordinal value into ``False`` and ``True`` respectively. Indentation is
        removed, blank lines are ignored.  A multi-level image can be created by:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.String(r'''
                    000000000
                    011112220
                    011112220
                    011112220
                    000000000
                    ''') - ord("0")
            >>> img.print()

        which has pixel values of 0, 1 and 2.

        .. note:: The default datatype for the image is ``uint8``

        :seealso: :meth:`Constant`
        """

        def str2array(s):
            if "|" in s:
                pixels = []
                for row in s.split("|"):
                    pixels.append([ord(c) - ord("0") for c in row])
                return pixels

            else:
                # string representation, the image in ASCII art format like:
                #
                #    ..........
                #    .########.
                #    .########..
                #    .########.
                #    .########.
                #    ..........

                pixels = []
                zeros = "."

                for row in s.split("\n"):
                    row = row.strip()
                    if len(row) > 0:
                        pixels.append([0 if c in zeros else ord(c) for c in row])
                return pixels

        try:
            if len(planes) == 1:
                pixels = np.array(str2array(planes[0]))
            else:
                pixels = np.array([str2array(plane) for plane in planes])
                pixels = np.moveaxis(pixels, 0, -1)
        except ValueError:
            raise ValueError("bad string, check all rows have same length")

        kwargs.setdefault("dtype", "uint8")
        return cls(pixels, colororder=colororder, **kwargs)

    @classmethod
    def Random(
        cls,
        *,
        size: int | Sequence[int] | None = None,
        colororder: str | None = None,
        dtype: Dtype | None = None,
        maxval: int | float | None = None,
        pdf: np.ndarray | None = None,
        like=None,
    ) -> Self:
        """
        Create image with random pixel values

        :param size: image size: size, (width, height) or (width, height, nplanes)
        :type size: int, 2-tuple or 3-tuple
        :param colororder: color plane names, defaults to None
        :type colororder: str
        :param dtype: NumPy datatype, defaults to 'uint8'
        :type dtype: str or NumPy dtype, optional
        :param maxval: maximum value for random values, defaults to None
        :type maxval: same as ``dtype``, optional
        :param pdf: probability density function for pixel values, defaults to None
        :type pdf: 1D or 2D array_like, optional
        :param like: template image supplying default ``size``, ``dtype`` and
            ``colororder`` when those are not given explicitly
        :type like: :class:`Image` or None, optional
        :return: image of random values
        :rtype: :class:`Image`

        The dimensions can be specified by a single scalar, a 2-tuple or a 3-tuple.  If
        a single scalar is given the image is square.  If a 2-tuple is given the image
        has one plane, if a 3-tuple is given the last element specifies the number of
        planes.

        Creates a new image where pixels are initialized to random values:

          - for an integer image the values are uniformly distributed in the range 0 to
            ``maxval``.  If ``maxval`` is not given then the values span the range 0 to
            the maximum positive value of the datatype.  For a floating image the values
            span the range 0.0 to 1.0.

          - if ``pdf`` is given then the pixel values are drawn from the given probability
            density function, which should be normalized to sum to 1.  If ``pdf`` is a 1D
            array then the same pdf is used for all planes, if it is a 2D array then each
            column gives the pdf for the corresponding plane.  This option is only
            supported for ``uint8`` images, and if given then ``maxval`` is ignored.


        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Random(size=3)
            >>> img.print()
            >>> img = Image.Random(size=3, colororder='RGB')
            >>> img.print()
            >>> img.red().print()
            >>> img = Image.Random(size=3, dtype='float32')
            >>> img.print
            >>> Image.Random(size=100).disp()

        .. plot::

            from machinevisiontoolbox import Image

            Image.Random(size=100).disp()

        We could, for example, create a random image with the same histogram as an existing image:

        .. code-block:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("street.png")
            >>> h = img.hist()
            >>> h.plot('pdf')
            >>> img2 = Image.Random(size=img.size, pdf=h.pdf)
            >>> img2.hist().plot('pdf')

        .. plot::

            from machinevisiontoolbox import Image
            img = Image.Read("street.png")
            h = img.hist()
            img2 = Image.Random(size=img.size, pdf=h.pdf)
            fig, (orig, random) = plt.subplots(1, 2, figsize=(10, 5))
            h.plot('pdf', ax=orig)
            img2.hist().plot('pdf', ax=random)
            orig.set_title("Original image histogram")
            random.set_title("Random image histogram")

        :seealso: :meth:`Constant`
        """

        size, dtype, colororder = _resolve_pattern_options(
            cls,
            size=size,
            dtype=dtype,
            colororder=colororder,
            like=like,
            default_size=None,
            default_dtype="uint8",
        )
        shape = _getshape(cls, None, None, colororder, size)
        dtype = np.dtype(dtype)

        if pdf is None:
            if np.issubdtype(dtype, np.integer):
                if maxval is None:
                    maxval = np.iinfo(dtype).max
                else:
                    maxval = int(maxval)
                im = np.random.randint(0, maxval, size=shape, dtype=dtype)
            elif np.issubdtype(dtype, np.floating):
                if maxval is None:
                    maxval = 1.0
                im = (np.random.rand(*shape) * maxval).astype(dtype)
            else:
                raise TypeError("dtype must be an integer or floating type")
        else:
            # pdf given, ignore maxval
            if maxval is not None:
                raise ValueError("Cannot specify both pdf and maxval")
            if dtype != np.dtype("uint8"):
                raise ValueError("pdf option only supported for uint8 dtype")
            if pdf.ndim == 1:
                # simple case, same pdf for all planes
                if pdf.shape[0] != 256:
                    raise ValueError("pdf must have length 256")
                im = np.random.choice(np.arange(256, dtype="uint8"), size=shape, p=pdf)
            elif pdf.ndim == 2:
                if pdf.shape[0] != 256:
                    raise ValueError("pdf must have length 256")
                if pdf.shape[1] != shape[2]:
                    raise ValueError(
                        "pdf must have same number of columns as planes in image"
                    )
                planes = []
                for i in range(shape[2]):
                    planes.append(
                        np.random.choice(
                            np.arange(255, dtype="uint8"), size=shape[:2], p=pdf[:, i]
                        )
                    )
                im = np.stack(planes, axis=2)
            else:
                raise ValueError("pdf must be a 1D or 2D array")

        return cls(im, colororder=colororder)

    @classmethod
    def Squares(
        cls,
        number: int,
        *,
        size: int | Sequence[int] | None = None,
        fg: Any = 1,
        bg: Any = 0,
        dtype: Dtype | None = None,
        colororder: str | None = None,
        like=None,
    ) -> Self:
        """
        Create image containing grid of squares

        :param number: number of squares horizontally and vertically
        :type number: int
        :param size: image size; scalar gives a square image, defaults to 256
        :type size: int or 2-tuple, optional
        :param fg: pixel value of the squares, defaults to 1
        :type fg: int, float, optional
        :param bg: pixel value of the background, defaults to 0
        :type bg: int, optional
        :param dtype: NumPy datatype, defaults to 'uint8'
        :type dtype: str or NumPy dtype, optional
        :param colororder: color plane names for the output image, defaults to None
        :type colororder: str or None, optional
        :param like: template image supplying default ``size``, ``dtype`` and
            ``colororder`` when those are not given explicitly
        :type like: :class:`Image` or None, optional
        :return: grid of squares
        :rtype: :class:`Image`

        The image background is set to ``bg`` and the circles are filled with ``fg``.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Squares(2, size=14, bg=1, fg=9)
            >>> img.print()

        ``number`` equal to 1, 2 and 8:

        .. plot::

            from machinevisiontoolbox import Image
            x = Image.Squares(1, size=100)
            y = Image.Squares(2, size=100)
            z = Image.Squares(8, size=100)
            Image.Hstack((x, y, z), sep=4, bgcolor=1).disp()

        .. note:: Image is square.
        """
        size, dtype, colororder = _resolve_pattern_options(
            cls,
            size=size,
            dtype=dtype,
            colororder=colororder,
            like=like,
            default_size=256,
            default_dtype="uint8",
        )
        dtype = np.dtype(dtype)
        shape = _getshape(cls, None, None, None, size)
        height, width = shape[:2]

        im = np.full((height, width), bg, dtype=dtype)
        # Compute the gap d: largest integer such that N*(2d) + (N+1)*d <= W
        dx = width // (3 * number + 1)
        dy = height // (3 * number + 1)
        d = min(dx, dy)
        # Give all remaining pixels to the squares so side >= 2*d (gap <= side/2)
        side_x = (width - (number + 1) * d) // number
        side_y = (height - (number + 1) * d) // number
        side = min(side_x, side_y)  # keep squares square

        # Centre the grid in both axes
        x_span = number * side + (number + 1) * d
        y_span = number * side + (number + 1) * d
        x_offset = (width - x_span) // 2
        y_offset = (height - y_span) // 2

        for r in range(number):
            y0 = y_offset + d + r * (side + d)
            for c in range(number):
                x0 = x_offset + d + c * (side + d)
                im[y0 : y0 + side, x0 : x0 + side] = fg

        return _pattern_image(cls, im, colororder)

    @classmethod
    def Circles(
        cls,
        number: int,
        *,
        size: int | Sequence[int] | None = None,
        fg: Any = 1,
        bg: Any = 0,
        dtype: Dtype | None = None,
        colororder: str | None = None,
        like=None,
    ) -> Self:
        """
        Create image containing grid of circles

        :param number: number of circles horizontally and vertically
        :type number: int
        :param size: image size; scalar gives a square image, defaults to 256
        :type size: int or 2-tuple, optional
        :param fg: pixel value of the circles, defaults to 1
        :type fg: int, float, optional
        :param bg: pixel value of the background, defaults to 0
        :type bg: int, optional
        :param dtype: NumPy datatype, defaults to 'uint8'
        :type dtype: str or NumPy dtype, optional
        :param colororder: color plane names for the output image, defaults to None
        :type colororder: str or None, optional
        :param like: template image supplying default ``size``, ``dtype`` and
            ``colororder`` when those are not given explicitly
        :type like: :class:`Image` or None, optional
        :return: grid of circles
        :rtype: :class:`Image`

        The image background is set to ``bg`` and the circles are filled with ``fg``.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Circles(2, size=14, bg=1, fg=9)
            >>> img.print()

        ``number`` equal to 1, 2 and 8:

        .. plot::

            from machinevisiontoolbox import Image
            x = Image.Circles(1, size=100)
            y = Image.Circles(2, size=100)
            z = Image.Circles(8, size=100)
            Image.Hstack((x, y, z), sep=4, bgcolor=1).disp()
        """
        size, dtype, colororder = _resolve_pattern_options(
            cls,
            size=size,
            dtype=dtype,
            colororder=colororder,
            like=like,
            default_size=256,
            default_dtype="uint8",
        )
        dtype = np.dtype(dtype)
        shape = _getshape(cls, None, None, None, size)
        height, width = shape[:2]
        im = np.full((height, width), bg, dtype=dtype)
        # Same layout as Squares: gap d, side fills remaining space
        dx = width // (3 * number + 1)
        dy = height // (3 * number + 1)
        d = min(dx, dy)
        side_x = (width - (number + 1) * d) // number
        side_y = (height - (number + 1) * d) // number
        side = min(side_x, side_y)  # keep cells square
        # Cell side must be odd so circle (diam = 2r+1) fills it exactly
        if side % 2 == 0:
            side -= 1

        # Radius fills the cell exactly: diam = 2*radius+1 = side
        radius = side // 2
        circle = Kernel.Circle(radius).K.astype(dtype) * (fg - bg) + bg

        # Centre the whole grid
        x_span = number * side + (number + 1) * d
        y_span = number * side + (number + 1) * d
        x_offset = (width - x_span) // 2
        y_offset = (height - y_span) // 2

        for r in range(number):
            y0 = y_offset + d + r * (side + d)
            for c in range(number):
                x0 = x_offset + d + c * (side + d)
                im[y0 : y0 + side, x0 : x0 + side] = circle

        return _pattern_image(cls, im, colororder)

    @classmethod
    def Ramp(
        cls,
        cycles: int = 2,
        dir: str = "x",
        *,
        size: int | Sequence[int] | None = None,
        dtype: Dtype | None = None,
        colororder: str | None = None,
        like=None,
    ) -> Self:
        """
        Create image of linear ramps

        :param dir: ramp direction: 'x' [default], 'y' or 'xy'
        :type dir: str, optional
        :param size: image size, width x height, defaults to 256x256
        :type size: int or 2-tuple, optional
        :param cycles: Number of complete ramps, defaults to 2
        :type cycles: int, optional
        :param dtype: NumPy datatype, defaults to 'float32'
        :type dtype: str or NumPy dtype, optional
        :param colororder: color plane names for the output image, defaults to None
        :type colororder: str or None, optional
        :param like: template image supplying default ``size``, ``dtype`` and
            ``colororder`` when those are not given explicitly
        :type like: :class:`Image` or None, optional
        :return: intensity ramps
        :rtype: :class:`Image`

        The direction ``'xy'`` creates a diagonal ramp.

        The ramps span the range:

        * float image: 0 to 1
        * int image: 0 to maximum positive value of the integer type

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> Image.Ramp(cycles=2, size=10).print()
            >>> Image.Ramp(cycles=3, size=10, dtype='uint8').print()

        Ramps in the x, y and diagonal directions:

        .. plot::

            from machinevisiontoolbox import Image
            x = Image.Ramp(100, dir='x')
            y = Image.Ramp(100, dir='y')
            xy = Image.Ramp(100, dir='xy')
            Image.Hstack((x, y, xy), sep=4, bgcolor=255).disp()


        """
        size, dtype, colororder = _resolve_pattern_options(
            cls,
            size=size,
            dtype=dtype,
            colororder=colororder,
            like=like,
            default_size=256,
            default_dtype="float32",
        )
        dtype = np.dtype(dtype)
        shape = _getshape(cls, None, None, None, size)
        height, width = shape[:2]

        if dir == "x":
            c = width / cycles
            x = np.arange(0, width)
            maxval = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
            s = np.expand_dims(np.mod(x, c) / (c - 1) * maxval, axis=0).astype(dtype)
            image = np.repeat(s, height, axis=0)
        elif dir == "y":
            c = height / cycles
            y = np.arange(0, height)
            maxval = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
            s = np.expand_dims(np.mod(y, c) / (c - 1) * maxval, axis=1).astype(dtype)
            image = np.repeat(s, width, axis=1)
        elif dir == "xy":
            c = width / cycles
            x = np.arange(0, width)
            maxval = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
            image = np.zeros((height, width), dtype=dtype)
            for row in range(height):
                image[row, :] = np.mod(x + row, c) / (c - 1) * maxval
            image = image.astype(dtype)

        else:
            raise ValueError("dir must be 'x', 'y' or 'xy'")

        return _pattern_image(cls, image, colororder)

    @classmethod
    def Sin(
        cls,
        cycles: int = 2,
        dir: str = "x",
        *,
        size: int | Sequence[int] | None = None,
        dtype: Dtype | None = None,
        colororder: str | None = None,
        like=None,
    ) -> Self:
        """
        Create image of sinusoidal intensity pattern

        :param dir: sinusoid direction: 'x' [default] or 'y'
        :type dir: str, optional
        :param size: image size, width x height, defaults to 256x256
        :type size: int or 2-tuple, optional
        :param cycles: Number of complete cycles, defaults to 2
        :type cycles: int, optional
        :param dtype: NumPy datatype, defaults to 'float32'
        :type dtype: str or NumPy dtype, optional
        :param colororder: color plane names for the output image, defaults to None
        :type colororder: str or None, optional
        :param like: template image supplying default ``size``, ``dtype`` and
            ``colororder`` when those are not given explicitly
        :type like: :class:`Image` or None, optional
        :return: sinusoidal pattern
        :rtype: :class:`Image`

        The sinusoids are offset to have a minimum value of zero, and span the range:

            * float image: 0 to 1
            * int image: 0 to maximum positive value of the integer type

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> Image.Sin(cycles=2, size=10).print()
            >>> Image.Sin(cycles=2, size=10, dtype='uint8').print()

        ``cycles`` equal to 1, 4 and 18:

        .. plot::

            from machinevisiontoolbox import Image
            x = Image.Sin(cycles=1, size=100)
            y = Image.Sin(cycles=4, size=100)
            z = Image.Sin(cycles=16, size=100)
            Image.Hstack((x, y, z), sep=4, bgcolor=1.0).disp()

        """
        size, dtype, colororder = _resolve_pattern_options(
            cls,
            size=size,
            dtype=dtype,
            colororder=colororder,
            like=like,
            default_size=256,
            default_dtype="float32",
        )
        dtype = np.dtype(dtype)
        shape = _getshape(cls, None, None, None, size)
        height, width = shape[:2]

        if dir == "x":
            c = width / cycles
            x = np.arange(0, width)
            maxval = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
            s = np.expand_dims((np.sin(x / c * 2 * np.pi) + 1) * maxval / 2, axis=0)
            image = np.repeat(s, height, axis=0).astype(dtype)
        elif dir == "y":
            c = height / cycles
            y = np.arange(0, height)
            maxval = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
            s = np.expand_dims((np.sin(y / c * 2 * np.pi) + 1) * maxval / 2, axis=1)
            image = np.repeat(s, width, axis=1).astype(dtype)
        else:
            raise ValueError("dir must be 'x' or 'y'")

        return _pattern_image(cls, image, colororder)

    @classmethod
    def Chequerboard(
        cls,
        square: int = 32,
        *,
        size: int | Sequence[int] | None = None,
        dtype: Dtype | None = None,
        colororder: str | None = None,
        like=None,
    ) -> Self:
        """Create chequerboard pattern

        :param size: image size, width x height, defaults to 256x256
        :type size: int or 2-tuple, optional
        :param square: cell dimension in pixels, defaults to 32
        :type square: int, optional
        :param dtype: image data type, defaults to "uint8"
        :type dtype: str or NumPy dtype, optional
        :param colororder: color plane names for the output image, defaults to None
        :type colororder: str or None, optional
        :param like: template image supplying default ``size``, ``dtype`` and
            ``colororder`` when those are not given explicitly
        :type like: :class:`Image` or None, optional
        :return: chequerboard pattern
        :rtype: :class:`Image`

        Create a chquerboard pattern with black and white square cells of side length ``square``.

        The pixel values within the cells are:

        * float image: black squares 0.0, white squares 1.0
        * int image: black squares 0, white squares maximum positive value of the integer type

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> Image.Chequerboard(square=2, size=8).print()

        ``square`` equal to 16, 32 and 64:

        .. plot::

            from machinevisiontoolbox import Image
            x = Image.Chequerboard(square=16, size=256)
            y = Image.Chequerboard(square=32, size=256)
            z = Image.Chequerboard(square=64, size=256)
            Image.Hstack((x, y, z), sep=4, bgcolor=255).disp()

        .. note:: There is no check for ``size`` being an integral multiple of ``square`` so the last row
            and column may be of different size to the others.
        """
        size, dtype, colororder = _resolve_pattern_options(
            cls,
            size=size,
            dtype=dtype,
            colororder=colororder,
            like=like,
            default_size=256,
            default_dtype="uint8",
        )
        dtype = np.dtype(dtype)

        if np.issubdtype(dtype, np.integer):
            maxval = np.iinfo(dtype).max
        else:
            maxval = 1.0

        shape = _getshape(cls, None, None, None, size)
        height, width = shape[:2]
        image = np.zeros((height, width), dtype=dtype)

        for row in range(height):
            for col in range(width):
                if (row // square) % 2 == (col // square) % 2:
                    image[row, col] = maxval

        return _pattern_image(cls, image, colororder)

    @classmethod
    def Polygons(
        cls,
        polygons: Polygon2 | list[Polygon2] | tuple[Polygon2, ...],
        *,
        size: int | Sequence[int] | None = None,
        color: Any = 1,
        bg: Any = 0,
        shift: int = 0,
        dtype: str | np.dtype = "uint8",
        colororder: str | None = None,
        like=None,
    ) -> Self:
        """
        Create an image containing filled polygons

        :param size: image size, defaults to None
        :type size: int or 2-tuple, optional
        :param polygons: polygon or list of polygons
        :type polygons: :class:`Polygon2` or list of :class:`Polygon2`
        :param color: pixel value for the polygons, defaults to 1
        :type color: int, float, optional
        :param bg: pixel value for the background, defaults to 0
        :type bg: int, float, optional
        :param shift: number of fractional bits in the vertex coordinates, defaults to 0
        :type shift: int, optional
        :param dtype: image data type, defaults to "uint8"
        :type dtype: str or NumPy dtype, optional
        :param colororder: color plane names for the output image, defaults to None
        :type colororder: str or None, optional
        :param like: template image supplying default ``size``, ``dtype`` and
            ``colororder`` when those are not given explicitly
        :type like: :class:`Image` or None, optional
        :return: image containing the polygons
        :rtype: :class:`Image`

        The image is initialized to ``bg`` and the pixels within the polygons are set to
        ``color``.

        ``polygons`` can be a single :class:`Polygon2` or a list of :class:`Polygon2`.
        If ``color`` is a single value then all polygons are filled with the same color,
        if ``color`` is a list then each polygon is filled with the corresponding color.

        The vertices of the polygons are rounded to the nearest integer. To achieve
        sub-pixel accuracy the ``shift`` parameter is passed to ``fillPoly`` and allows
        the vertices to be specified with sub-pixel precision. For example, a coordinate
        of 11 with a ``shift`` of 2 corresponds to a vertex at 11/4 = 2.75 pixels.

        .. note:: The polygon will be automatically clipped to the image boundaries,
            so vertices outside the image are allowed.

        .. warning:: The integer vertices are considered to lie at the centre of the corresponding
            pixels, so a vertex at (10, 10) corresponds to the pixel at row 10, column 10.
            The polygon edges are considered to lie halfway between the pixels, so a square
            with vertices at (10, 10), (20, 10), (20, 20) and (10, 20) corresponds to a
            filled square of pixels with rows from 10 to 20 and columns from 10 to 20.  It
            will have an area of 11x11 = 121 pixels, not 10x10 = 100 pixels.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> from spatialmath import Polygon2
            >>> p1 = Polygon2([(10, 10), (20, 10), (20, 20), (10, 20)])
            >>> p2 = Polygon2([(30, 30), (40, 30), (40, 40), (30, 40)])
            >>> img = Image.Polygons([p1, p2], size=50)
            >>> img

        :seealso: :meth:`sm.Polygon2` :meth:`cv2.fillPoly`
        """
        if isinstance(polygons, Polygon2):
            polygons = [polygons]

        size, dtype, colororder = _resolve_pattern_options(
            cls,
            size=size,
            dtype=dtype,
            colororder=colororder,
            like=like,
            default_size=None,
            default_dtype="uint8",
        )
        dtype = np.dtype(dtype)
        shape = _getshape(cls, None, None, None, size)
        height, width = shape[:2]

        if isinstance(color, Iterable):
            if len(color) != len(polygons):
                raise ValueError("fill must have same length as number of polygons")
            colors = color
        else:
            colors = [color] * len(polygons)

        im = np.full((height, width), bg, dtype=dtype)

        for polygon, color in zip(polygons, colors):
            vertices = (
                np.round(polygon.vertices()).astype("int32").T.reshape(-1, 1, 2)
            )  # Nx1x2
            cv2.fillPoly(img=im, pts=[vertices], color=color, shift=shift)

        return _pattern_image(cls, im, colororder)


if __name__ == "__main__":
    from pathlib import Path

    import pytest

    pytest.main(
        [
            str(
                Path(__file__).parent.parent.parent
                / "tests"
                / "test_image_constants.py"
            ),
            "-v",
        ]
    )
