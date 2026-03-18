"""
Factory class methods for constructing Image objects with common pixel patterns.
"""

from __future__ import annotations

import os
import os.path
import urllib
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Sequence

import cv2 as cv
import numpy as np
from spatialmath import Polygon2
from spatialmath import base as smb
from spatialmath.base import islistof, isscalar

# from numpy.lib.arraysetops import isin
from machinevisiontoolbox.base import float_image, int_image, name2color
from machinevisiontoolbox.base.imageio import convert, idisp, iread, iwrite
from machinevisiontoolbox.ImageSpatial import Kernel

# import spatialmath.base.argcheck as argcheck


# TODO
#  - get rid of w, h, just use size
#  - use colornames for fg and bg
#  - consistently use fg and bg for foreground and background values
#


def _getshape(
    cls,
    w: int | None,
    h: int | None,
    colororder: str | None,
    size: Sequence[int] | None,
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
        else:
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


class ImageConstantsMixin:
    # ======================= patterns ================================== #

    @classmethod
    def Zeros(
        cls,
        w: int | Sequence[int] | None = None,
        h: int | None = None,
        colororder: str | None = None,
        dtype: str = "uint8",
        size: Sequence[int] | None = None,
    ) -> "Image":
        """
        Create image with zero value pixels

        :param w: width, or (width, height)
        :type w: int, (int, int)
        :param h: height, defaults to None
        :type h: int, optional
        :param colororder: color plane names, defaults to None
        :type colororder: str
        :param dtype: NumPy datatype, defaults to 'uint8'
        :type dtype: str, optional
        :return: image of zero values
        :rtype: :class:`Image`

        Create a greyscale image of zero-valued pixels. If only one dimension is
        given the image is square.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> Image.Zeros(20)
            >>> Image.Zeros(10,20)
            >>> Image.Zeros(size=(10,20))
            >>> Image.Zeros(size=(10,20,3), colororder='RGB')
            >>> Image.Zeros(20, dtype='float', colororder="RGB") # create color image, all black

        :seealso: :meth:`Constant`
        """

        shape = _getshape(cls, w, h, colororder, size)
        return cls(np.zeros(shape, dtype=dtype), colororder=colororder)

    @classmethod
    def Constant(
        cls,
        w: int | Sequence[int] | None = None,
        h: int | None = None,
        value: Any = 0,
        colororder: str | None = None,
        dtype: str | None = None,
        size: Sequence[int] | None = None,
    ) -> "Image":
        """
        Create image with all pixels having same value

        :param w: width, or (width, height)
        :type w: int, (int, int)
        :param h: height, defaults to None
        :type h: int, optional
        :param value: value for all pixels, defaults to 0
        :type value: scalar, array_like, str
        :param colororder: color plane names, defaults to None
        :type colororder: str
        :param dtype: NumPy datatype, defaults to 'uint8'
        :type dtype: str, optional
        :return: image of constant values
        :rtype: :class:`Image`

        Creates a new image initialized to ``value``.  If ``value`` is iterable
        then the image has ``len(value)`` planes, each initialized to the
        corresponding element of ``value``.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Constant(10, value=17)
            >>> img
            >>> img.image[0, 0]
            >>> img = Image.Constant(10, 20, [100, 50, 200], colororder='RGB')
            >>> img
            >>> img.image[0, 0, :]
            >>> img = Image.Constant(10, value=range(6), colororder='ABCDEF')
            >>> img
            >>> img.image[0, 0, :]
            >>> img = Image.Constant(10, value='cyan')
            >>> img.image[0, 0, :]

        :note: If ``len(value) == 3`` and ``colororder`` is not specified
            then RGB is assumed.

        :seealso: :meth:`Zeros`
        """
        shape = _getshape(cls, w, h, colororder, size)

        if isinstance(value, float) and dtype is None:
            dtype = "float"
        if dtype is None:
            dtype = "uint8"

        if isinstance(value, str):
            # value given as a string, assume colorname
            value = name2color(value, dtype=dtype)

        if isinstance(value, Iterable):
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
    def String(cls, *planes, colororder=None, **kwargs):
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

        :note: Pixel values are determined by the unicode value of the
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
    def Random(cls, size, colororder=None, dtype="uint8", maxval=None):
        """
        Create image with random pixel values

        :param size: image size: size, (width, height) or (width, height, nplanes)
        :type size: int, 2-tuple or 3-tuple
        :param colororder: color plane names, defaults to None
        :type colororder: str
        :param dtype: NumPy datatype, defaults to 'uint8'
        :type dtype: str, optional
        :param maxval: maximum value for random values, defaults to None
        :type maxval: same as ``dtype``, optional
        :return: image of random values
        :rtype: :class:`Image`

        The dimensions can be specified by a single scalar, a 2-tuple or a 3-tuple.  If
        a single scalar is given the image is square.  If a 2-tuple is given the image
        has one plane, if a 3-tuple is given the last element specifies the number of
        planes.

        Creates a new image where pixels are initialized to uniformly distributed random values. For
        an integer image the values span the range 0 to the maximum positive
        value of the datatype.  For a floating image the values span the range 0.0 to 1.0.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Random(5)
            >>> img
            >>> img.image
            >>> img = Image.Random(5, colororder='RGB')
            >>> img
            >>> img.red().image
            >>> img = Image.Random(5, dtype='float32')
            >>> img.image

        Ramps in the x, y and diagonal directions:

        .. plot::

            from machinevisiontoolbox import Image
            Image.Random(100).disp()

        """
        shape = _getshape(cls, None, None, colororder, size)

        if maxval is None:
            if np.issubdtype(dtype, np.integer):
                maxval = np.iinfo(dtype).max
            else:
                maxval = 1.0
        if np.issubdtype(dtype, np.integer):
            im = np.random.randint(0, maxval, size=shape, dtype=dtype)
        elif np.issubdtype(dtype, np.floating):
            im = (np.random.rand(*shape) * maxval).astype(dtype)

        return cls(im, colororder=colororder)

    @classmethod
    def Squares(cls, number, size=256, fg=1, bg=0, dtype="uint8"):
        """
        Create image containing grid of squares

        :param number: number of squares horizontally and vertically
        :type number: int
        :param size: image width and height, defaults to 256
        :type size: int, optional
        :param fg: pixel value of the squares, defaults to 1
        :type fg: int, float, optional
        :param bg: pixel value of the background, defaults to 0
        :type bg: int, optional
        :param dtype: NumPy datatype, defaults to 'uint8'
        :type dtype: str, optional
        :return: grid of squares
        :rtype: :class:`Image`

        The image background is set to ``bg`` and the circles are filled with ``fg``.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Squares(2, size=14, bg=1, fg=9)
            >>> img.image

        ``number`` equal to 1, 2 and 8:

        .. plot::

            from machinevisiontoolbox import Image
            x = Image.Squares(1, size=100)
            y = Image.Squares(2, size=100)
            z = Image.Squares(8, size=100)
            Image.Hstack((x, y, z), sep=4, bgcolor=1).disp()

        :note: Image is square.
        """
        shape = _getshape(cls, None, None, None, size)

        im = np.full(shape, bg, dtype=dtype)
        d = size // (3 * number + 1)
        side = 2 * d + 1  # keep it odd
        sq = np.full((side, side), fg, dtype=dtype)
        s2 = side // 2
        for r in range(number):
            y0 = (r * 3 + 2) * d
            for c in range(number):
                x0 = (c * 3 + 2) * d
                im[y0 - s2 : y0 + s2 + 1, x0 - s2 : x0 + s2 + 1, ...] = sq

        return cls(im)

    @classmethod
    def Circles(cls, number, size=256, fg=1, bg=0, dtype="uint8"):
        """
        Create image containing grid of circles

        :param number: number of circles horizontally and vertically
        :type number: int
        :param size: image width and height, defaults to 256
        :type size: int, optional
        :param fg: pixel value of the circles, defaults to 1
        :type fg: int, float, optional
        :param bg: pixel value of the background, defaults to 0
        :type bg: int, optional
        :param dtype: NumPy datatype, defaults to 'uint8'
        :type dtype: str, optional
        :return: grid of circles
        :rtype: :class:`Image`

        The image background is set to ``bg`` and the circles are filled with ``fg``.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Circles(2, 14, bg=1, fg=9)
            >>> img.A

        ``number`` equal to 1, 2 and 8:

        .. plot::

            from machinevisiontoolbox import Image
            x = Image.Circles(1, size=100)
            y = Image.Circles(2, size=100)
            z = Image.Circles(8, size=100)
            Image.Hstack((x, y, z), sep=4, bgcolor=1).disp()
        """
        shape = _getshape(cls, None, None, None, size)
        im = np.full(shape, bg, dtype=dtype)
        d = size // (3 * number + 1)
        side = 2 * d + 1  # keep it odd
        s2 = side // 2
        circle = Kernel.Circle(s2).K.astype(dtype) * (fg - bg) + bg

        for r in range(number):
            y0 = (r * 3 + 2) * d
            for c in range(number):
                x0 = (c * 3 + 2) * d
                im[y0 - s2 : y0 + s2 + 1, x0 - s2 : x0 + s2 + 1, ...] = circle

        return cls(im)

    @classmethod
    def Ramp(cls, size=256, cycles=2, dir="x", dtype="float32"):
        """
        Create image of linear ramps

        :param dir: ramp direction: 'x' [default], 'y' or 'xy'
        :type dir: str, optional
        :param size: image size, width x height, defaults to 256x256
        :type size: int or 2-tuple, optional
        :param cycles: Number of complete ramps, defaults to 2
        :type cycles: int, optional
        :param dtype: NumPy datatype, defaults to 'float32'
        :type dtype: str, optional
        :return: intensity ramps
        :rtype: :class:`Image`

        The direction ``'xy'`` creates a diagonal ramp.

        The ramps span the range:

        * float image: 0 to 1
        * int image: 0 to maximum positive value of the integer type

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> Image.Ramp(10, 2).print()
            >>> Image.Ramp(10, 3, dtype='uint8').print()

        Ramps in the x, y and diagonal directions:

        .. plot::

            from machinevisiontoolbox import Image
            x = Image.Ramp(100, dir='x')
            y = Image.Ramp(100, dir='y')
            xy = Image.Ramp(100, dir='xy')
            Image.Hstack((x, y, xy), sep=4, bgcolor=255).disp()


        """
        if smb.isscalar(size):
            size = (size, size)

        if dir == "y":
            size = (size[1], size[0])

        c = size[0] / cycles
        if np.issubdtype(dtype, np.integer):
            max = np.iinfo(dtype).max
        else:
            max = 1.0

        x = np.arange(0, size[0])

        if dir in ("x", "y"):
            s = np.expand_dims(np.mod(x, c) / (c - 1) * max, axis=0).astype(dtype)
            image = np.repeat(s, size[1], axis=0)

            if dir == "y":
                image = image.T

        elif dir == "xy":
            image = np.zeros(size, dtype=dtype)
            for row in range(size[1]):
                image[row, :] = np.mod(x + row, c) / (c - 1) * max

        else:
            raise ValueError("dir must be 'x', 'y' or 'xy'")

        return cls(image, dtype=dtype)

    @classmethod
    def Sin(cls, size=256, cycles=2, dir="x", dtype="float32"):
        """
        Create image of sinusoidal intensity pattern

        :param dir: sinusoid direction: 'x' [default] or 'y'
        :type dir: str, optional
        :param size: image size, width x height, defaults to 256x256
        :type size: int or 2-tuple, optional
        :param cycles: Number of complete cycles, defaults to 2
        :type cycles: int, optional
        :param dtype: NumPy datatype, defaults to 'float32'
        :type dtype: str, optional
        :return: sinusoidal pattern
        :rtype: :class:`Image`

        The sinusoids are offset to have a minimum value of zero, and span the range:

            * float image: 0 to 1
            * int image: 0 to maximum positive value of the integer type

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> Image.Sin(10, 2).image
            >>> Image.Sin(10, 2, dtype='uint8').image

        ``cycles`` equal to 1, 4 and 18:

        .. plot::

            from machinevisiontoolbox import Image
            x = Image.Sin(100, 1)
            y = Image.Sin(100, 4)
            z = Image.Sin(100, 16)
            Image.Hstack((x, y, z), sep=4, bgcolor=1.0).disp()

        """
        if smb.isscalar(size):
            size = (size, size)
        if dir == "y":
            size = (size[1], size[0])

        image = np.zeros(size[0], dtype=dtype)
        c = size[0] / cycles
        x = np.arange(0, size[0])
        if np.issubdtype(dtype, np.integer):
            max = np.iinfo(dtype).max
        else:
            max = 1.0
        s = np.expand_dims((np.sin(x / c * 2 * np.pi) + 1) * max / 2, axis=0).astype(
            dtype
        )
        image = np.repeat(s, size[1], axis=0)
        if dir == "y":
            image = image.T

        return cls(image, dtype=dtype)

    @classmethod
    def Chequerboard(cls, size=256, square=32, dtype="uint8"):
        """Create chequerboard pattern

        :param size: image size, width x height, defaults to 256x256
        :type size: int or 2-tuple, optional
        :param square: cell dimension in pixels, defaults to 32
        :type square: int, optional
        :param dtype: image data type, defaults to "uint8"
        :type dtype: str, optional
        :return: chequerboard pattern
        :rtype: :class:`Image`

        Create a chquerboard pattern with black and white square cells of side length ``square``.

        The pixel values within the cells are:

        * float image: black squares 0.0, white squares 1.0
        * int image: black squares 0, white squares maximum positive value of the integer type

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> Image.Chequerboard(16, 2).image

        ``square`` equal to16, 32 and 64:

        .. plot::

            from machinevisiontoolbox import Image
            x = Image.Chequerboard(256, square=16)
            y = Image.Chequerboard(256, square=32)
            z = Image.Chequerboard(256, square=64)
            Image.Hstack((x, y, z), sep=4, bgcolor=255).disp()

        .. note:: There is no check for ``size`` being an integral multiple of ``square`` so the last row
            and column may be of different size to the others.
        """
        if np.issubdtype(dtype, np.integer):
            max = np.iinfo(dtype).max
        else:
            max = 1.0

        if smb.isscalar(size):
            size = (size, size)
        image = np.zeros((size[1], size[0]), dtype=dtype)

        for row in range(size[1]):
            for col in range(size[0]):
                if (row // square) % 2 == (col // square) % 2:
                    image[row, col] = max

        return cls(image, dtype=dtype)

    @classmethod
    def Polygons(cls, size, polygons, color=1, bg=0, shift=0, dtype="uint8"):
        """
        Create an image containing filled polygons

        :param polygons: polygon or list of polygons
        :type polygons: :class:`Polygon2` or list of :class:`Polygon2`
        :param color: pixel value for the polygons, defaults to 1
        :type color: int, float, optional
        :param bg: pixel value for the background, defaults to 0
        :type bg: int, float, optional
        :param shift: number of fractional bits in the vertex coordinates, defaults to 0
        :type shift: int, optional
        :param dtype: image data type, defaults to "uint8"
        :type dtype: str, optional
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

            >>> from machinevisiontoolbox import Image, Polygon2
            >>> p1 = Polygon2([(10, 10), (20, 10), (20, 20), (10, 20)])
            >>> p2 = Polygon2([(30, 30), (40, 30), (40, 40), (30, 40)])
            >>> img = Image.Polygons(50, [p1, p2])
            >>> img

        :seealso: :meth:`sm.Polygon2` :meth:`cv.fillPoly`
        """
        if isinstance(polygons, Polygon2):
            polygons = [polygons]

        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, cls):
            size = size.size
        elif isinstance(size, (tuple, list)):
            size = (size[1], size[0])

        if isinstance(color, Iterable):
            if len(color) != len(polygons):
                raise ValueError("fill must have same length as number of polygons")
            colors = color
        else:
            colors = [color] * len(polygons)

        im = np.full(size, bg, dtype="uint8")

        for polygon, color in zip(polygons, colors):
            vertices = (
                np.round(polygon.vertices()).astype("int32").T.reshape(-1, 1, 2)
            )  # Nx1x2
            cv.fillPoly(img=im, pts=[vertices], color=color, shift=shift)

        return cls(im, dtype=dtype)


if __name__ == "__main__":
    from pathlib import Path

    import pytest

    pytest.main(
        [str(Path(__file__).parent.parent.parent / "tests" / "test_constants.py"), "-v"]
    )
