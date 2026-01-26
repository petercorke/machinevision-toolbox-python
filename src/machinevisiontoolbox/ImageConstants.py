#!/usr/bin/env python
"""
Images class
@author: Dorian Tsai
@author: Peter Corke
"""

from collections.abc import Iterable
from pathlib import Path
import os.path
import os
import numpy as np
import cv2 as cv
from spatialmath import base as smb

# from numpy.lib.arraysetops import isin
from machinevisiontoolbox.base import int_image, float_image, name2color
from machinevisiontoolbox.ImageSpatial import Kernel
from spatialmath.base import isscalar, islistof
import warnings

# import spatialmath.base.argcheck as argcheck


from machinevisiontoolbox.base.imageio import idisp, iread, iwrite, convert
import urllib
import xml.etree.ElementTree as ET


class ImageConstantsMixin:
    # ======================= patterns ================================== #

    @classmethod
    def Zeros(cls, w, h=None, colororder=None, dtype="uint8"):
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
            >>> Image.Zeros(20, dtype='float', colororder="RGB") # create color image, all black

        :seealso: :meth:`Constant`
        """
        if h is None:
            if isinstance(w, (tuple, list)):
                h = w[1]
                w = w[0]
            else:
                h = w
        shape = [h, w]

        if colororder is not None:
            p = len(cls.colordict(colororder))
            shape.append(p)

        return cls(np.zeros(shape, dtype=dtype), colororder=colororder)

    @classmethod
    def Constant(cls, w, h=None, value=0, colororder=None, dtype=None):
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
        if h is None:
            if isinstance(w, (tuple, list)):
                h = w[1]
                w = w[0]
            else:
                h = w
        shape = (h, w)

        if isinstance(value, float) and dtype is None:
            dtype = "float"
        if dtype is None:
            dtype = "uint8"

        if isinstance(value, str):
            # value given as a string, assume colorname
            value = name2color(value, dtype=dtype)

        if isinstance(value, Iterable):
            # iterable
            if len(value) == 3 and colororder is None:
                colororder = "RGB"

            planes = []
            for bg in value:
                planes.append(np.full(shape, bg, dtype=dtype))
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

        :param size: image size, width x height, defaults to 256x256
        :type size: int or 2-tuple, optional
        :param colororder: color plane names, defaults to None
        :type colororder: str
        :param dtype: NumPy datatype, defaults to 'uint8'
        :type dtype: str, optional
        :param maxval: maximum value for random values, defaults to None
        :type maxval: same as ``dtype``, optional
        :return: image of random values
        :rtype: :class:`Image`

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
        if smb.isscalar(size):
            size = [size, size]
        else:
            size = [size[1], size[0]]

        if colororder is not None:
            size.append(len(cls.colordict(colororder)))

        if maxval is None:
            if np.issubdtype(dtype, np.integer):
                maxval = np.iinfo(dtype).max
            else:
                maxval = 1.0
        if np.issubdtype(dtype, np.integer):
            im = np.random.randint(0, maxval, size=size, dtype=dtype)
        elif np.issubdtype(dtype, np.floating):
            im = (np.random.rand(*size) * maxval).astype(dtype)

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
        im = np.full((size, size), bg, dtype=dtype)
        d = size // (3 * number + 1)
        side = 2 * d + 1  # keep it odd
        sq = np.full((side, side), fg, dtype=dtype)
        s2 = side // 2
        for r in range(number):
            y0 = (r * 3 + 2) * d
            for c in range(number):
                x0 = (c * 3 + 2) * d
                im[y0 - s2 : y0 + s2 + 1, x0 - s2 : x0 + s2 + 1] = sq

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
        im = np.full((size, size), bg, dtype=dtype)
        d = size // (3 * number + 1)
        side = 2 * d + 1  # keep it odd
        s2 = side // 2
        circle = Kernel.Circle(s2).K.astype(dtype) * (fg - bg) + bg

        for r in range(number):
            y0 = (r * 3 + 2) * d
            for c in range(number):
                x0 = (c * 3 + 2) * d
                im[y0 - s2 : y0 + s2 + 1, x0 - s2 : x0 + s2 + 1] = circle

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


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import pathlib
    import os.path

    from machinevisiontoolbox import Image

    # z = Image.Ramp(2, value=range(6), colororder='ABCDEF')
    # print(z)
    Image.Sin(256, 2).image
    Image.Sin(256, 2, dtype="uint8").image
    Image.Ramp(256, dir="y").disp(block=True)

    exec(
        open(
            pathlib.Path(__file__).parent.parent.absolute() / "tests" / "test_core.py"
        ).read()
    )  # pylint: disable=exec-used
