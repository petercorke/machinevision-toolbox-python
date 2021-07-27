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


class ImageConstantsMixin:

    # ======================= patterns ================================== #

    @classmethod
    def Zeros(cls, w, h=None, dtype='uint8'):
        """
        Create image with zero value pixels

        :param w: width, or (width, height)
        :type w: int, (int, int)
        :param h: height, defaults to None
        :type h: int, optional
        :param dtype: NumPy datatype, defaults to 'uint8'
        :type dtype: str, optional
        :return: image of zero values
        :rtype: Image instance
        """

        if isinstance(w, (tuple, list)) and h is None:
            h = w[1]
            w = w[0]
        return cls(np.zeros((h, w), dtype=dtype))

    @classmethod
    def Constant(cls, w, h, value=0, dtype='uint8'):
        """
        Create image with all pixels having same value

        :param w: width, or (width, height)
        :type w: int, (int, int)
        :param h: height, defaults to None
        :type h: int, optional
        :param value: value for all pixels, defaults to 1
        :type value: int or float
        :param dtype: NumPy datatype, defaults to 'uint8'
        :type dtype: str, optional
        :return: image of constant values
        :rtype: Image instance
        """
        if isinstance(w, (tuple, list)) and h is None:
            h = w[1]
            w = w[0]
        if isinstance(value, float):
            dtype = 'float'
        return cls(np.full((h, w), value, dtype=dtype))

    @classmethod
    def Squares(cls, number, size=256, fg=1, bg=0, dtype='uint8'):
        """
        Create image containing grid of squares

        :param number: number of squares horizontally and vertically
        :type number: int
        :param size: image width and height, defaults to 256
        :type size: int, optional
        :param fg: pixel value of the squares, defaults to 1
        :type fg: int or float, optional
        :param bg: pixel value of the background, defaults to 0
        :type bg: int, optional
        :param dtype: NumPy datatype, defaults to 'uint8'
        :type dtype: str, optional
        :return: grid of squares
        :rtype: Image instance

        .. notes::
            - Image is square
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
                im[y0-s2:y0+s2+1, x0-s2:x0+s2+1] = sq

        return cls(im)

    @classmethod
    def Circles(cls, number, size=256, fg=1, bg=0, dtype='uint8'):
        """
        Create image containing grid of circles

        :param number: number of circles horizontally and vertically
        :type number: int
        :param size: image width and height, defaults to 256
        :type size: int, optional
        :param fg: pixel value of the circles, defaults to 1
        :type fg: int or float, optional
        :param bg: pixel value of the background, defaults to 0
        :type bg: int, optional
        :param dtype: NumPy datatype, defaults to 'uint8'
        :type dtype: str, optional
        :return: grid of circles
        :rtype: Image instance

        .. notes::
            - Image is square
        """
        im = np.full((size, size), bg, dtype=dtype)
        d = size // (3 * number + 1)
        side = 2 * d + 1  # keep it odd
        s2 = side // 2
        circle = Kernel.Circle(s2).astype(dtype) * fg

        for r in range(number):
            y0 = (r * 3 + 2) * d
            for c in range(number):
                x0 = (c * 3 + 2) * d
                im[y0-s2:y0+s2+1, x0-s2:x0+s2+1] = circle

        return cls(im)

    @classmethod
    def Ramp(cls, dir='x', size=256, cycles=2, dtype='float32'):
        """
        Create image of linear ramps

        :param dir: ramp direction: 'x' [default] or 'y'
        :type dir: str, optional
        :param size: image width and height, defaults to 256
        :type size: int, optional
        :param cycles: Number of complete ramps, defaults to 2
        :type cycles: int, optional
        :param dtype: NumPy datatype, defaults to 'float32'
        :type dtype: str, optional
        :return: intensity ramps
        :rtype: Image instance

        The ramps span the range:

        * float image: 0 to 1
        * int image: 0 to intmax
        """
        c = size / cycles
        x = np.arange(0, size)
        s = np.expand_dims(np.mod(x, c) / (c - 1), axis=0)
        image = np.repeat(s, size, axis=0)

        if dir == 'y':
            image = image.T

        return cls(image, dtype=dtype)

    @classmethod
    def Sin(cls, dir='x', size=256, cycles=2, dtype='float32'):
        """
        Create image of sinusoidal intensity pattern

        :param dir: sinusoid direction: 'x' [default] or 'y'
        :type dir: str, optional
        :param size: image width and height, defaults to 256
        :type size: int, optional
        :param cycles: Number of complete cycles, defaults to 2
        :type cycles: int, optional
        :param dtype: NumPy datatype, defaults to 'float32'
        :type dtype: str, optional
        :return: sinusoidal pattern
        :rtype: Image instance

        The sinusoids are offset and span the range:

        * float image: 0 to 1
        * int image: 0 to intmax
        """
        c = size / cycles
        x = np.arange(0, size)
        s = np.expand_dims((np.sin(x / c * 2 * np.pi) + 1) / 2, axis=0)
        image = np.repeat(s, size, axis=0)
        if dir == 'y':
            image = image.T

        return cls(image, dtype=dtype)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    import pathlib
    import os.path
    
    exec(open(pathlib.Path(__file__).parent.parent.absolute() / "tests" / "test_core.py").read())  # pylint: disable=exec-used