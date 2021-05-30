#!/usr/bin/env python

import numpy as np
import spatialmath.base.argcheck as argcheck
import cv2 as cv

from machinevisiontoolbox.base import color
from machinevisiontoolbox.base import imageio

from scipy import interpolate

# import scipy as sp

# from scipy import signal
# from scipy import interpolate

# from collecitons import namedtuple
# from pathlib import Path


class ImageColorMixin:
    """
    Image processing color operations on the Image class
    """
    def mono(self, opt='r601'):
        """
        Convert color image to monochrome

        :param opt: greyscale conversion mode: 'r601' [default], 'r709' or
          'value'
        :type opt: str
        :return: Image with floating point pixel types
        :rtype: Image instance

        - ``IM.mono(im)`` is a greyscale equivalent of the color image ``im``

        Example:

        .. runblock:: pycon

            >>> im = Image('flowers1.png')
            >>> im
            >>> im_mono = im.mono()
            >>> im_mono

        :references:

            - Robotics, Vision & Control, Section 10.1, P. Corke,
              Springer 2011.
        """

        if not self.iscolor:
            return self

        if opt == 'r601':
            mono = 0.229 * self.red() + 0.587 * self.green() + \
                0.114 * self.blue()

        elif opt == 'r709':
            mono = 0.2126 * self.red() + 0.7152 * self.green() + \
                0.0722 * self.blue()

        elif opt == 'value':
            # 'value' refers to the V in HSV space, not the CIE L*
            # the mean of the max and min of RGB values at each pixel
            mn = self.A.min(axis=2)
            mx = self.A.max(axis=2)

            # # if np.issubdtype(im.dtype, np.float):
            # # NOTE let's make a new predicate for Image
            # if im.isfloat:
            #     mono = 0.5 * (mn + mx)
            #     mono = mono.astype(im.dtype)
            # else:
            #     z = (np.int32(mx) + np.int32(mn)) / 2
            #     mono = z.astype(im.dtype)
            mono = mn / 2 + mx / 2
        else:
            raise TypeError('unknown type for opt')

        return self.__class__(self.cast(mono.A))

    def chromaticity(self, which='RG'):
      if not self.iscolor:
        raise ValueError('cannot compute chromaticity for greyscale image')
      if self.nplanes != 3:
        raise ValueError('expecting 3 plane image')

      sum = np.sum(self.image, axis=2)
      r = self._plane(which[0]) / sum
      g = self._plane(which[1]) / sum

      return self.__class__(np.dstack((r, g)), colororder=which.lower())

    def chromaticity(self, chroma='rg'):

        # TODO
        im = self.asfloat()
        return self.__class__(color.tristim2cc(im), colororder=chroma)


    def tristim2cc(self):
        return self.__class__(color.tristim2cc(self.image), colororder='rg')


    def colorize(self, c=[1, 1, 1], colororder='RGB', alpha=False):
        """
        Colorize a greyscale image

        :param c: color to color image :type c: string or rgb-tuple :return
        out: Image  with float64 precision elements ranging from 0 to 1 :rtype:
        Image instance

        ``c`` is defined in terms of the specified color order.

        - ``IM.color()`` is a color image out, where each color plane is equal
          to image.

        - ``IM.imcolor(c)`` as above but each output pixel is ``c``(3,1) times
          the corresponding element of image.

        .. note::
            - Can convert a monochrome sequence (h,W,N) to a color image
              sequence (H,W,3,N).

        :references:

            - Robotics, Vision & Control, Section 10.1, P. Corke,
              Springer 2011.
        """

        # TODO, colorize all in list
        c = argcheck.getvector(c).astype(self.dtype)
        if self.iscolor:
          raise ValueError(self.image, 'Image must be greyscale')

        # alpha can be False, True, or scalar
        if alpha is False:
            out = np.dstack((c[0] * self.A,
                             c[1] * self.A,
                             c[2] * self.A))
        else:
          if alpha is True:
            alpha = 1

          out = np.dstack((c[0] * self.A,
                           c[1] * self.A,
                           c[2] * self.A,
                           alpha * np.ones(self.shape)))

        return self.__class__(out, colororder=colororder)

    def grey(self, colorspace=None):
      return self.colorspace('gray')

    def colorkmeans(self, k):
        # TODO
        # colorspace can be RGB, rg, Lab, ab
        
        data = self.column()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        if isinstance(k, int):
            # perform clustering
            ret, label, centres = cv.kmeans(
                    data=data,
                    K= k,
                    bestLabels=None,
                    criteria=criteria,
                    attempts=10,
                    flags=cv.KMEANS_RANDOM_CENTERS
                )
            return self.__class__(label.reshape(self.shape[:2])), centres, ret
        
        elif isinstance(k, np.ndarray):
            # assign pixels to given cluster centres
            centres = k.T  # M x K
            k = centres.shape[1]
            data = np.repeat(data[..., np.newaxis], k, axis=2)  # N x M x K

            # compute L2 norm over the error
            distance = np.linalg.norm(data - centres, axis=1)  # N x K

            # now find which cluster centre gave the smallest error 
            label = np.argmin(distance, axis=1)

            return self.__class__(label.reshape(self.shape[:2]))

    def colorspace(self, dst=None, src=None, **kwargs):
        """
        Transform a color image between color representations

        :param conv: color code for color conversion (OpenCV codes for now)
        :type conv: string (see below)
        :param kwargs: keywords/options for OpenCV's cvtColor
        :type kwargs: name/value pairs
        :return: out
        :rtype: numpy array, shape (N,M) or (N,3)

        :references:

            - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
        """

        # TODO other color cases
        # TODO check conv is valid

        # TODO conv string parsing

        # ensure floats? unsure if cv.cvtColor operates on ints
        imf = self.asfloat()

        if src is None:
            src = 'bgr'

        # options gamma, on by default if to is RGB or BGR
        # options white on by default

        out = []
        print('converting from', src, 'to', dst)

        out = color.colorspace_convert(imf, src, dst)
        print('conversion done')
        if out.ndim > 2:
          colororder = dst
        else:
          colororder = None
        return self.__class__(out, dtype=self.dtype, colororder=colororder)


        # for im in imf:
            # if conv == 'xyz2bgr':
            #     # note that using cv.COLOR_XYZ2RGB does not seem to work

            #     BGR_raw = cv.cvtColor(im.bgr, cv.COLOR_XYZ2BGR, **kwargs)

            #     B = BGR_raw[:, :, 0]
            #     G = BGR_raw[:, :, 1]
            #     R = BGR_raw[:, :, 2]

            #     # desaturate and rescale to constrain resulting RGB values
            #     # to [0,1]
            #     # add_white = -np.minimum(np.minimum(np.minimum(R, G), B), 0)
            #     # B += add_white
            #     # G += add_white
            #     # R += add_white
            #     mn = np.amin(BGR_raw, axis=2)
            #     BGR_raw += mn

            #     BGR = color.gamma_encode(BGR_raw)
            #     # inverse gamma correction
            #     B = color.gamma_encode(B)
            #     G = color.gamma_encode(G)
            #     R = color.gamma_encode(R)

            #     out.append(np.dstack((B, G, R)))  # BGR

            # elif conv == 'Lab2bgr':
            #     # convert source from Lab to xyz

            #     # in colorspace.m, image was parsed into a (251001,1,3)
            #     labim = np.reshape(im.image,
            #                        (im.shape[0], 1, im.shape[1]))

            #     fY = (labim[:, :, 0] + 16) / 116
            #     fX = fY + labim[:, :, 1] / 500
            #     fZ = fY - labim[:, :, 2] / 200
            #     # cie xyz whitepoint
            #     WhitePoint = np.r_[0.950456, 1, 1.088754]

            #     xyz = np.zeros(labim.shape)
            #     xyz[:, :, 0] = WhitePoint[0] * self._invf(fX)
            #     xyz[:, :, 1] = WhitePoint[1] * self._invf(fY)
            #     xyz[:, :, 2] = WhitePoint[2] * self._invf(fZ)

            #     # then call function again with conv = xyz2bgr
            #     xyz = self.__class__(xyz)

            #     out.append(xyz.colorspace('xyz2bgr').image)

            # else:
            #     raise ValueError('other conv options not yet implemented')
            #     # TODO other color conversion cases
            #     # out.append(cv.cvtColor(np.float32(im), **kwargs))

        return self.__class__(out)


    def _invf(self, fY):
        """
        Inverse f from colorspace.m
        """
        Y = fY ** 3
        Y[Y < 0.008856] = (fY[Y < 0.008856] - 4 / 29) * (108 / 841)
        return Y

    def gamma_encode(self, gamma):
        """
        Gamma encoding

        :param gamma: gamma value
        :type gam: string or float
        :return: gamma encoded version of image
        :rtype: Image instance

        - ``IM.gamma_encode(gamma)`` is the image with an gamma correction based
          applied.  This takes a linear luminance image and converts it to a 
          form suitable for display on a non-linear monitor.

        Example:

        .. autorun:: pycon

        .. note::

            - Gamma encoding is typically performed in a camera with
              GAMMA=0.45.
            - For images with multiple planes the gamma correction is applied
              to all planes.
            - For images sequences the gamma correction is applied to all
              elements.
            - For images of type double the pixels are assumed to be in the
              range 0 to 1.
            - For images of type int the pixels are assumed in the range 0 to
              the maximum value of their class.  Pixels are converted first to
              double, processed, then converted back to the integer class.

        :references:

            - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
        """

        out = []
        for im in self:
          out.append(color.gamma_encode(im.image, gamma))

        return self.__class__(out)

    def gamma_decode(self, gamma):
        """
        Gamma decoding

        :param gamma: gamma value
        :type gam: string or float
        :return: gamma decoded version of image
        :rtype: Image instance

        - ``IM.gamma_decode(gamma)`` is the image with an gamma correction
          applied.  This takes a gamma-corrected image and converts it to a
          linear luminance image.

        Example:

        .. autorun:: pycon

        .. note::

            - Gamma decoding should be applied to any color image prior to
              colometric operations.
            - Gamma decoding is typically performed in the display with
              GAMMA=2.2.
            - For images with multiple planes the gamma correction is applied
              to all planes.
            - For images sequences the gamma correction is applied to all
              elements.
            - For images of type double the pixels are assumed to be in the
              range 0 to 1.
            - For images of type int the pixels are assumed in the range 0 to
              the maximum value of their class.  Pixels are converted first to
              double, processed, then converted back to the integer class.

        :references:

            - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
        """

        out = []
        for im in self:
          out.append(color.gamma_decode(im.image, gamma))

        return self.__class__(out, colororder=self.colororder)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    import pathlib
    import os.path
    
    exec(open(pathlib.Path(__file__).parent.parent.absolute() / "tests" / "test_color.py").read())  # pylint: disable=exec-used
