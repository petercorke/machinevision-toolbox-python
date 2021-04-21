#!/usr/bin/env python

import numpy as np
import spatialmath.base.argcheck as argcheck
import cv2 as cv

import machinevisiontoolbox.base.color as color

from scipy import interpolate

# import scipy as sp

# from scipy import signal
# from scipy import interpolate

# from collecitons import namedtuple
# from pathlib import Path


class ImageProcessingColorMixin:
    """
    Image processing color operations on the Image class
    """

    def colorise(self, c=[1, 1, 1]):
        """
        Colorise a greyscale image

        :param c: color to color image :type c: string or rgb-tuple :return
        out: Image  with float64 precision elements ranging from 0 to 1 :rtype:
        Image instance

        - ``IM.color()`` is a color image out, where each color plane is equal
          to image.

        - ``IM.imcolor(c)`` as above but each output pixel is ``c``(3,1) times
          the corresponding element of image.

        Example:

        .. autorun:: pycon

        .. note::

            - Can convert a monochrome sequence (h,W,N) to a color image
              sequence (H,W,3,N).


        :references:

            - Robotics, Vision & Control, Section 10.1, P. Corke,
              Springer 2011.
        """

        c = argcheck.getvector(c).astype(self.dtype)
        c = c[::-1]  # reverse because of bgr

        # make sure im are greyscale
        img = self.mono()

        if img.iscolor is False:
            # only one plane to convert
            # recall opencv uses BGR
            out = [np.dstack((c[0] * im.image,
                              c[1] * im.image,
                              c[2] * im.image))
                   for im in img]
        else:
            raise ValueError(self.image, 'Image must be greyscale')

        return self.__class__(out)




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
        imf = self.float().image

        if src is None:
            src = 'bgr'

        # options gamma, on by default if to is RGB or BGR
        # options white on by default

        out = []
        print('converting from', src, 'to', dst)

        out = color.colorspace_convert(imf, src, dst)
        print('conversion done')
        return self.__class__(out, colororder=dst)


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

            if im.iscolor:
                R = color.gamma_encode(im.red, gamma)
                G = color.gamma_encode(im.green, gamma)
                B = color.gamma_encode(im.blue, gamma)
                out.append(np.dstack((R, G, B)))
            else:
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

            if im.iscolor:
                R = gamma_decode(m.red, gamma)
                G = gamma_decode(im.green, gamma)
                B = gamma_decode(im.blue, gamma)
                out.append(np.dstack((R, G, B)))
            else:
                out.append(gamma_decode(im.image, gamma))

        return self.__class__(out)

# --------------------------------------------------------------------------#
if __name__ == '__main__':

    # test run ImageProcessingColor.py
    print('ImageProcessingColor.py')

    from machinevisiontoolbox.Image import Image

    im = Image('monalisa.png')
    im.disp()

    imcs = Image.showcolorspace()
    imcs.disp()

    import code
    code.interact(local=dict(globals(), **locals()))
