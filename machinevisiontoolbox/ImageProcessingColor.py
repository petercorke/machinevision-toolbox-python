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

    def red(self):
        """
        Extract the red plane of a color image

        :raises ValueError: if image is not color
        :return out: greyscale image representing the red image plane
        :rtype: Image instance
        """
        if not self.iscolor:
            raise ValueError('cannot extract color plane from greyscale image')

        out = [im.rgb[:, :, 0] for im in self]
        # out = []
        # for im in self:
        #     out.append(im.image[:, :, 0])
        return self.__class__(out)

    def green(self):
        """
        Extract the green plane of a color image

        :raises ValueError: if image is not color
        :return out: greyscale image representing the green image plane
        :rtype: Image instance
        """
        if not self.iscolor:
            raise ValueError('cannot extract color plane from greyscale image')

        out = [im.rgb[:, :, 1] for im in self]
        # out = []
        # for im in self:
        #     out.append(im.image[:, :, 1])
        return self.__class__(out)

    def blue(self):
        """
        Extract the blue plane of a color image

        :raises ValueError: if image is not color
        :return out: greyscale image representing the blue image plane
        :rtype: Image instance
        """
        if not self.iscolor:
            raise ValueError('cannot extract color plane from greyscale image')

        out = [im.rgb[:, :, 2] for im in self]
        # out = []
        # for im in self:
        #     out.append(im.image[:, :, 2])
        return self.__class__(out)

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




    def colorspace(self, conv, **kwargs):
        """
        Transform a color image between color representations

        :param conv: color code for color conversion (OpenCV codes for now)
        :type conv: string (see below)
        :param kwargs: keywords/options for OpenCV's cvtColor
        :type kwargs: name/value pairs
        :return: out
        :rtype: numpy array, shape (N,M) or (N,3)

        - ``IM.colorspace(conv)`` transforms the color representation of image
          where ``conv`` is a string specifying the conversion. The image
          should be a real full double array of size (M,3) or (M,N,3). The
          output is the same size as ``IM``

        ``conv`` tells the source and destination color spaces,
        ``conv`` = 'dest<-src', or alternatively, ``conv`` = 'src->dest'.
        Supported color spaces are
        'RGB'              sRGB IEC 61966-2-1
        'YCbCr'            Luma + Chroma ("digitized" version of Y'PbPr)
        'JPEG-YCbCr'       Luma + Chroma space used in JFIF JPEG
        'YDbDr'            SECAM Y'DbDr Luma + Chroma
        'YPbPr'            Luma (ITU-R BT.601) + Chroma
        'YUV'              NTSC PAL Y'UV Luma + Chroma
        'YIQ'              NTSC Y'IQ Luma + Chroma
        'HSV' or 'HSB'     Hue Saturation Value/Brightness
        'HSL' or 'HLS'     Hue Saturation Luminance
        'HSI'              Hue Saturation Intensity
        'XYZ'              CIE 1931 XYZ
        'Lab'              CIE 1976 L*a*b* (CIELAB)
        'Luv'              CIE L*u*v* (CIELUV)
        'LCH'              CIE L*C*H* (CIELCH)
        'CAT02 LMS'        CIE CAT02 LMS

        .. note::

            - All conversions assume 2 degree observer and D65 illuminant.
              Color space names are case insensitive and spaces are ignored.
              When sRGB is the source or destination, it can be omitted. For
              example 'yuv<-' is short for 'yuv<-rgb'. For sRGB, the values
              should be scaled between 0 and 1.  Beware that transformations
              generally do not constrain colors to be "in gamut." Particularly,
              transforming from another space to sRGB may obtain R'G'B' values
              outside of the [0,1] range.  So the result should be clamped to
              [0,1] before displaying. image(min(max(B,0),1));  lamp B to [0,1]
              and display sRGB (Red Green Blue) is the (ITU-R BT.709
              gamma-corrected) standard red-green-blue representation of colors
              used in digital imaging.  The components should be scaled between
              0 and 1.  The space can be visualized geometrically as a cube.
            - Y'PbPr, Y'CbCr, Y'DbDr, Y'UV, and Y'IQ are related to sRGB by
              linear transformations.  These spaces separate a color into a
              grayscale luminance component Y and two chroma components.  The
              valid ranges of the components depends on the space.
            - HSV (Hue Saturation Value) is related to sRGB by
              H = hexagonal hue angle   (0 <= H < 360),
              S = C/V                   (0 <= S <= 1),
              V = max(R',G',B')         (0 <= V <= 1),
              where C = max(R',G',B') - min(R',G',B').
            - The hue angle H is computed on a hexagon.  The space is
              geometrically a hexagonal cone.
            - HSL (Hue Saturation Lightness) is related to sRGB by
              H = hexagonal hue angle                (0 <= H < 360),
              S = C/(1 - abs(2L-1))                     (0 <= S <= 1),
              L = (max(R',G',B') + min(R',G',B'))/2  (0 <= L <= 1),
              where H and C are the same as in HSV.  Geometrically, the space
              is a double hexagonal cone.
            - HSI (Hue Saturation Intensity) is related to sRGB by
              H = polar hue angle        (0 <= H < 360),
              S = 1 - min(R',G',B')/I    (0 <= S <= 1),
              I = (R'+G'+B')/3           (0 <= I <= 1).
              Unlike HSV and HSL, the hue angle H is computed on a circle
              rather than a hexagon.
            - CIE XYZ is related to sRGB by inverse gamma correction followed
              by a linear transform.  Other CIE color spaces are defined
              relative to XYZ.
            - CIE L*a*b*, L*u*v*, and L*C*H* are nonlinear functions of XYZ.
              The L* component is designed to match closely with human
              perception of lightness.  The other two components describe the
              chroma.
            - CIE CAT02 LMS is the linear transformation of XYZ using the
              MCAT02 chromatic adaptation matrix.  The space is designed to
              model the response of the three types of cones in the human eye,
              where L, M, S, correspond respectively to red ("long"), green
              ("medium"), and blue ("short").

        :references:

            - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
        """

        # TODO other color cases
        # TODO check conv is valid

        # TODO conv string parsing

        # ensure floats? unsure if cv.cvtColor operates on ints
        imf = self.float()

        out = []
        for im in imf:
            if conv == 'xyz2bgr':
                # note that using cv.COLOR_XYZ2RGB does not seem to work
                BGR_raw = cv.cvtColor(im.bgr, cv.COLOR_XYZ2BGR, **kwargs)

                # desaturate and rescale to constrain resulting RGB values
                # to [0,1]
                B = BGR_raw[:, :, 0]
                G = BGR_raw[:, :, 1]
                R = BGR_raw[:, :, 2]
                add_white = -np.minimum(np.minimum(np.minimum(R, G), B), 0)
                B += add_white
                G += add_white
                R += add_white

                # inverse gamma correction
                B = self._gammacorrection(B)
                G = self._gammacorrection(G)
                R = self._gammacorrection(R)

                out.append(np.dstack((B, G, R)))  # BGR

            elif conv == 'Lab2bgr':
                # convert source from Lab to xyz

                # in colorspace.m, image was parsed into a (251001,1,3)
                labim = np.reshape(im.image,
                                   (im.shape[0], 1, im.shape[1]))

                fY = (labim[:, :, 0] + 16) / 116
                fX = fY + labim[:, :, 1] / 500
                fZ = fY - labim[:, :, 2] / 200
                # cie xyz whitepoint
                WhitePoint = np.r_[0.950456, 1, 1.088754]

                xyz = np.zeros(labim.shape)
                xyz[:, :, 0] = WhitePoint[0] * self._invf(fX)
                xyz[:, :, 1] = WhitePoint[1] * self._invf(fY)
                xyz[:, :, 2] = WhitePoint[2] * self._invf(fZ)

                # then call function again with conv = xyz2bgr
                xyz = self.__class__(xyz)

                out.append(xyz.colorspace('xyz2bgr').image)

            else:
                raise ValueError('other conv options not yet implemented')
                # TODO other color conversion cases
                # out.append(cv.cvtColor(np.float32(im), **kwargs))

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
