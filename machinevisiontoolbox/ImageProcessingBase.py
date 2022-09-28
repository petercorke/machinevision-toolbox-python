#!/usr/bin/env python

from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import interpolate
import cv2 as cv

import spatialmath.base.argcheck as argcheck
from machinevisiontoolbox.base import color, int_image, float_image, plot_histogram

class ImageProcessingBaseMixin:
    """
    Image processing basic operations on the Image class
    """

    def int(self, intclass='uint8'):
        """
        Convert image to integer type

        :param intclass: either 'uint8', or any integer class supported by np
        :type intclass: str
        :return: Image with integer pixel types
        :rtype: Image instance

        - ``IM.int()`` is a copy of image with pixels converted to unsigned
          8-bit integer (uint8) elements in the range 0 to 255.

        - ``IM.int(intclass)`` as above but the output pixels are converted to
          the integer class ``intclass``.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image('flowers1.png', dtype='float64')
            >>> print(im)
            >>> im_int = im.int()
            >>> print(im_int)

        .. note::

            - Works for an image with arbitrary number of dimensions, eg. a
              color image or image sequence.
            - If the input image is floating point (single or double) the
              pixel values are scaled from an input range of [0,1] to a range
              spanning zero to the maximum positive value of the output integer
              class.
            - If the input image is an integer class then the pixels are cast
              to change type but not their value.

        :references:

            - Robotics, Vision & Control, Section 12.1, P. Corke,
              Springer 2011.
        """

        out = []
        for im in self:
            out.append(int_image(im.image, intclass))
        return self.__class__(out)

    def float(self, floatclass='float32'):
        """
        Convert image to float type

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

        Example:

        .. runblock:: pycon

            >>> im = Image('flowers1.png')
            >>> print(im)
            >>> im_float = im.float()
            >>> print(im_float)

        :references:

            - Robotics, Vision & Control, Section 12.1, P. Corke,
              Springer 2011.
        """

        out = []
        for im in self:
            out.append(float_image(im.image, floatclass))
        return self.__class__(out)

    def mono(self, opt='r601'):
        """
        Convert color image to monochrome

        :param opt: greyscale conversion option 'r601' [default] or 'r709'
        :type opt: string
        :return: Image with floating point pixel types
        :rtype: Image instance

        - ``IM.mono(im)`` is a greyscale equivalent of the color image ``im``

        Example:

        .. runblock:: pycon

            >>> im = Image('flowers1.png')
            >>> print(im)
            >>> im_mono = im.mono()
            >>> print(im_mono)

        :references:

            - Robotics, Vision & Control, Section 10.1, P. Corke,
              Springer 2011.
        """

        if not self.iscolor:
            return self

        out = []
        for im in [img.bgr for img in self]:
            if opt == 'r601':
                new = 0.229 * im[:, :, 2] + 0.587 * im[:, :, 1] + \
                    0.114 * im[:, :, 0]
                new = new.astype(im.dtype)
            elif opt == 'r709':
                new = 0.2126 * im[:, :, 0] + 0.7152 * im[:, :, 1] + \
                    0.0722 * im[:, :, 2]
                new = new.astype(im.dtype)
            elif opt == 'value':
                # 'value' refers to the V in HSV space, not the CIE L*
                # the mean of the max and min of RGB values at each pixel
                mn = im[:, :, 2].min(axis=2)
                mx = im[:, :, 2].max(axis=2)

                # if np.issubdtype(im.dtype, np.float):
                # NOTE let's make a new predicate for Image
                if im.isfloat:
                    new = 0.5 * (mn + mx)
                    new = new.astype(im.dtype)
                else:
                    z = (np.int32(mx) + np.int32(mn)) / 2
                    new = z.astype(im.dtype)
            else:
                raise TypeError('unknown type for opt')

            out.append(new)
        return self.__class__(out)

    def stretch(self, max=1, r=None):
        """
        Image normalisation

        :param max: M  pixels are mapped to the r 0 to M
        :type max: scalar integer or float
        :param r: r[0] is mapped to 0, r[1] is mapped to 1 (or max value)
        :type r: 2-tuple or numpy array (2,1)
        :return: Image with pixel values stretched to M across r
        :rtype: Image instance

        - ``IM.stretch()`` is a normalised image in which all pixel values lie
          in the r range of 0 to 1. That is, a linear mapping where the minimum
          value of ``im`` is mapped to 0 and the maximum value of ``im`` is
          mapped to 1.

        Example:

        .. runblock:: pycon

        .. note::

            - For an integer image the result is a float image in the range 0
              to max value

        :references:

            - Robotics, Vision & Control, Section 12.1, P. Corke,
              Springer 2011.
        """

        # TODO make all infinity values = None?

        out = []
        for im in [img.image for img in self]:
            if r is None:
                mn = np.min(im)
                mx = np.max(im)
            else:
                r = argcheck.getvector(r)
                mn = r[0]
                mx = r[1]

            zs = (im - mn) / (mx - mn) * max

            if r is not None:
                zs = np.maximum(0, np.minimum(max, zs))
            out.append(zs)

        return self.__class__(out)

    def thresh(self, t=None, opt='binary'):
        """
        Image threshold

        :param t: threshold
        :type t: scalar
        :param opt: threshold option (see below)
        :type opt: string
        :return imt: Image thresholded binary image
        :rtype imt: Image instance
        :return: threshold if opt is otsu or triangle
        :rtype: list of scalars

        - ``IM.thresh()`` uses Otsu's method for thresholding a greyscale
          image.

        - ``IM.thresh(t)`` as above but the threshold ``t`` is specified.

        - ``IM.thresh(t, opt)`` as above but the threshold option is specified.
          See opencv threshold types for threshold options
          https://docs.opencv.org/4.2.0/d7/d1b/group__imgproc__
          misc.html#gaa9e58d2860d4afa658ef70a9b1115576

        Example:

        .. runblock:: pycon

        :options:

            - 'binary' # TODO consider the LaTeX formatting of equations
            - 'binary_inv'
            - 'trunc'
            - 'tozero'
            - 'tozero_inv'
            - 'otsu'
            - 'triangle'

        .. note::

            - Converts a color image to greyscale.
            - For a uint8 class image the slider range is 0 to 255.
            - For a floating point class image the slider range is 0 to 1.0
        """

        # dictionary of threshold options from OpenCV
        threshopt = {
            'binary': cv.THRESH_BINARY,
            'binary_inv': cv.THRESH_BINARY_INV,
            'trunc': cv.THRESH_TRUNC,
            'tozero': cv.THRESH_TOZERO,
            'tozero_inv': cv.THRESH_TOZERO_INV,
            'otsu': cv.THRESH_OTSU,
            'triangle': cv.THRESH_TRIANGLE
        }

        if t is not None:
            if not argcheck.isscalar(t):
                raise ValueError(t, 't must be a scalar')
        else:
            # if no threshold is specified, we assume to use Otsu's method
            print('No threshold specified. Applying Otsu''s method.')
            opt = 'otsu'

        # ensure mono images
        if self.iscolor:
            imono = self.mono()
        else:
            imono = self

        out_t = []
        out_imt = []
        for im in [img.image for img in imono]:

            # for image int class, maxval = max of int class
            # for image float class, maxval = 1
            if np.issubdtype(im.dtype, np.integer):
                maxval = np.iinfo(im.dtype).max
            else:
                # float image, [0, 1] range
                maxval = 1.0

            threshvalue, imt = cv.threshold(im, t, maxval, threshopt[opt])
            out_t.append(threshvalue)
            out_imt.append(imt)

        if opt == 'otsu' or opt == 'triangle':
            return self.__class__(out_imt), out_t
        else:
            return self.__class__(out_imt)

    def otsu(self, levels=256, valley=None):
        """
        Otsu threshold selection

        :return t: Otsu's threshold
        :rtype t: float
        :return imt: Image thresholded to a binary image
        :rtype imt: Image instance

        - ``otsu(im)`` is an optimal threshold for binarizing an image with a
          bimodal intensity histogram.  ``t`` is a scalar threshold that
          maximizes the variance between the classes of pixels below and above
          the thresold ``t``.

        Example::

        .. runblock:: pycon

        .. note::

            - Converts a color image to greyscale.

        :references:

            - A Threshold Selection Method from Gray-Level Histograms, N. Otsu.
              IEEE Trans. Systems, Man and Cybernetics Vol SMC-9(1), Jan 1979,
              pp 62-66.
            - An improved method for image thresholding on the valley-emphasis
              method. H-F Ng, D. Jargalsaikhan etal. Signal and Info Proc.
              Assocn. Annual Summit and Conf (APSIPA). 2013. pp1-4
        """

        # mvt-mat has options on levels and valleys, which Opencv does not have
        # TODO best option is likely just to code the function itself, with
        # default option of simply calling OpenCV's Otsu implementation

        im = self.mono()

        if (valley is None):
            imt, t = im.thresh(opt='otsu')

        else:
            raise ValueError(valley, 'not implemented yet')
            # TODO implement otsu.m
            # TODO levels currently ignored

        return imt, t

    def nonzero(self):
        return np.nonzero(self.image)

    def meshgrid(self, step=1):
        """
        Domain matrices for image

        :param a1: array input 1
        :type a1: numpy array
        :param a2: array input 2
        :type a2: numpy array
        :return u: domain of image, horizontal
        :rtype u: numpy array
        :return v: domain of image, vertical
        :rtype v: numpy array

        - ``IM.imeshgrid()`` are matrices that describe the domain of image
          ``im (h,w)`` and are each ``(h,w)``. These matrices are used for the
          evaluation of functions over the image. The element ``u(r,c) = c``
          and ``v(r,c) = r``.

        - ``IM.imeshgrid(w, h)`` as above but the domain is ``(w,h)``.

        - ``IM.imeshgrid(s)`` as above but the domain is described by ``s``
          which can be a scalar ``(s,s)`` or a 2-vector ``s=[w,h]``.

        Example:

        .. runblock:: pycon

        """

        # TODO too complex, simplify
        # Use cases
        #  image.meshgrid()  spans image
        #  image.meshgrid(step=N) spans image with step

        # if not (argcheck.isvector(a1) or isinstance(a1, np.ndarray)
        #         or argcheck.isscalar(a1) or isinstance(a1, self.__class__)):
        #     raise ValueError(
        #         a1, 'a1 must be an Image, matrix, vector, or scalar')
        # if a2 is not None and (not (argcheck.isvector(a2) or
        #                             isinstance(a2, np.ndarray) or
        #                             argcheck.isscalar(a2) or
        #                             isinstance(a2, self.__class__))):
        #     raise ValueError(
        #         a2, 'a2 must be Image, matrix, vector, scalar or None')

        # if isinstance(a1, self.__class__):
        #     a1 = a1.image
        # if isinstance(a2, self.__class__):
        #     a2 = a2.image

        # if a2 is None:
        #     if a1.ndim <= 1 and len(a1) == 1:
        #         # if a1 is a single number
        #         # we specify a size for a square output image
        #         ai = np.arange(0, a1)
        #         u, v = np.meshgrid(ai, ai)
        #     elif a1.ndim <= 1 and len(a1) == 2:
        #         # if a1 is a 2-vector
        #         # we specify a size for a rectangular output image (w, h)
        #         a10 = np.arange(0, a1[0])
        #         a11 = np.arange(0, a1[1])
        #         u, v = np.meshgrid(a10, a11)
        #     elif (a1.ndim >= 2):  # and (a1.shape[2] > 2):
        #         u, v = np.meshgrid(np.arange(0, a1.shape[1]),
        #                            np.arange(0, a1.shape[0]))
        #     else:
        #         raise ValueError(a1, 'incorrect argument a1 shape')
        # else:
        #     # we assume a1 and a2 are two scalars
        #     u, v = np.meshgrid(np.arange(0, a1), np.arange(0, a2))

        u = np.arange(0, self.width, step)
        v = np.arange(0, self.height, step)

        return np.meshgrid(v, u, indexing='ij')


    def hist(self, nbins=256, opt=None):
        """
        Image histogram

        :param nbins: number of bins for histogram
        :type nbins: integer
        :param opt: histogram option
        :type opt: string
        :return hist: histogram h as a column vector, and corresponding bins x,
        cdf and normcdf
        :rtype hist: collections.namedtuple

        - ``IM.hist()`` is the histogram of intensities for image as a vector.
          For an image with  multiple planes, the histogram of each plane is
          given in a separate column. Additionally, the cumulative histogram
          and normalized cumulative histogram, whose maximum value is one, are
          computed.

        - ``IM.hist(nbins)`` as above with the number of bins specified

        - ``IM.hist(opt)`` as above with histogram options specified

        :options:

            - 'sorted' histogram but with occurrence sorted in descending
              magnitude order.  Bin coordinates X reflect this sorting.

        Example:

        .. runblock:: pycon

        .. note::

            - The bins spans the greylevel range 0-255.
            - For a floating point image the histogram spans the greylevel
              range 0-1.
            - For floating point images all NaN and Inf values are first
              removed.
            - OpenCV CalcHist only works on floats up to 32 bit, images are
              automatically converted from float64 to float32
        """

        # check inputs
        optHist = ['sorted']
        if opt is not None and opt not in optHist:
            raise ValueError(opt, 'opt is not a valid option')

        if self.isint:
            maxrange = np.iinfo(self.dtype).max
        else:
            # float image
            maxrange = 1.0

        out = []
        for im in self:
            # normal histogram case

            xc = []
            hc = []
            hcdf = []
            hnormcdf = []
            implanes = cv.split(im.image)
            for i in range(self.numchannels):
                # bin coordinates
                x = np.linspace(0, maxrange, nbins, endpoint=True).T
                h = cv.calcHist(implanes, [i], None, [nbins], [0, maxrange + 1])

                if opt == 'sorted':
                    h = np.sort(h, axis=0)
                    isort = np.argsort(h, axis=0)
                    x = x[isort]

                cdf = np.cumsum(h)
                normcdf = cdf / cdf[-1]

                xc.append(x)
                hc.append(h)
                hcdf.append(cdf)
                hnormcdf.append(normcdf)

            # stack into arrays
            xs = np.vstack(xc).T
            hs = np.hstack(hc)
            cs = np.vstack(hcdf).T
            ns = np.vstack(hnormcdf).T

            # TODO this seems too complex, why do we stack stuff as well
            # as have an array of hist tuples??

            hhhx = Histogram(hs, cs, ns, xs)
            out.append(hhhx)

        if len(out) == 1:
            return out[0]
        else:
            return out

    # helper function that was part of hist() in the Matlab toolbox
    # TODO consider moving this to ImpageProcessingBase.py
    def plothist(self, title=None, block=False, **kwargs):
        """
        plot first image histogram as a line plot (TODO as poly)
        NOTE convenient, but maybe not a great solution because we then need to
        duplicate all the plotting options as for idisp?
        """
        if title is None:
            title = self[0].filename

        hist = self[0].hist(**kwargs)
        x = hist[0].x
        h = hist[0].h
        fig, ax = plt.subplots()

        # line plot histogram style
        if self.iscolor:
            ax.plot(x[:, 0], h[:, 0], 'b', alpha=0.8)
            ax.plot(x[:, 1], h[:, 1], 'g', alpha=0.8)
            ax.plot(x[:, 2], h[:, 2], 'r', alpha=0.8)
        else:
            ax.plot(hist[0].x, hist[0].h, 'k', alpha=0.7)

        # polygon histogram style
        polygon_style = False
        if polygon_style:
            if self.iscolor:
                from matplotlib.patches import Polygon
                # TODO make sure pb goes to bottom of axes at the edges:
                pb = np.stack((x[:, 0], h[:, 0]), axis=1)
                polyb = Polygon(pb,
                                closed=True,
                                facecolor='b',
                                linestyle='-',
                                alpha=0.75)
                ax.add_patch(polyb)

                pg = np.stack((x[:, 1], h[:, 1]), axis=1)
                polyg = Polygon(pg,
                                closed=True,
                                facecolor='g',
                                linestyle='-',
                                alpha=0.75)
                ax.add_patch(polyg)

                pr = np.stack((x[:, 2], h[:, 2]), axis=1)
                polyr = Polygon(pr,
                                closed=True,
                                facecolor='r',
                                linestyle='-',
                                alpha=0.75)
                ax.add_patch(polyr)

                # addpatch seems to require a plot, so hack is to plot null and
                # make alpha=0
                ax.plot(0, 0, alpha=0)
            else:
                from matplotlib.patches import Polygon
                p = np.hstack((x, h))
                poly = Polygon(p,
                               closed=True,
                               facecolor='k',
                               linestyle='-',
                               alpha=0.5)
                ax.add_patch(poly)
                ax.plot(0, 0, alpha=0)

        ax.set_ylabel('count')
        ax.set_xlabel('bin')
        ax.grid()

        ax.set_title(title)

        plt.show(block=block)

        # him = im[2].hist()
        # fig, ax = plt.subplots()
        # ax.plot(him[i].x[:, 0], him[i].h[:, 0], 'b')
        # ax.plot(him[i].x[:, 1], him[i].h[:, 1], 'g')
        # ax.plot(him[i].x[:, 2], him[i].h[:, 2], 'r')
        # plt.show()

    def normhist(self, nbins=256, opt=None):
        """
        Histogram normalisaton

        :param nbins: number of bins for histogram
        :type nbins: integer
        :param opt: histogram option
        :type opt: string
        :return nim: Image with normalised image
        :rtype nim: Image instance

        - ``IM.normhist()`` is a histogram normalized version of the image.

        Example:

        .. runblock:: pycon

        .. note::

            - Highlights image detail in dark areas of an image.
            - The histogram of the normalized image is approximately uniform,
              that is, all grey levels ae equally likely to occur.
            - Color images automatically converted to grayscale
        """

        # check inputs
        optHist = ['sorted']
        if opt is not None and opt not in optHist:
            raise ValueError(opt, 'opt is not a valid option')

        img = self.mono()

        # if self.iscolor:
        #     raise ValueError(self, 'normhist does not support color images')

        # NOTE could alternatively just call cv.equalizeHist()? However,
        # cv.equalizeHist only accepts 8-bit images, while normhist can
        # accept float images as well.
        # return cv.equalizeHist(im) cdf = hist(im, 'cdf')

        hcnx = img.hist(nbins, opt)

        out = []
        i = 0
        for im in img:
            # j = 0  # channel (only 1 channel due to mono)
            if im.isfloat:
                f = interpolate.interp1d(np.squeeze(hcnx[i].x),
                                         np.squeeze(hcnx[i].normcdf),
                                         kind='nearest')
                nim = f(im.image.flatten())
            else:
                f = interpolate.interp1d(np.squeeze(hcnx[i].x),
                                         np.squeeze(hcnx[i].normcdf),
                                         kind='nearest')
                # turn image data to float but scaled to im.dtype max
                imy = im.float().image.flatten() * \
                    float(np.iinfo(im.dtype).max)
                nim = f(imy)

            # reshape back into image format
            nimr = nim.reshape(im.shape[0], im.shape[1], order='C')

            o = self.__class__(nimr)
            if im.isfloat:
                o = o.float()  # nim = np.float32(nim)
            else:
                o = o.int()

            i += 1
            out.append(o)

        return self.__class__(out)

    def replicate(self, M=1):
        """
        Expand image

        :param M: number of times to replicate image
        :type M: integer
        :return out: Image expanded image
        :rtype out: Image instance

        - ``IM.replicate(M)`` is an expanded version of the image (H,W) where
          each pixel is replicated into a (M,M) tile. If ``im`` is (H,W) the
          result is ((M*H),(M*W)) numpy array.

        Example:

        .. runblock:: pycon

        """

        out = []
        for im in self:
            if im.ndims > 2:
                # dealing with multiplane image
                # TODO replace with a list comprehension
                ir2 = []
                for i in range(im.numchannels):
                    im1 = self.__class__(im.image[:, :, i])
                    ir2 = np.append(im1.replicate(M))
                return ir2

            nr = im.shape[0]
            nc = im.shape[1]

            # replicate columns
            ir = np.zeros((M * nr, nc), dtype=im.dtype)
            for r in range(M):
                ir[r:-1:M, :] = im.image

            # replicate rows
            ir2 = np.zeros((M * nr, M * nc), dtype=im.dtype)
            for c in range(M):
                ir2[:, c:-1:M] = ir
            out.append(ir2)

        return self.__class__(out)

    def decimate(self, m=2, sigma=None):
        """
        Decimate an image

        :param m: decimation factor TODO probably not the correct term
        :type m: integer
        :param sigma: standard deviation for Gaussian kernel smoothing
        :type sigma: float
        :return out: Image decimated image
        :rtype out: Image instance

        - ``IM.idecimate(m)`` is a decimated version of the image whose size is
          reduced by m (an integer) in both dimensions.  The image is smoothed
          with a Gaussian kernel with standard deviation m/2 then subsampled.

        - ``IM.idecimate(m, sigma)`` as above but the standard deviation of the
          smoothing kernel is set to ``sigma``.

        .. note::

            - If the image has multiple planes, each plane is decimated.
            - Smoothing is used to eliminate aliasing artifacts and the
              standard deviation should be chosen as a function of the maximum
              spatial frequency in the image.

        Example:

        .. runblock:: pycon

        """

        if (m - np.ceil(m)) != 0:
            raise ValueError(m, 'decimation factor m must be an integer')

        if sigma is None:
            sigma = m / 2

        # smooth image
        ims = self.smooth(sigma)

        # decimate image
        out = []
        for im in ims:
            out.append(im.image[0:-1:m, 0:-1:m, :])

        return self.__class__(out)

    def testpattern(self, t, w, *args, **kwargs):
        """
        Create test images

        :param t: pattern type
        :type t: string
        :param w: image size of output pattern image
        :type w: integer or 2-element vector
        :param args: arguments for test patterns
        :type args: float (varies)
        :param kwargs: keyword arguments for test patterns? Not currently used
        :type kwargs: dictionary
        :return z: test pattern image
        :rtype z: numpy array

        - ``testpattern(type, w, args)`` creates a test pattern image.  If
          ``w`` is a scalar the output image has shape ``(w,w)`` else if
          ``w=(w,h)`` the output image shape is ``(w,h)``.  The image is
          specified by the string ``t`` and one or two (type specific)
          arguments:

        :options:

            - 'rampx' intensity ramp from 0 to 1 in the x-direction.
              ARGS is the number of cycles.
            - 'rampy' intensity ramp from 0 to 1 in the y-direction.
              ARGS is the number of cycles.
            - 'sinx' sinusoidal intensity pattern (from -1 to 1) in the
              x-direction. ARGS is the number of cycles.
            - 'siny' sinusoidal intensity pattern (from -1 to 1) in the
              y-direction. ARGS is the number of cycles.
            - 'dots' binary dot pattern.  ARGS are dot pitch (distance between
              centres); dot diameter.
            - 'squares' binary square pattern.  ARGS are pitch (distance
              between centres); square side length.
            - 'line'  a line.  ARGS are theta (rad), intercept.

        Example:

        .. runblock:: pycon

        """

        # check valid input
        topt = ['sinx', 'siny', 'rampx', 'rampy', 'line', 'squares', 'dots']
        if t not in topt:
            raise ValueError(t, 't is an unknown pattern type')

        w = argcheck.getvector(w)
        if len(w) == 1:
            w = np.int(w)
            z = np.zeros((w, w))
        elif len(w) == 2:
            # w = np.int(w)
            z = np.zeros((np.int(w[0]), np.int(w[1])))
        else:
            raise ValueError(w, 'w has more than two values')

        if t == 'sinx':
            if len(args) > 0:
                ncycles = args[0]
            else:
                ncycles = 1
            x = np.arange(0, z.shape[0])
            c = z.shape[0] / ncycles
            s = np.expand_dims(np.sin(x / c * ncycles * 2 * np.pi), axis=0)
            z = np.repeat(s, z.shape[1], axis=0)
            # z = matlib.repmat(np.sin(x / c * ncycles * 2 * np.pi),
            #                   z.shape[1], 1)

        elif t == 'siny':
            if len(args) > 0:
                ncycles = args[0]
            else:
                ncycles = 1
            c = z.shape[1] / ncycles
            y = np.arange(0, z.shape[1])
            y = np.expand_dims(y, axis=1)
            # z = matlib.repmat(np.sin(y / c * ncycles * 2 * np.pi),
            #                   1, z.shape[1])
            z = np.repeat(np.sin(y / c * ncycles * 2 * np.pi),
                          z.shape[1],
                          axis=1)

        elif t == 'rampx':
            if len(args) > 0:
                ncycles = args[0]
            else:
                ncycles = 1
            c = z.shape[0] / ncycles
            x = np.arange(0, z.shape[0])
            # z = matlib.repmat(np.mod(x, c) / (c - 1), z.shape[1], 1)
            s = np.expand_dims(np.mod(x, c) / (c - 1), axis=0)
            z = np.repeat(s, z.shape[1], axis=0)

        elif t == 'rampy':
            if len(args) > 0:
                ncycles = args[0]
            else:
                ncycles = 1
            c = z.shape[1] / ncycles
            y = np.arange(0, z.shape[1])
            y = np.expand_dims(y, axis=1)  # required due to 1D and 2D arrays
            # z = matlib.repmat(np.mod(y, c) / (c - 1), 1, z.shape[0])
            z = np.repeat(np.mod(y, c) / (c - 1), z.shape[0], axis=1)

        elif t == 'line':
            nr = z.shape[0]
            nc = z.shape[1]
            theta = args[0]
            c = args[1]

            if np.abs(np.tan(theta)) < 1:
                x = np.arange(0, nc)
                y = np.round(x * np.tan(theta) + c)
                # NOTE np.where seems to return a tuple, though it is
                # supposed to return an array
                s = np.where((y >= 1) * (y < nr))

            else:
                y = np.arange(0, nr)
                x = np.round((y - c) / np.tan(theta))
                # note: be careful about 1 vs 0, python vs matlab indexing
                s = np.where((x >= 1) * (x < nc))

            # s is a list - likely because np.where only returns an array if
            # you have two arrays as input
            # could probably np.where z for np.zeros(z.shape), np.ones(z.shape)
            for k in s[0]:
                z[int(y[k]), int(x[k])] = 1

        elif t == 'squares':
            nr = z.shape[0]
            nc = z.shape[1]
            pitch = args[0]
            d = args[1]
            if d > (pitch / 2):
                print('warning: squares will overlap')
            rad = np.int(np.floor(d / 2))
            d = 2 * rad
            for r in np.arange(pitch / 2, (nr - pitch / 2) + 1, pitch,
                               dtype=np.int):
                for c in np.arange(pitch / 2, (nc - pitch / 2) + 1, pitch,
                                   dtype=np.int):
                    # for r in range(pitch / 2.0, (nr - pitch / 2.0), pitch):
                    # for c in range(pitch / 2.0, (nc - pitch / 2.0), pitch):
                    z[r - rad:r + rad + 1, c - rad:c + rad + 1] = np.ones(d + 1)

        elif t == 'dots':
            nr = z.shape[0]
            nc = z.shape[1]
            pitch = args[0]
            d = args[1]
            if d > (pitch / 2.0):
                print('warning: dots will overlap')

            rad = np.int(np.floor(d / 2))
            d = 2 * rad
            s = self.kcircle(d / 2.0)

            # for r in range(pitch / 2, (nr - pitch / 2), pitch):
            # NOTE +1 is a hack to make np.arange include the endpoint. Surely
            # there's a better way?
            for r in np.arange(pitch / 2, (nr - pitch / 2) + 1, pitch,
                               dtype=np.int):
                for c in np.arange(pitch / 2, (nc - pitch / 2) + 1, pitch,
                                   dtype=np.int):
                    z[r - rad:r + rad + 1, c - rad:c + rad + 1] = s

        else:
            raise ValueError(t, 'unknown pattern type')
            z = []

        return self.__class__(z)



    def paste(self,
              pattern,
              pt,
              opt='set',
              centre=False,
              zero=True):
        """
        Paste an image into an image

        :param pattern: sub-image super-imposed onto onto canvas
        :type pattern: numpy array
        :param pt: coordinates where pattern is pasted
        :type pt: 2-element vector of integer coordinates
        :param opt: options for paste settings
        :type opt: string
        :param centre: True if pattern centered at pt, else topleft of pattern
        :type centre: boolean
        :param zero: zero-based coordinates (True) or 1-based coordinates
        :type zero: boolean
        :return out: Image with pasted image
        :rtype out: Image instance

        - ``IM.paste(pattern, pt)`` is the image canvas with the subimage
          ``pattern`` pasted in at the position ``pt=[U, V]``.

        - ``IM.paste(pattern, pt, centre)`` as above with centre specified. The
          pasted image is centred at ``pt``, otherwise ``pt`` is the top-left
          corner of the subimage in the image (default).

        - ``IM.paste(pattern, pt, zero)`` as above with zero specified. The
          coordinates of ``pt`` start at zero, by default (0, 0) is assumed.
          TODO shouldn't this be a point? like ``origin`` or something

        - ``IM.paste(pattern, pt, opt)`` as above with opt specified as below.

        :options:

            - 'set'       ``pattern`` overwrites the pixels in ``canvas``
              (default).
            - 'add'       ``pattern`` is added to the pixels in ``canvas``.
            - 'mean'      ``pattern`` is set to the mean of pixel values
              in ``canvas`` and ``pattern``.

        Example:

        .. runblock:: pycon

        .. note::

            - Pixels outside the pasted in region are unaffected.
        """

        # TODO can likely replace a lot of this with np.where?

        # check inputs
        pt = argcheck.getvector(pt)

        # TODO check optional inputs valid
        # TODO need to check that centre+point+pattern combinations are valid
        # for given canvas size
        out = []
        for canvas in self:
            cw = canvas.width
            ch = canvas.height
            pw = pattern.width
            ph = pattern.height

            pasteOpt = ['set', 'add', 'mean']
            if opt not in pasteOpt:
                raise ValueError(opt, 'opt is not a valid option for paste()')

            if centre:
                left = pt[0] - np.floor(pw / 2)
                top = pt[1] - np.floor(ph / 2)
            else:
                left = pt[0]  # x
                top = pt[1]  # y

            if not zero:
                left += 1
                top += 1

            # indexes must be integers
            top = np.int(top)
            left = np.int(left)

            if (top+ph) > ch:
                raise ValueError(ph, 'pattern falls off bottom edge')
            if (left+pw) > cw:
                raise ValueError(pw, 'pattern falls off right edge')

            if pattern.iscolor:
                npc = pattern.shape[2]
            else:
                npc = 1

            if canvas.iscolor:
                nc = canvas.shape[2]
            else:
                nc = 1

            if npc > nc:
                # pattern has multiple planes, replicate the canvas
                # sadly, this doesn't work because repmat doesn't work on 3D
                # arrays
                # o = np.matlib.repmat(canvas.image, [1, 1, npc])
                o = np.dstack([canvas.image for i in range(npc)])
            else:
                o = canvas.image

            if npc < nc:
                pim = np.dstack([pattern.image for i in range(nc)])
                # pattern.image = np.matlib.repmat(pattern.image, [1, 1, nc])
            else:
                pim = pattern.image

            if opt == 'set':
                if pattern.iscolor:
                    o[top:top+ph, left:left+pw, :] = pim
                else:
                    o[top:top+ph, left:left+pw] = pim

            elif opt == 'add':
                if pattern.iscolor:
                    o[top:top+ph, left:left+pw, :] = o[top:top+ph,
                                                       left:left+pw, :] + pim
                else:
                    o[top:top+ph, left:left+pw] = o[top:top+ph,
                                                    left:left+pw] + pim
            elif opt == 'mean':
                if pattern.iscolor:
                    old = o[top:top+ph, left:left+pw, :]
                    k = ~np.isnan(pim)
                    old[k] = 0.5 * (old[k] + pim[k])
                    o[top:top+ph, left:left+pw, :] = old
                else:
                    old = o[top:top+ph, left:left+pw]
                    k = ~np.isnan(pim)
                    old[k] = 0.5 * (old[k] + pim[k])
                    o[top:top+ph, left:left+pw] = old

            else:
                raise ValueError(opt, 'opt is not valid')

            out.append(o)

        return self.__class__(out)

    def peak2(self, npeaks=2, sc=1, interp=False):
        """
        Find peaks in a matrix

        :param npeaks: number of peaks to return (default all)
        :type npeaks: scalar
        :param sc: scale of peaks to consider
        :type sc: float
        :param interp:  interpolation done on peaks
        :type interp: boolean
        :return: peaks, xy locations, ap? TODO
        :rtype: collections.namedtuple

        - ``IM.peak2()`` are the peak values in the 2-dimensional signal
          ``IM``. Also returns the indices of the maxima in the matrix ``IM``.
          Use SUB2IND to convert these to row and column.

        - ``IM.peak2(npeaks)`` as above with the number of peaks to return
          specifieid (default all).

        - ``IM.peak2(sc)`` as above with scale ``sc`` specified. Only consider
          as peaks the largest value in the horizontal and vertical range +/- S
          units.

        - ``IM.peak2(interp)`` as above with interp specified. Interpolate peak
          (default no peak interpolation).

        Example:

        .. runblock:: pycon

        .. note::

            - A maxima is defined as an element that larger than its eight
              neighbours. Edges elements will never be returned as maxima.
            - To find minima, use PEAK2(-V).
            - The interp options fits points in the neighbourhood about the
              peak with a paraboloid and its peak position is returned.  In
              this case IJ will be non-integer.

        """

        # TODO check valid input

        # create a neighbourhood mask for non-local maxima suppression
        h = sc
        w = 2*h
        M = np.ones((w, w))
        M[h, h] = 0

        out = []
        for z in self:
            # compute the neighbourhood maximum
            znh = self.window(self.float(z), M, 'max', 'wrap')

            # find all pixels greater than their neighbourhood
            k = np.where(z > znh)

            # sort these local maxima into descending order
            ks = [np.argsort(z, axis=0)][:, :, -1]  # TODO check this
            k = k[ks]

            npks = np.min(np.length(k), npeaks)
            k = k[0:npks]

            y, x = np.unravel_index(k, z.shape)
            xy = np.stack((y, x), axis=2)

            # interpolate peaks if required
            if interp:
                # TODO see peak2.m, line 87-131
                raise ValueError(interp, 'interp not yet supported')
            else:
                xyp = xy
                zp = z(k)
                ap = []

            # TODO should xyp, etc be Images?
            o = namedtuple('peaks', 'xy' 'z' 'a')(xyp, zp, ap)
            out.append(o)

        return out

    def roi(self, reg=None, wh=None):
        """
        Extract region of interest

        :param reg: region
        :type reg: numpy array
        :param wh: width and/or height
        :type wh: 2-element vector of integers, or single integer
        :return: Image with roi as image
        :rtype: Image instance

        - ``IM.roi(rect)`` is a subimage of the image described by the
          rectangle ``rect=[umin, umax; vmin, vmax]``. The function returns the
          top, left, bottom and top coordinates of the selected region of
          interest, as vectors.

        - ``IM.roi(reg, wh)`` as above but the region is centered at
          ``reg=(U,V)`` and has a size ``wh``.  If ``wh`` is scalar then
          ``W=H=S`` otherwise ``S=(W,H)``.
        """

        # interpret reg
        if reg is not None and wh is not None:
            # reg = getself.__class__(reg)  # 2x2?
            wh = argcheck.getvector(wh)

            xc = reg[0]
            yc = reg[1]
            if len(wh) == 1:
                w = np.round(wh/2)
                h = w
            else:
                w = np.round(wh[0]/2)
                h = np.round(wh[1]/2)
            left = xc - w
            right = xc + w
            top = yc - h
            bot = yc + h

        elif reg is not None and wh is None:
            # reg = getself.__class__(reg)

            left = reg[0, 0]
            right = reg[0, 1]
            top = reg[1, 0]
            bot = reg[1, 1]

        else:
            raise ValueError(reg, 'reg cannot be None')

        # TODO check row/column ordering, and ndim check
        out = []
        for im in self:
            roi = im.image[top:bot, left:right, :]
            o = namedtuple('roi', ['roi',
                                   'left',
                                   'right',
                                   'top',
                                   'bot'])(self.__class__(roi),
                                           left,
                                           right,
                                           top,
                                           bot)
            out.append(o)

        return out

    def pixelswitch(self, mask, im2):
        """
        Pixel-wise image merge

        :param mask: image mask
        :type mask: numpy array
        :param im2: image 2
        :type im2: numpy array
        :return: out
        :rtype: Image instance

        - ``IM.pixelswitch(mask, im2)`` is an image where each pixel is
          selected from the corresponding pixel in  image or ``im2`` according
          to the corresponding pixel values in ``mask``.  If the element of
          ``mask`` is zero image is selected, otherwise ``im2`` is selected.

        - ``im2`` can contain a color descriptor which is one of: a scalar
          value corresponding to a greyscale, a 3-vector corresponding to a
          color value, or a string containing the name of a color which is
          found using COLORNAME.

        Example:

        .. runblock:: pycon

        .. note::

            - ``im1``, ``im2`` and ``mask`` must all have the same number of
              rows and columns (unless ``im1`` or ``im2`` are specifying a
              color)
            - If ``im1`` and ``im2`` are both greyscale then ``out`` is
              greyscale.
            - If either of ``im1`` or ``im2`` are color then ``out`` is color.
            - If either one image is double and one is integer then the integer
              image is first converted to a double image.
        """
        # TODO add possibility for alpha layering?

        # interpret im1, im2 to produce an Image for each
        im2 = self._checkimage(im2, mask)
        im2 = self.__class__(im2)

        out = []
        # 'for im in self:' was looping twice over a single-channel 2x2
        # greyscale image
        # below, converts self into a list of Image objects, each one of which
        # is im

        for im in [self.__class__(img) for img in self.listimages()]:
            # make consistent image.dtype
            if im.isfloat and im2.isint:
                im2 = im2.float()
            elif im.isint and im2.isfloat:
                im = im.float()

            # check consistent shape (height, width only):
            if np.any(im.shape[0:2] != im2.shape[0:2]):
                raise ValueError(im2, 'im1 and im2 must have the same shape')
            if np.any(im.shape[0:2] != mask.shape[0:2]):
                raise ValueError(mask, 'im1 and mask must have the same shape')
            # by extension, mask.shape == im2.shape

            # np.where returns im1 where mask == 0, and im2 where mask == 1
            # apply mask to each numchannel
            if self.iscolor:
                cmask = np.dstack([mask for i in range(im.numchannels)])
            else:
                # greyscale image
                cmask = mask
            o = np.array(np.where(cmask, im.image, im2.image))
            out.append(o)

        return self.__class__(out)

    def _checkimage(self, im, mask):
        """
        Check image and mask for pixelswitch

        :param im: image, possibly a color vector or identifier
        :type im: numpy array or scalar or string
        :param mask: mask
        :type mask: numpy array
        :return: out
        :rtype: Image instance

        - ``_checkimage(im, mask)`` is an image the same shape as ``mask``, and
          might be an image of all one color, depending on the value of ``im``
        """

        if isinstance(im, str):
            # image is a string color name
            col = color.colorname(im)
            if col is []:
                raise ValueError(im, 'unknown color')
            out = self.__class__(np.ones(mask.shape))
            out = out.colorise(col)

        elif argcheck.isscalar(im):
            # image is a  scalar, create a greyscale image the same size
            # as mask
            # TODO not certain if im.dtype works if im is scalar
            out = np.ones(mask.shape, dtype=im.dtype) * im

        elif im.ndim < 3 and max(im.shape) == 3:
            # or (3,) or (3,1)
            # image is a (1,3), create a color image the same size as mask
            out = self.__class__(np.ones(mask.shape, dtype=im.dtype))
            out = out.colorise(im)

        elif isinstance(im, self.__class__):
            # image class, check dimensions: (NOTE: im.size, not im.shape)
            # here, we are assuming mask is a 2D matrix
            if not np.any(im.size == mask.shape):
                raise ValueError(
                    im, 'input image size does not confirm with mask')
            out = im.image
        else:
            # actual image, check the dimensions
            if not np.any(im.shape == mask.shape):
                raise ValueError(
                    im, 'input image sizes (im or mask) do not conform')
        return out

class Histogram:

    def __init__(self, hs, cs, ns, xs):
        self.hs = hs # histogram
        self.cs = cs # cumulative histogram
        self.ns = ns # normalized cumulative histogram
        self.xs = xs  # x value
        # 'hist', 'h cdf normcdf x')

    def __str__(self):
        return f"histogram with {len(self.cs)} bins"

    def plot(self, type='histogram', block=False, **kwargs):

        if type == 'histogram':
            plot_histogram(self.xs.flatten(), self.hs.flatten(), block=block, 
            xlabel='pixel value', ylabel='number of pixels', **kwargs)
        elif type == 'cumulative':
            plot_histogram(self.xs.flatten(), self.cs.flatten(), block=block, 
            xlabel='pixel value', ylabel='cumulative number of pixels', **kwargs)
        elif type == 'normalized':
            plot_histogram(self.xs.flatten(), self.ns.flatten(), block=block, 
            xlabel='pixel value', ylabel='normalized cumulative number of pixels', **kwargs)

# --------------------------------------------------------------------------#
if __name__ == '__main__':

    print('ImageProcessingKernel.py')

    from machinevisiontoolbox import Image

    im = Image('penguins.png', grey=True)

    h = im.hist()
    print(h)
    h.plot(block=True)