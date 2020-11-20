#!/usr/bin/env python

import numpy as np
import spatialmath.base.argcheck as argcheck
import cv2 as cv
import scipy as sp

from scipy import signal


class ImageProcessingKernelMixin:
    """
    Image processing kernel operations on the Image class
    """

    def kgauss(self, sigma, hw=None):
        """
        Gaussian kernel

        :param sigma: standard deviation of Gaussian kernel
        :type sigma: float
        :param hw: width of the kernel
        :type hw: integer
        :return k: kernel
        :rtype: numpy array (N,H)

        - ``IM.kgauss(sigma)`` is a 2-dimensional Gaussian kernel of standard
          deviation ``sigma``, and centred within the matrix ``k`` whose
          half-width is ``hw=2*sigma`` and ``w=2*hw+1``.

        - ``IM.kgauss(sigma, hw)`` as above but the half-width ``hw`` is
          specified.

        Example:

        .. autorun:: pycon

        .. note::

            - The volume under the Gaussian kernel is one.
        """

        # make sure sigma, w are valid input
        if hw is None:
            hw = np.ceil(3 * sigma)

        wi = np.arange(-hw, hw + 1)
        x, y = np.meshgrid(wi, wi)

        m = 1.0 / (2.0 * np.pi * sigma ** 2) * \
            np.exp(-(np.power(x, 2) + np.power(y, 2)) / 2.0 / sigma ** 2)
        # area under the curve should be 1, but the discrete case is only
        # an approximation
        return m / np.sum(m)

    def klaplace(self):
        r"""
        Laplacian kernel

        :return k: kernel
        :rtype: numpy array (3,3)

        - ``IM.klaplace()`` is the Laplacian kernel:

        .. math::

            K = \begin{bmatrix}
                0 & 1 & 0 \\
                1 & -4 & 1 \\
                0 & 1 & 0
                \end{bmatrix}

        Example:

        .. autorun:: pycon

        .. note::

            - This kernel has an isotropic response to image gradient.
        """
        return np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]])

    def ksobel(self):
        r"""
        Sobel edge detector

        :return k: kernel
        :rtype: numpy array (3,3)

        - ``IM.ksobel()`` is the Sobel x-derivative kernel:

        .. math::

            K = \frac{1}{8} \begin{bmatrix}
                1 & 0 & -1 \\
                2 & 0 & -2 \\
                1 & 0 & -1
                \end{bmatrix}

        .. note::

            - This kernel is an effective vertical-edge detector
            - The y-derivative (horizontal-edge) kernel is K'
        """
        return np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]]) / 8.0

    def kdog(self, sigma1, sigma2=None, hw=None):
        """
        Difference of Gaussians kernel

        :param sigma1: standard deviation of first Gaussian kernel
        :type sigma1: float
        :param sigma2: standard deviation of second Gaussian kernel
        :type sigma2: float
        :param hw: half-width of Gaussian kernel
        :type hw: integer
        :return k: kernel
        :rtype: numpy array

        - ``IM.kdog(sigma1)`` is a 2-dimensional difference of Gaussian kernel
          equal to ``kgauss(sigma1) - kgauss(sigma2)``, where ``sigma1`` >
          ``sigma2. By default, ``sigma2 = 1.6 * sigma1``.  The kernel is
          centred within the matrix ``k`` whose half-width ``hw = 3xsigma1``
          and full width of the kernel is ``2xhw+1``.

        - ``IM.kdog(sigma1, sigma2)`` as above but sigma2 is specified
          directly.

        - ``IM.kdog(sigma1, sigma2, hw)`` as above but the kernel half-width is
          specified

        Example:

        .. autorun:: pycon

        .. note::

            - This kernel is similar to the Laplacian of Gaussian and is often
              used as an efficient approximation.
        """

        # sigma1 > sigma2
        if sigma2 is None:
            sigma2 = 1.6 * sigma1
        else:
            if sigma2 > sigma1:
                t = sigma1
                sigma1 = sigma2
                sigma2 = t

        # thus, sigma2 > sigma1
        if hw is None:
            hw = np.ceil(3.0 * sigma1)

        m1 = self.kgauss(sigma1, hw)  # thin kernel
        m2 = self.kgauss(sigma2, hw)  # wide kernel

        return m2 - m1

    def klog(self, sigma, hw=None):
        """
        Laplacian of Gaussian kernel

        :param sigma1: standard deviation of first Gaussian kernel
        :type sigma1: float
        :param hw: half-width of kernel
        :type hw: integer
        :return k: kernel
        :rtype: numpy array (2 * 3 * sigma + 1, 2 * 3 * sigma + 1)

        - ``IM.klog(sigma)`` is a 2-dimensional Laplacian of Gaussian kernel of
          width (standard deviation) sigma and centred within the matrix ``k``
          whose half-width is ``hw=3xsigma``, and ``w=2xhw+1``.

        - ``IM.klog(sigma, hw)`` as above but the half-width ``w`` is
          specified.

        Example:

        .. autorun:: pycon

        """

        if hw is None:
            hw = np.ceil(3.0 * sigma)
        wi = np.arange(-hw, hw + 1)
        x, y = np.meshgrid(wi, wi)

        return 1.0 / (np.pi * sigma ** 4.0) * \
            ((np.power(x, 2) + np.power(y, 2)) / (2.0 * sigma ** 2) - 1) * \
            np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2.0 * sigma ** 2))

    def kdgauss(self, sigma, hw=None):
        """
        Derivative of Gaussian kernel

        :param sigma1: standard deviation of first Gaussian kernel
        :type sigma1: float
        :param hw: half-width of kernel
        :type hw: integer
        :return k: kernel
        :rtype: numpy array (2 * 3 * sigma + 1, 2 * 3 * sigma + 1)

        - ``IM.kdgauss(sigma)`` is a 2-dimensional derivative of Gaussian
          kernel ``(w,w)`` of width (standard deviation) sigma and centred
          within the matrix ``k`` whose half-width ``hw = 3xsigma`` and
          ``w=2xhw+1``.

        - ``IM.kdgauss(sigma, hw)`` as above but the half-width is explictly
          specified.

        Example:

        .. autorun:: pycon

        .. note::

            - This kernel is the horizontal derivative of the Gaussian, dG/dx.
            - The vertical derivative, dG/dy, is k'.
            - This kernel is an effective edge detector.
        """
        if hw is None:
            hw = np.ceil(3.0 * sigma)

        wi = np.arange(-hw, hw + 1)
        x, y = np.meshgrid(wi, wi)

        return -x / sigma ** 2 / (2.0 * np.pi) * \
            np.exp(-np.power(x, 2) + np.power(y, 2) / 2.0 / sigma ** 2)

    def kcircle(self, r, hw=None):
        """
        Circular structuring element

        :param r: radius of circle structuring element, or 2-vector (see below)
        :type r: float, 2-tuple or 2-element vector of floats
        :param hw: half-width of kernel
        :type hw: integer
        :return k: kernel
        :rtype: numpy array (2 * 3 * sigma + 1, 2 * 3 * sigma + 1)

        - ``IM.kcircle(r)`` is a square matrix ``(w,w)`` where ``w=2r+1`` of
          zeros with a maximal centred circular region of radius ``r`` pixels
          set to one.

        - ``IM.kcircle(r,w)`` as above but the dimension of the kernel is
          explicitly specified.

        Example:

        .. autorun:: pycon

        .. note::

            - If ``r`` is a 2-element vector the result is an annulus of ones,
              and the two numbers are interpretted as inner and outer radii.
        """

        # check valid input:
        if not argcheck.isscalar(r):  # r.shape[1] > 1:
            r = argcheck.getvector(r)
            rmax = r.max()
            rmin = r.min()
        else:
            rmax = r

        if hw is not None:
            w = hw * 2 + 1
        elif hw is None:
            w = 2 * rmax + 1

        s = np.zeros((np.int(w), np.int(w)))
        c = np.floor(w / 2.0)

        if not argcheck.isscalar(r):
            s = self.kcircle(rmax, w) - self.kcircle(rmin, w)
        else:
            x, y = self.meshgrid(s)
            x = x - c
            y = y - c
            ll = np.where(np.round((x ** 2 + y ** 2 - r ** 2) <= 0))
            s[ll] = 1
        return s

    def smooth(self, sigma, hw=None, optmode='same', optboundary='fill'):
        """
        Smooth image

        :param sigma: standard deviation of the Gaussian kernel
        :type sigma: float
        :param hw: half-width of the kernel
        :type hw: float
        :param opt: convolution options np.convolve (see below)
        :type opt: string
        :return out: Image with smoothed image pixels
        :rtype: Image instance

        - ``IM.smooth(sigma)`` is the image after convolution with a Gaussian
          kernel of standard deviation ``sigma``

        - ``IM.smooth(sigma, hw)`` as above with kernel half-width ``hw``.

        - ``IM.smooth(sigma, opt)`` as above with options passed to np.convolve

        :options:

            - 'full'    returns the full 2-D convolution (default)
            - 'same'    returns OUT the same size as IM
            - 'valid'   returns  the valid pixels only, those where the kernel
              does not exceed the bounds of the image.

        Example:

        .. autorun:: pycon

        .. note::

            - By default (option 'full') the returned image is larger than the
              passed image.
            - Smooths all planes of the input image.
            - The Gaussian kernel has a unit volume.
            - If input image is integer it is converted to float, convolved,
              then converted back to integer.
        """

        if not argcheck.isscalar(sigma):
            raise ValueError(sigma, 'sigma must be a scalar')

        modeopt = {
            'full': 'full',
            'valid': 'valid',
            'same': 'same'
        }
        if optmode not in modeopt:
            raise ValueError(optmode, 'opt is not a valid option')

        boundaryopt = {
            'fill': 'fill',
            'wrap': 'wrap',
            'reflect': 'symm'
        }
        if optboundary not in boundaryopt:
            raise ValueError(optboundary, 'opt is not a valid option')

        is_int = False
        if np.issubdtype(self.dtype, np.integer):
            is_int = True
            img = self.float()
        else:
            img = self

        # make the smoothing kernel
        K = self.kgauss(sigma, hw)

        if img.iscolor:
            # could replace this with a nested list comprehension

            ims = []
            for im in img:
                o = np.dstack([signal.convolve2d(np.squeeze(im.image[:, :, i]),
                                                 K,
                                                 mode=modeopt[optmode],
                                                 boundary=boundaryopt[
                                                     optboundary])
                              for i in range(im.numchannels)])
                ims.append(o)

        elif not img.iscolor:
            ims = []
            for im in img:
                ims.append(signal.convolve2d(im.image,
                                             K,
                                             mode=modeopt[optmode],
                                             boundary=boundaryopt[
                                                 optboundary]))

        else:
            raise ValueError(self.iscolor, 'bad value for iscolor')

        if is_int:
            return self.__class__(ims).int()
        else:
            return self.__class__(ims)

    def sad(self, im2):
        """
        Sum of absolute differences

        :param im2: image 2
        :type im2: numpy array
        :return out: sad
        :rtype out: scalar

        - ``IM.sad(im2)`` is the sum of absolute differences between the two
          equally sized image patches of image and ``im2``. The result is a
          scalar that indicates image similarity, a value of 0 indicates
          identical pixel patterns and is increasingly positive as image
          dissimilarity increases.

        Example:

        .. autorun:: pycon

        """

        if not np.all(self.shape == im2.shape):
            raise ValueError(im2, 'im2 shape is not equal to self')

        # out = []
        # for im in self:
            # m = np.abs(im.image - im2.image)
            # out.append(np.sum(m))
        m = np.abs(self.image - im2.image)
        out = np.sum(m)
        return out

    def ssd(self, im2):
        """
        Sum of squared differences

        :param im2: image 2
        :type im2: numpy array
        :return out: ssd
        :rtype out: scalar

        - ``IM.ssd(im2)`` is the sum of squared differences between the two
          equally sized image patches image and ``im2``.  The result M is a
          scalar that indicates image similarity, a value of 0 indicates
          identical pixel patterns and is increasingly positive as image
          dissimilarity increases.

        Example:

        .. autorun:: pycon

        """

        if not np.all(self.shape == im2.shape):
            raise ValueError(im2, 'im2 shape is not equal to im1')
        m = np.power((self.image - im2.image), 2)
        return np.sum(m)

    def ncc(self, im2):
        """
        Normalised cross correlation

        :param im2: image 2
        :type im2: numpy array
        :return out: ncc
        :rtype out: scalar

        - ``IM.ncc(im2)`` is the normalized cross-correlation between the two
          equally sized image patches image and ``im2``. The result is a scalar
          in the interval -1 (non match) to 1 (perfect match) that indicates
          similarity.

        .. note::

            - A value of 1 indicates identical pixel patterns.
            - The ``ncc`` similarity measure is invariant to scale changes in
              image intensity.

        Example:

        .. autorun:: pycon

        """
        if not np.all(self.shape == im2.shape):
            raise ValueError(im2, 'im2 shape is not equal to im1')

        denom = np.sqrt(np.sum(self.image ** 2) * np.sum(im2.image ** 2))

        if denom < 1e-10:
            return 0
        else:
            return np.sum(self.image * im2.image) / denom

    def zsad(self, im2):
        """
        Zero-mean sum of absolute differences

        :param im2: image 2
        :type im2: numpy array
        :return out: zsad
        :rtype out: scalar

        - ``IM.zsad(im2)`` is the zero-mean sum of absolute differences between
          the two equally sized image patches image and ``im2``. The result is
          a scalar that indicates image similarity, a value of 0 indicates
          identical pixel patterns and is increasingly positive as image
          dissimilarity increases.

        Example:

        .. autorun:: pycon

        .. note::

            - The ``zsad`` similarity measure is invariant to changes in image
            brightness offset.
        """
        if not np.all(self.shape == im2.shape):
            raise ValueError(im2, 'im2 shape is not equal to im1')

        image = self.image - np.mean(self.image)
        image2 = im2.image - np.mean(im2.image)
        m = np.abs(image - image2)
        return np.sum(m)

    def zssd(self, im2):
        """
        Zero-mean sum of squared differences

        :param im2: image 2
        :type im2: numpy array
        :return out: zssd
        :rtype out: scalar

        - ``IM.zssd(im1, im2)`` is the zero-mean sum of squared differences
          between the two equally sized image patches image and ``im2``.  The
          result is a scalar that indicates image similarity, a value of 0
          indicates identical pixel patterns and is increasingly positive as
          image dissimilarity increases.

        Example:

        .. autorun:: pycon

        .. note::

            - The ``zssd`` similarity measure is invariant to changes in image
              brightness offset.
        """
        if not np.all(self.shape == im2.shape):
            raise ValueError(im2, 'im2 shape is not equal to im1')

        image = self.image - np.mean(self.image)
        image2 = im2.image - np.mean(im2.image)
        m = np.power(image - image2, 2)
        return np.sum(m)

    def zncc(self, im2):
        """
        Zero-mean normalized cross correlation

        :param im2: image 2 :type im2: numpy array :return out: zncc :rtype
        out: scalar

        - ``IM.zncc(im2)`` is the zero-mean normalized cross-correlation
          between the two equally sized image patches image and ``im2``.  The
          result is a scalar in the interval -1 to 1 that indicates similarity.
          A value of 1 indicates identical pixel patterns.

        Example:

        .. autorun:: pycon

        .. note::

            - The ``zncc`` similarity measure is invariant to affine changes
              in image intensity (brightness offset and scale).

        """
        if not np.all(self.shape == im2.shape):
            raise ValueError(im2, 'im2 shape is not equal to im1')

        image = self.image - np.mean(self.image)
        image2 = im2.image - np.mean(im2.image)
        denom = np.sqrt(np.sum(np.power(image, 2) *
                               np.sum(np.power(image2, 2))))

        if denom < 1e-10:
            return 0
        else:
            return np.sum(image * image2) / denom

    def pyramid(self, sigma=1, N=None):
        """
        Pyramidal image decomposition

        :param sigma: standard deviation of Gaussian kernel
        :type sigma: float
        :param N: number of pyramid levels to be computed
        :type N: int
        :return pyrimlist: list of Images for each pyramid level computed
        :rtype pyrimlist: list

        - ``IM.pyramid()`` is a pyramid decomposition of image using Gaussian
          smoothing with standard deviation of 1. The return is a list array of
          images each one having dimensions half that of the previous image.
          The pyramid is computed down to a non-halvable image size.

        - ``IM.pyramid(sigma)`` as above but the Gaussian standard deviation is
          ``sigma``.

        - ``IM.pyramid(sigma, N)`` as above but only ``N`` levels of the
          pyramid are computed.

        Example:

        .. autorun:: pycon

        .. note::

            - Converts a color image to greyscale.
            - Works for greyscale images only.
        """

        # check inputs, greyscale only
        im = self.mono()

        if not argcheck.isscalar(sigma):
            raise ValueError(sigma, 'sigma must be a scalar')

        if N is None:
            N = max(im.shape)
        else:
            if (not argcheck.isscalar(N)) and (N >= 0) and \
               (N <= max(im.shape)):
                raise ValueError(N, 'N must be a scalar and \
                    0 <= N <= max(im.shape)')

        # TODO options to accept different border types,
        # note that the Matlab implementation is hard-coded to 'same'

        # return cv.buildPyramid(im, N, borderType=cv.BORDER_REPLICATE)
        # Python version does not seem to be implemented

        # list comprehension approach
        # TODO pyr = [cv.pyrdown(inputs(i)) for i in range(N) if conditional]

        impyr = im.image
        pyr = [impyr]
        for i in range(N):
            if impyr.shape[0] == 1 or impyr.shape[1] == 1:
                break
            impyr = cv.pyrDown(impyr, borderType=cv.BORDER_REPLICATE)
            pyr.append(impyr)

        # output list of Image objects
        pyrimlist = [self.__class__(p) for p in pyr]
        return pyrimlist

    def window(self, se, func, opt='border', **kwargs):
        """
        Generalized spatial operator

        :param se: structuring element
        :type se: numpy array
        :param func: function to operate
        :type funct: reference to a callable function
        :param opt: border option
        :type opt: string
        :return out: Image after function has operated on every pixel by func
        :rtype out: Image instance

        - ``IM.window(se, func)`` is an image where each pixel is the result of
          applying the function ``func`` to a neighbourhood centred on the
          corresponding pixel in image. The neighbourhood is defined by the
          size of the structuring element ``se`` which should have odd side
          lengths. The elements in the neighbourhood corresponding to non-zero
          elements in ``se`` are packed into a vector (in column order from top
          left) and passed to the specified callable function ``func``. The
          return value of ``func`` becomes the corresponding pixel value.

        - ``IM.window(se, func, opt)`` as above but performance of edge pixels
          can be controlled.

        :options:

            - 'replicate'     the border value is replicated (default)
            - 'none'          pixels beyond the border are not included in the
              window
            - 'trim'          output is not computed for pixels where the
              structuring element crosses the image border, hence output image
              has reduced dimensions TODO

        Example:

        .. autorun:: pycon

        .. note::

            - The structuring element should have an odd side length.
            - Is slow since the function ``func`` must be invoked once for
              every output pixel.
            - The input can be logical, uint8, uint16, float or double, the
              output is always double
        """
        # replace window's mex function with scipy's ndimage.generic_filter

        # border options:
        edgeopt = {
            'border': 'nearest',
            'none': 'constant',
            'wrap': 'wrap'
        }
        if opt not in edgeopt:
            raise ValueError(opt, 'opt is not a valid edge option')

        if not callable(func):
            raise TypeError(func, 'func not callable')

        out = []
        for im in self:
            out.append(sp.ndimage.generic_filter(im.image,
                                                 func,
                                                 footprint=se,
                                                 mode=edgeopt[opt]))
        return self.__class__(out)

    def similarity(self, T, metric=None):
        """
        Locate template in image

        :param T: template image
        :type T: numpy array
        :param metric: similarity metric function
        :type metric: callable function reference
        :return S: Image similarity image
        :rtype S: Image instance

        - ``IM.similarity(T)`` is an image where each pixel is the ``zncc``
          similarity of the template ``T`` (M,M) to the (M,M) neighbourhood
          surrounding the corresonding input pixel in image.  ``S`` is same
          size as image.

        - ``IM.similarity(T, metric)`` as above but the similarity metric is
          specified by the function ``metric`` which can be any of @sad, @ssd,
          @ncc, @zsad, @zssd.

        Example:

        .. autorun:: pycon

        .. note::

            - For NCC and ZNCC the maximum in S corresponds to the most likely
              template location.  For SAD, SSD, ZSAD and ZSSD the minimum value
              corresponds to the most likely location.
            - Similarity is not computed for those pixels where the template
              crosses the image boundary, and these output pixels are set
              to NaN.
            - The ZNCC function is a MEX file and therefore the fastest
            - User provided similarity metrics can be used, the function
              accepts two regions and returns a scalar similarity score.

        :references:

            - Robotics, Vision & Control, Section 12.4, P. Corke,
              Springer 2011.
        """

        # check inputs
        if ((T.shape[0] % 2) == 0) or ((T.shape[1] % 2) == 0):
            raise ValueError(T, 'template T must have odd dimensions')

        if metric is None:
            metric = self.zncc
        if not callable(metric):
            raise TypeError(metric, 'metric not a callable function')

        # to use metric, T must be an image class
        T = self.__class__(T)

        hc = np.floor(T.shape[0] / 2)
        hr = np.floor(T.shape[1] / 2)

        out = []
        for im in self:
            S = np.empty(im.shape)

            # TODO can probably replace these for loops with comprehensions
            for c in range(start=hc + 1, stop=im.shape[0] - hc):
                for r in range(start=hr + 1, stop=im.shape[1] - hr):
                    S[r, c] = T.metric(im.image[r-hr:r+hr, c-hc:c+hc])
            out.append(S)

        return self.__class__(out)

    def convolve(self, K, optmode='same', optboundary='wrap'):
        """
        Image convolution

        :param K: kernel
        :type K: numpy array
        :param optmode: option for convolution
        :type optmode: string
        :param optboundary: option for boundary handling
        :type optboundary: string
        :return C: Image convolved image
        :rtype C: Image instance

        - ``IM.convolve(K)`` is the convolution of image with the kernel ``K``

        - ``IM.convolve(K, optmode)`` as above but specifies the convolution
          mode. See scipy.signal.convolve2d for details, mode options below

        - ``IM.convolve(K, optboundary)`` as above but specifies the boundary
          handling options

        :options:

            - 'same'    output image is same size as input image (default)
            - 'full'    output image is larger than the input image
            - 'valid'   output image is smaller than the input image, and
              contains only valid pixels TODO

        Example:

        .. autorun:: pycon

        .. note::

            - If the image is color (has multiple planes) the kernel is
              applied to each plane, resulting in an output image with the same
              number of planes.
            - If the kernel has multiple planes, the image is convolved with
              each plane of the kernel, resulting in an output image with the
              same number of planes.
            - This function is a convenience wrapper for the MATLAB function
              CONV2.
            - Works for double, uint8 or uint16 images.  Image and kernel must
              be of the same type and the result is of the same type.
            - This function replaces iconv().

        :references:

            - Robotics, Vision & Control, Section 12.4, P. Corke,
              Springer 2011.
        """

        # TODO check images are of the same type

        # TODO check opt is valid string based on conv2 options
        modeopt = {
            'full': 'full',
            'valid': 'valid',
            'same': 'same'
        }
        if optmode not in modeopt:
            raise ValueError(optmode, 'opt is not a valid option')

        boundaryopt = {
            'fill': 'fill',
            'wrap': 'wrap',
            'reflect': 'symm'
        }
        if optboundary not in boundaryopt:
            raise ValueError(optboundary, 'opt is not a valid option')

        out = []
        for im in self:
            if im.iscolor and K.ndim == 2:
                # image has multiple planes:
                C = np.dstack([signal.convolve2d(im.image[:, :, i],
                                                 K,
                                                 mode=modeopt[optmode],
                                                 boundary=boundaryopt[
                                                     optboundary])
                               for i in range(im.nchannels)])

            elif not im.iscolor and K.ndim == 2:
                # simple case, convolve image with kernel, both are 2D
                C = signal.convolve2d(im.image,
                                      K,
                                      mode=modeopt[optmode],
                                      boundary=boundaryopt[optboundary])

            elif not im.iscolor and K.ndim == 3:
                # kernel has multiple planes:
                C = np.dstack([signal.convolve2d(im.image,
                                                 K.image[:, :, i],
                                                 mode=modeopt[optmode],
                                                 boundary=boundaryopt[
                                                     optboundary])
                               for i in range(K.shape[2])])
            else:
                raise ValueError(
                    im, 'image and kernel cannot both have muliple planes')
            out.append(C)

        return self.__class__(out)

    def canny(self, sigma=1, th0=None, th1=None):
        """
        Canny edge detection

        :param sigma: standard deviation for Gaussian kernel smoothing
        :type sigma: float
        :param th0: lower threshold
        :type th0: float
        :param th1: upper threshold
        :type th1: float
        :return E: Image with edge image
        :rtype E: Image instance

        - ``IM.canny()`` is an edge image obtained using the Canny edge
          detector algorithm.  Hysteresis filtering is applied to the gradient
          image: edge pixels > ``th1`` are connected to adjacent pixels >
          ``th0``, those below ``th0`` are set to zero.

        - ``IM.canny(sigma, th0, th1)`` as above, but the standard deviation of
          the Gaussian smoothing, ``sigma``, lower and upper thresholds
          ``th0``, ``th1`` can be specified

        Example:

        .. autorun:: pycon

        .. note::

            - Produces a zero image with single pixel wide edges having
              non-zero values.
            - Larger values correspond to stronger edges.
            - If th1 is zero then no hysteresis filtering is performed.
            - A color image is automatically converted to greyscale first.

        :references:

            - "A Computational Approach To Edge Detection", J. Canny,
              IEEE Trans. Pattern Analysis and Machine Intelligence,
              8(6):679–698, 1986.

        """

        # convert to greyscale:
        img = self.mono()

        # set defaults (eg thresholds, eg one as a function of the other)
        if th0 is None:
            if np.issubdtype(th0, np.float):
                th0 = 0.1
            else:
                # isint
                th0 = np.round(0.1 * np.iinfo(img.dtype).max)
        if th1 is None:
            th1 = 1.5 * th0

        # compute gradients Ix, Iy using guassian kernel
        dg = self.kdgauss(sigma)

        out = []
        for im in img:

            Ix = np.abs(im.convolve(dg, 'same'))
            Iy = np.abs(im.convolve(np.transpose(dg), 'same'))

            # Ix, Iy must be 16-bit input image
            Ix = np.array(Ix, dtype=np.int16)
            Iy = np.array(Iy, dtype=np.int16)

            out.append((cv.Canny(Ix, Iy, th0, th1, L2gradient=True)))

        return self.__class__(out)


# --------------------------------------------------------------------------#
if __name__ == '__main__':

    print('ImageProcessingKernel.py')
