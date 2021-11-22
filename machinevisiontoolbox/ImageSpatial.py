#!/usr/bin/env python

import numpy as np
import spatialmath.base.argcheck as argcheck
import cv2 as cv
import scipy as sp

from scipy import signal


class Kernel:
    """
    Image processing kernel operations on the Image class
    """

    @staticmethod
    def Gauss(sigma, h=None):
        r"""
        Gaussian kernel

        :param sigma: standard deviation of Gaussian kernel
        :type sigma: float
        :param h: half width of the kernel
        :type h: integer, optional
        :return k: Gaussian kernel
        :rtype: ndarray(2h+1, 2h+1)

        Returns a 2-dimensional Gaussian kernel of standard deviation ``sigma``

        .. math::

            K = \frac{1}{2\pi \sigma^2} e^{-(x^2 + y^2) / 2 \sigma^2}
        
        The kernel is centred within a square array with side length given by:

        - :math:`2 \mbox{ceil}(3 \sigma) + 1`, or
        - :math:`2 h + 1`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.Gauss(sigma=1, h=2)
            >>> K.shape
            >>> K
            >>> K = Kernel.Gauss(sigma=2)
            >>> K.shape

        .. note::

            - The volume under the Gaussian kernel is one.
            - If the kernel is strongly truncated, ie. it is non-zero at the 
              edges of the window then the volume will be less than one.

        :seealso: :meth:`.DGauss`
        """

        # make sure sigma, w are valid input
        if h is None:
            h = np.ceil(3 * sigma)

        wi = np.arange(-h, h + 1)
        x, y = np.meshgrid(wi, wi)

        m = 1.0 / (2.0 * np.pi * sigma ** 2) * \
            np.exp(-(x ** 2 + y ** 2) / 2.0 / sigma ** 2)
        # area under the curve should be 1, but the discrete case is only
        # an approximation
        return m / np.sum(m)

    @staticmethod
    def Laplace():
        r"""
        Laplacian kernel

        :return k: Laplacian kernel
        :rtype: ndarray(3,3)

        Returns the Laplacian kernel

        .. math::

            K = \begin{bmatrix}
                0 & 1 & 0 \\
                1 & -4 & 1 \\
                0 & 1 & 0
                \end{bmatrix}

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.Laplace()
            >>> K

        .. note::

            - This kernel has an isotropic response to image gradient.

        :seealso: :meth:`.LoG`
        """
        # fmt: off
        return np.array([[ 0,  1,  0],
                         [ 1, -4,  1],
                         [ 0,  1,  0]])
        # fmt: on

    @staticmethod
    def Sobel():
        r"""
        Sobel edge detector

        :return k: Sobel kernel
        :rtype: ndarray(3,3)

        - ``IM.ksobel()`` is the Sobel x-derivative kernel:

        .. math::

            K = \frac{1}{8} \begin{bmatrix}
                1 & 0 & -1 \\
                2 & 0 & -2 \\
                1 & 0 & -1
                \end{bmatrix}

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.Sobel()
            >>> K

        .. note::

            - This kernel is an effective vertical-edge detector
            - The y-derivative (horizontal-edge) kernel is ``K.T``

        :seealso: :meth:`.DGauss`
        """
        # fmt: off
        return np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]]) / 8.0
        # fmt: on

    @staticmethod
    def DoG(sigma1, sigma2=None, h=None):
        """
        Difference of Gaussians kernel

        :param sigma1: standard deviation of first Gaussian kernel
        :type sigma1: float
        :param sigma2: standard deviation of second Gaussian kernel
        :type sigma2: float
        :param h: half-width of Gaussian kernel
        :type h: int, optional
        :return k: difference of Gaussian kernel
        :rtype: ndarray(2h+1, 2h+1)

        Returns a 2-dimensional difference of Gaussian kernel
        equal to :math:`G(\sigma_1) - G(\sigma_2)` where :math:`\sigma_1 > \sigma_2`. 
        By default, :math:`\sigma_2 = 1.6 \sigma_1`. 
        
        The kernel is centred within a square array with side length given by:

        - :math:`2 \mbox{ceil}(3 \sigma) + 1`, or
        - :math:`2 h + 1`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.DoG(1)
            >>> K

        .. note::

            - This kernel is similar to the Laplacian of Gaussian and is often
              used as an efficient approximation.
            - This is a "Mexican hat" shaped kernel

        :seealso: :meth:`.LoG` :meth:`.Gauss`
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
        if h is None:
            h = np.ceil(3.0 * sigma1)

        m1 = Kernel.Gauss(sigma1, h)  # thin kernel
        m2 = Kernel.Gauss(sigma2, h)  # wide kernel

        return m2 - m1

    @staticmethod
    def LoG(sigma, h=None):
        r"""
        Laplacian of Gaussian kernel

        :param sigma1: standard deviation of first Gaussian kernel
        :type sigma: float
        :param h: half-width of kernel
        :type h: int, optional
        :return k: kernel
        :rtype: ndarray(2h+1, 2h+1)

        Returns a 2-dimensional Laplacian of Gaussian kernel with
        standard deviation ``sigma``

        .. math::

            K = \frac{1}{\pi \sigma^4} \left(\frac{x^2 + y^2}{2 \sigma^2} -1\right) e^{-(x^2 + y^2) / 2 \sigma^2}

        The kernel is centred within a square array with side length given by:

        - :math:`2 \mbox{ceil}(3 \sigma) + 1`, or
        - :math:`2 h + 1`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.LoG(1)
            >>> K

        .. note:: This is a "Mexican hat" shaped kernel

        :seealso: :meth:`.Laplace` :meth:`.DoG` :meth:`.Gauss`
        """

        if h is None:
            h = np.ceil(3.0 * sigma)
        wi = np.arange(-h, h + 1)
        x, y = np.meshgrid(wi, wi)

        return 1.0 / (np.pi * sigma ** 4.0) * \
            ((x ** 2 + y ** 2) / (2.0 * sigma ** 2) - 1) * \
            np.exp(-(x **2 + y** 2) / (2.0 * sigma ** 2))

    @staticmethod
    def DGauss(sigma, h=None):
        r"""
        Derivative of Gaussian kernel

        :param sigma1: standard deviation of first Gaussian kernel
        :type sigma1: float
        :param h: half-width of kernel
        :type h: int, optional
        :return k: kernel
        :rtype: ndarray(2h+1, 2h+1)

        Returns a 2-dimensional derivative of Gaussian
        kernel with standard deviation ``sigma``

        .. math::

            K = \frac{-x}{2\pi \sigma^2} e^{-(x^2 + y^2) / 2 \sigma^2}

        The kernel is centred within a square array with side length given by:

        - :math:`2 \mbox{ceil}(3 \sigma) + 1`, or
        - :math:`2 h + 1`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.DGauss(1)
            >>> K

        .. note::

            - This kernel is the horizontal derivative of the Gaussian, dG/dx.
            - The vertical derivative, dG/dy, is ``K.T``.
            - This kernel is an effective edge detector.

        :seealso: :meth:`.Gauss` :meth:`.Sobel`
        """
        if h is None:
            h = np.ceil(3.0 * sigma)

        wi = np.arange(-h, h + 1)
        x, y = np.meshgrid(wi, wi)

        return -x / sigma ** 2 / (2.0 * np.pi) * \
            np.exp(-(x ** 2 + y ** 2) / 2.0 / sigma ** 2)

    @staticmethod
    def Circle(radius, h=None):
        """
        Circular structuring element

        :param r: radius of circle structuring element, or 2-vector (see below)
        :type r: float, 2-tuple or 2-element vector of floats
        :param h: half-width of kernel
        :type h: int
        :return k: circular kernel
        :rtype: ndarray(2h+1, 2h+1)

        Returns a circular kernel of radius ``r`` pixels.  Values inside the
        circle are set to one.

        The kernel is centred within a square array with side length given 
        by :math:`2 h + 1`.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.Circle(2)
            >>> K
            >>> K = Kernel.Circle([2, 3])
            >>> K

        .. note::

            - If ``r`` is a 2-element vector the result is an annulus of ones,
              and the two numbers are interpretted as inner and outer radii.
        """

        # check valid input:
        if not argcheck.isscalar(radius):  # r.shape[1] > 1:
            radius = argcheck.getvector(radius)
            rmax = radius.max()
            rmin = radius.min()
        else:
            rmax = radius

        if h is not None:
            w = h * 2 + 1
        elif h is None:
            w = 2 * rmax + 1

        s = np.zeros((np.int(w), np.int(w)))
        c = np.floor(w / 2.0)

        if not argcheck.isscalar(radius):
            s = self.kcircle(rmax, w) - self.kcircle(rmin, w)
        else:
            x = np.arange(w) - c
            X, Y = np.meshgrid(x, x)
            ll = np.where(np.round((X ** 2 + Y ** 2 - radius ** 2) <= 0))
            s[ll] = 1
        return s

    @staticmethod
    def Box(h, normalize=True):
        """
        Square structuring element

        :param r: radius of circle structuring element, or 2-vector (see below)
        :type r: float, 2-tuple or 2-element vector of floats
        :param h: half-width of kernel
        :type h: int
        :return k: kernel
        :rtype: ndarray(2h+1, 2h+1)

        Returns a square kernel with unit volume.

        The kernel is centred within a square array with side length given 
        by :math:`2 h + 1`.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.Box(2)
            >>> K

        """

        # check valid input:

        wi = 2 * h + 1
        k = np.ones((wi, wi))
        if normalize:
            k /= np.sum(k)

        return k

class ImageSpatialMixin:
    
    def smooth(self, sigma, h=None, optmode='same', optboundary='fill'):
        """
        Smooth image

        :param sigma: standard deviation of the Gaussian kernel
        :type sigma: float
        :param h: half-width of the kernel
        :type h: float
        :param opt: convolution options np.convolve (see below)
        :type opt: string
        :return out: Image with smoothed image pixels
        :rtype: Image instance

        - ``IM.smooth(sigma)`` is the image after convolution with a Gaussian
          kernel of standard deviation ``sigma``

        - ``IM.smooth(sigma, h)`` as above with kernel half-width ``h``.

        - ``IM.smooth(sigma, opt)`` as above with options passed to np.convolve

        :options:

            - 'full'    returns the full 2-D convolution (default)
            - 'same'    returns OUT the same size as IM
            - 'valid'   returns  the valid pixels only, those where the kernel
              does not exceed the bounds of the image.

        Example:

        .. runblock:: pycon

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

        # make the smoothing kernel
        K = Kernel.Gauss(sigma, h)

        return self.convolve(K)

        # modeopt = {
        #     'full': 'full',
        #     'valid': 'valid',
        #     'same': 'same'
        # }
        # if optmode not in modeopt:
        #     raise ValueError(optmode, 'opt is not a valid option')

        # boundaryopt = {
        #     'fill': 'fill',
        #     'wrap': 'wrap',
        #     'reflect': 'symm'
        # }
        # if optboundary not in boundaryopt:
        #     raise ValueError(optboundary, 'opt is not a valid option')

        # is_int = False
        # if np.issubdtype(self.dtype, np.integer):
        #     is_int = True
        #     img = self.float()
        # else:
        #     img = self

        # # make the smoothing kernel
        # K = Kernel.Gauss(sigma, h)

        # if img.iscolor:
        #     # could replace this with a nested list comprehension

        #     ims = []
        #     for im in img:
        #         o = np.dstack([signal.convolve2d(np.squeeze(im.image[:, :, i]),
        #                                          K,
        #                                          mode=modeopt[optmode],
        #                                          boundary=boundaryopt[
        #                                              optboundary])
        #                       for i in range(im.numchannels)])
        #         ims.append(o)

        # elif not img.iscolor:
        #     ims = []
        #     for im in img:
        #         ims.append(signal.convolve2d(im.image,
        #                                      K,
        #                                      mode=modeopt[optmode],
        #                                      boundary=boundaryopt[
        #                                          optboundary]))

        # else:
        #     raise ValueError(self.iscolor, 'bad value for iscolor')

        # if is_int:
        #     return self.__class__(ims).int()
        # else:
        #     return self.__class__(ims)

    # def replicate(self, M=1):
    #     """
    #     Expand image

    #     :param M: number of times to replicate image
    #     :type M: integer
    #     :return out: Image expanded image
    #     :rtype out: Image instance

    #     - ``IM.replicate(M)`` is an expanded version of the image (H,W) where
    #       each pixel is replicated into a (M,M) tile. If ``im`` is (H,W) the
    #       result is ((M*H),(M*W)) numpy array.

    #     Example:

    #     .. runblock:: pycon

    #     """

    #     out = []
    #     for im in self:
    #         if im.ndims > 2:
    #             # dealing with multiplane image
    #             # TODO replace with a list comprehension
    #             ir2 = []
    #             for i in range(im.numchannels):
    #                 im1 = self.__class__(im.image[:, :, i])
    #                 ir2 = np.append(im1.replicate(M))
    #             return ir2

    #         nr = im.shape[0]
    #         nc = im.shape[1]

    #         # replicate columns
    #         ir = np.zeros((M * nr, nc), dtype=im.dtype)
    #         for r in range(M):
    #             ir[r:-1:M, :] = im.image

    #         # replicate rows
    #         ir2 = np.zeros((M * nr, M * nc), dtype=im.dtype)
    #         for c in range(M):
    #             ir2[:, c:-1:M] = ir
    #         out.append(ir2)

    #     return self.__class__(out)

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

        .. runblock:: pycon

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

        .. runblock:: pycon

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

        .. runblock:: pycon

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

        .. runblock:: pycon

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

        .. runblock:: pycon

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

        .. runblock:: pycon

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

        .. runblock:: pycon

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

        .. runblock:: pycon

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


        if isinstance(se, int):
            s = 2 * se + 1
            se = np.full((s, s), True)

        out = sp.ndimage.generic_filter(self.A,
                                            func,
                                            footprint=se,
                                            mode=edgeopt[opt])
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

        .. runblock:: pycon

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
        :seealso: `cv2.matchTemplate<https://docs.opencv.org/master/df/dfb/group__imgproc__object.html#ga586ebfb0a7fb604b35a23d85391329be>`_
        """

        # check inputs
        if ((T.shape[0] % 2) == 0) or ((T.shape[1] % 2) == 0):
            raise ValueError(T, 'template T must have odd dimensions')

        if metric is None:
            metric = self.zncc
        # if not callable(metric):
        #     raise TypeError(metric, 'metric not a callable function')

        metricdict = {
            'ssd': cv.TM_SQDIFF,
            'zssd': cv.TM_SQDIFF,
            'ncc': cv.TM_CCOEFF_NORMED,
            'zncc': cv.TM_CCOEFF_NORMED
        }

        try:
            method = metricdict[metric]
        except KeyError:
            raise ValueError('bad metric specified')

        if metric[0] == 'z':
            # remove offset from template
            T_im = T.A
            T_im -= np.mean(T_im)

        im = self.A
        if metric[0] == 'z':
            # remove offset from image
            im = im - np.mean(im)
            
        out = cv.matchTemplate(im, T_im, method=method)

        return self.__class__(out)

    def convolve(self, K, mode='same', border='reflect', value=0):
        """
        Image convolution

        :param K: kernel
        :type K: numpy array
        :param mode: option for convolution
        :type mode: str
        :param border: option for boundary handling
        :type border: str
        :return C: Image convolved image
        :rtype C: Image instance

        - ``IM.convolve(K)`` is the convolution of image with the kernel ``K``

        - ``IM.convolve(K, mode)`` as above but specifies the convolution
          mode. See scipy.signal.convolve2d for details, mode options below

        - ``IM.convolve(K, boundary)`` as above but specifies the boundary
          handling options

        :options:

            - 'same'    output image is same size as input image (default)
            - 'full'    output image is larger than the input image
            - 'valid'   output image is smaller than the input image, and
              contains only valid pixels TODO

        Example:

        .. runblock:: pycon

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

        if isinstance(K, self.__class__):
            K = K.A

        K = argcheck.getmatrix(K, shape=[None,None], dtype='float32')

        # OpenCV does correlation, not convolution, so we flip the kernel
        # to compensate.  Flip horizontally and vertically.
        K = np.flip(K)
        kh, kw = K.shape
        kh //= 2
        kw //= 2

        # TODO check images are of the same type

        # TODO check opt is valid string based on conv2 options
        modeopt = ['valid', 'same', 'full']

        if mode not in modeopt:
            raise ValueError(mode, 'opt is not a valid option')

        borderopt = {
            'replicate': cv.BORDER_REPLICATE,
            'zero': cv.BORDER_CONSTANT,
            'pad': cv.BORDER_CONSTANT,
            'wrap': cv.BORDER_WRAP,
            'reflect': cv.BORDER_REFLECT
        }
        if border not in borderopt:
            raise ValueError(border, 'opt is not a valid option')

        # TODO options are wrong, only borderType

        img = self.A
        if border == "pad" and value != 0:
            img = cv.copyMakeBorder(a, kv, kv, kh, kh, cv.BORDER_CONSTANT, value=value)
        elif mode == "full":
            img = cv.copyMakeBorder(a, kv, kv, kh, kh, boundaryopt[boundary], value=value)

        out = cv.filter2D(img, ddepth=-1, kernel=K, 
            borderType=borderopt[border])

        if mode == "valid":
            if out.ndim == 2:
                out = out[kh:-kh, kw:-kw]
            else:
                out = out[kh:-kh, kw:-kw, :]
        return self.__class__(out, colororder=self.colororder)

    # def sobel(self, kernel=None):
    #     if kernel is None:
    #         kernel = Kernel.Sobel()

    #     Iu = self.convolve(kernel)
    #     Iv = self.convolve(kernel.T)
    #     return Iu, Iv

    def gradients(self, kernel=None):
        if kernel is None:
            kernel = Kernel.Sobel()

        Iu = self.convolve(kernel)
        Iv = self.convolve(kernel.T)
        return Iu, Iv

    def zerocross(self):
        min = cv.morphologyEx(self.image, cv.MORPH_ERODE, np.ones((3,3)))
        max = cv.morphologyEx(self.image, cv.MORPH_DILATE, np.ones((3,3)))
        zeroCross = np.logical_or(
            np.logical_and(min < 0, self.image > 0), 
            np.logical_and(max > 0, self.image < 0)
        )
        return self.__class__(zeroCross)


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

        .. runblock:: pycon

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
        dg = Kernel.DGauss(sigma)

        sigma = 0.3333

        Ix = self.convolve(dg)
        Iy = self.convolve(np.transpose(dg))

        # Ix, Iy must be 16-bit input image
        Ix = np.array(Ix.A, dtype=np.int16)
        Iy = np.array(Iy.A, dtype=np.int16)

        v = np.mean(self.A)
        # apply automatic Canny edge detection using the computed median
        lower = (max(0, (1.0 - sigma) * v))
        upper = (min(1, (1.0 + sigma) * v))

        out = cv.Canny(self.to_int(), lower, upper, L2gradient=False)

        return self.__class__(out)


# --------------------------------------------------------------------------#
if __name__ == '__main__':

    print('ImageProcessingKernel.py')
    from machinevisiontoolbox import Image

    image = Image('monalisa.png', grey=True)
    blur = image.smooth(3, 5)
    blur.disp(block=True)