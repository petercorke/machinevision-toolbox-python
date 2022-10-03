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
        :return: Gaussian kernel
        :rtype: ndarray(2h+1, 2h+1)

        Return the 2-dimensional Gaussian kernel of standard deviation ``sigma``

        .. math::

            \mathbf{K} = \frac{1}{2\pi \sigma^2} e^{-(u^2 + v^2) / 2 \sigma^2}
        
        The kernel is centred within a square array with side length given by:

        - :math:`2 \mbox{ceil}(3 \sigma) + 1`, or
        - :math:`2 \mathtt{h} + 1`

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

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.1.1, P. Corke, Springer 2023.

        :seealso: :meth:`DGauss`
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

        :return: Laplacian kernel
        :rtype: ndarray(3,3)

        Return the Laplacian kernel

        .. math::

            \mathbf{K} = \begin{bmatrix}
                0 & 1 & 0 \\
                1 & -4 & 1 \\
                0 & 1 & 0
                \end{bmatrix}

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> Kernel.Laplace()

        .. note::

            - This kernel has an isotropic response to image gradient.

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.1.3, P. Corke, Springer 2023.

        :seealso: :meth:`LoG` :meth:`zerocross`
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

        :return: Sobel kernel
        :rtype: ndarray(3,3)

        Return the Sobel kernel for horizontal gradient

        .. math::

            \mathbf{K} = \frac{1}{8} \begin{bmatrix}
                1 & 0 & -1 \\
                2 & 0 & -2 \\
                1 & 0 & -1
                \end{bmatrix}

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> Kernel.Sobel()

        .. note::

            - This kernel is an effective vertical-edge detector
            - The y-derivative (horizontal-edge) kernel is ``K.T``

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.1.3, P. Corke, Springer 2023.

        :seealso: :meth:`DGauss`
        """
        # fmt: off
        return np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]]) / 8.0
        # fmt: on

    @staticmethod
    def DoG(sigma1, sigma2=None, h=None):
        r"""
        Difference of Gaussians kernel

        :param sigma1: standard deviation of first Gaussian kernel
        :type sigma1: float
        :param sigma2: standard deviation of second Gaussian kernel
        :type sigma2: float, optional
        :param h: half-width of Gaussian kernel
        :type h: int, optional
        :return: difference of Gaussian kernel
        :rtype: ndarray(2h+1, 2h+1)

        Return the 2-dimensional difference of Gaussian kernel defined by two
        standard deviation values:

        .. math::

            \mathbf{K} = G(\sigma_1) - G(\sigma_2)
            
        where :math:`\sigma_1 > \sigma_2`. 
        By default, :math:`\sigma_2 = 1.6 \sigma_1`. 
        
        The kernel is centred within a square array with side length given by:

        - :math:`2 \mbox{ceil}(3 \sigma) + 1`, or
        - :math:`2\mathtt{h} + 1`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> Kernel.DoG(1)

        .. note::

            - This kernel is similar to the Laplacian of Gaussian and is often
              used as an efficient approximation.
            - This is a "Mexican hat" shaped kernel

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.1.3, P. Corke, Springer 2023.

        :seealso: :meth:`LoG` :meth:`Gauss`
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

        :param sigma: standard deviation of first Gaussian kernel
        :type sigma: float
        :param h: half-width of kernel
        :type h: int, optional
        :return: kernel
        :rtype: ndarray(2h+1, 2h+1)

        Return a 2-dimensional Laplacian of Gaussian kernel with
        standard deviation ``sigma``

        .. math::

            \mathbf{K} = \frac{1}{\pi \sigma^4} \left(\frac{u^2 + v^2}{2 \sigma^2} -1\right) e^{-(u^2 + v^2) / 2 \sigma^2}

        The kernel is centred within a square array with side length given by:

        - :math:`2 \mbox{ceil}(3 \sigma) + 1`, or
        - :math:`2\mathtt{h} + 1`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> Kernel.LoG(1)

        .. note:: This is the classic "Mexican hat" shaped kernel

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.1.3, P. Corke, Springer 2023.

        :seealso: :meth:`Laplace` :meth:`DoG` :meth:`Gauss` :meth:`zerocross`
        """

        if h is None:
            h = np.ceil(3.0 * sigma)
        wi = np.arange(-h, h + 1)
        x, y = np.meshgrid(wi, wi)

        log = 1.0 / (np.pi * sigma ** 4.0) * \
            ((x ** 2 + y ** 2) / (2.0 * sigma ** 2) - 1) * \
            np.exp(-(x **2 + y** 2) / (2.0 * sigma ** 2))

        # ensure that the mean is zero, for a truncated kernel this may not
        # be the case
        log -= log.mean()

        return log

    @staticmethod
    def DGauss(sigma, h=None):
        r"""
        Derivative of Gaussian kernel

        :param sigma: standard deviation of first Gaussian kernel
        :type sigma: float
        :param h: half-width of kernel
        :type h: int, optional
        :return: kernel
        :rtype: ndarray(2h+1, 2h+1)

        Returns a 2-dimensional derivative of Gaussian
        kernel with standard deviation ``sigma``

        .. math::

            \mathbf{K} = \frac{-x}{2\pi \sigma^2} e^{-(x^2 + y^2) / 2 \sigma^2}

        The kernel is centred within a square array with side length given by:

        - :math:`2 \mbox{ceil}(3 \sigma) + 1`, or
        - :math:`2\mathtt{h} + 1`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> Kernel.DGauss(1)

        .. note::

            - This kernel is the horizontal derivative of the Gaussian, :math:`dG/dx`.
            - The vertical derivative, :math:`dG/dy`, is the transpose of this kernel.
            - This kernel is an effective edge detector.

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.1.3, P. Corke, Springer 2023.

        :seealso: :meth:`Gauss` :meth:`Sobel`
        """
        if h is None:
            h = np.ceil(3.0 * sigma)

        wi = np.arange(-h, h + 1)
        x, y = np.meshgrid(wi, wi)

        return -x / sigma ** 2 / (2.0 * np.pi) * \
            np.exp(-(x ** 2 + y ** 2) / 2.0 / sigma ** 2)

    @staticmethod
    def Circle(radius, h=None, normalize=False, dtype='uint8'):
        r"""
        Circular structuring element

        :param radius: radius of circular structuring element
        :type radius: scalar, array_like(2)
        :param h: half-width of kernel
        :type h: int
        :param normalize: normalize volume of kernel to one, defaults to False
        :type normalize: bool, optional
        :param dtype: data type for image, defaults to ``uint8``
        :type dtype: str or NumPy dtype, optional
        :return: circular kernel
        :rtype: ndarray(2h+1, 2h+1)

        Returns a circular kernel of radius ``radius`` pixels. Sometimes referred
        to as a tophat kernel. Values inside the circle are set to one,
        outside are set to zero.

        If ``radius`` is a 2-element vector the result is an annulus of ones,
        and the two numbers are interpretted as inner and outer radii
        respectively.

        The kernel is centred within a square array with side length given 
        by :math:`2\mathtt{h} + 1`.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> Kernel.Circle(2)
            >>> Kernel.Circle([2, 3])


        :references:
            - Robotics, Vision & Control for Python, Section 11.5.1.1, P. Corke, Springer 2023.

        :seealso: :meth:`Box`
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

        s = np.zeros((int(w), int(w)), dtype=dtype)
        c = np.floor(w / 2.0)

        if not argcheck.isscalar(radius):
            # circle case
            x = np.arange(w) - c
            X, Y = np.meshgrid(x, x)
            r2 = (X ** 2 + Y ** 2)
            ll = np.where((r2 >= rmin**2) & (r2 <= rmax**2))
            s[ll] = 1
        else:
            # annulus case
            x = np.arange(w) - c
            X, Y = np.meshgrid(x, x)
            ll = np.where(np.round((X ** 2 + Y ** 2 - radius ** 2) <= 0))
            s[ll] = 1

        if normalize:
            k /= np.sum(k)
        return s

    @staticmethod
    def Box(h, normalize=True):
        r"""
        Square structuring element

        :param h: half-width of kernel
        :type h: int
        :param normalize: normalize volume of kernel to one, defaults to True
        :type normalize: bool, optional
        :return: kernel
        :rtype: ndarray(2h+1, 2h+1)

        Returns a square kernel with unit volume.

        The kernel is centred within a square array with side length given 
        by :math:`2\mathtt{h} + 1`.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> Kernel.Box(2)
            >>> Kernel.Box(2, normalize=False)

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.1.1, P. Corke, Springer 2023.

        :seealso: :meth:`Circle`
        """
        # check valid input:
        wi = 2 * h + 1
        k = np.ones((wi, wi))
        if normalize:
            k /= np.sum(k)

        return k

class ImageSpatialMixin:

    @staticmethod
    def _bordertype_cv(border, exclude=None):
        """
        Border handling options for OpenCV

        :param border: border handling option, one of: 'constant', 'replicate', 
            'reflect', 'mirror', 'wrap', 'pad', 'none'
        :type border: str
        :param exclude: list of excluded values, defaults to None
        :type exclude: list or tuple, optional
        :raises ValueError: ``border`` value is not valid
        :raises ValueError: ``border`` value is excluded
        :return: OpenCV key
        :rtype: int

        Map an MVTB border handling key to the OpenCV flag value
        """

        # border options:
        border_opt = {
            'constant':    cv.BORDER_CONSTANT,
            'replicate':   cv.BORDER_REPLICATE,
            'reflect':     cv.BORDER_REFLECT,
            'mirror':      cv.BORDER_REFLECT_101,
            'reflect_101': cv.BORDER_REFLECT_101,
            'wrap':        cv.BORDER_WRAP,
            'pad':         cv.BORDER_CONSTANT,
            'none':        cv.BORDER_ISOLATED,
        }
        if exclude is not None and border in exclude:
            raise ValueError('border option not supported')

        try:
            return border_opt[border]
        except KeyError:
            raise ValueError(border, 'border is not a valid option')

    # border options:
    _border_opt = {
        'constant':   cv.BORDER_CONSTANT,
        'replicate':  cv.BORDER_REPLICATE,
        'reflect':    cv.BORDER_REFLECT,
        'mirror':     cv.BORDER_REFLECT_101,
        'wrap':       cv.BORDER_WRAP,
        'pad':        cv.BORDER_CONSTANT,
        'none':       cv.BORDER_ISOLATED,
    }

    @staticmethod
    def _border_args_cv(border, morpho=False, allow=None, disallow=None):
        if disallow is not None and border in disallow:
            raise ValueError(f"border option {border} not supported")
        if allow is not None and border not in allow:
            raise ValueError(f"border option {border} not supported")
        
        if isinstance(border, str):
            # given as string, convert to OpenCV flag value
            try:
                return dict(borderType=_border_opt[border])
            except KeyError:
                raise ValueError(border, 'border is not a valid option')
        elif isinstance(border, int) or isinstance(border, float):
            # given as a numeric value, assume 'pad'
            return dict(bordertype=_border_opt['pad'], borderValu=border_value)

    @staticmethod
    def _bordertype_sp(border, exclude=None):
        """
        Border handling options for SciPy

        :param border: border handling option, one of: 'constant', 'replicate', 
            'reflect', 'mirror', 'wrap'
        :type border: str
        :param exclude: list of excluded values, defaults to None
        :type exclude: list or tuple, optional
        :raises ValueError: ``border`` value is not valid
        :raises ValueError: ``border`` value is excluded
        :return: SciPy key
        :rtype: str

        Map an MVTB border handling key to the SciPy ndimage flag value
        """
        # border options:
        border_opt = {
            'constant':   'constant',
            'replicate':  'nearest',
            'reflect':    'reflect',
            'mirror':     'mirror',
            'wrap':       'wrap',
        }
        if exclude is not None and border in exclude:
            raise ValueError('border option not supported')

        try:
            return border_opt[border]
        except KeyError:
            raise ValueError(border, 'border is not a valid option')

    def smooth(self, sigma, h=None, mode='same', border='reflect', bordervalue=0):
        r"""
        Smooth image

        :param sigma: standard deviation of the Gaussian kernel
        :type sigma: float
        :param h: half-width of the kernel
        :type h: int
        :param mode: option for convolution, see :meth:`convolve`, defaults to 'same'
        :type mode: str, optional
        :param border: option for boundary handling, see :meth:`convolve`, defaults to 'reflect'
        :type border: str, optional
        :param bordervalue: padding value, see :meth:`convolve`, defaults to 0
        :type bordervalue: scalar, optional
        :return: smoothed image
        :rtype: :class:`Image`

        Smooth the image by convolving with a Gaussian kernel of standard
        deviation ``sigma``.  If ``h`` is not given the kernel half width is set
        to :math:`2 \mbox{ceil}(3 \sigma) + 1`.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('monalisa.png')
            >>> img.smooth(sigma=3).disp()

        .. note::
            - Smooths all planes of the input image.
            - The Gaussian kernel has a unit volume.
            - If input image is integer it is converted to float, convolved,
              then converted back to integer.

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.1, P. Corke, Springer 2023.

        :seealso: :meth:`machinevisiontoolbox.Kernel.Gauss` :meth:`convolve`
        """

        if not argcheck.isscalar(sigma):
            raise ValueError(sigma, 'sigma must be a scalar')

        # make the smoothing kernel
        K = Kernel.Gauss(sigma, h)

        return self.convolve(K, mode=mode, border=border, bordervalue=bordervalue)

    def convolve(self, K, mode='same', border='reflect', bordervalue=0):
        """
        Image convolution

        :param K: kernel
        :type K: ndarray(N,M)
        :param mode: option for convolution, defaults to 'same'
        :type mode: str, optional
        :param border: option for boundary handling, defaults to 'reflect'
        :type border: str, optional
        :param bordervalue: padding value, defaults to 0
        :type bordervalue: scalar, optional
        :return: convolved image
        :rtype: :class:`Image` instance

        Computes the convolution of image with the kernel ``K``.

        There are two options that control what happens at the edge of the image
        where the convolution window lies outside the image border.  ``mode``
        controls the size of the resulting image, while ``border`` controls how
        pixel values are extrapolated outside the image border. 

        ===========   ===========================================================================
        ``mode``      description
        ===========   ===========================================================================
        ``'same'``    output image is same size as input image (default)
        ``'full'``    output image is larger than the input image, add border to input image
        ``'valid'``   output image is smaller than the input image and contains only valid pixels
        ===========   ===========================================================================

        ================  ====================================================
        ``border``        description
        ================  ====================================================
        ``'replicate'``   replicate border pixels outwards
        ``'pad'``         outside pixels are set to ``value``
        ``'wrap'``        borders are joined, left to right, top to bottom
        ``'reflect'``     outside pixels reflect inside pixels
        ``'reflect101'``  outside pixels reflect inside pixels except for edge
        ``'none'``          do not look outside of border
        ================  ====================================================

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> img = Image.Read('monalisa.png')
            >>> img.convolve(K=np.ones((11,11))).disp()

        .. note::
            - The kernel is typically square with an odd side length.
            - The result has the same datatype as the input image.  For a kernel
              where the results could be negative (eg. edge detection kernel) 
              this will cause issues such as value wraparound.
            - If the image is color (has multiple planes) the kernel is
              applied to each plane, resulting in an output image with the same
              number of planes.

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.1, P. Corke, Springer 2023.

        :seealso: :meth:`smooth` :func:`opencv.filter2D` :func:`opencv.copyMakeBorder`
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

        img = self.A
        if border == "pad" and value != 0:
            img = cv.copyMakeBorder(a, kv, kv, kh, kh, 
                    cv.BORDER_CONSTANT, value=bordervalue)
        elif mode == "full":
            img = cv.copyMakeBorder(a, kv, kv, kh, kh, 
                    self._bordertype_cv(border), value=bordervalue)

        out = cv.filter2D(img, ddepth=-1, kernel=K, 
            borderType=self._bordertype_cv(border))

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



    def gradients(self, kernel=None, mode='same', border='reflect', bordervalue=0):
        """
        Compute horizontal and vertical gradients

        :param kernel: derivative kerne, defaults to Sobel
        :type kernel: 2D ndarray, optional
        :param mode: option for convolution, see :meth:`convolve`, defaults to 'same'
        :type mode: str, optional
        :param border: option for boundary handling, see :meth:`convolve`, defaults to 'reflect'
        :type border: str, optional
        :param bordervalue: padding value, , see :meth:`convolve`, defaults to 0
        :type bordervalue: scalar, optional
        :return: gradient images
        :rtype: :class:`Image`, :class:`Image`

        Compute horizontal and vertical gradient images.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('monalisa.png', grey=True)
            >>> Iu, Iv = img.gradients()

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.1.3, P. Corke, Springer 2023.

        :seealso: :obj:`Kernel.Sobel` :obj:`Kernel.DGauss` :obj:`Kernel.DoG`
        """
        if kernel is None:
            kernel = Kernel.Sobel()

        Iu = self.convolve(kernel, mode=mode, border=border, bordervalue=bordervalue)
        Iv = self.convolve(kernel.T, mode=mode, border=border, bordervalue=bordervalue)
        return Iu, Iv

    def direction(horizontal, vertical):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        r"""
        Gradient direction

        :param im: vertical gradient image
        :type im: :class:`Image`
        :return: gradient direction in radians
        :rtype: :class:`Image`

        Compute the per-pixel gradient direction from two images comprising the
        horizontal and vertical gradient components.

        .. math::

            \theta_{u,v} = \tan^{-1} \frac{\mat{I}_{v: u,v}}{\mat{I}_{u: u,v}}

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('monalisa.png', grey=True)
            >>> Iu, Iv = img.gradients()
            >>> direction = Iu.direction(Iv)

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.1.3, P. Corke, Springer 2023.

        :seealso: :meth:`gradients`
        """
        if horizontal.shape != vertical.shape:
            raise ValueError('images must the same shape')
        return horizontal.__class__(np.arctan2(vertical.A, horizontal.A))

    def Harris_corner_strength(self, k=0.04, h=2):
        """
        Harris corner strength image

        :param k: Harris parameter, defaults to 0.04
        :type k: float, optional
        :param h: kernel half width, defaults to 2
        :type h: int, optional
        :return: Harris corner strength image
        :rtype: :class:`Image`

        Returns an image containing Harris corner strength values.  This is
        positive for high gradient in orthogonal directions, and negative
        for high gradient in a single direction.

        :references:
            - Robotics, Vision & Control for Python, Section 12.3.1, P. Corke, Springer 2023.

        :seealso: :meth:`gradients` :meth:`Harris`
        """
        dst = cv.cornerHarris(self.mono().image, 2, 2 * h + 1, k)
        return self.__class__(dst)

    def window(self, func, h=None, se=None, border='reflect', bordervalue=0, **kwargs):
        r"""
        Generalized spatial operator

        :param func: function applied to window
        :type func: callable
        :param h: half width of structuring element
        :type h: int, optional
        :param se: structuring element
        :type se: ndarray(N,M), optional
        :param border: option for boundary handling, see :meth:`convolve`, defaults to 'reflect'
        :type border: str, optional
        :param bordervalue: padding value, defaults to 0
        :type bordervalue: scalar, optional
        :raises ValueError: ``border`` is not a valid option
        :raises TypeError: ``func`` not callable
        :raises ValueError: single channel images only
        :return: transformed image
        :rtype: :class:`Image`

        Returns an image where each pixel is the result of applying the function
        ``func`` to a neighbourhood centred on the corresponding pixel in image.
        The return value of ``func`` becomes the corresponding pixel value.

        The neighbourhood is defined in two ways:
        
        - If ``se`` is given then it is the the size of the structuring element
          ``se`` which should have odd side lengths. The elements in the
          neighbourhood corresponding to non-zero elements in ``se`` are packed
          into a vector (in column order from top left) and passed to the
          specified callable function ``func``. 
        - If ``se`` is None then ``h`` is the half width of a :math:`w \times
          w` square structuring element of ones, where :math:`w =2h+1`.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> img = Image.Read('monalisa.png', grey=True)
            >>> out = img.window(np.median, h=3)

        .. note::
            - The structuring element should have an odd side length.
            - Is slow since the function ``func`` must be invoked once for
              every output pixel.

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.3, P. Corke, Springer 2023.

        :seealso: :func:`scipy.ndimage.generic_filter`
        """
        # replace window's mex function with scipy's ndimage.generic_filter
        if self.iscolor:
            raise ValueError('single channel images only')

        if not callable(func):
            raise TypeError(func, 'func not callable')

        if h is not None and se is None:
            w = 2 * h + 1
            se = np.ones((w, w))

        out = sp.ndimage.generic_filter(self.A,
                                            func,
                                            footprint=se,
                                            mode=self._bordertype_sp(border),
                                            cval=bordervalue)
        return self.__class__(out)

    def zerocross(self):
        """
        Compute zero crossing

        :return: boolean image
        :rtype: :class:`Image` instance
            
        Compute a zero-crossing image, where pixels are true if they are adjacent to 
        a change in sign.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> U, V = Image.meshgrid(None, 6, 6)
            >>> img = Image(U - V - 2, dtype='float')
            >>> img.print()
            >>> img.zerocross().print()

        .. note:: Use morphological filtering with 3x3 structuring element, can
            lead to erroneous values in border pixels.

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.1.3, P. Corke, Springer 2023.

        :seealso: :meth:`Laplace` :meth:`LoG`
        """
        min = cv.morphologyEx(self.image, cv.MORPH_ERODE, np.ones((3,3)))
        max = cv.morphologyEx(self.image, cv.MORPH_DILATE, np.ones((3,3)))
        zeroCross = np.logical_or(
            np.logical_and(min < 0, self.image > 0), 
            np.logical_and(max > 0, self.image < 0)
        )
        return self.__class__(zeroCross)

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
    #     :rtype out: :class:`Image`

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


    def scalespace(self, n, sigma=1):
        """
        Compute image scalespace sequence

        :param n: number of steps
        :type n: omt
        :param sigma: Gaussian filter width, defaults to 1
        :type sigma: scalar, optional
        :return: Gaussian and difference of Gaussian sequences, scale factors
        :rtype: list of :class:`Image`, list of :class:`Image`, list of float

        Compute a scalespace image sequence by consecutively smoothing the input
        image with a Gaussian of width ``sigma``.  The difference between
        consecutive smoothings is the difference of Gaussian which is an 
        approximation to the Laplacian of Gaussian.

        Examples::

            >>> mona = Image.Read("monalisa.png", dtype="float");
            >>> G, L, scales = mona.scalespace(8, sigma=8);

        .. note:: The two image sequences have the same length, the original image is
            not included in the list of smoothed images.

        :references:
            - Robotics, Vision & Control for Python, Section 12.3.2, P. Corke, Springer 2023.

        :seealso: :meth:`pyramid` :meth:`smooth` :class:`Kernel.Gauss` :class:`Kernel.LoG`
        """
        im = self.copy()
        g = [im]
        scale = 0.5
        scales = [scale]
        lap = []

        for i in range(n-1):
            im = im.smooth(sigma)
            scale = np.sqrt(scale ** 2 + sigma ** 2)
            scales.append(scale)
            g.append(im)
            x = (g[-1] - g[-2]) * scale ** 2 
            lap.append(x)

        return g, lap, scales

    def pyramid(self, sigma=1, N=None, border='replicate', bordervalue=0):
        """
        Pyramidal image decomposition

        :param sigma: standard deviation of Gaussian kernel
        :type sigma: float
        :param N: number of pyramid levels to be computed, defaults to all
        :type N: int, optional
        :param border: option for boundary handling, see :meth:`convolve`, defaults to 'replicate'
        :type border: str, optional
        :param bordervalue: padding value, defaults to 0
        :type bordervalue: scalar, optional
        :return: list of images at each pyramid level
        :rtype: list of :class:`Image`

        Returns a pyramid decomposition of the input image using Gaussian
        smoothing with standard deviation of ``sigma``. The return is a list
        array of images each one having dimensions half that of the previous
        image. The pyramid is computed down to a non-halvable image size.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('monalisa.png')
            >>> pyramid = img.pyramid(4)
            >>> len(pyramid)
            >>> pyramid

        .. note::
            - Works for greyscale images only.
            - Converts a color image to greyscale.

        :references:
            - Robotics, Vision & Control for Python, Section 12.3.2, P. Corke, Springer 2023.

        :seealso: :meth:`smooth` :meth:`scalespace`
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
            impyr = cv.pyrDown(impyr, borderType=self._bordertype_cv(border, exclude=('constant')))
            pyr.append(impyr)

        # output list of Image objects
        pyrimlist = [self.__class__(p) for p in pyr]
        return pyrimlist
        
    def canny(self, sigma=1, th0=None, th1=None):
        """
        Canny edge detection

        :param sigma: standard deviation for Gaussian kernel smoothing, defaults to 1
        :type sigma: float, optional
        :param th0: lower threshold
        :type th0: float
        :param th1: upper threshold
        :type th1: float
        :return: edge image
        :rtype: :class:`Image` instance

        Computes an edge image obtained using the Canny edge detector algorithm.
        Hysteresis filtering is applied to the gradient image: edge pixels >
        ``th1`` are connected to adjacent pixels > ``th0``, those below ``th0``
        are set to zero.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('monalisa.png')
            >>> edges = img.canny()

        .. note::

            - Produces a zero image with single pixel wide edges having
              non-zero values.
            - Larger values correspond to stronger edges.
            - If ``th1`` is zero then no hysteresis filtering is performed.
            - A color image is automatically converted to greyscale first.

        :references:
            - "A Computational Approach To Edge Detection", J. Canny,
              IEEE Trans. Pattern Analysis and Machine Intelligence,
              8(6):679–698, 1986.
            - Robotics, Vision & Control for Python, Section 11.5.1.3, P. Corke, Springer 2023.

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

    def rank(self, footprint=None, h=None, rank=-1, border='replicate', bordervalue=0):
        r"""
        Rank filter

        :param footprint: filter footprint or structuring element
        :type footprint: ndarray(N,M), optional
        :param h: half width of structuring element
        :type h: int, optional
        :param rank: rank of filter
        :type rank: int, str
        :param border: option for boundary handling, defaults to 'replicate'
        :type border: str, optional
        :param bordervalue: padding value, defaults to 0
        :type bordervalue: scalar, optional
        :return: rank filtered image
        :rtype: :class:`Image`

        Return a rank filtered version of image.  Only pixels corresponding to
        non-zero elements of the structuring element are ranked, and the value
        that is ``rank`` in rank becomes the corresponding output pixel value.
        The highest rank, the maximum, is rank 0.  The rank can also be given
        as a string: 'min|imumum', 'max|imum', 'med|ian', long or short versions
        are supported.

        The structuring element is given as:
        
            - ``footprint`` a 2D Numpy array containing zero or one values, or
            - ``h`` which is the half width :math:`w=2h+1` of an array of ones

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> img = Image(np.arange(25).reshape((5,5)))
            >>> img.print()
            >>> img.rank(h=1, rank=0).print()  # maximum filter
            >>> img.rank(h=1, rank=8).print()  # minimum filter
            >>> img.rank(h=1, rank=4).print()  # median filter
            >>> img.rank(h=1, rank='median').print()  # median filter

        .. note::
            - The footprint should have an odd side length.
            - The input can be logical, uint8, uint16, float or double, the
              output is always double.

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.3, P. Corke, Springer 2023.

        :seealso: :obj:`scipy.ndimage.rank_filter`
        """
        if h is not None:
            w = 2 * h + 1
            footprint = np.ones((w, w))

        n = np.sum(footprint)
            
        if isinstance(rank, str):
            if rank in ('min', 'minimum'):
                rank = n - 1
            elif rank in ('max', 'maximum'):
                rank = 0
            elif rank in ('med', 'median'):
                rank = n // 2
        elif not isinstance(rank, int):
            raise TypeError(rank, 'rank must be int or str')

        if rank < 0:
            raise ValueError('rank must be >= 0')

        r = int(footprint.sum() - rank - 1)

        out = sp.ndimage.rank_filter(self.A,
                                    r,
                                    footprint=footprint,
                                    mode=self._bordertype_sp(border))
        return self.__class__(out)

    def medianfilter(self, h=1, **kwargs):
        r"""
        Median filter

        :param h: half width of structuring element, defaults to 1
        :type h: int, optional
        :param kwargs: options passed to :meth:`rank`
        :return: median filtered image
        :rtype: :class:`Image` instance

        Return the median filtered image.  For every :math:`w \times w, w=2h+1`
        window take the median value as the output pixel value.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> img = Image(np.arange(25).reshape((5,5)))
            >>> img.A
            >>> img.medianfilter(h=1).A  # median filter
            >>> img = Image.Read('monalisa.png')
            >>> img.medianfilter(h=5).disp()  # ameliorate background cracking

        .. note:: This filter is effective for removing impulse (aka
            salt and pepper) noise.

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.3, P. Corke,
              Springer 2023.

        :seealso: :meth:`rank`
        """
        w = 2 * h + 1
        r = int((w ** 2 - 1) / 2)
        return self.rank(h=h, rank=r, **kwargs)

    def distance_transform(self, invert=False, norm="L2", h=1):
        """
        Distance transform

        :param invert: consider inverted image, defaults to False
        :type invert: bool, optional
        :param norm: distance metric: 'L1' or 'L2' [default]
        :type norm: str, optional
        :param h: half width of window, defaults to 1
        :type h: int, optional
        :return: distance transform of image
        :rtype: :class:`Image`

        Compute the distance transform. For each zero input pixel, compute its
        distance to the nearest non-zero input pixel.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> pixels = np.zeros((5,5))
            >>> pixels[2, 1:3] = 1
            >>> img = Image(pixels)
            >>> img.distance_transform().print(precision=3)
            >>> img.distance_transform(norm="L1").print()

        .. note::
            - The output image is the same size as the input image.
            - Distance is computed using a sliding window and is an 
              approximation of true distance.
            - For non-zero input pixels the corresponding output pixels are set
              to zero.
            - The signed-distance function is ``image.distance_transform() - image.distance_transform(invert=True)``

        :references:
            - Robotics, Vision & Control for Python, Section 11.6.4, P. Corke, Springer 2023.
        
        :seealso: `opencv.distanceTransform <https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga8a0b7fdfcb7a13dde018988ba3a43042>`_
        """
        # OpenCV does distance to nearest zero pixel
        # this function does distance to nearest non-zero pixel by default,
        # and the OpenCV thing if invert=True
        if invert:
            # distance to nearest zero pixel
            im = self.to_int()
        else:
            # distance to nearest non-zero pixel, invert the image
            im = self.invert().to_int()

        normdict = {
            "L1": cv.DIST_L1,
            "L2": cv.DIST_L2,
        }

        out = cv.distanceTransform(im, distanceType=normdict[norm], maskSize=2*h+1)
        return self.__class__(out)

    # ======================= labels ============================= #

    def labels_binary(self, connectivity=4, ltype='int32'):
        """
        Blob labelling

        :param connectivity: number of neighbours used for connectivity: 4 [default] or 8
        :type connectivity: int, optional
        :param ltype: output image type: 'int32' [default], 'uint16'
        :type ltype: string, optional
        :return: label image, number of regions
        :rtype: :class:`Image`, int

        Compute labels of connected components in the input greyscale or binary
        image. Regions are sets of contiguous pixels with the same value.

        The method returns the label image and the number of labels N, so labels
        lie in the range [0, N-1].The value in the label image in an integer
        indicating which region the corresponding input pixel belongs to.  The
        background has label 0.  

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Squares(2, 15)
            >>> img.print()
            >>> labels, N = img.labels_binary()
            >>> N
            >>> labels.print()

        .. note::
            - This algorithm is variously known as region labelling,
              connectivity analysis, region coloring, connected component analysis,
              blob labelling.
            - The output image is the same size as the input image.
            - The input image can be binary or greyscale.
            - Connectivity is performed using 4 nearest neighbours by default.
            - 8-way connectivity introduces ambiguities, a chequerboard is
              two blobs.

        :references:
            - Robotics, Vision & Control for Python, Section 12.1.2.1, P. Corke, Springer 2023.

        :seealso: :meth:`blobs` `cv2.connectedComponents <https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gaedef8c7340499ca391d459122e51bef5>`_
            :meth:`labels_graphseg` :meth:`labels_MSER` 
        """
        if not (connectivity in [4, 8]):
            raise ValueError(conn, 'connectivity must be 4 or 8')

        # make labels uint32s, unique and never recycled?
        # set ltype to default to cv.CV_32S
        if ltype == 'int32':
            ltype = cv.CV_32S
            dtype = np.int32
        elif ltype == 'uint16':
            ltype = cv.CV_16U
            dtype = np.uint16
        else:
            raise TypeError(ltype, 'ltype must be either int32 or uint16')

        retval, labels = cv.connectedComponents(
            image=self.to_int(),
            connectivity=connectivity,
            ltype=ltype
        )
        return self.__class__(labels), retval


    def labels_MSER(self, **kwargs):
        """
        Blob labelling using MSER

        :param kwargs: arguments passed to ``MSER_create``
        :return: label image, number of regions
        :rtype: :class:`Image`, int

        Compute labels of connected components in the input greyscale image.
        Regions are sets of contiguous pixels that form stable regions across a
        range of threshold values.

        The method returns the label image and the number of labels N, so labels
        lie in the range [0, N-1].The value in the label image in an integer
        indicating which region the corresponding input pixel belongs to.  The
        background has label 0.  

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Squares(2, 15)
            >>> img.print()
            >>> labels, N = img.labels_MSER()
            >>> N
            >>> labels.print()

        :references:
            - Linear time maximally stable extremal regions,
              David Nistér and Henrik Stewénius,
              In Computer Vision–ECCV 2008, pages 183–196. Springer, 2008.
            - Robotics, Vision & Control for Python, Section 12.1.2.2, P. Corke, Springer 2023.

        :seealso: :meth:`labels_binary` :meth:`labels_graphseg` :meth:`blobs` `opencv.MSER_create <https://docs.opencv.org/3.4/d3/d28/classcv_1_1MSER.html>`_
        """

        mser = cv.MSER_create(**kwargs)
        regions, _ = mser.detectRegions(self.to_int())

        if len(regions) < 256:
            dtype = np.uint8
        else:
            dtype=np.uint32

        out = np.zeros(self.shape, dtype=dtype)

        for i, points in enumerate(regions):
            # print('region ', i, points.shape[0])
            out[points[:,1], points[:,0]] = i

        return self.__class__(out, dtype=dtype), len(regions)

    def labels_graphseg(self, sigma=0.5, k=2000, minsize=100):
        """
        Blob labelling using graph-based segmentation

        :param kwargs: arguments passed to ``MSER_create``
        :return: label image, number of regions
        :rtype: :class:`Image`, int

        Compute labels of connected components in the input color image. Regions
        are sets of contiguous pixels that are similar with respect to their
        surrounds.

        The method returns the label image and the number of labels N, so labels
        lie in the range [0, N-1].The value in the label image in an integer
        indicating which region the corresponding input pixel belongs to.  The
        background has label 0.  

        :references:
            - Efficient graph-based image segmentation,
              Pedro F Felzenszwalb and Daniel P Huttenlocher,
              volume 59, pages 167–181. Springer, 2004.
            - Robotics, Vision & Control for Python, Section 12.1.2.2, P. Corke, Springer 2023.

        :seealso: :meth:`labels_binary` :meth:`labels_MSER` :meth:`blobs` `opencv.createGraphSegmentation <https://docs.opencv.org/3.4/d5/df0/group__ximgproc__segmentation.html#ga5e3e721c5f16e34d3ad52b9eeb6d2860>`_
        """
        # P. Felzenszwalb, D. Huttenlocher: "Graph-Based Image Segmentation
        segmenter = cv.ximgproc.segmentation.createGraphSegmentation(
            sigma=0.5,
            k=2000,
            min_size=100)
        out = segmenter.processImage(self.to_int())

        return self.__class__(out), np.max(out)




    # -------------------- similarity operations -------------------------- #

    def sad(image1, image2):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Sum of absolute differences

        :param image2: second image
        :type image2: :class:`Image`
        :raises ValueError: image2 shape is not equal to self
        :return: sum of absolute differences
        :rtype: scalar

        Returns a simple image disimilarity measure which is the sum of absolute
        differences between the image and ``image2``.   The result is a scalar
        and a value of 0 indicates identical pixel patterns and is increasingly
        positive as image dissimilarity increases.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img1 = Image([[10, 11], [12, 13]])
            >>> img2 = Image([[10, 11], [10, 13]])
            >>> img1.sad(img2)
            >>> img1.sad(img2+10)
            >>> img1.sad(img2*2)

        .. note:: Not invariant to pixel value scale or offset.

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.2, P. Corke, Springer 2023.

        :seealso: :meth:`zsad` :meth:`ssd` :meth:`ncc`
        """

        if not np.all(image1.shape == image2.shape):
            raise ValueError('image2 shape is not equal to image1')

        # out = []
        # for im in self:
            # m = np.abs(im.image - image2.image)
            # out.append(np.sum(m))
        m = np.abs(image1.image - image2.image)
        out = np.sum(m)
        return out

    def ssd(image1, image2):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Sum of squared differences

        :param image2: second image
        :type image2: :class:`Image`
        :raises ValueError: image2 shape is not equal to self
        :return: sum of squared differences
        :rtype: scalar

        Returns a simple image disimilarity measure which is the sum of the
        squared differences between the image and ``image2``.   The result is a
        scalar and a value of 0 indicates identical pixel patterns and is
        increasingly positive as image dissimilarity increases.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img1 = Image([[10, 11], [12, 13]])
            >>> img2 = Image([[10, 11], [10, 13]])
            >>> img1.ssd(img2)
            >>> img1.ssd(img2+10)
            >>> img1.ssd(img2*2)

        .. note:: Not invariant to pixel value scale or offset.

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.2, P. Corke, Springer 2023.

        :seealso: :meth:`zssd` :meth:`sad` :meth:`ncc`
        """

        if not np.all(image1.shape == image2.shape):
            raise ValueError('image2 shape is not equal to image1')
        m = np.power((image1.image - image2.image), 2)
        return np.sum(m)

    def ncc(image1, image2):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Normalised cross correlation

        :param image2: second image
        :type image2: :class:`Image`
        :raises ValueError: image2 shape is not equal to self
        :return: normalised cross correlation
        :rtype: scalar

        Returns an image similarity measure which is the normalized
        cross-correlation between the image and ``image2``. The result is a
        scalar in the interval -1 (non match) to 1 (perfect match) that
        indicates similarity.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img1 = Image([[10, 11], [12, 13]])
            >>> img2 = Image([[10, 11], [10, 13]])
            >>> img1.ncc(img2)
            >>> img1.ncc(img2+10)
            >>> img1.ncc(img2*2)

        .. note:: 
            - The ``ncc`` similarity measure is invariant to scale changes in
              image intensity.

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.2, P. Corke, Springer 2023.

        :seealso: :meth:`zncc` :meth:`sad` :meth:`ssd`
        """
        if not np.all(image1.shape == image2.shape):
            raise ValueError('image2 shape is not equal to image1')

        denom = np.sqrt(np.sum(image1.image ** 2) * np.sum(image2.image ** 2))

        if denom < 1e-10:
            return 0
        else:
            return np.sum(image1.image * image2.image) / denom

    def zsad(image1, image2):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Zero-mean sum of absolute differences

        :param image2: second image
        :type image2: :class:`Image`
        :raises ValueError: image2 shape is not equal to self
        :return: sum of absolute differences
        :rtype: scalar

        Returns a simple image disimilarity measure which is the zero-mean sum
        of absolute differences between the image and ``image2``.   The result
        is a scalar and a value of 0 indicates identical pixel patterns
        (relative to their mean values) and is increasingly positive as image
        dissimilarity increases.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img1 = Image([[10, 11], [12, 13]])
            >>> img2 = Image([[10, 11], [10, 13]])
            >>> img1.zsad(img2)
            >>> img1.zsad(img2+10)
            >>> img1.zsad(img2*2)

        .. note:: 
            - The ``zsad`` similarity measure is invariant to changes in image
              brightness offset.

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.2, P. Corke, Springer 2023.

        :seealso: :meth:`zsad` :meth:`ssd` :meth:`ncc`
        """
        if not np.all(image1.shape == image2.shape):
            raise ValueError('image2 shape is not equal to image1')

        image1 = image1.image - np.mean(image1.image)
        image2 = image2.image - np.mean(image2.image)
        m = np.abs(image1 - image2)
        return np.sum(m)

    def zssd(image1, image2):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Zero-mean sum of squared differences

        :param image2: second image
        :type image2: :class:`Image`
        :raises ValueError: image2 shape is not equal to self
        :return: sum of squared differences
        :rtype: scalar

        Returns a simple image disimilarity measure which is the zero-mean sum of the
        squared differences between the image and ``image2``.   The result is a
        scalar and a value of 0 indicates identical pixel patterns (relative to their maen) and is
        increasingly positive as image dissimilarity increases.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img1 = Image([[10, 11], [12, 13]])
            >>> img2 = Image([[10, 11], [10, 13]])
            >>> img1.zssd(img2)
            >>> img1.zssd(img2+10)
            >>> img1.zssd(img2*2)

        .. note:: 
            - The ``zssd`` similarity measure is invariant to changes in image
              brightness offset.

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.2, P. Corke, Springer 2023.

        :seealso: :meth:`zssd` :meth:`sad` :meth:`ncc`
        """

        if not np.all(image1.shape == image2.shape):
            raise ValueError('image2 shape is not equal to image1')

        image1 = image1.image - np.mean(image1.image)
        image2 = image2.image - np.mean(image2.image)
        m = np.power(image1 - image2, 2)
        return np.sum(m)

    def zncc(image1, image2):  # lgtm[py/not-named-self] pylint: disable=no-self-argument
        """
        Zero-mean normalized cross correlation

        :param image2: second image
        :type image2: :class:`Image`
        :raises ValueError: image2 shape is not equal to self
        :return: normalised cross correlation
        :rtype: scalar

        Returns an image similarity measure which is the zero-mean normalized
        cross-correlation between the image and ``image2``. The result is a
        scalar in the interval -1 (non match) to 1 (perfect match) that
        indicates similarity.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img1 = Image([[10, 11], [12, 13]])
            >>> img2 = Image([[10, 11], [10, 13]])
            >>> img1.zncc(img2)
            >>> img1.zncc(img2+10)
            >>> img1.zncc(img2*2)

        .. note:: 
            - The ``zncc`` similarity measure is invariant to affine changes (offset and scale factor)
              in image intensity (brightness offset and scale).

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.2, P. Corke, Springer 2023.

        :seealso: :meth:`zncc` :meth:`sad` :meth:`ssd`
        """

        if not np.all(image1.shape == image2.shape):
            raise ValueError('image2 shape is not equal to image1')

        image1 = image1.image - np.mean(image1.image)
        image2 = image2.image - np.mean(image2.image)
        denom = np.sqrt(np.sum(np.power(image1, 2) *
                               np.sum(np.power(image2, 2))))

        if denom < 1e-10:
            return 0
        else:
            return np.sum(image1 * image2) / denom
            

    def similarity(self, T, metric='zncc'):
        """
        Locate template in image

        :param T: template image
        :type T: ndarray(N,M)
        :param metric: similarity metric, one of: 'ssd', 'zssd', 'ncc', 'zncc' [default]
        :type metric: str
        :raises ValueError: template T must have odd dimensions
        :raises ValueError: bad metric specified
        :return: similarity image
        :rtype: :class:`Image` instance

        Compute a similarity image where each output pixel is the similarity of
        the template ``T`` to the same-sized neighbourhood surrounding the
        corresonding input pixel in image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> crowd = Image.Read("wheres-wally.png", mono=True, dtype="float")
            >>> T = Image.Read("wally.png", mono=True, dtype="float")
            >>> sim = crowd.similarity(T, "zncc")
            >>> sim.disp(colormap="signed", colorbar=True);

        .. note::

            - For NCC and ZNCC the maximum similarity value corresponds to the most likely
              template location.  For SSD and ZSSD the minimum value
              corresponds to the most likely location.
            - Similarity is not computed for those pixels where the template
              crosses the image boundary, and these output pixels are set
              to NaN.

        :references:
            - Robotics, Vision & Control for Python, Section 11.5.2, P. Corke, Springer 2023.

        :seealso: `cv2.matchTemplate <https://docs.opencv.org/master/df/dfb/group__imgproc__object.html#ga586ebfb0a7fb604b35a23d85391329be>`_
        """

        # check inputs
        if ((T.shape[0] % 2) == 0) or ((T.shape[1] % 2) == 0):
            raise ValueError('template T must have odd dimensions')

        metricdict = {
            'ssd': cv.TM_SQDIFF,
            'zssd': cv.TM_SQDIFF,
            'ncc': cv.TM_CCOEFF_NORMED,
            'zncc': cv.TM_CCOEFF_NORMED
        }

        im = self.A
        T_im = T.A
        if metric[0] == 'z':
            T_im -= np.mean(T_im)  # remove offset from template
            im = im - np.mean(im)  # remove offset from image

        try:
            out = cv.matchTemplate(im, T_im, method=metricdict[metric])
        except KeyError:
            raise ValueError('bad metric specified')
        return self.__class__(out)




# --------------------------------------------------------------------------#
if __name__ == '__main__':
    from machinevisiontoolbox import *

    img = Image(np.array(np.tile(np.r_[-2, -1, 1, 2, 3], (4,1))), dtype='float')
    img.zerocross().A

    print('ImageProcessingKernel.py')
    from machinevisiontoolbox import *

    print(Kernel.Circle([2,3]))

    image = Image.Read('monalisa.png', grey=True)
    blur = image.convolve(Kernel.Gauss(5))
    blur.disp(block=True)
