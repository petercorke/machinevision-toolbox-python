"""
Spatial operations: distance transforms, connected components, labelling, and histograms.
"""

import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal, overload

import cv2
import numpy as np
import scipy as sp
import spatialmath.base.argcheck as argcheck

from machinevisiontoolbox._image_typing import ImageMixinBase
from machinevisiontoolbox.Kernel import Kernel

if TYPE_CHECKING:
    from machinevisiontoolbox.ImageWholeFeatures import CannyHistogram


class ImageSpatialMixin(ImageMixinBase):
    @classmethod
    def Gauss(cls, sigma: float, h: int | None = None) -> Kernel:
        return Kernel.Gauss(sigma=sigma, h=h)

    @classmethod
    def Sobel(cls, direction: str = "x") -> Kernel:
        K = Kernel.Sobel()
        if direction.lower() in ("x", "u"):
            return K
        elif direction.lower() in ("y", "v"):
            return K.T
        else:
            raise ValueError("direction must be one of 'x' or 'y'")

    @classmethod
    def Laplace(cls) -> Kernel:
        return Kernel.Laplace()

    @classmethod
    def DoG(
        cls, sigma1: float, sigma2: float | None = None, h: int | None = None
    ) -> Kernel:
        return Kernel.DoG(sigma1=sigma1, sigma2=sigma2, h=h)

    @classmethod
    def LoG(cls, sigma: float, h: int | None = None) -> Kernel:
        return Kernel.LoG(sigma=sigma, h=h)

    @classmethod
    def DGauss(cls, sigma: float, h: int | None = None) -> Kernel:
        return Kernel.DGauss(sigma=sigma, h=h)

    @staticmethod
    def _bordertype_cv(
        border: str, exclude: list[str] | tuple[str, ...] | None = None
    ) -> int:
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
            "constant": cv2.BORDER_CONSTANT,
            "replicate": cv2.BORDER_REPLICATE,
            "reflect": cv2.BORDER_REFLECT,
            "mirror": cv2.BORDER_REFLECT_101,
            "reflect_101": cv2.BORDER_REFLECT_101,
            "wrap": cv2.BORDER_WRAP,
            "pad": cv2.BORDER_CONSTANT,
            "none": cv2.BORDER_ISOLATED,
        }
        if exclude is not None and border in exclude:
            raise ValueError("border option not supported")

        try:
            return border_opt[border]
        except KeyError:
            raise ValueError(
                f"border must be one of {list(border_opt)}, got {border!r}"
            )

    @staticmethod
    def _bordertype_sp(
        border: str, exclude: list[str] | tuple[str, ...] | None = None
    ) -> str:
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
            "constant": "constant",
            "replicate": "nearest",
            "reflect": "reflect",
            "mirror": "mirror",
            "wrap": "wrap",
        }
        if exclude is not None and border in exclude:
            raise ValueError("border option not supported")

        try:
            return border_opt[border]
        except KeyError:
            raise ValueError(
                f"border must be one of {list(border_opt)}, got {border!r}"
            )

    def smooth(
        self,
        sigma: float = 0,
        h: int | None = None,
        mode: str = "same",
        border: str = "reflect",
        bordervalue: int | float = 0,
    ) -> Any:
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
        to :math:`2 \mbox{ceil}(3 \sigma) + 1`.  If ``sigma`` is not given it is computed from ``h`` using the rule of thumb that
        :math:`\sigma = 0.3 \mathtt{h} + 0.8` which ensures that most of the Gaussian is
        contained within the window.

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
            - |RVC3|, Section 11.5.1.

        :seealso: :meth:`machinevisiontoolbox.Kernel.Gauss` :meth:`convolve`
        """

        if not argcheck.isscalar(sigma):
            raise ValueError(f"sigma must be a scalar, got {sigma!r}")

        # make the smoothing kernel
        K = Kernel.Gauss(sigma, h)

        return self.convolve(K, mode=mode, border=border, bordervalue=bordervalue)

    def convolve(
        self,
        K: Kernel | np.ndarray | Any,
        mode: str = "same",
        border: str = "reflect",
        bordervalue: int | float = 0,
    ) -> Any:
        """
        Image convolution

        :param K: convolution kernel
        :type K: :class:`~machinevisiontoolbox.Kernel.Kernel` or ndarray, optional
        :param mode: option for convolution, defaults to 'same'
        :type mode: str, optional
        :param border: option for boundary handling, defaults to 'reflect'
        :type border: str, optional
        :param bordervalue: padding value, defaults to 0
        :type bordervalue: scalar, optional
        :return: input image convolved with kernel
        :rtype:
            :class:`.Image`

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

        .. warning:: treats NaNs in input image or kernels as standard IEEE-754 floating-point
            values, meaning any convolution operation involving a NaN results in NaN.
            Because convolution is a sum of products, a single NaN within the kernel's
            window propagates to make the output pixel NaN.  Use :meth:`fixbad` for handling NaN values.

        :references:
            - |RVC3|, Section 11.5.1.

        .. important:: Uses OpenCV functions ``cv2.filter2D`` and ``cv2.copyMakeBorder`` which accept multiple-channel, CV_8U, CV_16U, CV_16S, CV_32F or CV_64F images.

        :seealso:
            :class:`~machinevisiontoolbox.Kernel.Kernel`
            :meth:`~smooth`
            `cv2.filter2D <https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04>`_
            `cv2.copyMakeBorder <https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga2ac1049c2c3dd25c2b41bffe17658a36>`_
            :meth:`.fixbad`
        """

        if isinstance(K, self.__class__):
            # kernel is an Image instance
            K = K._A
        elif isinstance(K, Kernel):
            # kernel is a numpy array
            K = K.K

        K = argcheck.getmatrix(K, shape=[None, None], dtype="float32")

        # OpenCV does correlation, not convolution, so we flip the kernel
        # to compensate.  Flip horizontally and vertically.
        K = np.flip(K)
        kh, kw = K.shape
        kh //= 2
        kw //= 2

        # TODO check images are of the same type

        # TODO check opt is valid string based on conv2 options
        modeopt = ["valid", "same", "full"]

        if mode not in modeopt:
            raise ValueError(f"mode must be one of {modeopt}, got {mode!r}")

        self._opencv_type_check(
            self._A, "multiple-channel", "CV_8U", "CV_16U", "CV_16S", "CV_32F", "CV_64F"
        )
        img = self._A
        crop = False
        if border == "pad" and bordervalue != 0 and mode == "same":
            # OpenCV can only pad with zero, so pad explicitly, filter, then
            # crop back to the original size
            img = cv2.copyMakeBorder(
                src=img,
                top=kh,
                bottom=kh,
                left=kw,
                right=kw,
                borderType=cv2.BORDER_CONSTANT,
                value=bordervalue,
            )
            crop = True
        elif mode == "full":
            img = cv2.copyMakeBorder(
                src=img,
                top=kh,
                bottom=kh,
                left=kw,
                right=kw,
                borderType=cv2.BORDER_CONSTANT,
                value=bordervalue,
            )

        out = cv2.filter2D(
            src=img, ddepth=-1, kernel=K, borderType=self._bordertype_cv(border)
        )

        if mode == "valid" or crop:
            if out.ndim == 2:
                out = out[kh : out.shape[0] - kh, kw : out.shape[1] - kw]
            else:
                out = out[kh : out.shape[0] - kh, kw : out.shape[1] - kw, :]
        return self.__class__(out, colororder=self.colororder)

    # def sobel(self, kernel=None):
    #     if kernel is None:
    #         kernel = Kernel.Sobel()

    #     Iu = self.convolve(kernel)
    #     Iv = self.convolve(kernel.T)
    #     return Iu, Iv

    def gradients(
        self,
        K: Kernel | np.ndarray | None = None,
        mode: str = "same",
        border: str = "reflect",
        bordervalue: int | float = 0,
    ) -> tuple[Any, Any]:
        """
        Compute horizontal and vertical gradients

        :param K: derivative kernel, defaults to Sobel
        :type K: :class:`~machinevisiontoolbox.Kernel.Kernel` or ndarray, optional
        :param mode: option for convolution, see :meth:`convolve`, defaults to 'same'
        :type mode: str, optional
        :param border: option for boundary handling, see :meth:`convolve`, defaults to 'reflect'
        :type border: str, optional
        :param bordervalue: padding value, , see :meth:`convolve`, defaults to 0
        :type bordervalue: scalar, optional
        :return: horizontal and vertical gradient images
        :rtype: 2-tuple of :class:`Image`

        Compute horizontal and vertical gradient images.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('monalisa.png', grey=True)
            >>> Iu, Iv = img.gradients()

        :references:
            - |RVC3|, Section 11.5.1.3.

        :seealso: :class:`~machinevisiontoolbox.Kernel.Kernel`
        """
        if K is None:
            K = Kernel.Sobel()

        Iu = self.convolve(K, mode=mode, border=border, bordervalue=bordervalue)
        Iv = self.convolve(K.T, mode=mode, border=border, bordervalue=bordervalue)
        return Iu, Iv

    def direction(
        horizontal: Any, vertical: Any
    ) -> Any:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
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
            - |RVC3|, Section 11.5.1.3.

        :seealso: :meth:`gradients`
        """
        if horizontal.shape != vertical.shape:
            raise ValueError("images must the same shape")
        return horizontal.__class__(np.arctan2(vertical._A, horizontal._A))

    def Harris_corner_strength(self, k: float = 0.04, h: int = 2) -> Any:
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
            - |RVC3|, Section 12.3.1.

        .. important:: Uses OpenCV function ``cv2.cornerHarris`` which accepts single-channel, CV_8U, CV_32F or CV_64F images (color images are automatically converted to greyscale).

        :seealso:
            :meth:`gradients`
            :meth:`Harris`
            `cv2.cornerHarris <https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345>`_
        """
        self._opencv_type_check(
            self._A, "multiple-channel", "CV_8U", "CV_32F", "CV_64F"
        )
        dst = cv2.cornerHarris(src=self.mono()._A, blockSize=2, ksize=2 * h + 1, k=k)
        return self.__class__(dst)

    def window(
        self,
        func: Callable[..., Any],
        h: int | None = None,
        se: np.ndarray | None = None,
        border: str = "reflect",
        bordervalue: int | float = 0,
        **kwargs: Any,
    ) -> Any:
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
            - |RVC3|, Section 11.5.3.

        :seealso: :func:`scipy.ndimage.generic_filter`
        """
        # replace window's mex function with scipy's ndimage.generic_filter
        if self.iscolor:
            raise ValueError("single channel images only")

        if not callable(func):
            raise TypeError(f"func not callable, got {func!r}")

        if h is not None and se is None:
            w = 2 * h + 1
            se = np.ones((w, w))

        out = sp.ndimage.generic_filter(
            self._A,
            func,
            footprint=se,
            mode=self._bordertype_sp(border),
            cval=bordervalue,
        )
        return self.__class__(out)

    def zerocross(self) -> Any:
        """
        Compute zero crossing

        :return: boolean image
        :rtype: :class:`Image` instance

        Compute a zero-crossing image, where pixels are true if they are adjacent to
        a change in sign.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image, meshgrid
            >>> U, V = meshgrid(6, 6)
            >>> img = Image(U - V - 2, dtype='float')
            >>> img.print()
            >>> img.zerocross().print()

        .. note:: Uses morphological filtering with 3x3 structuring element, which can
            lead to erroneous values in border pixels.

        :references:
            - |RVC3|, Section 11.5.1.3.

        :seealso:
            :meth:`Laplace`
            :meth:`LoG`
        """
        min = cv2.morphologyEx(self._A, cv2.MORPH_ERODE, np.ones((3, 3)))
        max = cv2.morphologyEx(self._A, cv2.MORPH_DILATE, np.ones((3, 3)))
        zeroCross = np.logical_or(
            np.logical_and(min < 0, self._A > 0),
            np.logical_and(max > 0, self._A < 0),
        )
        return self.__class__(zeroCross)

    def scalespace(
        self, n: int, sigma: int | float = 1
    ) -> tuple[list[Any], list[Any], list[float]]:
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

        Examples:

            .. code-block:: python

                mona = Image.Read("monalisa.png", dtype="float");
                G, L, scales = mona.scalespace(8, sigma=8);


        .. note:: The two image sequences have the same length, the original image is
            not included in the list of smoothed images.

        :references:
            - |RVC3|, Section 12.3.2.

        :seealso:
            :meth:`pyramid`
            :meth:`smooth`
            :class:`~.Gauss`
            :class:`~.LoG`
        """
        im = self.copy()
        g = [im]
        scale = 0.5
        scales = [scale]
        lap = []

        for i in range(n - 1):
            im = im.smooth(sigma)
            scale = np.sqrt(scale**2 + sigma**2)
            scales.append(scale)
            g.append(im)
            x = (g[-1] - g[-2]) * scale**2
            lap.append(x)

        return g, lap, scales

    def pyramid(
        self,
        sigma: int | float = 1,
        N: int | None = None,
        border: str = "replicate",
        bordervalue: int | float = 0,
    ) -> list[Any]:
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
            - |RVC3|, Section 12.3.2.

        .. important:: Uses OpenCV function ``cv2.pyrDown`` which accepts single-channel, CV_8U, CV_16U, CV_16S, CV_32F or CV_64F images (color images are automatically converted to greyscale).

        :seealso:
            :meth:`smooth`
            :meth:`scalespace`
            `cv2.pyrDown <https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaf9bba239dfca11654cb7f50f889fc2ff>`_
        """

        # check inputs, greyscale only
        im = self.mono()
        self._opencv_type_check(
            im._A, "single-channel", "CV_8U", "CV_16U", "CV_16S", "CV_32F", "CV_64F"
        )

        if not argcheck.isscalar(sigma):
            raise ValueError(f"sigma must be a scalar, got {sigma!r}")

        if N is None:
            N = max(im.shape)
        else:
            if (not argcheck.isscalar(N)) and (N >= 0) and (N <= max(im.shape)):
                raise ValueError(
                    f"N must be a scalar and 0 <= N <= max(im.shape), got {N!r}",
                )

        # TODO options to accept different border types,
        # note that the Matlab implementation is hard-coded to 'same'

        # return cv2.buildPyramid(im, N, borderType=cv2.BORDER_REPLICATE)
        # Python version does not seem to be implemented

        # list comprehension approach
        # TODO pyr = [cv2.pyrdown(inputs(i)) for i in range(N) if conditional]

        impyr = im._A
        pyr = [impyr]
        for i in range(N):
            if impyr.shape[0] == 1 or impyr.shape[1] == 1:
                break
            impyr = cv2.pyrDown(
                src=impyr, borderType=self._bordertype_cv(border, exclude=("constant",))
            )
            pyr.append(impyr)

        # output list of Image objects
        pyrimlist = [self.__class__(p) for p in pyr]
        return pyrimlist

    @overload
    def canny(
        self,
        sigma: float | None = ...,
        th0: float | None = ...,
        th1: float | None = ...,
        percentile: float = ...,
        k: float = ...,
        hist: Literal[False] = ...,
    ) -> Any:
        ...

    @overload
    def canny(
        self,
        sigma: float | None = ...,
        th0: float | None = ...,
        th1: float | None = ...,
        percentile: float = ...,
        k: float = ...,
        *,
        hist: Literal[True],
    ) -> "tuple[Any, CannyHistogram]":
        ...

    def canny(
        self,
        sigma: float | None = None,
        th0: float | None = None,
        th1: float | None = None,
        percentile: float = 90,
        k: float = 0.5,
        hist: bool = False,
    ) -> Any:
        r"""
        Canny edge detection

        :param sigma: standard deviation of a Gaussian used to smooth the image
            before edge detection, defaults to None (no smoothing)
        :param th0: lower hysteresis threshold, defaults to None
        :param th1: upper hysteresis threshold, defaults to None
        :param percentile: percentile of the gradient magnitude used as the upper
            threshold when no threshold is given, defaults to 90
        :param k: ratio of the lower threshold to the upper threshold, defaults to 0.5
        :param hist: also return a histogram of the gradient magnitude, defaults to False
        :raises ValueError: a threshold is given as well as a non-default ``percentile``
        :return: edge image, or edge image and gradient histogram if ``hist`` is True

        Computes an edge image obtained using the Canny edge detector algorithm.
        Hysteresis filtering is applied to the gradient image: edge pixels >
        ``th1`` are connected to adjacent pixels > ``th0``, those below ``th0``
        are set to zero.

        The thresholds apply to the gradient magnitude computed by OpenCV, which
        is :math:`|G_x| + |G_y|` where :math:`G_x` and :math:`G_y` are the
        outputs of the 3x3 Sobel kernels applied to the image after conversion
        to ``uint8``.  This is not normalized, so a step of height 255 gives a
        gradient of 1020.

        The thresholds are set in one of two ways:

        1. Explicitly.  ``th0`` and ``th1`` are passed to OpenCV.  If only one is
           given the other is set using ``k``: ``th0 = k * th1`` or
           ``th1 = th0 / k``.
        2. Automatically, if neither is given.  The upper threshold is the
           ``percentile`` of the gradient magnitude of the (optionally smoothed)
           image, and the lower threshold is ``k`` times that.  A percentile
           of 80 to 90 is typical.

        With ``hist=True`` the gradient magnitude, which is the quantity that
        is thresholded, is also returned as a :class:`CannyHistogram`.  Its
        attributes ``th0`` and ``th1`` are the thresholds that were used, and
        its ``plot`` method shows them on the histogram.  This helps in
        choosing thresholds.

        .. rubric:: Examples

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('monalisa.png')
            >>> edges = img.canny()
            >>> edges = img.canny(sigma=2, percentile=80)
            >>> edges = img.canny(th0=100, th1=300)
            >>> edges, h = img.canny(hist=True)
            >>> h.th0, h.th1

        .. note::
            - Produces a zero image with single pixel wide edges having
              non-zero values.
            - A color image is automatically converted to greyscale first.

        .. versionchanged:: 2.5.0
            ``sigma``, ``th0`` and ``th1`` now take effect.  Previously they
            were ignored and the thresholds were always set from the mean of
            the image.  ``sigma`` now defaults to None, no smoothing.  The
            automatic thresholds are now set from the ``percentile`` of the
            gradient magnitude and ``k``.  The arguments ``percentile``,
            ``k`` and ``hist`` are new.

        :references:
            - "A Computational Approach To Edge Detection", J. Canny,
              IEEE Trans. Pattern Analysis and Machine Intelligence,
              8(6):679–698, 1986.
            - |RVC3|, Section 11.5.1.3.

        .. seealso:: :meth:`smooth` :meth:`~machinevisiontoolbox.ImageWholeFeatures.Histogram`
        """
        if not 0 < percentile < 100:
            raise ValueError("percentile must be in the range (0, 100)")
        if not 0 < k <= 1:
            raise ValueError("k must be in the range (0, 1]")
        explicit = th0 is not None or th1 is not None
        if explicit and percentile != 90:
            raise ValueError("percentile cannot be used when a threshold is given")

        # convert to greyscale, and smooth if requested
        img = self.mono()
        if sigma is not None and sigma > 0:
            img = img.smooth(sigma=sigma)
        gray = img.array_as("uint8")

        # We call cv2.Sobel rather than gradients() deliberately.  The thresholds
        # are compared by cv2.Canny with its own gradient magnitude, which is the
        # unnormalized 3x3 Sobel with the L1 norm |Gx| + |Gy| (apertureSize=3,
        # L2gradient=False), 8 times the gain of Kernel.Sobel().  Using the same
        # calculation means that a percentile of this gradient, and the
        # thresholds shown in the histogram, are exactly the values that Canny
        # applies.  gradients() would differ in scale, in border handling
        # (reflect versus OpenCV's reflect_101), and on a uint8 image in its
        # output type.
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.abs(gx) + np.abs(gy)

        if explicit:
            if th0 is None:
                th0 = k * th1  # type: ignore[operator]
            elif th1 is None:
                th1 = th0 / k
        else:
            th1 = float(np.percentile(grad, percentile))
            th0 = k * th1

        out = cv2.Canny(
            image=gray,
            threshold1=float(th0),
            threshold2=float(th1),  # type: ignore[arg-type]
            L2gradient=False,
        )
        edges = self.__class__(out)

        if hist:
            from machinevisiontoolbox.ImageWholeFeatures import CannyHistogram

            return edges, CannyHistogram(self.__class__(grad), float(th0), float(th1))  # type: ignore[arg-type]
        return edges

    def rankfilter(
        self,
        footprint: np.ndarray | None = None,
        h: int | None = None,
        rank: int | str = "median",
        border: str = "replicate",
        bordervalue: int | float = 0,
    ) -> Any:
        r"""
        Rank filter

        :param footprint: filter footprint or structuring element
        :type footprint: ndarray(N,M), optional
        :param h: half width of structuring element
        :type h: int, optional
        :param rank: rank of filter, defaults to ``"median"``
        :type rank: int, str, optional
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
            >>> img.rankfilter(h=1, rank=0).print()  # maximum filter
            >>> img.rankfilter(h=1, rank=8).print()  # minimum filter
            >>> img.rankfilter(h=1, rank=4).print()  # median filter
            >>> img.rankfilter(h=1, rank='median').print()  # median filter

        .. note::
            - The footprint should have an odd side length.
            - The input can be logical, uint8, uint16, float or double, the
              output is always double.

        :references:
            - |RVC3|, Section 11.5.3.

        .. versionchanged:: 2.5.0
            ``rank`` now defaults to ``"median"``.  The previous default of -1
            was always rejected, so ``rank`` had to be given.

        :seealso: :obj:`scipy.ndimage.rank_filter`
        """
        if h is not None:
            w = 2 * h + 1
            footprint = np.ones((w, w))

        if footprint is None:
            raise ValueError("either footprint or h must be given")

        n = int(np.sum(footprint))

        if isinstance(rank, str):
            if rank in ("min", "minimum"):
                rank = n - 1
            elif rank in ("max", "maximum"):
                rank = 0
            elif rank in ("med", "median"):
                rank = n // 2
            else:
                raise ValueError(f"unknown rank name, got {rank!r}")
        elif not isinstance(rank, int):
            raise TypeError(f"rank must be int or str, got {rank!r}")

        if rank < 0:
            raise ValueError("rank must be >= 0")

        r = int(footprint.sum() - rank - 1)

        out = []
        for plane in self.planes():
            out.append(
                sp.ndimage.rank_filter(
                    plane._A, r, footprint=footprint, mode=self._bordertype_sp(border)
                )
            )
        if self.iscolor:
            return self.__class__(np.dstack(out), colororder=self.colororder)
        return self.__class__(out[0], colororder=self.colororder)

    def rank(
        self,
        footprint: np.ndarray | None = None,
        h: int | None = None,
        rank: int | str = "median",
        border: str = "replicate",
        bordervalue: int | float = 0,
    ) -> Any:
        """Deprecated wrapper for :meth:`rankfilter`.

        .. deprecated:: 2.0.0
            Use :meth:`rankfilter` instead.
        """
        warnings.warn(
            "Deprecated in 2.0.0: use rankfilter() instead of rank().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.rankfilter(
            footprint=footprint,
            h=h,
            rank=rank,
            border=border,
            bordervalue=bordervalue,
        )

    def medianfilter(self, h: int = 1, **kwargs: Any) -> Any:
        r"""
        Median filter

        :param h: half width of structuring element, defaults to 1
        :type h: int, optional
        :param kwargs: options passed to :meth:`rankfilter`
        :return: median filtered image
        :rtype: :class:`Image` instance

        Return the median filtered image.  For every :math:`w \times w, w=2h+1`
        window take the median value as the output pixel value.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> array = np.arange(25).reshape((5,5))
            >>> array[2, 2] = 0
            >>> img = Image(array)
            >>> img.print()
            >>> img.medianfilter(h=1).print()  # median filter
            >>> img = Image.Read('monalisa.png')
            >>> img.medianfilter(h=5).disp()  # ameliorate background cracking

        .. note:: This filter is effective for removing impulse (aka
            salt and pepper) noise.

        :references:
            - |RVC3|, Section 11.5.3.

        :seealso: :meth:`rankfilter`
        """
        w = 2 * h + 1
        r = int((w**2 - 1) / 2)
        return self.rankfilter(h=h, rank=r, **kwargs)

    def distance_transform(
        self, invert: bool = False, norm: str = "L2", h: int = 1
    ) -> Any:
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
            >>> pixels = np.zeros((5,5), dtype=np.uint8)
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
            - |RVC3|, Section 11.6.4.

        .. important:: Uses OpenCV function ``cv2.distanceTransform`` which accepts single-channel, CV_8U images.

        :seealso: `opencv.distanceTransform <https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga8a0b7fdfcb7a13dde018988ba3a43042>`_
        """
        self._opencv_type_check(self._A, "single-channel", "CV_8U")
        # OpenCV does distance to nearest zero pixel
        # this function does distance to nearest non-zero pixel by default,
        # and the OpenCV thing if invert=True
        if invert:
            # distance to nearest zero pixel
            im = self.array_as("uint8")
        else:
            # distance to nearest non-zero pixel, invert the image
            im = self.invert().array_as("uint8")

        normdict = {
            "L1": cv2.DIST_L1,
            "L2": cv2.DIST_L2,
        }

        out = cv2.distanceTransform(
            src=im, distanceType=normdict[norm], maskSize=2 * h + 1
        )
        return self.__class__(out)

    # ======================= labels ============================= #

    def labels_binary(
        self, connectivity: int = 4, ltype: str = "int32"
    ) -> tuple[Any, int]:
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
            >>> img = Image.Squares(2, size=15)
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
            - |RVC3|, Section 12.1.2.1.

        .. important:: Uses OpenCV function ``cv2.connectedComponents`` which accepts single-channel, CV_8U images.

        :seealso:
            :meth:`blobs`
            `cv2.connectedComponents <https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gaedef8c7340499ca391d459122e51bef5>`_
            :meth:`labels_graphseg`
            :meth:`labels_MSER`
        """
        if not (connectivity in [4, 8]):
            raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")

        self._opencv_type_check(self._A, "single-channel", "CV_8U")
        # make labels uint32s, unique and never recycled?
        # set ltype to default to cv2.CV_32S
        if ltype == "int32":
            cvtype = cv2.CV_32S
        elif ltype == "uint16":
            cvtype = cv2.CV_16U
        else:
            raise TypeError(f"ltype must be either int32 or uint16, got {ltype!r}")

        retval, labels = cv2.connectedComponents(
            image=self.array_as("uint8"), connectivity=connectivity, ltype=cvtype
        )
        return self.__class__(labels), retval

    def labels_MSER(self, **kwargs: Any) -> tuple[Any, int]:
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
            >>> img = Image.Squares(2, size=15)
            >>> img.print()
            >>> labels, N = img.labels_MSER()
            >>> N
            >>> labels.print()

        :references:
            - Linear time maximally stable extremal regions,
              David Nistér and Henrik Stewénius,
              In Computer Vision–ECCV 2008, pages 183–196. Springer, 2008.
            - |RVC3|, Section 12.1.2.2.

        :seealso:
            :meth:`labels_binary`
            :meth:`labels_graphseg`
            :meth:`blobs`
            `opencv.MSER_create <https://docs.opencv.org/4.x/d3/d28/classcv_1_1MSER.html>`_
        """

        mser = cv2.MSER.create(**kwargs)
        regions, _ = mser.detectRegions(self.array_as("uint8"))

        dtype: Any = np.uint8 if len(regions) < 256 else np.uint32
        out = np.zeros(self.shape, dtype=dtype)

        for i, points in enumerate(regions):
            # print('region ', i, points.shape[0])
            pts = np.asarray(points)
            out[pts[:, 1], pts[:, 0]] = i

        return self.__class__(out, dtype=dtype), len(regions)

    def labels_graphseg(
        self, sigma: float = 0.5, k: int = 2000, minsize: int = 100
    ) -> tuple[Any, int]:
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
            - |RVC3|, Section 12.1.2.2.

        :seealso:
            :meth:`labels_binary`
            :meth:`labels_MSER`
            :meth:`blobs`
            `opencv.createGraphSegmentation <https://docs.opencv.org/4.x/d5/df0/group__ximgproc__segmentation.html#ga5e3e721c5f16e34d3ad52b9eeb6d2860>`_
        """
        # P. Felzenszwalb, D. Huttenlocher: "Graph-Based Image Segmentation
        segmenter = cv2.ximgproc.segmentation.createGraphSegmentation(
            sigma=0.5, k=2000, min_size=100
        )
        out = segmenter.processImage(self.array_as("uint8"))

        return self.__class__(out), int(np.max(out)) + 1

    # -------------------- similarity operations -------------------------- #

    def sad(
        image1: Any, image2: Any
    ) -> float:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
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
            - |RVC3|, Section 11.5.2.

        :seealso:
            :meth:`zsad`
            :meth:`ssd`
            :meth:`ncc`
        """

        if not np.all(image1.shape == image2.shape):
            raise ValueError("image2 shape is not equal to image1")

        # out = []
        # for im in self:
        # m = np.abs(im._A - image2._A)
        # out.append(np.sum(m))
        # convert first: subtracting uint8 arrays would wrap around at 256
        m = np.abs(image1._A.astype(float) - image2._A)
        out = np.sum(m)
        return out

    def ssd(
        image1: Any, image2: Any
    ) -> float:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
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
            - |RVC3|, Section 11.5.2.

        :seealso:
            :meth:`zssd`
            :meth:`sad`
            :meth:`ncc`
        """

        if not np.all(image1.shape == image2.shape):
            raise ValueError("image2 shape is not equal to image1")
        # convert first: subtracting uint8 arrays would wrap around at 256
        m = np.power(image1._A.astype(float) - image2._A, 2)
        return np.sum(m)

    def ncc(
        image1: Any, image2: Any
    ) -> float:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
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
            - |RVC3|, Section 11.5.2.

        :seealso:
            :meth:`zncc`
            :meth:`sad`
            :meth:`ssd`
        """
        if not np.all(image1.shape == image2.shape):
            raise ValueError("image2 shape is not equal to image1")

        # convert first: squaring uint8 arrays would wrap around at 256
        a = image1._A.astype(float)
        b = image2._A.astype(float)
        denom = np.sqrt(np.sum(a**2) * np.sum(b**2))

        if denom < 1e-10:
            return 0
        else:
            return np.sum(a * b) / denom

    def zsad(
        image1: Any, image2: Any
    ) -> float:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
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
            - |RVC3|, Section 11.5.2.

        :seealso:
            :meth:`zsad`
            :meth:`ssd`
            :meth:`ncc`
        """
        if not np.all(image1.shape == image2.shape):
            raise ValueError("image2 shape is not equal to image1")

        image1 = image1._A - np.mean(image1._A)
        image2 = image2._A - np.mean(image2._A)
        m = np.abs(image1 - image2)
        return np.sum(m)

    def zssd(
        image1: Any, image2: Any
    ) -> float:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
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
            - |RVC3|, Section 11.5.2.

        :seealso:
            :meth:`zssd`
            :meth:`sad`
            :meth:`ncc`
        """

        if not np.all(image1.shape == image2.shape):
            raise ValueError("image2 shape is not equal to image1")

        image1 = image1._A - np.mean(image1._A)
        image2 = image2._A - np.mean(image2._A)
        m = np.power(image1 - image2, 2)
        return np.sum(m)

    def zncc(
        image1: Any, image2: Any
    ) -> float:  # lgtm[py/not-named-self] pylint: disable=no-self-argument
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
            - |RVC3|, Section 11.5.2.

        :seealso:
            :meth:`zncc`
            :meth:`sad`
            :meth:`ssd`
        """

        if not np.all(image1.shape == image2.shape):
            raise ValueError("image2 shape is not equal to image1")

        image1 = image1._A - np.mean(image1._A)
        image2 = image2._A - np.mean(image2._A)
        denom = np.sqrt(np.sum(np.power(image1, 2) * np.sum(np.power(image2, 2))))

        if denom < 1e-10:
            return 0
        else:
            return np.sum(image1 * image2) / denom

    def similarity(self, T: Any, metric: str = "zncc") -> Any:
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
        corresponding input pixel in image.

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
            - |RVC3|, Section 11.5.2.

        .. important:: Uses OpenCV function ``cv2.matchTemplate`` which accepts multiple-channel, CV_8U or CV_32F images.

        :seealso: `cv2.matchTemplate <https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#ga586ebfb0a7fb604b35a23d85391329be>`_
        """

        # check inputs
        if ((T.shape[0] % 2) == 0) or ((T.shape[1] % 2) == 0):
            raise ValueError("template T must have odd dimensions")

        self._opencv_type_check(self._A, "multiple-channel", "CV_8U", "CV_32F")
        metricdict = {
            "ssd": cv2.TM_SQDIFF,
            "zssd": cv2.TM_SQDIFF,
            "ncc": cv2.TM_CCOEFF_NORMED,
            "zncc": cv2.TM_CCOEFF_NORMED,
        }

        im = self._A
        T_im = T._A
        if metric[0] == "z":
            # remove the offset from the template and from the image, into new
            # float32 arrays: the template belongs to the caller, and an 8-bit
            # image cannot be changed in place by a float mean
            T_im = T_im.astype(np.float32) - np.mean(T_im, dtype=np.float32)
            im = im.astype(np.float32) - np.mean(im, dtype=np.float32)

        try:
            out = cv2.matchTemplate(image=im, templ=T_im, method=metricdict[metric])
        except KeyError:
            raise ValueError("bad metric specified")
        return self.__class__(out)


if __name__ == "__main__":
    from pathlib import Path

    import pytest

    pytest.main(
        [
            str(
                Path(__file__).parent.parent.parent / "tests" / "test_image_spatial.py"
            ),
            "-v",
        ]
    )
