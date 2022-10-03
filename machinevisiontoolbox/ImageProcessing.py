#!/usr/bin/env python

from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.arraysetops import isin
import scipy as sp
from scipy import interpolate
import cv2 as cv
from pathlib import Path
import os.path
from spatialmath.base import argcheck, getvector, e2h, h2e, transl2
from machinevisiontoolbox.base import iread, iwrite, colorname, \
    int_image, float_image, idisp, sphere_rotate, name2color

class ImageProcessingMixin:

    # ======================= image processing ============================= #

    def LUT(self, lut, colororder=None):
        """
        Apply lookup table

        :param lut: lookup table
        :type lut: array_like(256), ndarray(256,N)
        :param colororder: colororder for output image, optional
        :type colororder: str or dict
        :return: transformed image
        :rtype: :class:`Image`

        For a greyscale image the LUT can be:

            - (256,)
            - (256,N) in which case the resulting image has ``N`` planes created
              my applying the I'th column of the LUT to the input image
  
        For a color image the LUT can be:

            - (256,) and applied to every plane, or
            - (256,N) where the LUT columns are applied to the ``N`` planes of
              the input image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> img = Image([[100, 150], [200, 250]])
            >>> img.LUT(np.arange(255, -1, -1, dtype='uint8')).A

        .. note:: Works only for ``uint8`` and ``int8`` image and LUT.

        :references:
            - Robotics, Vision & Control for Python, Section 11.3, P. Corke, Springer 2023.

        :seealso: `cv2.LUT <https://docs.opencv.org/master/d2/de8/group__core__array.html#gab55b8d062b7f5587720ede032d34156f>`_
        """
        image = self.to_int()
        lut = np.array(lut).astype(np.uint8)
        if lut.ndim == 2:
            lut = lut[np.newaxis, ...]
            if self.nplanes == 1:
                image = np.dstack((image,) * lut.shape[2])

        out = cv.LUT(image, lut)
        if colororder is None:
            colororder = self.colororder

        return self.__class__(self.like(out), colororder=colororder)

    def apply(self, func, vectorize=False):
        """
        Apply a function to an image

        :param func: function to apply to image or pixel
        :type func: callable
        :return: transformed image
        :rtype: :class:`Image`

        If ``vectorize`` is False the function is called with a single argument
        which is the underlying NumPy array, and it must return a NumPy array.
        The return array can have different dimensions to its argument.

        If ``vectorize`` is True the function is called for every pixel with a
        single argument which is a scalar or a 1d-array of length equal to the
        number of color planes. The return array will have the same dimensions
        to its argument.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> import math
            >>> img = Image([[1, 2], [3, 4]])
            >>> img.apply(np.sqrt).image
            >>> img.apply(lambda x: math.sqrt(x), vectorize=True).image

        .. note:: Slow when ``vectorize=True`` which involves a large number
            of calls to ``func``.

        :references:
            - Robotics, Vision & Control for Python, Section 11.3, P. Corke, Springer 2023.

        :seealso: :meth:`apply2`
        """
        if vectorize:
            func = np.vectorize(func)
        return self.__class__(func(self.A), colororder=self.colororder)

    def apply2(self, other, func, vectorize=False):
        """
        Apply a function to two images

        :param func: function to apply to image or pixel
        :type func: callable
        :raises ValueError: images must have same size
        :return: transformed image
        :rtype: :class:`Image`

        If ``vectorize`` is False the function is called with two arguments
        which are the underlying NumPy arrays, and it must return a NumPy array.
        The return array can have different dimensions to its arguments.

        If ``vectorize`` is True the function is called for every pixel in both
        images with two arguments which are the corresponding pixel values as a
        scalar or 1d-array of length equal to the number of color planes. The
        function returns a scalar or a 1d-array. The return array will have the
        same dimensions to its argument.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> import math
            >>> img1 = Image([[1, 2], [3, 4]])
            >>> img2 = Image([[5, 6], [7, 8]])
            >>> img1.apply2(img2, np.hypot).image
            >>> img1.apply2(img2, lambda x, y: math.hypot(x,y), vectorize=True).image

        .. note:: Slow when ``vectorize=True`` which involves a large number
            of calls to ``func``.

        :references:
            - Robotics, Vision & Control for Python, Section 11.4, P. Corke, Springer 2023.

        :seealso: :meth:`apply`
        """
        if self.size != other.size:
            raise ValueError('two images must have same size')
        if vectorize:
            func = np.vectorize(func)
        return self.__class__(func(self.A, other.A), colororder=self.colororder)

    def clip(self, min, max):
        """
        Clip pixel values

        :param min: minimum value
        :type min: int or float
        :param max: maximum value
        :type max: int or float
        :return: transformed image
        :rtype: :class:`Image`

        Pixels in the returned image will have values in the range [``min``, ``max``]
        inclusive.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> img.clip(2, 3).A

        :seealso: :func:`numpy.clip`
        """
        return self.__class__(np.clip(self.A, min, max), colororder=self.colororder)


    def roll(self, ru=0, rv=0):
        """
        Roll image by row or column

        :param ru: roll in the column direction, defaults to 0
        :type ru: int, optional
        :param rv: roll in the row direction, defaults to 0
        :type rv: int, optional
        :return: rolled image
        :rtype: :class:`Image` instance

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> img.roll(ru=1).A
            >>> img.roll(ru=-1).A
            >>> img.roll(rv=1).A
            >>> img.roll(ru=1, rv=-1).A

        :seealso: :func:`numpy.roll`
        """
        return self.__class__(np.roll(self.image, (ru, rv), (1, 0)))

    def normhist(self):
        """
        Histogram normalisaton

        :return: normalised image
        :rtype: :class:`Image` instance

        Return a histogram normalized version of the image which highlights
        image detail in low-contrast areas of an image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[10, 20, 30], [40, 41, 42], [70, 80, 90]])
            >>> img.normhist().A

        .. note::
            - The histogram of the normalized image is approximately uniform,
              that is, all grey levels ae equally likely to occur.
            - Color images automatically converted to grayscale

        :references:
            - Robotics, Vision & Control for Python, Section 11.3, P. Corke, Springer 2023.

        :seealso: `cv2.equalizeHist <https://docs.opencv.org/master/d6/dc7/group__imgproc__hist.html#ga7e54091f0c937d49bf84152a16f76d6e>`_
        """
        out = cv.equalizeHist(self.to_int())
        return self.__class__(self.like(out))

    def stretch(self, max=1, range=None, clip=True):
        """
        Image normalisation

        :param max: maximum value of output pixels, defaults to 1
        :type max: int, float, optional
        :param range: range[0] is mapped to 0, range[1] is mapped to max
        :type range: array_like(2), optional
        :param clip: clip pixel values to interval [0, max], defaults to True
        :type clip: bool, optional
        :return: Image with pixel values stretched to M across r
        :rtype: :class:`Image`

        Returns a normalised image in which all pixel values are linearly mapped
        to the interval of 0.0 to ``max``. That is, the minimum pixel value is
        mapped to 0 and the maximum pixel value is mapped to ``max``.  
        
        If ``range`` is specified then ``range[0]`` is mapped to 0.0 and
        ``range[1]`` is mapped to ``max``.  If ``clip`` is False then pixels
        less than ``range[0]`` will be mapped to a negative value and pixels
        greater than ``range[1]`` will be mapped to a value greater than
        ``max``.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> img.stretch().A

        :references:

            - Robotics, Vision & Control, Section 12.1, P. Corke,
              Springer 2011.
        """

        # TODO make all infinity values = None?

        im = self.A
        if range is None:
            mn = np.min(im)
            mx = np.max(im)
        else:
            range = argcheck.getvector(range)
            mn = range[0]
            mx = range[1]

        zs = (im - mn) / (mx - mn) * max

        if range is not None and clip:
            zs = np.maximum(0, np.minimum(max, zs))
        return self.__class__(zs)

    def thresh(self, t=None, opt='binary'):
        r"""
        Image threshold

        :param t: threshold value
        :type t: scalar, str
        :param option: threshold option, defaults to 'binary'
        :type option: str, optional
        :return: thresholded image
        :rtype: :class:`Image`

        Apply a threshold ``t`` to the image.  Various thresholding options are
        supported:

        ================  =====================================================================================================================
        Option             Function
        ================  =====================================================================================================================
        ``'binary'``      :math:`Y_{u,v} = \left\{ \begin{array}{l} m \mbox{, if } X_{u,v} > t \\ 0 \mbox{, otherwise} \end{array} \right.`
        ``'binary_inv'``  :math:`Y_{u,v} = \left\{ \begin{array}{l} 0 \mbox{, if } X_{u,v} > t \\ m \mbox{, otherwise} \end{array} \right.`
        ``'truncate'``    :math:`Y_{u,v} = \left\{ \begin{array}{l} t \mbox{, if } X_{u,v} > t \\ X_{u,v} \mbox{, otherwise} \end{array} \right.`
        ``'tozero'``      :math:`Y_{u,v} = \left\{ \begin{array}{l} X_{u,v} \mbox{, if } X_{u,v} > t \\ 0 \mbox{, otherwise} \end{array} \right.`
        ``'tozero_inv'``  :math:`Y_{u,v} = \left\{ \begin{array}{l} 0 \mbox{, if } X_{u,v} > t \\ X_{u,v} \mbox{, otherwise} \end{array} \right.`
        ================  =====================================================================================================================

        where :math:`m` is the maximum value of the image datatype.

        If threshold ``t`` is a string then the threshold is determined
        automatically:

        +---------------+-----------------------------------------------------+
        |threshold      | algorithm                                           |
        +===============+=====================================================+
        |``'otsu'``     | Otsu's method finds the threshold that minimizes    |
        |               | the within-class variance. This technique is        |
        |               | effective for a bimodal greyscale histogram.        |
        +---------------+-----------------------------------------------------|
        |``'triangle'`` | The triangle method constructs a line between the   |
        |               | histogram peak and the farthest end of the          |
        |               | histogram. The threshold is the point of maximum    |
        |               | distance between the line and the histogram. This   |
        |               | technique is effective when the object pixels       |
        |               | produce a weak peak in the histogram.               |
        +---------------+-----------------------------------------------------+

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> img.thresh(5).image

        .. note::

            - The threshold is applied to all color planes
            - If threshold is 'otsu' or 'triangle' the image must be greyscale,
              and the computed threshold is also returned.
              
        :references:
            - A Threshold Selection Method from Gray-Level Histograms, N. Otsu.
              IEEE Trans. Systems, Man and Cybernetics Vol SMC-9(1), Jan 1979,
              pp 62-66.
            - Automatic measurement of sister chromatid exchange frequency"
              Zack (Zack GW, Rogers WE, Latt SA (1977),
              J. Histochem. Cytochem. 25 (7): 741–53.
            - Robotics, Vision & Control for Python, Section 12.1.1, P. Corke, Springer 2023.


        :seealso: :meth:`ithresh` :meth:`adaptive_threshold` :meth:`otsu` `opencv.threshold <https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57>`_
        """

        # dictionary of threshold options from OpenCV
        options_dict = {
            'binary': cv.THRESH_BINARY,
            'binary_inv': cv.THRESH_BINARY_INV,
            'truncate': cv.THRESH_TRUNC,
            'tozero': cv.THRESH_TOZERO,
            'tozero_inv': cv.THRESH_TOZERO_INV,
        }
        threshold_dict = {
            'otsu': cv.THRESH_OTSU,
            'triangle': cv.THRESH_TRIANGLE
        }

        flag = options_dict[opt]
        if isinstance(t, str):
            # auto threshold requested
            flag |= threshold_dict[t]

            threshvalue, imt = cv.threshold(
                src=self.to_int(),
                thresh=0.0,
                maxval=self.maxval,
                type=flag)
            return self.__class__(self.like(imt)), self.like(int(threshvalue), maxint=255)

        elif argcheck.isscalar(t):
            # threshold is given
            _, imt = cv.threshold(
                src=self.image,
                thresh=t,
                maxval=self.maxval,
                type=flag)
            return self.__class__(imt)

        else:
            raise ValueError(t, 't must be a string or scalar')

    def ithresh(self):
        r"""
        Interactive thresholding

        :return: selected threshold value
        :rtype: scalar

        The image is displayed with a binary threshold displayed in a simple
        Matplotlib GUI along with the histogram and a slider for threshold
        value.  Adjusting the slider changes the thresholded image view.

        The displayed image is

        .. math:: Y_{u,v} = \left\{ \begin{array}{l} m \mbox{, if } X_{u,v} > t \\ 0 \mbox{, otherwise} \end{array} \right.

        :references:
            - Robotics, Vision & Control for Python, Section 12.1.1.1, P. Corke, Springer 2023.

        :seealso: :meth:`thresh` :meth:`adaptive_threshold` :meth:`otsu` `opencv.threshold <https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57>`_
        """

        # ACKNOWLEDGEMENT: https://matplotlib.org/devdocs/gallery/widgets/range_slider.html
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        from matplotlib import colors

        #N = 128
        Ncolors = 256
        img = self.image
        t = int((img.max() + img.min()) / 2)

        x = np.linspace(self.min(), self.max(), Ncolors)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plt.subplots_adjust(bottom=0.25)

        def colormap(t):
            
            X = np.tile(x > t, (3, 1)).T  # N x 3 colormap
            X = np.hstack([X, np.ones((Ncolors, 1))]) # N x 4
            return colors.LinearSegmentedColormap.from_list('threshold_colormap', X)

        im = axs[0].imshow(img, cmap="gray")
        im.set_cmap(colormap(t))
        axs[1].hist(img.flatten(), bins='auto')
        axs[1].set_title('Histogram of pixel intensities')

        # Create the Slider
        slider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
        slider = Slider(slider_ax, "Threshold", img.min(), img.max(), t)

        # Create the Vertical lines on the histogram
        lower_limit_line = axs[1].axvline(slider.val, color='k')

        thresh = t

        def update(val):
            # The val passed to a callback by the Slider

            # Update the image's colormap
            # im.norm.vmin = val
            # im.norm.vmax = val

            nonlocal thresh

            im.set_cmap(colormap(val))

            # Update the position of the vertical line
            lower_limit_line.set_xdata([val, val])

            # Redraw the figure to ensure it updates
            fig.canvas.draw_idle()
            thresh = val

        slider.on_changed(update)
        plt.show(block=True)
        return thresh

    # def ithresh2(self):

    #     # ACKNOWLEDGEMENT: https://matplotlib.org/devdocs/gallery/widgets/range_slider.html
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     from matplotlib.widgets import RangeSlider

    #     #N = 128
    #     img = self.image

    #     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    #     plt.subplots_adjust(bottom=0.25)

    #     im = axs[0].imshow(img)
    #     axs[1].hist(img.flatten(), bins='auto')
    #     axs[1].set_title('Histogram of pixel intensities')

    #     # Create the RangeSlider
    #     slider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
    #     slider = RangeSlider(slider_ax, "Threshold", img.min(), img.max())

    #     # Create the Vertical lines on the histogram
    #     lower_limit_line = axs[1].axvline(slider.val[0], color='k')
    #     upper_limit_line = axs[1].axvline(slider.val[1], color='k')


    #     def update(val):
    #         # The val passed to a callback by the RangeSlider will
    #         # be a tuple of (min, max)

    #         # Update the image's colormap
    #         im.norm.vmin = val[0]
    #         im.norm.vmax = val[1]

    #         # Update the position of the vertical lines
    #         lower_limit_line.set_xdata([val[0], val[0]])
    #         upper_limit_line.set_xdata([val[1], val[1]])

    #         # Redraw the figure to ensure it updates
    #         fig.canvas.draw_idle()

    #     slider.on_changed(update)
    #     plt.show(block=True)

    def adaptive_threshold(self, C=0, width=3):
        #TODO options
        # looks like Niblack

        im = self.to_int()

        out = cv.adaptiveThreshold(
            src=im,
            maxValue=255,
            adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv.THRESH_BINARY,
            blockSize=width*2+1,
            C=C
        )
        return self.__class__(self.like(out))

    
    def otsu(self):
        """
        Otsu threshold selection

        :return: Otsu's threshold
        :rtype: scalar

        Compute the optimal threshold for binarizing an image with a
        bimodal intensity histogram.  ``t`` is a scalar threshold that
        maximizes the variance between the classes of pixels below and above
        the thresold ``t``.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('street.png')
            >>> img.otsu()

        .. note::

            - Converts a color image to greyscale.
            - OpenCV implementation gives slightly different result to 
              MATLAB Machine Vision Toolbox.

        :references:
            - A Threshold Selection Method from Gray-Level Histograms, N. Otsu.
              IEEE Trans. Systems, Man and Cybernetics Vol SMC-9(1), Jan 1979,
              pp 62-66.
            - An improved method for image thresholding on the valley-emphasis
              method. H-F Ng, D. Jargalsaikhan etal. Signal and Info Proc.
              Assocn. Annual Summit and Conf (APSIPA). 2013. pp1-4
            - Robotics, Vision & Control for Python, Section 12.1.1, P. Corke, Springer 2023.

        :seealso: :meth:`thresh` :meth:`ithresh` :meth:`adaptive_threshold`  `opencv.threshold <https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57>`_
        """
        _, t = self.thresh(t='otsu')
        return t

    def blend(self, image2, alpha, beta=None, gamma=0):
        r"""
        Image blending

        :param image2: second image
        :type image2: :class:`Image`
        :param alpha: fraction of image
        :type alpha: float
        :param beta: fraction of ``image2``, defaults to 1-``alpha``
        :type beta: float, optional
        :param gamma: gamma nonlinearity, defaults to 0
        :type gamma: int, optional
        :raises ValueError: images are not same size
        :raises ValueError: images are of different type
        :return: blended image
        :rtype: :class:`Image`

        The resulting image is
        
        .. math::

            \mathbf{Y} = \alpha \mathbf{X}_1 + \beta \mathbf{X}_2 + \gamma

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img1 = Image.Constant(3, value=4)
            >>> img2 = Image([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> img1.blend(img2, 0.5, 2).A

        .. note::
            - For integer images the result is saturated.
            - For a multiplane image each plane is processed independently.

        :seealso: :meth:`choose` `cv2.addWeighted <https://docs.opencv.org/master/d2/de8/group__core__array.html#gafafb2513349db3bcff51f54ee5592a19>`_
        """

        if self.shape != image2.shape:
            raise ValueError('images are not the same size')
        if self.isint != image2.isint:
            raise ValueError('images must be both int or both floating type')
            
        if beta is None:
            beta = 1 - alpha
        out = cv.addWeighted(self.A, alpha, image2.A, beta, gamma)
        return self.__class__(out, colororder=self.colororder)

    def choose(self, image2, mask):
        r"""
        Pixel-wise image merge

        :param image2: second image
        :type image2: :class:`Image`, array_like(3), str
        :param mask: image mask
        :type mask: ndarray(H,W)
        :raises ValueError: image and mask must be same size
        :raises ValueError: image and image2 must be same size
        :return: merged images
        :rtype: :class:`Image`

        Return an image where each pixel is selected from the corresponding
        pixel in self or ``image2`` according to the corresponding pixel values
        in ``mask``.  If the element of ``mask`` is zero/false the pixel value
        from self is selected, otherwise the pixel value from ``image2`` is selected:
        
        .. math::

            \mathbf{Y}_{u,v} = \left\{ \begin{array}{ll}
                \mathbf{X}_{1:u,v} & \mbox{if } \mathbf{M}_{u,v} = 0 \\
                \mathbf{X}_{2:u,v} & \mbox{if } \mathbf{M}_{u,v} > 0
                \end{array} \right.

        If ``image2`` is a scalar or 1D array it is taken as the pixel value,
        and must have the same number of elements as the channel depth of self.
        If ``image2`` is a string it is taken as a colorname which is looked up
        using ``name2color``.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img1 = Image.Constant(3, value=10)
            >>> img2 = Image.Constant(3, value=80)
            >>> img = Image([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> img1.choose(img2, img >=5).image
            >>> img1 = Image.Constant(3, value=[0,0,0])
            >>> img1.choose('red', img>=5).red().image

        .. note::
            - If image and ``image2`` are both greyscale then the result is
              greyscale.
            - If either of image and ``image2`` are color then the result is color.
            - If one image is double and the other is integer, then the integer
              image is first converted to a double image.
            - ``image2`` can contain a color descriptor which is one of: a scalar
              value corresponding to a greyscale, a 3-vector corresponding to a
              color value, or a string containing the name of a color which is
              found using :meth:`name2color`.

        :seealso: :func:`~machinevisiontoolbox.base.color.name2color` `opencv.bitwise_and <https://docs.opencv.org/master/d2/de8/group__core__array.html#ga60b4d04b251ba5eb1392c34425497e14>`_
        """
        im1 = self.image

        if isinstance(mask, self.__class__):
            mask = mask.A > 0
        elif not isinstance(mask, np.ndarray):
            raise ValueError('bad type for mask')

        mask = mask.astype(np.uint8)
        if im1.shape[:2] != mask.shape:
            raise ValueError('image and mask must be same size')

        if isinstance(image2, self.__class__):
            # second image is Image type
            im2 = image2.image
            if im1.shape != im2.shape:
                raise ValueError('image and image2 must be same size')
        else:
            # second image is scalar, 3-vector or str
            dt = self.dtype
            shape = self.shape[:2]
            if isinstance(image2, (int, float)):
                # scalar
                im2 = np.full(shape, image2, dtype=dt)
            else:
                # possible color value
                if isinstance(image2, str):
                    # it's a colorname, look it up
                    color = self.like(name2color(image2))
                else:
                    try:
                        color = argcheck.getvector(image2, 3)
                    except:
                        raise ValueError('expecting a scalar, string or 3-vector')
                if self.isbgr:
                    color = color[::-1]
                im2 = np.dstack((
                    np.full(shape, color[0], dtype=dt),
                    np.full(shape, color[1], dtype=dt),
                    np.full(shape, color[2], dtype=dt)))
            if im1.ndim == 2 and im2.ndim > 2:
                im1 = np.repeat(np.atleast_3d(im1), im2.shape[2], axis=2)

        m = cv.bitwise_and(mask, np.uint8([1]))
        m_not = cv.bitwise_xor(mask, np.uint8([1]))

        out = cv.bitwise_and(im1, im1, mask=m_not) \
              + cv.bitwise_and(im2, im2, mask=mask)
        
        return self.__class__(out, colororder=self.colororder)

    def paste(self,
              pattern,
              pt,
              method='set',
              position='topleft',
              copy=False,
              zero=True):
        """
        Paste an image into an image

        :param pattern: image to be pasted
        :type pattern: :class:`Image`, ndarray(H,W)
        :param pt: coordinates (u,v) where pattern is pasted
        :type pt: array_like(2)
        :param method: options for image merging, one of: ``'set'`` [default],
            ``'mean'``, ``'add'``
        :type method: str
        :param position: ``pt`` is one of: ``'topleft'`` [default] or  ``'centre'``
        :type position: str, optional
        :param copy: copy image before pasting, defaults to False
        :type copy: bool, optional
        :param zero: zero-based coordinates (True, default) or 1-based coordinates (False)
        :type zero: bool, optional
        :raises ValueError: pattern is positioned outside the bounds of the image
        :return: original image with pasted pattern
        :rtype: :class:`Image`

        Pastes the ``pattern`` into the image which is modified inplace.  The
        pattern can be incorporated into the specified image by:

        ==========  ================================================================
        method      description
        ==========  ================================================================
        ``'set'``   overwrites the pixels in image
        ``'add'``   adds to the pixels in image
        ``'mean'``  sets pixels to the mean of the pixel values in pattern and image
        ==========  ================================================================

        The ``position`` of the pasted ``pattern`` in the image can be specified
        by its top left corner (umin, vmin) or its centre in the image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img1 = Image.Constant(5, value=10)
            >>> pattern = Image([[11, 12], [13, 14]])
            >>> img1.copy().paste(pattern, (1,2)).image
            >>> img1.copy().paste(pattern, (1,2), method='add').image
            >>> img1.copy().paste(pattern, (1,2), method='mean').image

        .. note::
            - Pixels outside the pasted region are unaffected.
            - If ``copy`` is False the image is modified in place
            - For ``position='centre'`` an odd sized pattern is assumed.  For
              an even dimension the centre pixel is the one at dimension / 2.
            - Multi-plane images are supported.
            - If the pattern is multiplane and the image is singleplane, the image planes
              are replicated and colororder is taken from the pattern.
            - If the image is multiplane and the pattern is singleplane, the pattern planes
              are replicated.
        """

        # TODO can likely replace a lot of this with np.where?

        # check inputs
        pt = argcheck.getvector(pt)

        if isinstance(pattern, np.ndarray):
            pattern = self.__class__(pattern)

        # TODO check optional inputs valid
        # TODO need to check that centre+point+pattern combinations are valid
        # for given canvas size

        cw = self.width
        ch = self.height
        pw = pattern.width
        ph = pattern.height
        colororder = self.colororder

        if position in ('centre', 'center'):
            left = pt[0] - pw // 2
            top = pt[1] - ph // 2
        elif position == 'topleft':
            left = pt[0]  # x
            top = pt[1]  # y
        else:
            raise ValueError('bad position specified')

        if not zero:
            left += 1
            top += 1

        # indices must be integers
        left = int(left)
        top = int(top)

        if (top+ph) > ch:
            raise ValueError(ph, 'pattern falls off bottom edge')
        if (left+pw) > cw:
            raise ValueError(pw, 'pattern falls off right edge')

        npc = pattern.nplanes
        nc = self.nplanes

        if npc > nc:
            # pattern has multiple planes, replicate the canvas
            # sadly, this doesn't work because repmat doesn't work on 3D
            # arrays
            # o = np.matlib.repmat(canvas.image, [1, 1, npc])
            o = np.dstack([self.A for i in range(npc)])
            colororder = pattern.colororder
        else:
            if copy:
                o = self.image.copy()
            else:
                o = self.image

        if npc < nc:
            pim = np.dstack([pattern.A for i in range(nc)])
            # pattern.image = np.matlib.repmat(pattern.image, [1, 1, nc])
        else:
            pim = pattern.image

        if method == 'set':
            if pattern.iscolor:
                o[top:top+ph, left:left+pw, :] = pim
            else:
                o[top:top+ph, left:left+pw] = pim

        elif method == 'add':
            if pattern.iscolor:
                o[top:top+ph, left:left+pw, :] = o[top:top+ph,
                                                    left:left+pw, :] + pim
            else:
                o[top:top+ph, left:left+pw] = o[top:top+ph,
                                                left:left+pw] + pim
        elif method == 'mean':
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

        elif method == 'blend':
            # compute the mean using float32 to avoid overflow issues
            bg = o[top:top+ph, left:left+pw].astype(np.float32)
            fg = pim.astype(np.float32)
            blend = 0.5 * (bg + fg)
            blend = blend.astype(self.dtype)

            # make masks for foreground and background
            fg_set = (fg > 0).astype(np.uint8)
            bg_set = (bg > 0).astype(np.uint8)
            
            # blend is valid
            blend_mask = cv.bitwise_and(fg_set, bg_set)

            # only fg is valid
            fg_mask = cv.bitwise_and(fg_set, cv.bitwise_xor(bg_set, 1))

            # only bg is valid
            bg_mask = cv.bitwise_and(cv.bitwise_xor(fg_set, 1), bg_set)

            # merge them
            out = cv.bitwise_and(blend, blend, mask=blend_mask) \
                + cv.bitwise_and(bg, bg, mask=bg_mask) \
                + cv.bitwise_and(fg, fg, mask=fg_mask)
            o[top:top+ph, left:left+pw] = out

        else:
            raise ValueError('method is not valid')

        if copy:
            return self.__class__(o, copy=copy, colororder=colororder)
        else:
            self.A = o
            return self

    def invert(self):
        r"""
        Invert image

        :return: _description_
        :rtype: _type_

        For an integer image

        .. math:: Y_{u,v} = \left\{ \begin{array}{l} p_{\mbox{max}} \mbox{, if } X_{u,v} = 0 \\ p_{\mbox{min}} \mbox{, otherwise} \end{array}\right.

        where :math:`p_{\mbox{min}}` and :math:`p_{\mbox{max}}` are respectively
        the minimum and maximum value of the datatype.

        For a float image

        .. math:: Y_{u,v} = \left\{ \begin{array}{l} 1.0 \mbox{, if } X_{u,v} = 0 \\ 0.0 \mbox{, otherwise} \end{array}\right.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[0, 1], [2, 3]])
            >>> img.invert().image
        """
        if self.isint:
            out = np.where(self.image == 0, self.like(self.maxval), self.like(self.minval))
        elif self.isfloat:
            out = np.where(self.image == 0, 1.0, 0.0)
        return self.__class__(out)




    # def scalespace(self, n, sigma=1):

    #     im = self.copy()
    #     g = []
    #     scale = 0.5
    #     scales = []
    #     lap = []
    #     L = Kernel.Laplace()
    #     scale = sigma

    #     for i in range(n):
    #         im = im.smooth(sigma)
    #         g.append(im)
    #         lap.append(im.convolve(L))
    #         scales.append(scale)

    #         scale = np.sqrt(scale ** 2 + sigma ** 2)
    #         scales.append(scale)
    #         g.append(im)
    #         x = (g[-1] - g[-2]) * scale ** 2 
    #         lap.append(x)

    #     return g, lap, scales
# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    import pathlib
    import os.path
    from machinevisiontoolbox import Image
    # a = Image.Read('street.png')
    # a.ithresh()

    a = Image.Read('castle2.png')
    b = a.labels_MSER()

    #exec(open(pathlib.Path(__file__).parent.parent.absolute() / "tests" / "test_processing.py").read())  # pylint: disable=exec-used