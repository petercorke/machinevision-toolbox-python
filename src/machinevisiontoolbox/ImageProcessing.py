"""
Spatial filtering, convolution, edge detection, and general image processing operations.
"""

from __future__ import annotations

import os.path
from collections import namedtuple
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from warnings import warn

import cv2
import matplotlib.pyplot as plt
import numpy as np

# from numpy.lib.arraysetops import isin
import scipy as sp
from scipy import interpolate
from spatialmath.base import argcheck, e2h, getvector, h2e, transl2

from machinevisiontoolbox.base import (
    colorname,
    float_image,
    idisp,
    int_image,
    iread,
    iwrite,
    name2color,
)

if TYPE_CHECKING:
    from machinevisiontoolbox.ImageCore import Image
    from machinevisiontoolbox._image_typing import _ImageBase


class ImageProcessingMixin(_ImageBase if TYPE_CHECKING else object):
    # ======================= image processing ============================= #

    def LUT(self, lut: Any, colororder: str | dict[str, Any] | None = None) -> "Image":
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
              by applying the I'th column of the LUT to the input image

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
            - |RVC3|, Section 11.3.

        .. important:: Uses OpenCV function ``cv2.LUT`` which accepts multiple-channel, CV_8U images.

        :seealso: `cv2.LUT <https://docs.opencv.org/4.x/d2/de8/group__core__array.html#gab55b8d062b7f5587720ede032d34156f>`_
        """
        self._opencv_type_check(self._A, "multiple-channel", "CV_8U")
        image = self.array_as("uint8")
        lut = np.array(lut).astype(np.uint8)
        if lut.ndim == 2:
            lut = lut[np.newaxis, ...]
            if self.nplanes == 1:
                image = np.dstack((image,) * lut.shape[2])

        out = cv2.LUT(src=image, lut=lut)
        if colororder is None:
            colororder = self.colororder

        return self.__class__(self.like(out), colororder=colororder)

    def apply(
        self, func: Callable[..., Any], vectorize: bool = False, **kwargs
    ) -> "Image":
        """
        Apply a function to an image

        :param func: function to apply to image or pixel
        :type func: callable
        :param vectorize: if True apply function to each pixel, defaults to False
        :type vectorize: bool, optional
        :param kwargs: additional keyword arguments to pass to function
        :return: transformed image
        :rtype: :class:`Image`

        If ``vectorize`` is False:

        - the function is called with a single argument which is
          the underlying NumPy array
        - the function must return a NumPy array, which can have different dimensions
          to its argument.  This allows for a large number of NumPy or OpenCV functions to
          be applied to an image.
        - For a multiplane image the function is called with a 3D array, and can
          return an array with the same or a different number of channels.
        - the returned NumPy array is encapsulated in a new ``Image``.

        If ``vectorize`` is True:

        - the function is called for every pixel with a single argument which is a
          scalar.
        - for a color image the same function is called for each plane with the corresponding pixel value as a scalar.
        - the return array will have the same dimensions (width, height, planes)
          as its argument.

        The function ``func`` is called with the image or pixel value as the first argument,
        followed by any additional keyword arguments.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> import math
            >>> img = Image([[1, 2], [3, 4]])
            >>> img.apply(np.sqrt).print()
            >>> img.apply(lambda x: math.sqrt(x), vectorize=True).print()

        .. note::
            - Slow when ``vectorize=True`` which involves a large number
              of calls to ``func``.


        :references:
            - |RVC3|, Section 11.3.

        :seealso: :meth:`apply2` :meth:`numpy.vectorize`
        """
        if vectorize:
            func = np.vectorize(func, **kwargs)
        return self.__class__(func(self._A, **kwargs), colororder=self.colororder)

    def apply2(
        self,
        other: "Image",
        func: Callable[..., Any],
        vectorize: bool = False,
        **kwargs,
    ) -> "Image":
        """
        Apply a function to two images

        :param func: function to apply to image or pixel
        :type func: callable
        :raises ValueError: images must have same size
        :param vectorize: if True apply function to each pixel, defaults to False
        :type vectorize: bool, optional
        :param kwargs: additional keyword arguments passed to the function
        :return: transformed image
        :rtype: :class:`Image`

        If ``vectorize`` is False:

        - the function ``func`` is called with two arguments which are
          the underlying NumPy array of ``self`` and ``other``.
        - the function must return a NumPy array, which can have different dimensions
          to its arguments.  This allows for a large number of NumPy or OpenCV functions to
          be applied to an image.
        - For a multiplane image the function is called with a 3D array, and can
          return an array with the same or a different number of channels.
        - the returned NumPy array is encapsulated in a new ``Image``.

        If ``vectorize`` is True:

        - the function ``func`` is called for every pixel with two arguments which are the corresponding scalar pixel values from
          ``self`` and ``other``.
        - the images must have the same width, height, and number of channels
        - for a color image the function is called for every pixel on every plane.
        - the return array will have the same dimensions (width, height, planes)
          as its argument.


        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> import math
            >>> img1 = Image([[1, 2], [3, 4]])
            >>> img2 = Image([[5, 6], [7, 8]])
            >>> img1.apply2(img2, np.hypot).print()
            >>> img1.apply2(img2, lambda x, y: math.hypot(x,y), vectorize=True).print()

        .. note::
            - Slow when ``vectorize`` is ``True`` which involves a large number
              of calls to ``func``.

        :references:
            - |RVC3|, Section 11.4.

        :seealso: :meth:`apply` :meth:`numpy.vectorize`
        """
        if self.size != other.size:
            raise ValueError("two images must have same size")
        if vectorize:
            func = np.vectorize(func)
        return self.__class__(func(self._A, other._A), colororder=self.colororder)

    def clip(self, min: int | float, max: int | float) -> "Image":
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
        return self.__class__(np.clip(self._A, min, max), colororder=self.colororder)

    def roll(
        self,
        ru: int = 0,
        rv: int = 0,
        dx: int | None = None,
        dy: int | None = None,
    ) -> "Image":
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
        if dx is not None:
            ru = dx
        if dy is not None:
            rv = dy

        return self.__class__(np.roll(self._A, (ru, rv), (1, 0)))

    def normhist(self) -> "Image":
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
            - Color images automatically converted to greyscale

        :references:
            - |RVC3|, Section 11.3.

        .. important:: Uses OpenCV function ``cv2.equalizeHist`` which accepts single-channel, CV_8U images.

        :seealso: `cv2.equalizeHist <https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html#ga7e54091f0c937d49bf84152a16f76d6e>`_
        """
        self._opencv_type_check(self._A, "single-channel", "CV_8U")
        out = cv2.equalizeHist(src=self.array_as("uint8"))
        return self.__class__(self.like(out))

    def stretch(
        self, max: int | float = 1, range: Any = None, clip: bool = True
    ) -> "Image":
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

            - |RVC3|, Section 12.1.
        """

        # TODO make all infinity values = None?

        im = self._A
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

    def thresh(self, *args: Any, **kwargs: Any) -> Any:
        """
        Image threshold

        .. deprecated:: 1.0.3
            Use :meth:`threshold` instead
        """
        warn(
            "Deprecated in 1.0.3: use threshold() instead of thresh().",
            DeprecationWarning,
            stacklevel=2,
        )
        if "t" in kwargs and "threshold" not in kwargs:
            kwargs["threshold"] = kwargs.pop("t")
        if "opt" in kwargs and "method" not in kwargs:
            kwargs["method"] = kwargs.pop("opt")
        return self.threshold(*args, **kwargs)

    def threshold(
        self,
        threshold: int | float | str | None = None,
        opt: str | None = None,
        *,
        method: str = "binary",
        nbins: int = 256,
        p: float = 50.0,
        as_bool: bool = False,
        t: int | float | str | None = None,
    ) -> Any:
        r"""
        Image threshold

        :param threshold: threshold value, or automatic selector name (``'otsu'``,
            ``'triangle'`` or ``'percentile'``)
        :type threshold: scalar or str
        :param method: thresholding method for scalar threshold values, defaults to
            ``'binary'``
        :type method: str, optional
        :param nbins: number of bins for histogram-based automatic methods, defaults to
            256
        :type nbins: int, optional
        :param p: percentile used when ``threshold='percentile'``, defaults to 50
        :type p: float, optional
        :param as_bool: return binary outputs as ``bool`` if ``True``, otherwise
            ``uint8`` with values 0/255, defaults to ``False``
        :type as_bool: bool, optional
        :return: thresholded image
        :rtype: :class:`Image`

        Apply a threshold ``threshold`` to the image.  The threshold condition is for
        the value **greater than** ``threshold``. Various thresholding options are
        supported where :math:`t` is the threshold value:

        .. list-table::
            :header-rows: 1

            * - ``method``
              - Function
              - Return data type
            * - ``'binary'``
              - :math:`Y_{u,v} = \left\{ \begin{array}{l} T \mbox{,if } X_{u,v} > t \\ F \mbox{, otherwise} \end{array} \right.`
              - ``uint8`` or ``bool``
            * - ``'binary_inv'``
              - :math:`Y_{u,v} = \left\{ \begin{array}{l} F \mbox{, if } X_{u,v} > t \\ T \mbox{, otherwise} \end{array}\right.`
              - ``uint8`` or ``bool``
            * - ``'truncate'``
              - :math:`Y_{u,v} = \left\{ \begin{array}{l} t \mbox{,if } X_{u,v} > t \\ X_{u,v} \mbox{, otherwise} \end{array} \right.`
              - same as :math:`X`
            * - ``'tozero'``
              - :math:`Y_{u,v} = \left\{ \begin{array}{l} X_{u,v} \mbox{, if } X_{u,v} > t \\ 0 \mbox{, otherwise} \end{array}\right.`
              - same as :math:`X`
            * - ``'tozero_inv'``
              - :math:`Y_{u,v} = \left\{ \begin{array}{l} 0 \mbox{, if } X_{u,v} > t \\ X_{u,v} \mbox{, otherwise} \end{array}\right.`
              - same as :math:`X`

        For the case where the return data type is ``uint8`` the return pixel values are
        either :math:`F=0` or :math:`T=255`. For the case where ``as_bool`` is ``True``
        then the return data type is ``bool`` and the pixel values are either
        :math:`F=False` or :math:`T=True`.

        If ``threshold`` is a string then the threshold is determined automaticly
        prior to executing the logic above.  The following automatic threshold selection
        methods are supported:

        +-----------------+-----------------------------------------------------+
        |threshold        | algorithm                                           |
        +=================+=====================================================+
        |``'otsu'``       | Otsu's method finds the threshold that minimizes    |
        |                 | the within-class variance. This technique is        |
        |                 | effective for a bimodal greyscale histogram.        |
        +-----------------+-----------------------------------------------------+
        |``'triangle'``   | The triangle method constructs a line between the   |
        |                 | histogram peak and the farthest end of the          |
        |                 | histogram. The threshold is the point of maximum    |
        |                 | distance between the line and the histogram. This   |
        |                 | technique is effective when the object pixels       |
        |                 | produce a weak peak in the histogram.               |
        +-----------------+-----------------------------------------------------+
        |``'percentile'`` | Select threshold from the image percentile given    |
        |                 | by ``p`` (0 to 100).                                |
        +-----------------+-----------------------------------------------------+

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> img.threshold(5).print()
            >>> img.threshold('otsu')
            >>> img.threshold('percentile', p=90)


        :references:
            - A Threshold Selection Method from Gray-Level Histograms, N. Otsu. IEEE
              Trans. Systems, Man and Cybernetics Vol SMC-9(1), Jan 1979, pp 62-66.
            - Automatic measurement of sister chromatid exchange frequency" Zack (Zack
              GW, Rogers WE, Latt SA (1977), J. Histochem. Cytochem. 25 (7): 741–53.
            - |RVC3|, Section 12.1.1.

        .. note:: Uses NumPy thresholding and toolbox-native Otsu and triangle
            threshold selection, not OpenCV functions, to support a wider range of datatypes and automatic threshold selection methods.

        :seealso:
            :meth:`threshold_interactive` :meth:`threshold_adaptive_` :meth:`otsu`
            :meth:`triangle`
        """
        if t is not None:
            warn(
                "Deprecated in 1.1.0: pass threshold as the first argument instead of using t=.",
                DeprecationWarning,
                stacklevel=2,
            )
            if threshold is not None:
                raise ValueError("specify either threshold or t, not both")
            threshold = t

        if threshold is None:
            raise ValueError("threshold must be specified")

        if opt is not None:
            warn(
                "Deprecated in 1.1.0: use method= instead of opt=.",
                DeprecationWarning,
                stacklevel=2,
            )
            if method != "binary":
                raise ValueError("specify either method or opt, not both")
            method = opt

        mode = method.lower()
        arr = self._A
        autothresh = None

        if as_bool:
            false_value: bool | int = False
            true_value: bool | int = True
            binary_dtype = np.bool_
        else:
            false_value = 0
            true_value = 255
            binary_dtype = np.uint8

        if isinstance(threshold, str):
            auto = threshold.lower()
            if auto == "otsu":
                if self.iscolor:
                    raise ValueError(
                        "Otsu thresholding only works for greyscale images"
                    )
                autothresh = self.otsu(nbins=nbins)

            elif auto == "triangle":
                if self.iscolor:
                    raise ValueError(
                        "triangle thresholding only works for greyscale images"
                    )
                autothresh = self.triangle(nbins=nbins)
            elif auto == "percentile":
                autothresh = np.percentile(arr, p)
            else:
                raise ValueError(auto, "unknown automatic threshold method")

            threshold = autothresh

        if not argcheck.isscalar(threshold):
            raise ValueError(threshold, "threshold must be a scalar")

        if mode == "binary":
            imt = np.where(arr > threshold, true_value, false_value).astype(
                binary_dtype
            )
        elif mode in ("/binary", "binary_inv"):
            imt = np.where(arr > threshold, false_value, true_value).astype(
                binary_dtype
            )
        elif mode == "truncate":
            imt = np.minimum(arr, threshold)
        elif mode == "tozero":
            imt = np.where(arr > threshold, arr, 0)
        elif mode in ("/tozero", "tozero_inv"):
            imt = np.where(arr <= threshold, arr, 0)
        else:
            raise ValueError(mode, "unknown threshold method")

        if mode in ("binary", "binary_inv"):
            out = self.__class__(imt)
        else:
            out = self.__class__(self.like(imt))

        if autothresh is not None:
            return out, self.cast(autothresh)
        return out

    def ithresh(self, threshold: float | None = None, opt: str = "binary") -> Any:
        """
        Interactive thresholding

        .. deprecated:: 1.0.3
            Use :meth:`threshold_interactive` instead
        """
        if threshold is not None:
            return self.threshold(threshold=threshold, method=opt)

        warn(
            "Deprecated in 1.0.3: use threshold_interactive() instead of ithresh().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.threshold_interactive()

    def threshold_interactive(
        self, title: str | None = None, mode: str = "binary", block: bool = False
    ) -> Any:
        r"""
        Interactive thresholding

        :param title: title for the window, defaults to ``None``
        :type title: str, optional
        :param mode: threshold display mode, one of ``'binary'`` or ``'tozero'``,
            defaults to ``'binary'``
        :type mode: str, optional
        :param block: block execution while the interactive window is open,
            defaults to ``False``
        :type block: bool, optional
        :raises ValueError: called for a color image
        :raises ValueError: unknown mode
        :return: selected threshold value
        :rtype: scalar

        The image is displayed in a simple Matplotlib GUI along with the histogram
        and a slider for threshold value.  Adjusting the slider changes the
        thresholded image view.

        If ``mode='binary'`` the displayed image is:

        .. math:: Y_{u,v} = \left\{ \begin{array}{l} 1 \mbox{, if } X_{u,v} > t \\ 0 \mbox{, otherwise} \end{array} \right.

        If ``mode='tozero'`` the displayed image is:

        .. math:: Y_{u,v} = \left\{ \begin{array}{l} X_{u,v} \mbox{, if } X_{u,v} > t \\ 0 \mbox{, otherwise} \end{array} \right.

        :references:
            - |RVC3|, Section 12.1.1.1.

        :seealso: :meth:`threshold` :meth:`threshold_adaptive` :meth:`otsu` `opencv.threshold <https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57>`_
        """

        # ACKNOWLEDGEMENT: https://matplotlib.org/devdocs/gallery/widgets/range_slider.html
        import matplotlib.pyplot as plt
        import numpy as np

        if self.iscolor:
            raise ValueError("interactive thresholding only works for greyscale images")

        mode = mode.lower()
        if mode not in ("binary", "tozero"):
            raise ValueError(mode, "unknown threshold mode")

        img = self._A
        img_min = float(np.min(img))
        img_max = float(np.max(img))

        if img_min == img_max:
            return self.cast(img_min)

        t = 0.5 * (img_max + img_min)

        def apply_mode(threshold):
            if mode == "binary":
                return (img > threshold).astype(np.uint8)
            return np.where(img > threshold, img, 0)

        # With the ipympl widget backend every draw_idle() encodes the full
        # canvas as PNG and sends it over the WebSocket.  Rapid slider events
        # flood the queue and the slider freezes.  Use an ipywidgets
        # FloatSlider (continuous_update=False) instead so updates only fire
        # on mouse-release, not during the drag.
        _backend = plt.get_backend().lower()
        _use_ipywidgets = "widget" in _backend or "ipympl" in _backend
        if _use_ipywidgets:
            try:
                import ipywidgets as widgets
                from IPython.display import display
            except ImportError:
                _use_ipywidgets = False

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plt.subplots_adjust(bottom=0.1 if _use_ipywidgets else 0.25)
        try:
            if title is not None and fig.canvas.manager is not None:
                fig.canvas.manager.set_window_title(title)  # for 3.4 onward
        except Exception:
            pass

        im = axs[0].imshow(apply_mode(t), cmap="gray")
        if mode == "binary":
            im.set_clim(0, 1)
        else:
            im.set_clim(img_min, img_max)

        counts, edges = np.histogram(img.flatten(), bins=256, range=(img_min, img_max))
        axs[1].stairs(
            counts,
            edges,
            fill=True,
            linewidth=0,
            antialiased=False,
        )
        axs[1].set_xlim(img_min, img_max)
        axs[1].set_title("Histogram of pixel intensities")
        axs[1].grid(True)

        # Create the vertical line on the histogram
        lower_limit_line = axs[1].axvline(t, color="k", linestyle="--")

        if _use_ipywidgets:
            thresh_holder = [t]

            slider = widgets.FloatSlider(
                value=t,
                min=img_min,
                max=img_max,
                step=(img_max - img_min) / 256,
                description="Threshold",
                continuous_update=False,
                layout=widgets.Layout(width="80%"),
            )

            def update(change):
                val = change["new"]
                im.set_data(apply_mode(val))
                lower_limit_line.set_xdata([val, val])
                fig.canvas.draw_idle()
                thresh_holder[0] = val

            slider.observe(update, names="value")
            plt.show()
            display(slider)
            return self.cast(thresh_holder[0])

        # Native backend: use a matplotlib Slider
        from matplotlib.widgets import Slider

        slider_ax = plt.axes((0.20, 0.1, 0.60, 0.03))
        slider = Slider(slider_ax, "Threshold", img_min, img_max, valinit=t)

        thresh = t

        def update(val):
            # The val passed to a callback by the Slider

            nonlocal thresh

            im.set_data(apply_mode(val))

            # Update the position of the vertical line
            lower_limit_line.set_xdata([val, val])

            # Redraw the figure to ensure it updates
            fig.canvas.draw_idle()
            thresh = val

        slider.on_changed(update)
        plt.show(block=block)
        return self.cast(thresh)

    def threshold_adaptive(
        self,
        C: int = 0,
        h: int = 3,
        method: str = "mean",
        blocksize: int | None = None,
    ) -> "Image":
        r"""
        Adaptive threshold

        :param C: _description_, defaults to 0
        :type C: int, optional
        :param h: half-width of window, defaults to 3
        :type h: int, optional
        :return: thresholded image
        :rtype: :class:`Image`

        The threshold at each pixel is the mean over a :math:`w \times w, w=2h+1`
        window minus ``C``.  ``h`` should reflect the scale of the objects
        that are to be segmented from the background.

        :references:
            - |RVC3|, Section 12.1.1.1.

        .. important:: Uses OpenCV function ``cv2.adaptiveThreshold`` which accepts single-channel, CV_8U images.

        :seealso:
            :meth:`threshold`
            :meth:`threshold_interactive`
            :meth:`otsu`
            `opencv.adaptiveThreshold <https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3>`_
        """
        # TODO options
        # looks like Niblack

        if self.iscolor:
            raise ValueError("adaptive thresholding only works for greyscale images")
        im = self.array_as("uint8")  # only accepts 8-channel image

        if blocksize is not None:
            h = (blocksize - 1) // 2

        method_dict = {
            "mean": cv2.ADAPTIVE_THRESH_MEAN_C,
            "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        }
        adaptive_method = method_dict[method.lower()]

        out = cv2.adaptiveThreshold(
            src=im,
            maxValue=255,
            adaptiveMethod=adaptive_method,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=h * 2 + 1,
            C=C,
        )
        return self.__class__(self.like(out))

    def adaptive_threshold(self, *args: Any, **kwargs: Any) -> Any:
        """
        Adaptive threshold

        .. deprecated:: 1.1.0
            Use :meth:`threshold_adaptive` instead.  Mentioned on page 484 of |RVC3| as adaptive_threshold but implemented as threshold_adaptive for consistency with other method names.
        """
        warn(
            "Deprecated in 1.1.0: use threshold_adaptive() instead of adaptive_threshold().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.threshold_adaptive(*args, **kwargs)

    def otsu(self, nbins: int = 256) -> Any:
        """
        Otsu threshold selection

        :param nbins: number of bins for histogram computation, defaults to 256
        :type nbins: int, optional

        :return: Otsu's threshold
        :rtype: scalar

        Compute the optimal threshold for binarizing an image with a
        bimodal intensity histogram.  ``t`` is a scalar threshold that
        maximizes the variance between the classes of pixels below and above
        the threshold ``t``.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('street.png')
            >>> img.otsu()

        .. note::
            - Converts a color image to greyscale.
            - Implementation gives slightly different result to
              MATLAB Machine Vision Toolbox.
            - Works for greyscale integer and floating-point images.

        :references:
            - A Threshold Selection Method from Gray-Level Histograms, N. Otsu.
              IEEE Trans. Systems, Man and Cybernetics Vol SMC-9(1), Jan 1979,
              pp 62-66.
            - An improved method for image thresholding on the valley-emphasis
              method. H-F Ng, D. Jargalsaikhan etal. Signal and Info Proc.
              Assocn. Annual Summit and Conf (APSIPA). 2013. pp1-4
            - |RVC3|, Section 12.1.1.

        :seealso: :meth:`triangle` :meth:`threshold` :meth:`threshold_interactive` :meth:`threshold_adaptive`
        """
        image = self.mono() if self.iscolor else self

        pixels = image._A.reshape(-1)

        if pixels.size == 0:
            raise ValueError("cannot compute Otsu threshold for an empty image")
        if nbins < 2:
            raise ValueError("nbins must be >= 2")

        pmin = float(np.min(pixels))
        pmax = float(np.max(pixels))
        if pmin == pmax:
            return pixels[0]

        counts, edges = np.histogram(pixels, bins=nbins, range=(pmin, pmax))
        values = (edges[:-1] + edges[1:]) / 2.0

        probabilities = counts.astype(np.float64) / counts.sum()
        cumulative_probabilities = np.cumsum(probabilities)
        cumulative_means = np.cumsum(probabilities * values.astype(np.float64))
        global_mean = cumulative_means[-1]

        between_class_variance = np.full(values.shape, -np.inf, dtype=np.float64)
        valid = np.logical_and(
            cumulative_probabilities > 0, cumulative_probabilities < 1
        )

        numerator = (
            global_mean * cumulative_probabilities[valid] - cumulative_means[valid]
        ) ** 2
        denominator = cumulative_probabilities[valid] * (
            1.0 - cumulative_probabilities[valid]
        )
        between_class_variance[valid] = numerator / denominator

        return values[np.argmax(between_class_variance)]

    def triangle(self, nbins: int = 256) -> Any:
        """
        Triangle threshold selection.

        :param nbins: number of bins for histogram computation, defaults to 256
        :type nbins: int, optional

        :return: triangle threshold
        :rtype: scalar

        Compute an automatic threshold using the triangle algorithm from the
        image histogram.  Works for greyscale integer and floating-point images.

        :seealso: :meth:`threshold` :meth:`otsu`
        """
        image = self.mono() if self.iscolor else self

        pixels = image._A.reshape(-1)

        if pixels.size == 0:
            raise ValueError("cannot compute triangle threshold for an empty image")
        if nbins < 2:
            raise ValueError("nbins must be >= 2")

        pmin = float(np.min(pixels))
        pmax = float(np.max(pixels))
        if pmin == pmax:
            return pixels[0]

        counts, edges = np.histogram(pixels, bins=nbins, range=(pmin, pmax))
        values = (edges[:-1] + edges[1:]) / 2.0

        hist = counts.astype(np.float64)
        peak = int(np.argmax(hist))
        nonzero = np.flatnonzero(hist)
        left = int(nonzero[0])
        right = int(nonzero[-1])

        if peak == left and peak == right:
            return values[peak]

        if (peak - left) >= (right - peak):
            end = left
        else:
            end = right

        lo = min(peak, end)
        hi = max(peak, end)
        if hi == lo:
            return values[peak]

        x = np.arange(lo, hi + 1, dtype=np.float64)
        y = hist[lo : hi + 1]

        x1 = float(peak)
        y1 = float(hist[peak])
        x2 = float(end)
        y2 = float(hist[end])

        dx = x2 - x1
        dy = y2 - y1
        den = np.hypot(dx, dy)
        if den == 0:
            return values[peak]

        distances = np.abs(dy * x - dx * y + x2 * y1 - y2 * x1) / den
        k = int(np.argmax(distances))
        return values[lo + k]

    def blend(
        self,
        image2: "Image",
        alpha: float,
        beta: float | None = None,
        gamma: float = 0,
        dtype: str | None = None,
    ) -> "Image":
        r"""
        Image blending

        :param image2: second image
        :type image2: :class:`Image`
        :param alpha: fraction of image
        :type alpha: float
        :param beta: fraction of ``image2``, defaults to 1-``alpha``
        :type beta: float, optional
        :param gamma: image offset, defaults to 0
        :type gamma: int, optional
        :param dtype: data type of the result, defaults to same as ``self``
        :type dtype: str, optional
        :raises ValueError: images are not same size
        :raises ValueError: images are of different type
        :return: blended image
        :rtype: :class:`Image`

        The resulting image isn a linear blend of the two input images ``self`` and ``image2`` according to:

        .. math::

            \mathbf{Y} = \alpha \mathbf{X}_1 + \beta \mathbf{X}_2 + \gamma

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img1 = Image.Constant(4, size=3)
            >>> img2 = Image([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> img1.blend(img2, 0.5, 2).A

        .. note::
            - For ``uint8`` images the result may be saturated.  To avoid this specify ``dtype='float'`` and the result will be a floating point image.
            - For a multiplane image each plane is processed independently.

        .. important:: Uses OpenCV function ``cv2.addWeighted`` which accepts multiple-channel, CV_8U, CV_16U, CV_16S, CV_32F or CV_64F images.

        :seealso: :meth:`choose` `cv2.addWeighted <https://docs.opencv.org/4.x/d2/de8/group__core__array.html#gafafb2513349db3bcff51f54ee5592a19>`_
        """

        if self.shape != image2.shape:
            raise ValueError("images are not the same size")
        if self.isint != image2.isint:
            raise ValueError("images must be both int or both floating type")

        if dtype is None:
            dtype = -1
        elif dtype in ("float", "float32"):
            dtype = cv2.CV_32F
        elif dtype in ("double", "float64"):
            dtype = cv2.CV_64F
        else:
            raise ValueError("dtype must be 'float', 'double', 'float32', or 'float64'")

        if beta is None:
            beta = 1 - alpha
        self._opencv_type_check(
            self._A, "multiple-channel", "CV_8U", "CV_16U", "CV_16S", "CV_32F", "CV_64F"
        )
        out = cv2.addWeighted(
            src1=self._A,
            alpha=alpha,
            src2=image2._A,
            beta=beta,
            gamma=gamma,
            dtype=dtype,
        )
        return self.__class__(out, colororder=self.colororder)

    def choose(self, image2: Any, mask: Any) -> "Image":
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
            >>> img1 = Image.Constant(10, size=3)
            >>> img2 = Image.Constant(80, size=3)
            >>> img = Image([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> img1.choose(img2, img >=5).print()
            >>> img1 = Image.Constant([0,0,0], size=3)
            >>> img1.choose('red', img>=5).red().print()

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

        .. important:: Uses OpenCV functions ``cv2.bitwise_and`` and ``cv2.bitwise_xor`` which accept multiple-channel, CV_8U, CV_16U, CV_16S, CV_32S, CV_32F or CV_64F images.

        :seealso: :func:`~machinevisiontoolbox.base.color.name2color` `opencv.bitwise_and <https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga60b4d04b251ba5eb1392c34425497e14>`_
        """
        im1 = self._A

        if isinstance(mask, self.__class__):
            mask = mask._A > 0
        elif not isinstance(mask, np.ndarray):
            raise ValueError("bad type for mask")

        mask = mask.astype(np.uint8)
        if im1.shape[:2] != mask.shape:
            raise ValueError("image and mask must be same size")

        if isinstance(image2, self.__class__):
            # second image is Image type
            im2 = image2._A
            if im1.shape != im2.shape:
                raise ValueError("image and image2 must be same size")
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
                        raise ValueError("expecting a scalar, string or 3-vector")
                if self.isbgr:
                    color = color[::-1]
                im2 = np.dstack(
                    (
                        np.full(shape, color[0], dtype=dt),
                        np.full(shape, color[1], dtype=dt),
                        np.full(shape, color[2], dtype=dt),
                    )
                )
            if im1.ndim == 2 and im2.ndim > 2:
                im1 = np.repeat(np.atleast_3d(im1), im2.shape[2], axis=2)

        ones = np.ones_like(mask, dtype=np.uint8)
        m = cv2.bitwise_and(src1=mask, src2=ones)
        m_not = cv2.bitwise_xor(src1=mask, src2=ones)

        out = cv2.bitwise_and(src1=im1, src2=im1, mask=m_not) + cv2.bitwise_and(
            src1=im2, src2=im2, mask=mask
        )

        return self.__class__(out, colororder=self.colororder)

    def paste(
        self,
        pattern: Any,
        pt: Any,
        method: str = "set",
        position: str = "topleft",
        copy: bool = False,
        zero: bool = True,
    ) -> "Image":
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
            >>> img1 = Image.Constant(10, size=5)
            >>> pattern = Image([[11, 12], [13, 14]])
            >>> img1.copy().paste(pattern, (1,2)).print()
            >>> img1.copy().paste(pattern, (1,2), method='add').print()
            >>> img1.copy().paste(pattern, (1,2), method='mean').print()

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

        if position in ("centre", "center"):
            left = pt[0] - pw // 2
            top = pt[1] - ph // 2
        elif position == "topleft":
            left = pt[0]  # x
            top = pt[1]  # y
        else:
            raise ValueError("bad position specified")

        if not zero:
            left += 1
            top += 1

        # indices must be integers
        left = int(left)
        top = int(top)

        if (top + ph) > ch:
            raise ValueError(ph, "pattern falls off bottom edge")
        if (left + pw) > cw:
            raise ValueError(pw, "pattern falls off right edge")

        npc = pattern.nplanes
        nc = self.nplanes

        if npc > nc:
            # pattern has multiple planes, replicate the canvas
            # sadly, this doesn't work because repmat doesn't work on 3D
            # arrays
            # o = np.matlib.repmat(canvas._A, [1, 1, npc])
            o = np.dstack([self._A for i in range(npc)])
            colororder = pattern.colororder
        else:
            if copy:
                o = self._A.copy()
            else:
                o = self._A

        if npc < nc:
            pim = np.dstack([pattern._A for i in range(nc)])
            # pattern._A = np.matlib.repmat(pattern._A, [1, 1, nc])
        else:
            pim = pattern._A

        if method == "set":
            if pattern.iscolor:
                o[top : top + ph, left : left + pw, :] = pim
            else:
                o[top : top + ph, left : left + pw] = pim

        elif method == "add":
            if pattern.iscolor:
                o[top : top + ph, left : left + pw, :] = (
                    o[top : top + ph, left : left + pw, :] + pim
                )
            else:
                o[top : top + ph, left : left + pw] = (
                    o[top : top + ph, left : left + pw] + pim
                )
        elif method == "mean":
            if pattern.iscolor:
                old = o[top : top + ph, left : left + pw, :]
                k = ~np.isnan(pim)
                old[k] = 0.5 * (old[k] + pim[k])
                o[top : top + ph, left : left + pw, :] = old
            else:
                old = o[top : top + ph, left : left + pw]
                k = ~np.isnan(pim)
                old[k] = 0.5 * (old[k] + pim[k])
                o[top : top + ph, left : left + pw] = old

        elif method == "blend":
            # compute the mean using float32 to avoid overflow issues
            bg = o[top : top + ph, left : left + pw].astype(np.float32)
            fg = pim.astype(np.float32)
            blend = 0.5 * (bg + fg)
            blend = blend.astype(self.dtype)

            # make masks for foreground and background
            fg_set = (fg > 0).astype(np.uint8)
            bg_set = (bg > 0).astype(np.uint8)

            # blend is valid
            blend_mask = cv2.bitwise_and(src1=fg_set, src2=bg_set)

            # only fg is valid
            fg_mask = cv2.bitwise_and(
                src1=fg_set, src2=cv2.bitwise_xor(src1=bg_set, src2=1)
            )

            # only bg is valid
            bg_mask = cv2.bitwise_and(
                src1=cv2.bitwise_xor(src1=fg_set, src2=1), src2=bg_set
            )

            # merge them
            out = (
                cv2.bitwise_and(src1=blend, src2=blend, mask=blend_mask)
                + cv2.bitwise_and(src1=bg, src2=bg, mask=bg_mask)
                + cv2.bitwise_and(src1=fg, src2=fg, mask=fg_mask)
            )
            o[top : top + ph, left : left + pw] = out

        else:
            raise ValueError("method is not valid")

        if copy:
            return self.__class__(o, copy=copy, colororder=colororder)
        else:
            self._A = o
            return self

    def invert(self) -> "Image":
        r"""
        Invert image

        :return: _description_
        :rtype: _type_

        Low becomes high and high becomes low.  The resulting image has the same
        datatype and size as the original image.  The transformation is:

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
            >>> img.invert().print()
        """
        if self.isint:
            out = self.maxval + self.minval - self._A
        elif self.isfloat:
            out = 1.0 - self._A
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


if __name__ == "__main__":
    from pathlib import Path

    import pytest

    # tests = Path(__file__).parent.parent.parent / "tests"
    # pytest.main(
    #     [
    #         str(tests / "test_image_processing.py"),
    #         str(tests / "test_image_processing_kernel.py"),
    #         "-v",
    #     ]
    # )
