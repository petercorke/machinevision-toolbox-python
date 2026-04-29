"""
Whole-image feature computation: moments, Hu invariants, histograms, and entropy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.patches import Polygon
from matplotlib.ticker import ScalarFormatter
from spatialmath import SE3, base

if TYPE_CHECKING:
    from machinevisiontoolbox._image_typing import _ImageBase

from machinevisiontoolbox.base import findpeaks, findpeaks2d
from machinevisiontoolbox.base.imageio import safe_plt_show, set_window_title
from machinevisiontoolbox.mvtb_types import *


class ImageWholeFeaturesMixin(_ImageBase if TYPE_CHECKING else object):

    # ------------------ scalar statistics ----------------------------- #

    def sum(self, **kwargs: Any) -> np.generic | np.ndarray:
        r"""
        Sum of all pixels

        :param kwargs: additional keyword arguments passed to :func:`numpy.nansum`
        :type kwargs: dict
        :return: sum

        Computes the sum of pixels in the image:

        .. math::

            \sum_{uvc} I_{uvc}

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.sum()  # sum over all planes
            >>> img.sum(axis=(0,1))  # per-plane sum

        .. note::
            - The return value type is the same as the image type.
            - NaN values are ignored in the sum
            - For ``axis`` remember the axes are in NumPy order: 0=rows, 1=columns, 2=planes.

        :seealso: :func:`numpy.sum` :meth:`numnan` :meth:`~~machinevisiontoolbox.ImageWholeFeatures.ImageWholeFeaturesMixin.mpq`
            :meth:`~machinevisiontoolbox.ImageWholeFeatures.ImageWholeFeaturesMixin.npq`
            :meth:`~machinevisiontoolbox.ImageWholeFeatures.ImageWholeFeaturesMixin.upq`
        """
        return np.nansum(self._A, **kwargs)

    def min(self, **kwargs: Any) -> np.generic | np.ndarray:
        """
        Minimum value of all pixels

        :param kwargs: additional keyword arguments passed to :func:`numpy.nanmin`
        :type kwargs: dict
        :return: minimum value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.min() # min over all planes
            >>> img.min(axis=(0,1))  # per-plane min

        .. note::
            - The return value type is the same as the image type.
            - NaN values are ignored
            - For ``axis`` remember the axes are in NumPy order: 0=rows, 1=columns, 2=planes.

        :seealso: :meth:`stats` :meth:`hist` :meth:`Hough` :meth:`max` :func:`numpy.nanmin` :meth:`numnan`
        """
        return np.nanmin(self._A, **kwargs)

    def max(self, **kwargs: Any) -> np.generic | np.ndarray:
        """
        Maximum value of all pixels

        :param kwargs: additional keyword arguments passed to :func:`numpy.nanmax`
        :type kwargs: dict
        :return: maximum value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.max()  # max over all planes
            >>> img.max(axis=(0,1))  # per-plane max

        .. note::
            - The return value type is the same as the image type.
            - NaN values are ignored
            - For ``axis`` remember the axes are in NumPy order: 0=rows, 1=columns, 2=planes.

        :seealso: :meth:`stats` :meth:`hist` :meth:`min` :func:`numpy.nanmax` :meth:`numnan`
        """
        return np.nanmax(self._A, **kwargs)

    def mean(self, **kwargs: Any) -> np.generic | np.ndarray:
        """
        Mean value of all pixels

        :param kwargs: additional keyword arguments passed to :func:`numpy.nanmean`
        :type kwargs: dict
        :return: mean value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.mean()  # mean over all planes
            >>> img.mean(axis=(0,1))  # per-plane mean

        .. note::
            - The return value type is the same as the image type.
            - NaN values are ignored
            - For ``axis`` remember the axes are in NumPy order: 0=rows, 1=columns, 2=planes.

        :seealso: :meth:`stats` :meth:`hist` :meth:`Hough` :meth:`std` :meth:`median` :func:`numpy.nanmean` :meth:`numnan`
        """
        return np.nanmean(self._A, **kwargs)

    def std(self, **kwargs: Any) -> np.generic | np.ndarray:
        """
        Standard deviation of all pixels

        :param kwargs: additional keyword arguments passed to :func:`numpy.nanstd`
        :type kwargs: dict
        :return: standard deviation value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.std()  # std over all planes
            >>> img.std(axis=(0,1))  # per-plane std

        .. note::
            - The return value type is the same as the image type.
            - NaN values are ignored
            - For ``axis`` remember the axes are in NumPy order: 0=rows, 1=columns, 2=planes.

        :seealso: :meth:`stats` :meth:`hist` :meth:`mean` :meth:`var` :func:`numpy.nanstd` :meth:`numnan`
        """
        return np.nanstd(self._A, **kwargs)

    def var(self, **kwargs: Any) -> np.generic | np.ndarray:
        """
        Variance of all pixels

        :param kwargs: additional keyword arguments passed to :func:`numpy.nanvar`
        :type kwargs: dict
        :return: variance value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.var()  # variance over all planes
            >>> img.var(axis=(0,1))  # per-plane variance

        .. note::
            - The return value type is the same as the image type.
            - NaN values are ignored
            - For ``axis`` remember the axes are in NumPy order: 0=rows, 1=columns, 2=planes.

        :seealso: :meth:`stats` :meth:`hist` :meth:`std` :func:`numpy.nanvar` :meth:`numnan`
        """
        return np.nanvar(self._A, **kwargs)

    def median(self, **kwargs: Any) -> np.generic | np.ndarray:
        """
        Median value of all pixels

        :param kwargs: additional keyword arguments passed to :func:`numpy.nanmedian`
        :type kwargs: dict
        :return: median value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.median()  # median over all planes
            >>> img.median(axis=(0,1))  # per-plane median

        .. note::
            - The return value type is the same as the image type.
            - NaN values are ignored
            - For ``axis`` remember the axes are in NumPy order: 0=rows, 1=columns, 2=planes.

        :seealso: :meth:`stats` :meth:`hist` :meth:`mean` :meth:`std` :func:`numpy.nanmedian` :meth:`numnan`
        """
        return np.nanmedian(self._A, **kwargs)

    @staticmethod
    def _format_stats(stats: dict[str, float | int]) -> str:
        s = (
            f"span=[{stats['min']:g}, {stats['max']:g}]; "
            f"mean={stats['mean']:g}, "
            f"𝜎={stats['sdev']:g}; "
            f"median={stats['median']:g}"
        )
        nnan = stats["nnan"]
        ninf = stats["ninf"]
        if nnan + ninf > 0:
            s += " (contains "
            if nnan > 0:
                s += f"{nnan}xNaN{'s' if nnan > 1 else ''}"
            if ninf > 0:
                s += f" {ninf}xInf{'s' if ninf > 1 else ''}"
            s += ")"
        return s

    @property
    def stats(self) -> dict[str, Any]:
        """
        Pixel value statistics

        :return: statistics dictionary; for greyscale keys are ``min``, ``max``,
            ``mean``, ``sdev``, ``median``, ``nnan``, ``ninf``; for color images
            returns a dictionary mapping plane name to that same dictionary
        :rtype: dict

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('street.png')
            >>> img.stats
            >>> img = Image.Read('flowers1.png')
            >>> img.stats

        .. note::

            - Returns statistics as a dictionary and does not print.
            - NaN values are ignored for scalar statistics and reported via
              ``nnan`` and ``ninf`` counts.

        :seealso: :meth:`printstats` :meth:`hist` :meth:`min` :meth:`max` :meth:`mean` :meth:`std` :meth:`median`
        """

        return self._stats

    def printstats(self) -> None:
        """
        Print pixel value statistics

        Prints a concise summary of image statistics.

        :seealso: :meth:`stats`
        """

        stats = self._stats
        if self.iscolor and self.colororder is not None:
            colororder = self.colororder
            for k in sorted(stats.keys(), key=lambda x: colororder[x]):
                print(f"{k:s}: {self._format_stats(stats[k])}")
        else:
            print(self._format_stats(stats))

    # ------------------ histogram ------------------------------------- #
    def hist(
        self,
        nbins: int = 256,
        sorted: bool | str = False,
        span: Literal["dtype"] | tuple[float, float] | None = "dtype",
        clip: bool = False,
        opt: str | None = None,
    ) -> "Histogram":
        """
        Image histogram

        :param nbins: number of histogram bins, defaults to 256
        :type nbins: int, optional
        :param sorted: sort bins by count rather than value, defaults to False
        :type sorted: bool
        :param span: histogram span definition, defaults to ``'dtype'``;
            ``'dtype'`` means full span of image dtype if integer or [0,1] for float, ``None`` means finite data
            min/max, ``(a,b)`` means explicit range
        :type span: str, tuple(float, float), None
        :param clip: clip out-of-span values to span endpoints before binning,
            defaults to False. If False, out-of-span values are ignored.
        :type clip: bool, optional
        :param opt: deprecated histogram option string, use ``sorted`` instead;
            supported legacy value is ``'sorted'``
        :type opt: str, optional
        :return: histogram of image
        :rtype: :class:`~machinevisiontoolbox.ImageWholeFeatures.Histogram`

        Returns an object that summarizes the distribution of
        pixel values in each color plane.

        Example:

        .. code-block:: python

            from machinevisiontoolbox import Image
            img = Image.Read('street.png')
            type(hist)
            hist = img.hist()
            hist
            hist.plot()


        .. plot::

            from machinevisiontoolbox import Image
            img = Image.Read('street.png')
            img.hist().plot()

        Example:

        .. code-block:: python

            from machinevisiontoolbox import Image
            img = Image.Read('flowers1.png')
            hist = img.hist()
            hist
            hist.plot(style='stack')


        .. plot::

            from machinevisiontoolbox import Image
            img = Image.Read('flowers1.png')
            img.hist().plot(style='stack')

        Example:

        .. code-block:: python

            from machinevisiontoolbox import Image
            img = Image.Read('flowers1.png')
            hist = img.hist()
            hist
            hist.plot(style='overlay')


        .. plot::

            from machinevisiontoolbox import Image
            img = Image.Read('flowers1.png')
            img.hist().plot(style='overlay')

        .. note::
            - Histogram horizontal range is controlled by ``span``.
            - ``span='dtype'`` uses full datatype span (for example uint8: 0-255,
                float: 0.0-1.0).
            - ``span=None`` uses the finite min/max of image data.
            - Values outside ``span`` are ignored unless ``clip=True``,
                in which case they are clipped to the span endpoints.
            - For floating point images all NaN and Inf values are removed before
                computing the histogram.
            - Histogram is computed using ``numpy.histogram`` independently for each plane.

        :references:
            - |RVC3|, Section 14.4.3.

        :seealso:
            :meth:`stats`
            :class:`~machinevisiontoolbox.ImageWholeFeatures.Histogram`
            :func:`numpy.histogram`
        """

        return Histogram(
            self,
            nbins=nbins,
            sorted=sorted,
            span=span,
            clip=clip,
            opt=opt,
        )

    def _default_hist(self) -> "Histogram":
        """Return cached default histogram, computing it on first access.

        The cached histogram corresponds to ``hist()`` called with default
        arguments.
        """
        hist = getattr(self, "_hist_default_cache", None)
        if hist is None:
            hist = self.hist()
            self._hist_default_cache = hist
        return hist

    @property
    def h(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Histogram count values and bin values


        :return: tuple ``(h, x)`` where ``h`` is histogram counts and ``x`` is
            left-hand bin values
        :rtype: tuple(ndarray(N) or ndarray(N,P), ndarray(N))

        This is the raw histogram count: the number of pixels in the image
        plane with grey values in each bin.  For a greyscale image ``h`` is a
        1D array; for a multiplane (color) image it is a 2D array with the
        histogram of each plane as a column.

        .. versionchanged:: 2.0.0
            Returns ``(h, x)`` instead of only ``h``.  This convenience
            property now uses a lazily cached default histogram.

        :seealso: :meth:`pdf` :meth:`cf` :meth:`cdf` :meth:`Histogram.h`
        """
        hist = self._default_hist()
        return hist.h, hist.x

    @property
    def pdf(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Empirical probability density function (PDF) and bin values

        :return: tuple ``(pdf, x)`` where ``pdf`` is the normalised histogram
            counts and ``x`` is left-hand bin values
        :rtype: tuple(ndarray(N) or ndarray(N,P), ndarray(N))

        This is the histogram count normalised such that the sum is 1.0,
        computed by dividing the histogram counts by the total number of pixels
        in each plane.  For a greyscale image ``pdf`` is a 1D array; for a
        multiplane (color) image it is a 2D array with the PDF of each plane
        as a column.

        :seealso: :meth:`h` :meth:`cf` :meth:`cdf` :meth:`Histogram.pdf`
        """
        hist = self._default_hist()
        return hist.pdf, hist.x

    @property
    def cf(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Cumulative frequency and bin values

        :return: tuple ``(cf, x)`` where ``cf`` is cumulative histogram count
            values and ``x`` is left-hand bin values
        :rtype: tuple(ndarray(N) or ndarray(N,P), ndarray(N))

        This is the cumulative histogram count, which increases from 0 to the
        total number of pixels in the image plane.  For a greyscale image
        ``cf`` is a 1D array; for a multiplane (color) image it is a 2D array
        with the cumulative histogram of each plane as a column.

        .. versionchanged:: 2.0.0
            Was previously called ``cdf``, and now returns ``(cf, x)`` instead of only ``cf``.  This convenience
            property now uses a lazily cached default histogram.

        :seealso: :meth:`h` :meth:`pdf` :meth:`cdf` :meth:`Histogram.cf`
        """
        hist = self._default_hist()
        return hist.cf, hist.x

    @property
    def cdf(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Cumulative distribution function and bin values

        :return: tuple ``(cdf, x)`` where ``cdf`` is normalised cumulative
            histogram values and ``x`` is left-hand bin values
        :rtype: tuple(ndarray(N) or ndarray(N,P), ndarray(N))

        The CDF is the cumulative frequency (``cf``) normalised to the range 0 to 1,
        computed by dividing the cumulative frequency by the total number of
        pixels in each plane.  For a greyscale image ``cdf`` is a 1D array;
        for a multiplane (color) image it is a 2D array with the normalised
        cumulative histogram of each plane as a column.

        .. versionchanged:: 2.0.0
            Was previously called ``ncdf``, and now returns ``(cdf, x)`` instead of only ``cdf``.  This convenience
            property now uses a lazily cached default histogram.

        :seealso: :meth:`h` :meth:`cf` :meth:`pdf` :meth:`Histogram.cdf`
        """
        hist = self._default_hist()
        return hist.cdf, hist.x

    @property
    def ncdf(self) -> np.ndarray:
        """
        Normalized cumulative distribution function

        :return: ``cdf`` is the normalised cumulative histogram values
        :rtype: ndarray(N) or ndarray(N,P)

        .. deprecated:: 1.1.0
            Use ``hist().cdf`` instead.
        """
        hist = self._default_hist()
        return hist.cdf

    def peaks(self, **kwargs: Any) -> np.ndarray | list[np.ndarray]:
        """
        Histogram peaks

        :param kwargs: parameters passed to
            :func:`~machinevisiontoolbox.base.findpeaks.findpeaks`
        :return: positions of histogram peaks
        :rtype: ndarray(M) or list of ndarray

        For a greyscale image returns an array of grey values corresponding to
        local maxima.  For a color image returns a list of such arrays, one per
        plane.

        :seealso: :meth:`h` :meth:`Histogram.peaks`
        """
        hist = self._default_hist()
        return hist.peaks(**kwargs)

    # ------------------ moments --------------------------------------- #

    def mpq(self, p: int, q: int) -> float:
        r"""
        Image moments

        :param p: u exponent
        :type p: int
        :param q: v exponent
        :type q: int
        :return: moment
        :type: scalar

        Computes the pq'th moment of the image:

        .. math::

            m(I) = \sum_{uv} I_{uv} u^p v^q

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('shark1.png')
            >>> img.mpq(1, 0)

        .. note::
            - Supports single channel images only.
            - ``mpq(0, 0)`` is the same as ``sum()`` but less efficient.

        :references:
            - |RVC3|, Section 12.1.3.4.

        :seealso: :meth:`sum` :meth:`npq` :meth:`upq`
        """

        if not isinstance(p, int) or not isinstance(q, int):
            raise TypeError(p, "p, q must be an int")

        im = self.mono()._A
        X, Y = self.meshgrid()
        return np.sum(im * (X**p) * (Y**q))

    def upq(self, p: int, q: int) -> float:
        r"""
        Central image moments

        :param p: u exponent
        :type p: int
        :param q: v exponent
        :type q: int
        :return: moment
        :type: scalar

        Computes the pq'th central moment of the image:

        .. math::

            \mu(I) = \sum_{uv} I_{uv} (u-u_0)^p (v-v_0)^q

        where :math:`u_0 = m_{10}(I) / m_{00}(I)` and :math:`v_0 = m_{01}(I) / m_{00}(I)`.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('shark1.png')
            >>> img.upq(2, 2)

        .. note::
            - The central moments are invariant to translation.
            - Supports single channel images only.

        :references:
            - |RVC3|, Section 12.1.3.4.

        :seealso: :meth:`sum` :meth:`mpq` :meth:`upq`
        """

        if not isinstance(p, int) or not isinstance(q, int):
            raise TypeError(p, "p, q must be an int")

        m00 = self.mpq(0, 0)
        xc = self.mpq(1, 0) / m00
        yc = self.mpq(0, 1) / m00

        im = self.mono()._A
        X, Y = self.meshgrid()

        return np.sum(im * ((X - xc) ** p) * ((Y - yc) ** q))

    def npq(self, p: int, q: int) -> float:
        r"""
        Normalized central image moments

        :param p: u exponent
        :type p: int
        :param q: v exponent
        :type q: int
        :return: moment
        :type: scalar

        Computes the pq'th normalized central moment of the image:

        .. math::

            \nu(I) = \frac{\mu_{pq}(I)}{m_{00}(I)} = \frac{1}{m_{00}(I)} \sum_{uv} I_{uv} (u-u_0)^p (v-v_0)^q

        where :math:`u_0 = m_{10}(I) / m_{00}(I)` and :math:`v_0 = m_{01}(I) / m_{00}(I)`.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('shark1.png')
            >>> img.npq(2, 2)

        .. note::
            - The normalized central moments are invariant to translation and
              scale.
            - Supports single channel images only.

        :references:
            - |RVC3|, Section 12.1.3.4.

        :seealso: :meth:`sum` :meth:`mpq` :meth:`upq`
        """
        if not isinstance(p, int) or not isinstance(q, int):
            raise TypeError(p, "p, q must be an int")
        if (p + q) < 2:
            raise ValueError(p + q, "normalized moments only valid for p+q >= 2")

        g = (p + q) / 2 + 1

        return self.upq(p, q) / self.mpq(0, 0) ** g

    def moments(self, binary: bool = False) -> dict[str, float]:
        """
        Image moments

        :param binary: if True, all non-zero pixels are treated as 1's
        :type binary: bool
        :return: image moments
        :type: dict

        Compute multiple moments of the image and return them as a dict

        ==========================  ===============================================================================
        Moment type                 dict keys
        ==========================  ===============================================================================
        moments                     ``m00`` ``m10`` ``m01`` ``m20`` ``m11`` ``m02`` ``m30`` ``m21`` ``m12`` ``m03``
        central moments             ``mu20`` ``mu11`` ``mu02`` ``mu30`` ``mu21`` ``mu12`` ``mu03`` |
        normalized central moments  ``nu20`` ``nu11`` ``nu02`` ``nu30`` ``nu21`` ``nu12`` ``nu03`` |
        ==========================  ===============================================================================

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('shark1.png')
            >>> img.moments()

        .. note::
            - Converts a color image to greyscale.

        :references:
            - |RVC3|, Section 12.1.3.4.

        .. important:: Uses OpenCV function ``cv2.moments`` which accepts single-channel, CV_8U, CV_16U, CV_16S, CV_32F or CV_64F images (color images are automatically converted to greyscale).

        :seealso: :meth:`mpq` :meth:`npq` :meth:`upq` `opencv.moments <https://docs.opencv.org/4.x/d8/d23/classcv_1_1Moments.html>`_
        """
        return cv2.moments(array=self.mono().to_int(), binaryImage=binary)

    def humoments(self) -> np.ndarray:
        """
        Hu image moment invariants

        :return: Hu image moments
        :rtype: ndarray(7)

        Computes the seven Hu image moment invariants of the image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('shark1.png', dtype='float')
            >>> img.humoments()

        .. note::
            - Image is assumed to be a binary image of a single connected
              region
            - These invariants are a function of object shape and are invariant
              to position, orientation and scale.

        :references:
            - M-K. Hu, Visual pattern recognition by moment invariants. IRE
              Trans. on Information Theory, IT-8:pp. 179-187, 1962.
            - |RVC3|, Section 12.1.3.6.

        .. important:: Uses OpenCV functions ``cv2.moments`` and ``cv2.HuMoments`` which accept single-channel, CV_8U, CV_16U, CV_16S, CV_32F or CV_64F images.

        :seealso: :meth:`moments` `opencv.HuMoments <https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gab001db45c1f1af6cbdbe64df04c4e944>`_
        """

        # TODO check for binary image

        self._opencv_type_check(
            self._A, "single-channel", "CV_8U", "CV_16U", "CV_16S", "CV_32F", "CV_64F"
        )
        moments = cv2.moments(array=self._A)
        hu = cv2.HuMoments(m=moments)
        return hu.flatten()

    # ------------------ pixel values --------------------------------- #

    def nonzero(self) -> np.ndarray:
        """
        Find non-zero pixel values as 2D coordinates

        :return: coordinates of non-zero pixels
        :rtype: ndarray(2,N)

        The (u,v) coordinates are given as columns of the returned array.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> pix = np.zeros((10,10)); pix[2,3]=10; pix[4,5]=11; pix[6,7]=12
            >>> img = Image(pix)
            >>> img.nonzero()

        :references:
            - |RVC3|, Section 12.1.3.2.

        :seealso: :meth:`flatnonzero`
        """
        v, u = np.nonzero(self._A)
        return np.vstack((u, v))

    def flatnonzero(self) -> np.ndarray:
        """
        Find non-zero pixel values as 1D indices

        :return: index of non-zero pixels
        :rtype: ndarray(N)

        The coordinates are given as 1D indices into a flattened version of
        the image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> pix = np.zeros((10,10)); pix[2,3]=10; pix[4,5]=11; pix[6,7]=12
            >>> img = Image(pix)
            >>> img.flatnonzero()
            >>> img.view1d()[23]

        :seealso: :meth:`view1d` :meth:`nonzero`
        """
        return np.flatnonzero(self._A)

    def peak2d(
        self,
        npeaks: int = 2,
        scale: int = 1,
        interp: bool = False,
        positive: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Find local maxima in image

        :param npeaks: number of peaks to return, defaults to 2
        :type npeaks: int, optional
        :param scale: scale of peaks to consider
        :type scale: int
        :param interp:  interpolate the peak positions, defaults to False
        :type interp: bool, optional
        :param positive:  only select peaks that are positive, defaults to False
        :type positive: bool, optional
        :return: peak magnitude and positions
        :rtype: ndarray(npeaks), ndarray(2,npeaks)

        Find the positions of the local maxima in the image.  A local maxima
        is the largest value within a sliding window of width
        :math:`2 \mathtt{scale}+1`.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> peak = np.array([[10, 20, 30], [40, 50, 45], [15, 20, 30]])
            >>> img = Image(np.pad(peak, 3, constant_values=10))
            >>> img.A
            >>> img.peak2d(interp=True)

        .. note::
            - Edges elements will never be returned as maxima.
            - To find minima, use ``peak2d(-image)``.
            - The ``interp`` option fits points in the neighbourhood about the
              peak with a paraboloid and its peak position is returned.

        :seealso: :meth:`~machinevisiontoolbox.base.findpeaks.findpeaks2d`
        """

        ret = findpeaks2d(self._A, npeaks=npeaks, scale=scale, interp=interp)
        return ret[:, -1], ret[:, :2].T


class Histogram:
    def __init__(
        self,
        image: ImageWholeFeaturesMixin,
        nbins: int = 256,
        sorted: bool | str = False,
        span: Literal["dtype"] | tuple[float, float] | None = "dtype",
        clip: bool = False,
        opt: str | None = None,
    ) -> None:
        """
        Create histogram instance

        :param image: image to histogram
        :type image: Image
        :param nbins: number of histogram bins, defaults to 256
        :type nbins: int, optional
        :param sorted: sort bins by count rather than value, defaults to False
        :type sorted: bool
        :param span: histogram span definition, defaults to ``'dtype'``;
            ``'dtype'`` means full span of image type, ``None`` means finite data
            min/max, ``(a,b)`` means explicit range
        :type span: str, tuple(float, float), None
        :param clip: clip out-of-span values to span endpoints before binning,
            defaults to False. If False, out-of-span values are ignored.
        :type clip: bool, optional
        :param opt: deprecated histogram option string, use ``sorted`` instead;
            supported legacy value is ``'sorted'``
        :type opt: str, optional

        Create :class:`Histogram` instance by computing the histogram over each image plane.

        :seealso: :meth:`~machinevisiontoolbox.ImageFeatures.ImageFeaturesMixin.hist`
        """
        if nbins <= 0:
            raise ValueError("nbins must be > 0")

        import warnings

        if isinstance(sorted, str):
            if sorted == "sorted":
                warnings.warn(
                    "Deprecated in 1.1.0: use hist(sorted=True) instead of hist(sorted='sorted').",
                    DeprecationWarning,
                    stacklevel=2,
                )
                sorted = True
            else:
                raise ValueError("sorted string value must be 'sorted'")

        if opt is not None:
            if opt != "sorted":
                raise ValueError("opt value must be 'sorted'")
            warnings.warn(
                "Deprecated in 1.1.0: use hist(sorted=True) instead of hist(opt='sorted').",
                DeprecationWarning,
                stacklevel=2,
            )
            sorted = True

        if not isinstance(sorted, bool):
            raise TypeError("sorted must be bool")
        if not isinstance(clip, bool):
            raise TypeError("clip must be bool")

        if span == "dtype":
            if np.issubdtype(image.dtype, np.bool_):
                xrange = (0.0, 1.0)
            elif np.issubdtype(image.dtype, np.integer):
                dtype_info = np.iinfo(image.dtype)
                xrange = (float(dtype_info.min), float(dtype_info.max))
            else:
                xrange = (0.0, 1.0)
        elif span is None:
            values = image.A.reshape(-1)
            if np.issubdtype(values.dtype, np.floating):
                values = values[np.isfinite(values)]
            if values.size == 0:
                raise ValueError("cannot determine span from empty finite image data")
            xrange = (float(np.min(values)), float(np.max(values)))
        elif isinstance(span, tuple) and len(span) == 2:
            xrange = (float(span[0]), float(span[1]))
        else:
            raise ValueError("span must be 'dtype', None or a 2-tuple")

        if not np.isfinite(xrange[0]) or not np.isfinite(xrange[1]):
            raise ValueError("span limits must be finite")
        if xrange[1] <= xrange[0]:
            raise ValueError("span max must be greater than span min")

        x = np.linspace(xrange[0], xrange[1], nbins, endpoint=True)

        hc = []
        for i in range(image.nplanes):
            if image.nplanes == 1:
                plane = image._A
            else:
                plane = image._A[..., i]

            values = plane.reshape(-1)
            if np.issubdtype(values.dtype, np.floating):
                values = values[np.isfinite(values)]
            if clip:
                values = np.clip(values, xrange[0], xrange[1])

            h, _ = np.histogram(values, bins=nbins, range=xrange)
            hc.append(h.astype(np.int64))

        hs = np.column_stack(hc)

        self.nplanes = hs.shape[1]
        self._sorted = sorted
        if self._sorted:
            if self.nplanes != 1:
                raise ValueError("sorted histogram is only supported for mono images")
            order = np.argsort(hs[:, 0])[::-1]
            hs = hs[order, :]
            x = np.arange(nbins, dtype=float)

        if self.nplanes == 1:
            self._h = hs[:, 0]
        else:
            self._h = hs

        if image.isint:
            x = x.astype(int)
        self._x = x.flatten()
        self._xrange = xrange
        self.isfloat = image.isfloat
        self.colordict = image.colororder
        # 'hist', 'h cdf normcdf x')

    def __str__(self) -> str:
        """
        Histogram summary as a string

        :return: concise summary of histogram
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Random(size=256)
            >>> h = im.hist(100)
            >>> print(h)
        """
        s = f"histogram with {len(self.x)} bins"
        if self.nplanes > 1:
            s += f" x {self._h.shape[1]} planes"
        s += f": xrange {self.x[0]} - {self.x[-1]}, yrange {int(np.min(self._h))} - {int(np.max(self._h))}"
        return s

    def __repr__(self) -> str:
        """
        Print histogram summary

        :return: print concise summary of histogram
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Random(size=256)
            >>> h = im.hist(100)
            >>> h
        """
        return f"Histogram(nbins={len(self.x)}, nplanes={self.nplanes}, xrange=({self.x[0]}, {self.x[-1]}), yrange=({int(np.min(self._h))}, {int(np.max(self._h))}))"

    @property
    def x(self) -> np.ndarray:
        """
        Histogram bin values

        :return: array of left-hand bin values
        :rtype: ndarray(N)

        Bin :math:`i` contains grey values in the range :math:`[x_{[i]}, x_{[i+1]})`.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> hist = Image.Read('flowers1.png').hist()
            >>> with np.printoptions(threshold=10):
            ...     hist.x
        """
        return self._x

    @property
    def h(self) -> np.ndarray:
        """
        Histogram count values

        :return: array of histogram count values
        :rtype: ndarray(N) or ndarray(N,P)

        This is the raw histogram count, which is the number of pixels in the image plane with grey values equal to the index of the bin.

        For a greyscale image this is a 1D array, for a multiplane (color) image
        this is a 2D array with the histograms of each plane as columns.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> hist = Image.Read('flowers1.png').hist()
            >>> with np.printoptions(threshold=10):
            ...     hist.h

        :seealso: :meth:`pdf` :meth:`cf` :meth:`cdf`
        """
        return self._h

    @property
    def pdf(self) -> np.ndarray:
        """
        Empirical probability density function (PDF)

        :return: array of cumulative histogram count values
        :rtype: ndarray(N) or ndarray(N,P)

        This the histogram count normalized such that the sum is 1.0.  This is computed by dividing the histogram counts by the total number of pixels in each plane.

        For a greyscale image this is a 1D array, for a multiplane (color) image
        this is a 2D array with the cumulative histograms of each plane as columns.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> hist = Image.Read('flowers1.png').hist()
            >>> with np.printoptions(threshold=10):
            ...     hist.pdf

        :seealso: :meth:`h` :meth:`cf` :meth:`cdf`
        """
        y = self._h.astype(float)
        sum = np.sum(y, axis=0)
        if self.nplanes == 1:
            y = y / sum
        else:
            y = y / sum[np.newaxis, :]
        return y

    @property
    def cf(self) -> np.ndarray:
        """
        Cumulative histogram count values

        :return: array of cumulative histogram count values
        :rtype: ndarray(N) or ndarray(N,P)

        This the cumulative histogram count, which increases from 0 to the total number of pixels in the image plane.

        For a greyscale image this is a 1D array, for a multiplane (color) image
        this is a 2D array with the cumulative histograms of each plane as columns.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> hist = Image.Read('flowers1.png').hist()
            >>> with np.printoptions(threshold=10):
            ...     hist.cf

        .. versionchanged:: 2.0.0
            In earlier releases this method was named ``cdf``.

        :seealso: :meth:`h` :meth:`pdf` :meth:`cdf`
        """
        return np.cumsum(self._h, axis=0)

    @property
    def cdf(self) -> np.ndarray:
        """
        Cumulative distribution function

        :return: array of normalized cumulative histogram count values
        :rtype: ndarray(N) or ndarray(N,P)

        The cdf is the cumulative frequency normalized to the range 0 to 1.  This is computed by dividing the cumulative frequency by the total number of pixels in each plane.

        For a greyscale image this is a 1D array, for a multiplane (color) image
        this is a 2D array with the normalized cumulative histograms of each
        plane as columns.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> hist = Image.Read('flowers1.png').hist()
            >>> with np.printoptions(threshold=10):
            ...    hist.cdf

        .. versionchanged:: 2.0.0
            In earlier releases this method was named ``ncdf``.

        :seealso: :meth:`h` :meth:`pdf` :meth:`cf`
        """
        y = np.cumsum(self._h, axis=0)

        if self.nplanes == 1:
            y = y / y[-1]
        else:
            y = y / y[-1, :]
        return y

    @property
    def ncdf(self) -> np.ndarray:
        """
        Normalized cumulative distribution function

        :return: ``cdf`` is the normalised cumulative histogram values
        :rtype: ndarray(N) or ndarray(N,P)

        .. deprecated:: 2.0.0
            Use ``hist().cdf`` instead.
        """
        return self.cdf

    def plot(
        self,
        type="frequency",
        block=False,
        filled=None,
        stats=True,
        style="stack",
        cursor=False,
        alpha=0.5,
        title=None,
        log=False,
        samescale=True,
        ax=None,
        bar=None,
        **kwargs,
    ):
        """
        Plot histogram

        :param type: histogram type, one of: 'frequency' [default], 'cdf', 'ncdf'
        :type type: str, optional
        :param block: hold plot, defaults to False
        :type block: bool, optional
        :param filled: use a filled stairs plot, defaults to True for frequency plot, False for
            other plots
        :type filled: bool, optional
        :param stats: draw vertical lines for mean (solid) and median (dashed), defaults to True
        :type stats: bool, optional
        :param style: Style for multiple plots, one of: 'stack' [default], 'overlay'
        :type style: str, optional
        :param cursor: enable interactive data cursor for stacked line plots, defaults
            to False.  Cursor is ignored for ``overlay`` style.
        :type cursor: bool, optional
        :param alpha: transparency for overlay plot, defaults to 0.5
        :type alpha: float, optional
        :param title: plot title, defaults to None
        :type title: str, optional
        :param log: use logarithmic y-axis, defaults to False
        :type log: bool, optional
        :param samescale: synchronize the y-axis scale for all subplots, defaults to True
        :type samescale: bool, optional
        :param ax: Matplotlib axes to plot on, defaults to None (new figure)
        :type ax: matplotlib.axes.Axes, optional
        :param bar: deprecated alias for ``filled``, use ``filled=`` instead
        :param kwargs: additional keyword arguments passed to Matplotlib plotting functions, ``plt.stairs`` for ``solid=True`` and ``plt.plot`` for ``solid=False``, and ``plt.Polygon`` for ``style='overlay'``
        :raises ValueError: invalid histogram type
        :raises ValueError: cannot use overlay style for 1-channel histogram

        Plots the histogram using Matplotlib.  For a color image, the histograms of each
        plane are plotted separately.  The ``type`` option selects the type of histogram
        to plot: ``frequency``, ``cdf`` or ``ncdf`` (normalized cumulative in the range 0 to 1).

        The ``style`` option selects the style for plotting multiple planes:

        - ``stack``: plot each plane in a separate subplot. The ``cursor`` option
          enables an interactive data cursor which displays the histogram values at the
          cursor position.  The ``filled`` option selects whether to use a filled stairs
          plot.
        - ``overlay``: plot all planes in the same axes with different colors. The
          ``filled`` option selects whether to use a filled plot. The ``alpha`` option
          controls the transparency of the bars in the ''overlay'' style.

        ``samescale`` controls whether the y-axis scale is the same for all planes (if
        False, each plane is scaled independently).  By default, the y-axis scale is the
        same for all planes, which allows for direct comparison of histogram values
        across planes.

        The ``log`` option uses a logarithmic scale for the y-axis, which can be useful
        for visualizing histograms with a large dynamic range, zero values are ignored in log plots.

        If ``ax`` is provided, the histogram is plotted on the given Matplotlib axes.
        For multi-plane histograms the ``style`` is overridden to 'overlay' since all
        the plots must be on one axes.   If ``ax`` is not provided, a new figure and
        axes are created.

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('street.png')
            >>> hist = im.hist()
            >>> print(hist)
            >>> hist.plot() # standard frequency plot
            >>> hist.plot(type='cdf') # CDF plot

        .. plot::

            from machinevisiontoolbox import Image
            im = Image.Read('street.png')
            hist = im.hist()
            hist.plot() # standard frequency plot

        .. plot::

            from machinevisiontoolbox import Image
            im = Image.Read('street.png')
            hist = im.hist()
            hist.plot(type='cdf') # CDF plot

        For multi-plane (color) images, the histograms of each plane can be plotted separately in stacked subplots or overlaid on the same axes.

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.hist().plot(style='stack', filled=True)
            >>> im.hist().plot(style='overlay', filled=True, alpha=0.5)

        .. plot::

            from machinevisiontoolbox import Image
            im = Image.Read('flowers1.png')
            im.hist().plot(style='stack', filled=True)

        .. plot::

            from machinevisiontoolbox import Image
            im = Image.Read('flowers1.png')
            im.hist().plot(style='overlay', filled=True, alpha=0.5)

        """

        x = self._x[:]

        import warnings

        if bar is not None:
            warnings.warn(
                "Deprecated in 1.1.0: use solid= instead of bar=.",
                DeprecationWarning,
                stacklevel=2,
            )
            if filled is None:
                filled = bar

        if type not in ("frequency", "pdf"):
            stats = False  # only show stats for frequency and pdf plot

        if ax is not None and self.nplanes > 1:
            # if an axis is provided, we must plot all planes on the same axis
            style = "overlay"

        if style == "overlay":
            # for overlay style, we use the same y-axis scale for all planes
            samescale = True

        # figure vertical axis labels and scaling based on histogram type
        if type == "frequency":
            y = self.h
            if samescale:
                maxy = np.max(y)
            else:
                maxy = np.max(y, axis=0)
            ylabel1 = "frequency"
            ylabel2 = "frequency"
            if filled is None:
                filled = True
        elif type in ("pdf", "probability"):
            y = self.pdf
            if samescale:
                maxy = np.max(y)
            else:
                maxy = np.max(y, axis=0)
            ylabel1 = "PDF"
            ylabel2 = "probability density"
        elif type in ("cf", "cumulative"):
            y = self.cf
            maxy = y[
                -1, 0
            ]  # last row values are all the same, total number of pixels in plane
            ylabel1 = "cumulative frequency"
            ylabel2 = "cumulative frequency"
        elif type in ("cdf", "normalized"):
            y = self.cdf
            maxy = 1
            ylabel1 = "CDF"
            ylabel2 = "normalized cumulative frequency"
        else:
            raise ValueError("unknown type")

        if self.nplanes == 1:
            y = y[..., np.newaxis]

        if style == "stack":
            # Ensure one y-limit per plotted plane, even if maxy is a NumPy scalar.
            maxy_arr = np.asarray(maxy)
            if maxy_arr.ndim == 0:
                maxy = np.repeat(float(maxy_arr), self.nplanes)
            else:
                maxy = maxy_arr.reshape(-1)

        hist_counts = self.h
        if self.nplanes == 1:
            hist_counts = hist_counts[..., np.newaxis]

        def plane_stats(counts: np.ndarray) -> tuple[float | None, float | None]:
            total = float(np.sum(counts))
            if total <= 0:
                return None, None

            mean = float(np.sum(self.x * counts) / total)
            cdf = np.cumsum(counts)
            median_idx = int(np.searchsorted(cdf, 0.5 * total, side="left"))
            median = float(self.x[min(median_idx, len(self.x) - 1)])
            return mean, median

        if self._sorted:
            xrange = (self.x[0], self.x[-1])
            xlabel = "bin rank"
        else:
            xrange = (self._xrange[0], self._xrange[1])
            xlabel = "pixel value"

        colors: list[str] = []
        if self.colordict is not None:
            colors = list(self.colordict.keys())
            n = len(colors)
            # ylabel1 += ' (' + ','.join(colors) + ')'
        else:
            n = 1
            if style == "overlay":
                raise ValueError("cannot use overlay style for monochrome image")

        if self._sorted and style == "overlay":
            raise ValueError("cannot use overlay style for sorted histogram")

        if style == "stack":
            # ------------------------------------------------------------------------
            #  stack style plots each plane in a separate subplot
            # ------------------------------------------------------------------------
            if ax is None:
                # axes are passed in, can only handle
                fig, axes = plt.subplots(n, 1)
            else:
                axes = ax
            axes = np.atleast_1d(axes)
            for i, ax in enumerate(axes):
                bin_width = x[1] - x[0]
                ax.stairs(
                    y[:, i], np.append(x, x[-1] + bin_width), fill=filled, **kwargs
                )
                ax.grid()
                if n == 1:
                    ax.set_ylabel(ylabel1)
                else:
                    ax.set_ylabel(ylabel1 + " (" + colors[i] + ")")
                ax.set_xlim(*xrange)
                ax.set_ylim(0, maxy[i])
                ax.yaxis.set_major_formatter(
                    ScalarFormatter(useOffset=False, useMathText=True)
                )
                if stats:
                    mean, median = plane_stats(hist_counts[:, i])
                    if mean is not None and median is not None:
                        ax.axvline(
                            mean,
                            color="k",
                            linestyle="-",
                            linewidth=1.0,
                            label="mean",
                        )
                        ax.axvline(
                            median,
                            color="k",
                            linestyle="--",
                            linewidth=1.0,
                            label="median",
                        )
                        ax.legend(loc="best")
                if log:
                    ax.set_yscale("log")
            axes[-1].set_xlabel(xlabel)

            if cursor:
                # ------------------------------------------------------------------------
                #  add a data picking cursor to the stacked line plot
                # ------------------------------------------------------------------------
                x_interp = np.asarray(x, dtype=float)
                xpad = 0.01 * (x_interp[-1] - x_interp[0])

                cursor_lines = []
                cursor_markers = []
                cursor_labels = []

                for i, ax in enumerate(axes):
                    y0 = float(y[0, i])
                    vline = ax.axvline(
                        x_interp[0],
                        color="k",
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.8,
                        visible=False,
                    )
                    marker = ax.plot(
                        [x_interp[0]],
                        [y0],
                        marker="o",
                        markersize=4,
                        color="k",
                        linestyle="None",
                        visible=False,
                    )[0]
                    label = ax.text(
                        x_interp[0],
                        y0,
                        "",
                        fontsize="small",
                        visible=False,
                        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                    )
                    cursor_lines.append(vline)
                    cursor_markers.append(marker)
                    cursor_labels.append(label)

                def on_move(event):
                    if event.xdata is None or event.inaxes not in axes:
                        for vline, marker, label in zip(
                            cursor_lines, cursor_markers, cursor_labels
                        ):
                            vline.set_visible(False)
                            marker.set_visible(False)
                            label.set_visible(False)
                        fig.canvas.draw_idle()
                        return

                    xv = float(np.clip(event.xdata, x_interp[0], x_interp[-1]))
                    for i, (vline, marker, label) in enumerate(
                        zip(cursor_lines, cursor_markers, cursor_labels)
                    ):
                        yv = float(np.interp(xv, x_interp, y[:, i]))
                        vline.set_xdata([xv, xv])
                        marker.set_data([xv], [yv])
                        label.set_position((xv + xpad, yv))
                        if np.isdtype(self._x.dtype, "integral"):
                            label.set_text(f"({xv:.0f}, {yv:.3g})")
                        else:
                            label.set_text(f"({xv:.3g}, {yv:.3g})")
                        vline.set_visible(True)
                        marker.set_visible(True)
                        label.set_visible(True)

                    fig.canvas.draw_idle()

                fig.canvas.mpl_connect("motion_notify_event", on_move)

        elif style == "overlay":
            # ------------------------------------------------------------------------
            #  overlay style plots each plane in the same subplot
            # ------------------------------------------------------------------------
            x = np.r_[xrange[0], x, xrange[1]]
            _, ax = plt.subplots(1, 1)

            patchcolor = []
            goodcolors = [c for c in "rgbykcm"]
            if self.colordict is None:
                self.colordict = {c: i for i, c in enumerate(goodcolors[:n])}
                colors = list(self.colordict.keys())
                patchcolor = [c.lower() for c in colors]
            else:
                for color, i in self.colordict.items():
                    if color.lower() in "rgbykcm":
                        patchcolor.append(color.lower())
                    else:
                        patchcolor.append(goodcolors.pop(0))

            if filled:
                for i in range(n):
                    yi = np.r_[0, y[:, i], 0]
                    p1 = np.array([x, yi]).T
                    poly1 = Polygon(
                        p1,
                        closed=True,
                        facecolor=patchcolor[i],
                        alpha=alpha,
                        label=colors[i],
                        **kwargs,
                    )
                    ax.add_patch(poly1)
            else:
                for i in range(n):
                    ax.plot(x, np.r_[0, y[:, i], 0], color=patchcolor[i], **kwargs)

            if stats:
                mean, median = plane_stats(np.sum(hist_counts, axis=1))
                if mean is not None and median is not None:
                    ax.axvline(
                        mean,
                        color="k",
                        linestyle="-",
                        linewidth=1.0,
                        label="mean",
                    )
                    ax.axvline(
                        median,
                        color="k",
                        linestyle="--",
                        linewidth=1.0,
                        label="median",
                    )
            ax.set_xlim(*xrange)
            ax.set_ylim(0, maxy)
            ax.yaxis.set_major_formatter(
                ScalarFormatter(useOffset=False, useMathText=True)
            )

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel2)
            if log:
                ax.set_yscale("log")

            ax.grid(True)
            ax.legend(loc="best")

        else:
            raise ValueError("unknown style")

        if title is not None:
            set_window_title(title)
        safe_plt_show(block=block)

    def peaks(self, **kwargs: Any) -> np.ndarray | list[np.ndarray]:
        r"""
        Histogram peaks

        :param kwargs: parameters passed to :func:`~machinevisiontoolbox.base.findpeaks.findpeaks`
        :return: positions of histogram peaks
        :rtype: ndarray(M), list of ndarray

        For a greyscale image return an array of grey values corresponding to local
        maxima.  For a color image return a list of arrays of grey values corresponding
        to local maxima in each plane.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> hist = Image.Read('street.png').hist()
            >>> hist.peaks(scale=20)

        .. plot::

            from machinevisiontoolbox import Image
            hist = Image.Read('street.png').hist()
            x = hist.peaks(scale=20)
            idx = np.searchsorted(hist.x, x)
            idx = np.clip(idx, 0, len(hist.x) - 1)
            plt.plot(x, hist.h[idx], 'ko', markersize=10)

        Note the first peak (at 8) is excluded because the maxima window ($x \pm 20$) extends outside the histogram range.

        :seealso: :func:`~machinevisiontoolbox.base.findpeaks.findpeaks`
        """
        if self.nplanes == 1:
            # greyscale image
            x = findpeaks(self.h, self.x, **kwargs)
            return x[0]

        else:
            xp = []
            for i in range(self.nplanes):
                x = findpeaks(self.h[:, i], self.x, **kwargs)
                xp.append(x[0])
            return xp

    # # helper function that was part of hist() in the Matlab toolbox
    # # TODO consider moving this to ImpageProcessingBase.py
    # def plothist(self, title=None, block=False, **kwargs):
    #     """
    #     plot first image histogram as a line plot (TODO as poly)
    #     NOTE convenient, but maybe not a great solution because we then need to
    #     duplicate all the plotting options as for idisp?
    #     """
    #     if title is None:
    #         title = self[0].filename

    #     hist = self[0].hist(**kwargs)
    #     x = hist[0].x
    #     h = hist[0].h
    #     fig, ax = plt.subplots()

    #     # line plot histogram style
    #     if self.iscolor:
    #         ax.plot(x[:, 0], h[:, 0], 'b', alpha=0.8)
    #         ax.plot(x[:, 1], h[:, 1], 'g', alpha=0.8)
    #         ax.plot(x[:, 2], h[:, 2], 'r', alpha=0.8)
    #     else:
    #         ax.plot(hist[0].x, hist[0].h, 'k', alpha=0.7)

    #     # polygon histogram style
    #     polygon_style = False
    #     if polygon_style:
    #         if self.iscolor:
    #             from matplotlib.patches import Polygon
    #             # TODO make sure pb goes to bottom of axes at the edges:
    #             pb = np.stack((x[:, 0], h[:, 0]), axis=1)
    #             polyb = Polygon(pb,
    #                             closed=True,
    #                             facecolor='b',
    #                             linestyle='-',
    #                             alpha=0.75)
    #             ax.add_patch(polyb)

    #             pg = np.stack((x[:, 1], h[:, 1]), axis=1)
    #             polyg = Polygon(pg,
    #                             closed=True,
    #                             facecolor='g',
    #                             linestyle='-',
    #                             alpha=0.75)
    #             ax.add_patch(polyg)

    #             pr = np.stack((x[:, 2], h[:, 2]), axis=1)
    #             polyr = Polygon(pr,
    #                             closed=True,
    #                             facecolor='r',
    #                             linestyle='-',
    #                             alpha=0.75)
    #             ax.add_patch(polyr)

    #             # addpatch seems to require a plot, so hack is to plot null and
    #             # make alpha=0
    #             ax.plot(0, 0, alpha=0)
    #         else:
    #             from matplotlib.patches import Polygon
    #             p = np.hstack((x, h))
    #             poly = Polygon(p,
    #                            closed=True,
    #                            facecolor='k',
    #                            linestyle='-',
    #                            alpha=0.5)
    #             ax.add_patch(poly)
    #             ax.plot(0, 0, alpha=0)

    #     ax.set_ylabel('count')
    #     ax.set_xlabel('bin')
    #     ax.grid()

    #     ax.set_title(title)

    #     plt.show(block=block)

    #     # him = im[2].hist()
    #     # fig, ax = plt.subplots()
    #     # ax.plot(him[i].x[:, 0], him[i].h[:, 0], 'b')
    #     # ax.plot(him[i].x[:, 1], him[i].h[:, 1], 'g')
    #     # ax.plot(him[i].x[:, 2], him[i].h[:, 2], 'r')
    #     # plt.show()


if __name__ == "__main__":
    from pathlib import Path

    import pytest

    pytest.main(
        [
            str(
                Path(__file__).parent.parent.parent
                / "tests"
                / "test_image_whole_features.py"
            ),
            "-v",
        ]
    )
