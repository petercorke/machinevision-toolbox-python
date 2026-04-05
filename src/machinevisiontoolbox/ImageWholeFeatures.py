"""
Whole-image feature computation: moments, Hu invariants, histograms, and entropy.
"""

from __future__ import annotations

from typing import Any

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.patches import Polygon
from matplotlib.ticker import ScalarFormatter
from spatialmath import SE3, base

from machinevisiontoolbox._image_typing import _ImageBase
from machinevisiontoolbox.base import findpeaks, findpeaks2d, set_window_title
from machinevisiontoolbox.mvtb_types import *


class ImageWholeFeaturesMixin(_ImageBase):

    # ------------------ scalar statistics ----------------------------- #

    def sum(self, *args, **kwargs) -> int | float:
        r"""
        Sum of all pixels

        :param args: additional positional arguments to :func:`numpy.sum`
        :param kwargs: additional keyword arguments to :func:`numpy.sum`
        :return: sum

        Computes the sum of pixels in the image:

        .. math::

            \sum_{uvc} I_{uvc}

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.sum()  # R+G+B
            >>> img.sum(axis=(0,1)) # sum(R), sum(G), sum(B)
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.sum(axis=2)


        .. note::
            - The return value type is the same as the image type.
            - By default the result is a scalar computed over all pixels,
              if the ``axis`` option is given the results is a 1D or 2D NumPy
              array.
            - NaN values are ignored in the sum

        :seealso: :func:`numpy.sum` :meth:`numnan` :meth:`~~machinevisiontoolbox.ImageWholeFeatures.ImageWholeFeaturesMixin.mpq`
            :meth:`~machinevisiontoolbox.ImageWholeFeatures.ImageWholeFeaturesMixin.npq`
            :meth:`~machinevisiontoolbox.ImageWholeFeatures.ImageWholeFeaturesMixin.upq`
        """
        return np.nansum(self._A, *args, **kwargs)

    def min(self, *args, **kwargs) -> int | float:
        """
        Minimum value of all pixels

        :param args: additional positional arguments to :func:`numpy.nanmin`
        :param kwargs: additional keyword arguments to :func:`numpy.nanmin`
        :return: minimum value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.min()
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.min(axis=(0,1)) # minimum over each plane

        .. note::
            - The return value type is the same as the image type.
            - By default the result is a scalar computed over all pixels,
              if the ``axis`` option is given the results is a 1D or 2D NumPy
              array.
            - NaN values are ignored

        :seealso: :meth:`stats` :meth:`hist` :meth:`Hough` :meth:`max` :func:`numpy.nanmin` :meth:`numnan`
        """
        return np.nanmin(self._A, *args, **kwargs)

    def max(self, *args, **kwargs) -> int | float:
        """
        Maximum value of all pixels

        :param args: additional positional arguments to :func:`numpy.nanmax`
        :param kwargs: additional keyword arguments to :func:`numpy.nanmax`
        :return: maximum value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.max()
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.max(axis=(0,1)) # maximum over each plane

        .. note::
            - The return value type is the same as the image type.
            - By default the result is a scalar computed over all pixels,
              if the ``axis`` option is given the results is a 1D or 2D NumPy
              array.
            - NaN values are ignored

        :seealso: :meth:`stats` :meth:`hist` :meth:`min` :func:`numpy.nanmax` :meth:`numnan`
        """
        return np.nanmax(self._A, *args, **kwargs)

    def mean(self, *args, **kwargs) -> float:
        """
        Mean value of all pixels

        :param args: additional positional arguments to :func:`numpy.nanmean`
        :param kwargs: additional keyword arguments to :func:`numpy.nanmean`
        :return: mean value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.mean()
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.mean(axis=(0,1)) # mean over each plane

        .. note::
            - The return value type is the same as the image type.
            - By default the result is a scalar computed over all pixels,
              if the ``axis`` option is given the results is a 1D or 2D NumPy
              array.
            - NaN values are ignored

        :seealso: :meth:`stats` :meth:`hist` :meth:`Hough` :meth:`std` :meth:`median` :func:`numpy.nanmean` :meth:`numnan`
        """
        return np.nanmean(self._A, *args, **kwargs)

    def std(self, *args, **kwargs) -> float | np.ndarray:
        """
        Standard deviation of all pixels

        :param args: additional positional arguments to :func:`numpy.nanstd`
        :param kwargs: additional keyword arguments to :func:`numpy.nanstd`
        :return: standard deviation value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.std()
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.std(axis=(0,1)) # std over each plane

        .. note::
            - The return value type is the same as the image type.
            - By default the result is a scalar computed over all pixels,
              if the ``axis`` option is given the results is a 1D or 2D NumPy
              array.
            - NaN values are ignored

        :seealso: :meth:`stats` :meth:`hist` :meth:`mean` :meth:`var` :func:`numpy.nanstd` :meth:`numnan`
        """
        result = np.nanstd(self._A, *args, **kwargs)
        if np.ndim(result) == 0:
            return float(result)
        return result

    def var(self, *args, **kwargs) -> float | np.ndarray:
        """
        Variance of all pixels

        :param args: additional positional arguments to :func:`numpy.nanvar`
        :param kwargs: additional keyword arguments to :func:`numpy.nanvar`
        :return: variance value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.var()
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.var(axis=(0,1)) # variance over each plane

        .. note::
            - The return value type is the same as the image type.
            - By default the result is a scalar computed over all pixels,
              if the ``axis`` option is given the results is a 1D or 2D NumPy
              array.
            - NaN values are ignored

        :seealso: :meth:`stats` :meth:`hist` :meth:`std` :func:`numpy.nanvar` :meth:`numnan`
        """
        result = np.nanvar(self._A, *args, **kwargs)
        if np.ndim(result) == 0:
            return float(result)
        return result

    def median(self, *args, **kwargs) -> int | float:
        """
        Median value of all pixels

        :param args: additional positional arguments to :func:`numpy.nanmedian`
        :param kwargs: additional keyword arguments to :func:`numpy.nanmedian`
        :return: median value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.median()
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.median(axis=(0,1)) # median over each plane

        .. note::
            - The return value type is the same as the image type.
            - By default the result is a scalar computed over all pixels,
              if the ``axis`` option is given the results is a 1D or 2D NumPy
              array.
            - NaN values are ignored

        :seealso: :meth:`stats` :meth:`hist` :meth:`mean` :meth:`std` :func:`numpy.nanmedian` :meth:`numnan`
        """
        return np.nanmedian(self._A, *args, **kwargs)

    def stats(self) -> dict[str, Any]:
        """
        Display pixel value statistics

        :return: statistics dictionary; for greyscale keys are ``min``, ``max``,
            ``mean``, ``sdev``, ``median``, ``nan``, ``inf``; for color images
            returns a dictionary mapping plane name to that same dictionary
        :rtype: dict

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.stats()

        .. note::

            - Statistics are printed to standard output and also returned as a
              dictionary.
            - NaN values are ignored, but their number is displayed as part of the statistics if non zero.

        :seealso: :meth:`hist` :meth:`min` :meth:`max` :meth:`mean` :meth:`std` :meth:`median`
        """

        def plane_stats(plane):
            return {
                "min": float(np.nanmin(plane)),
                "max": float(np.nanmax(plane)),
                "mean": float(np.nanmean(plane)),
                "sdev": float(np.nanstd(plane)),
                "median": float(np.nanmedian(plane)),
                "nan": int(np.sum(np.isnan(plane))),
                "inf": int(np.sum(np.isinf(plane))),
            }

        def printstats(plane):
            stats = plane_stats(plane)
            s = (
                f"range={stats['min']:g} - {stats['max']:g}; "
                f"mean={stats['mean']:g}, "
                f"𝜎={stats['sdev']:g}; "
                f"median={stats['median']:g}"
            )
            nnan = stats["nan"]
            ninf = stats["inf"]
            if nnan + ninf > 0:
                s += " (contains "
                if nnan > 0:
                    s += f"{nnan}xNaN{'s' if nnan > 1 else ''}"
                if ninf > 0:
                    s += f" {ninf}xInf{'s' if ninf > 1 else ''}"
                s += ")"
            print(s)

        if self.iscolor and self.colororder is not None:
            all_stats = {}
            for k, v in sorted(self.colororder.items(), key=lambda x: x[1]):
                print(f"{k:s}: ", end="")
                printstats(self._A[..., v])
                all_stats[k] = plane_stats(self._A[..., v])
            return all_stats
        else:
            printstats(self._A)
            return plane_stats(self._A)

    # ------------------ histogram ------------------------------------- #
    def hist(self, nbins: int = 256, opt: str | None = None) -> "Histogram":
        """
        Image histogram

        :param nbins: number of histogram bins, defaults to 256
        :type nbins: int, optional
        :param opt: histogram option, defaults to None; options are 'sorted' to sort the histogram by count rather than bin value
        :type opt: str
        :return: histogram of image
        :rtype: :class:`~machinevisiontoolbox.ImageWholeFeatures.Histogram`

        Returns an object that summarizes the distribution of
        pixel values in each color plane.

        Example::

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('street.png')
            >>> type(hist)
            >>> hist = img.hist()
            >>> hist
            >>> hist.plot()

        .. plot::

            from machinevisiontoolbox import Image
            img = Image.Read('street.png')
            img.hist().plot()

        Example::

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> hist = img.hist()
            >>> hist
            >>> hist.plot(style='stack')

        .. plot::

            from machinevisiontoolbox import Image
            img = Image.Read('flowers1.png')
            img.hist().plot(style='stack')

        Example::

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> hist = img.hist()
            >>> hist
            >>> hist.plot(style='overlay')

        .. plot::

            from machinevisiontoolbox import Image
            img = Image.Read('flowers1.png')
            img.hist().plot(style='overlay')

        .. note::
            - For a `uint8` image the bins spans the greylevel range 0-255.
            - For a floating point image the histogram spans the greylevel range
              0.0 to 1.0 with 256 bins.
            - For floating point images all NaN and Inf values are first
              removed.
            - Computed using OpenCV `calcHist`. Only works on floats up to 32 bit,
              float64 images are automatically converted to float32.

        :references:
            - |RVC3|, Section 14.4.3.

        .. important:: Uses OpenCV function ``cv2.calcHist`` which accepts single-channel, CV_8U or CV_32F images (float64 images are automatically converted to float32).

        :seealso:
            :meth:`stats`
            :class:`~machinevisiontoolbox.ImageWholeFeatures.Histogram`
            `opencv.calcHist <https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d>`_
        """

        # check inputs
        optHist = ["sorted"]
        if opt is not None and opt not in optHist:
            raise ValueError(opt, "opt is not a valid option")

        if self.isint:
            xrange = [0, np.iinfo(self.dtype).max]
        else:
            # float image
            xrange = [0.0, 1.0]

        xc = []
        hc = []
        hcdf = []
        hnormcdf = []

        # ensure that float image is converted to float32
        if self._A.dtype == np.dtype("float64"):
            implanes = cv.split(m=self._A.astype("float32"))
        else:
            implanes = cv.split(m=self._A)

        for i in range(self.nplanes):
            # bin coordinates
            x = np.linspace(xrange[0], xrange[1], nbins, endpoint=True).T
            # h = cv.calcHist(implanes, [i], None, [nbins], [0, maxrange + 1])
            h = cv.calcHist(
                images=implanes,
                channels=[i],
                mask=None,
                histSize=[nbins],
                ranges=xrange,
            )
            if i == 0:
                xc.append(x)
            hc.append(h)

        # stack into arrays
        xs = np.vstack(xc).T
        hs = np.hstack(hc)

        # TODO this seems too complex, why do we stack stuff as well
        # as have an array of hist tuples??
        # xs, xc are the same, and same for all plots

        hhhx = Histogram(hs, xs, self.isfloat)
        hhhx.colordict = self.colororder

        return hhhx

    @property
    def x(self) -> np.ndarray:
        """
        Histogram bin values

        :return: array of left-hand bin values
        :rtype: ndarray(N)
        """
        return self.hist().x

    @property
    def h(self) -> np.ndarray:
        """
        Histogram count values

        :return: array of histogram count values
        :rtype: ndarray(N) or ndarray(N,P)
        """
        return self.hist().h

    @property
    def cdf(self) -> np.ndarray:
        """
        Cumulative histogram values

        :return: array of cumulative histogram values
        :rtype: ndarray(N) or ndarray(N,P)
        """
        return self.hist().cdf

    @property
    def ncdf(self) -> np.ndarray:
        """
        Normalised cumulative histogram values

        :return: array of normalised cumulative histogram values
        :rtype: ndarray(N) or ndarray(N,P)
        """
        return self.hist().ncdf

    def peaks(self, **kwargs) -> np.ndarray | list[np.ndarray]:
        """
        Histogram peaks

        :param kwargs: parameters passed to histogram peak finder
        :return: positions of histogram peaks
        :rtype: ndarray(M), list of ndarray
        """
        return self.hist().peaks(**kwargs)

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

        .. important:: Uses OpenCV function ``cv2.moments`` which accepts single-channel, CV_8U, CV_16U, CV_16S, CV_32F or CV_64F images (colour images are automatically converted to greyscale).

        :seealso: :meth:`mpq` :meth:`npq` :meth:`upq` `opencv.moments <https://docs.opencv.org/4.x/d8/d23/classcv_1_1Moments.html>`_
        """
        return cv.moments(array=self.mono().to_int(), binaryImage=binary)

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
        moments = cv.moments(array=self._A)
        hu = cv.HuMoments(m=moments)
        return hu.flatten()

    # ------------------ pixel values --------------------------------- #

    def nonzero(self):
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

    def flatnonzero(self):
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

    def peak2d(self, npeaks=2, scale=1, interp=False, positive=True):
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
    def __init__(self, h, x, isfloat=False):
        """
        Create histogram instance

        :param h: image histogram
        :type h: ndarray(N), ndarray(N,P)
        :param x: image values
        :type x: ndarray(N)
        :param isfloat: pixel values are floats, defaults to False
        :type isfloat: bool, optional

        Create :class:`Histogram` instance from histogram data provided
        as Numpy arrays.

        :seealso: :meth:`~machinevisiontoolbox.ImageFeatures.ImageFeaturesMixin.hist`
        """
        self.nplanes = h.shape[1]

        if self.nplanes == 1:
            h = h[:, 0]

        self._h = h  # histogram
        self._x = x.flatten()  # x value
        self.isfloat = isfloat
        self.colordict: dict[str, int] | None = None
        # 'hist', 'h cdf normcdf x')

    def __str__(self):
        """
        Histogram summary as a string

        :return: concise summary of histogram
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Random(256)
            >>> h = im.hist(100)
            >>> print(h)
        """
        s = f"histogram with {len(self.x)} bins"
        if self.nplanes > 1:
            s += f" x {self._h.shape[1]} planes"
        s += f": xrange {self.x[0]} - {self.x[-1]}, yrange {np.min(self._h)} - {np.max(self.h)}"
        return s

    def __repr__(self):
        """
        Print histogram summary

        :return: print concise summary of histogram
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Random(256)
            >>> h = im.hist(100)
            >>> h
        """
        return str(self)

    @property
    def x(self):
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
            >>>     hist.x
        """
        return self._x

    @property
    def h(self):
        """
        Histogram count values

        :return: array of histogram count values
        :rtype: ndarray(N) or ndarray(N,P)

        For a greyscale image this is a 1D array, for a multiplane (color) image
        this is a 2D array with the histograms of each plane as columns.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> hist = Image.Read('flowers1.png').hist()
            >>> with np.printoptions(threshold=10):
            >>>     hist.h
        """
        return self._h

    @property
    def cdf(self):
        """
        Cumulative histogram values

        :return: array of cumulative histogram count values
        :rtype: ndarray(N) or ndarray(N,P)

        For a greyscale image this is a 1D array, for a multiplane (color) image
        this is a 2D array with the cumulative histograms of each plane as columns.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> hist = Image.Read('flowers1.png').hist()
            >>> with np.printoptions(threshold=10):
            >>>     hist.cdf
        """
        return np.cumsum(self._h, axis=0)

    @property
    def ncdf(self):
        """
        Normalized cumulative histogram values

        :return: array of normalized cumulative histogram count values
        :rtype: ndarray(N) or ndarray(N,P)

        For a greyscale image this is a 1D array, for a multiplane (color) image
        this is a 2D array with the normalized cumulative histograms of each
        plane as columns.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> hist = Image.Read('flowers1.png').hist()
            >>> with np.printoptions(threshold=10):
            >>>     hist.ncdf
        """
        y = np.cumsum(self._h, axis=0)

        if self.nplanes == 1:
            y = y / y[-1]
        else:
            y = y / y[-1, :]
        return y

    def plot(
        self,
        type="frequency",
        block=False,
        bar=None,
        style="stack",
        alpha=0.5,
        title=None,
        **kwargs,
    ):
        """
        Plot histogram

        :param type: histogram type, one of: 'frequency' [default], 'cdf', 'ncdf'
        :type type: str, optional
        :param block: hold plot, defaults to False
        :type block: bool, optional
        :param bar: histogram bar plot, defaults to True for frequency plot, False for
            other plots
        :type bar: bool, optional
        :param style: Style for multiple plots, one of: 'stack' [default], 'overlay'
        :type style: str, optional
        :param alpha: transparency for overlay plot, defaults to 0.5
        :type alpha: float, optional
        :raises ValueError: invalid histogram type
        :raises ValueError: cannot use overlay style for 1-channel histogram
        """

        # if type == 'histogram':
        #     plot_histogram(self.xs.flatten(), self.hs.flatten(), block=block,
        #     xlabel='pixel value', ylabel='number of pixels', **kwargs)
        # elif type == 'cumulative':
        #     plot_histogram(self.xs.flatten(), self.cs.flatten(), block=block,
        #     xlabel='pixel value', ylabel='cumulative number of pixels', **kwargs)
        # elif type == 'normalized':
        #     plot_histogram(self.xs.flatten(), self.ns.flatten(), block=block,
        #     xlabel='pixel value', ylabel='normalized cumulative number of pixels', **kwargs)
        # fig = plt.figure()
        x = self._x[:]

        if type == "frequency":
            y = self.h
            maxy = np.max(y)
            ylabel1 = "frequency"
            ylabel2 = "frequency"
            if bar is not False:
                bar = True
        elif type in ("cdf", "cumulative"):
            y = self.cdf
            maxy = np.max(y[-1, :])
            ylabel1 = "cumulative frequency"
            ylabel2 = "cumulative frequency"
        elif type in ("ncdf", "normalized"):
            y = self.ncdf
            ylabel1 = "norm. cumulative freq."
            ylabel2 = "normalized cumulative frequency"
            maxy = 1
        else:
            raise ValueError("unknown type")

        if self.nplanes == 1:
            y = y[..., np.newaxis]

        if self.isfloat:
            xrange = (0.0, 1.0)
        else:
            xrange = (0, 255)

        if self.colordict is not None:
            colors = list(self.colordict.keys())
            n = len(colors)
            # ylabel1 += ' (' + ','.join(colors) + ')'
        else:
            n = 1
            if style == "overlay":
                raise ValueError("cannot use overlay style for monochrome image")

        if style == "stack":
            for i in range(n):
                ax = plt.subplot(n, 1, i + 1)
                if False:
                    # ax.bar(x, y[:, i], width=x[1] - x[0], bottom=0, **kwargs)
                    ax.bar(x, y[:, i])

                else:
                    ax.plot(x, y[:, i], **kwargs)
                ax.grid()
                if n == 1:
                    ax.set_ylabel(ylabel1)
                else:
                    ax.set_ylabel(ylabel1 + " (" + colors[i] + ")")
                # ax.set_xlim(*xrange)
                # ax.set_ylim(0, maxy)
                ax.yaxis.set_major_formatter(
                    ScalarFormatter(useOffset=False, useMathText=True)
                )
            ax.set_xlabel("pixel value")

        elif style == "overlay":
            x = np.r_[0, x, 255]
            ax = plt.subplot(1, 1, 1)

            patchcolor = []
            goodcolors = [c for c in "rgbykcm"]
            if self.colordict is None:
                self.colordict = {c: i for i, c in enumerate(goodcolors[:n])}
            else:
                for color, i in self.colordict.items():
                    if color.lower() in "rgbykcm":
                        patchcolor.append(color.lower())
                    else:
                        patchcolor.append(goodcolors.pop(0))

            for i in range(n):
                yi = np.r_[0, y[:, i], 0]
                p1 = np.array([x, yi]).T
                poly1 = Polygon(
                    p1, closed=True, facecolor=patchcolor[i], alpha=alpha, **kwargs
                )
                ax.add_patch(poly1)
            ax.set_xlim(*xrange)
            ax.set_ylim(0, maxy)
            ax.yaxis.set_major_formatter(
                ScalarFormatter(useOffset=False, useMathText=True)
            )

            ax.set_xlabel("pixel value")
            ax.set_ylabel(ylabel2)

            ax.grid(True)
            plt.legend(colors)
        if title is not None:
            set_window_title(title)
        plt.show(block=block)

    def peaks(self, **kwargs):
        """
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
            >>> import numpy as np
            >>> hist = Image.Read('street.png').hist()
            >>> hist.peaks(scale=20)

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
            str(Path(__file__).parent.parent.parent / "tests" / "test_wholefeature.py"),
            "-v",
        ]
    )
