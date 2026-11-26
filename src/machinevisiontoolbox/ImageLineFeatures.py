"""
Detection and description of line features (Hough, LSD) in images.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from spatialmath import base
from spatialmath.base import set_window_title

from machinevisiontoolbox._image_typing import ImageMixinBase

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from machinevisiontoolbox._image_typing import _ImageBase


class ImageLineFeaturesMixin(ImageMixinBase):
    """
    Line features are common in in many human-built environments.

    """

    def Hough(self, **kwargs: Any) -> HoughFeature:
        """
        Find Hough line features

        :return: Hough lines
        :rtype: :class:`~machinevisiontoolbox.ImageLineFeatures.HoughFeature`

        Compute the Hough transform of the image and return an object that
        represents the lines found within the image.

        .. seealso:: :class:`~machinevisiontoolbox.ImageLineFeatures.HoughFeature`
        """
        return HoughFeature(self, **kwargs)


# --------------------- supporting classes -------------------------------- #


class HoughFeature:
    def __init__(self, image: _ImageBase, ntheta: int = 180, drho: int = 1) -> None:
        r"""
        Hough line features

        :param image: greyscale image
        :type image: :class:`Image`
        :param ntheta: number of steps in the :meth:`\theta` direction, defaults to 180
        :type ntheta: int, optional
        :param drho: increment size in the :meth:`\rho` direction, defaults to 1
        :type drho: int, optional

        Create a Hough line feature object.  It can be used to detect:

        - lines using the classical Hough algorithm :meth:`lines`
        - line segments using the probabilistic Hough algorithm  :meth:`lines_p`

        The Hough accumulator is a 2D array that counts votes for lines

        .. math:: u \cos \theta + v \sin \theta = \rho

        with quantized parameters :math:`\theta` and :math:`\rho`.  The parameter
        :math:`\theta` is quantized into ``ntheta`` steps spanning the interval
        :math:`[-\pi, \pi)`, while :math:`\rho` is quantized into steps of
        ``drho`` spanning the vertical dimension of the image.

        .. note::

            - Lines are not detected until :meth:`lines` or  :meth:`lines_p`
              is called.  This instance simply holds parameters.
            - The OpenCV implementation works with ``uint8`` images only, other formats
              will be converted to ``uint8``.

        .. rubric:: References

        - |RVC3|, Section 12.2.

        .. seealso:: :meth:`lines` :meth:`lines_p`
        """

        self.image = image.array_as(np.uint8)
        self.dtheta = np.pi / ntheta
        self.drho = drho
        self.A: np.ndarray | None = None
        self.extent: tuple[float, float, float, float] | None = None
        self.votes: list[int] | None = None
        self.t: int | None = None

    def lines(self, minvotes: int) -> np.ndarray:
        r"""
        Get Hough lines

        :param minvotes: only return lines with at least this many votes
        :type minvotes: int
        :return: Hough lines, one per row as :math:`(\theta, \rho)`
        :rtype: ndarray(n,2)

        Return a set of lines that have at least ``minvotes`` of support.  Each
        line is described by :math:`(\theta, \rho)` such that

        .. math:: u \cos \theta + v \sin \theta = \rho

        .. rubric:: References

        - |RVC3|, Section 12.2.

        .. important:: Uses OpenCV function ``cv2.HoughLines`` which accepts single-channel, CV_8U images (images are pre-converted to uint8 when the :class:`Hough` object is created).

        .. seealso:: :meth:`plot_lines` :meth:`lines_p` `opencv.HoughLines <https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a>`_
        """
        lines = cv2.HoughLines(
            image=self.image, rho=self.drho, theta=self.dtheta, threshold=minvotes
        )
        # OpenCV 4 and 5.0 return None when there are no lines, 5.1 returns an
        # empty array
        if lines is None or len(lines) == 0:
            return np.zeros((0, 2))
        else:
            return np.array((lines[:, 0, 1], lines[:, 0, 0])).T

    def lines_p(
        self,
        minvotes: int,
        minlinelength: int = 30,
        maxlinegap: int = 10,
        seed: int | None = None,
    ) -> np.ndarray:
        r"""
        Get probabilistic Hough lines

        :param minvotes: only return lines with at least this many votes
        :type minvotes: int
        :param minlinelength: minimum line length. Line segments shorter than that are rejected.
        :type minlinelength: int
        :param maxlinegap: maximum allowed gap between points on the same line to link them.
        :type maxlinegap: int
        :return: Hough lines, one per row as :math:`(u_1, v_1, u_2, v_2)`
        :rtype: ndarray(n,4)

        Return a set of line segments that have at least ``minvotes`` of support.  Each
        line segment is described by its end points :math:`(u_1, v_1)` and
        :math:`(u_2, v_2)`.

        .. important:: Uses OpenCV function ``cv2.HoughLinesP`` which accepts single-channel, CV_8U images (images are pre-converted to uint8 when the :class:`Hough` object is created).

        .. seealso:: :meth:`plot_lines` :meth:`lines` `opencv.HoughLinesP <https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb>`_
        """
        if seed is not None:
            cv2.setRNGSeed(seed)

        lines = cv2.HoughLinesP(
            image=self.image,
            rho=self.drho,
            theta=self.dtheta,
            threshold=minvotes,
            minLineLength=minlinelength,
            maxLineGap=maxlinegap,
        )
        if lines is None or len(lines) == 0:
            return np.zeros((0, 4))
        # --- OpenCV 4/5 compat -----------------------------------------
        # OpenCV 5's HoughLinesP returns a true (N,4) array; OpenCV 4
        # returns (N,1,4) (the old std::vector<T>-wrapping convention).
        # Verified empirically 2026-08-03 (opencv-contrib-python 5.0.0.93
        # vs 4.13.0.92). Drop the singleton middle dim only if present.
        # -----------------------------------------------------------------
        elif lines.ndim == 3 and lines.shape[1] == 1:
            return lines[:, 0, :]
        else:
            return lines

    def plot_lines(self, lines: np.ndarray, *args: Any, **kwargs: Any) -> None:
        r"""
        Plot Hough lines

        :param lines: Hough or probabilistic Hough lines
        :type lines: ndarray(n,2), ndarray(n,4)
        :param args: positional arguments passed to Matplotlib :obj:`~matplotlib.pyplot.plot`
        :param kwargs: arguments passed to Matplotlib :obj:`~matplotlib.pyplot.plot`

        Detected lines are given as rows of ``lines``:

        - for Hough lines, each row is :math:`(\theta, \rho)`, and  lines are
          clipped by the bounds of the current plot.
        - for probabilistic Hough lines, each row is :math:`(u_1, v_1, u_2, v_2)`,
          and lines segments are drawn on the current plot.

        .. seealso:: :meth:`lines` :meth:`lines_p`
        """
        if lines.shape[1] == 2:
            # Hough lines
            theta, rho = lines.T
            homlines = np.vstack((np.cos(theta), np.sin(theta), -rho))
            base.plot_homline(homlines, *args, **kwargs)
        else:
            for line in lines:
                plt.plot(line[[0, 2]], line[[1, 3]], *args, **kwargs)

    def accumulator(self) -> None:
        r"""
        Compute the Hough accumulator

        Every set pixel :math:`(u, v)` of the image votes for all the lines that
        pass through it, one for each quantized :math:`\theta`, with

        .. math:: \rho = u \cos \theta + v \sin \theta

        quantized into steps of ``drho``.  It creates the following attributes
        for the instance:

        - ``A`` is the Hough "accumulator" array of integer vote counts, rows
          represent :math:`\rho` and columns represent :math:`\theta`.
        - ``votes`` is a list of the number of lines found versus threshold, it
          can be used to select an optimal threshold.  Element ``t`` is the number
          of lines that :meth:`lines` returns for ``minvotes=t``, to within
          rounding, since a line is a cell of ``A`` that exceeds the threshold
          and is a local maximum.
        - ``t`` is the highest threshold at which a line is still found.
        - ``extent`` is :math:`[\theta_{\mbox{min}}, \theta_{\mbox{max}},
          \rho_{\mbox{min}}, \rho_{\mbox{max}}]`, the edges of the array
          for plotting.

        .. warning:: The OpenCV ``HoughLines`` function does not expose its
            accumulator array, so it is computed here directly by voting, with
            the same quantization.  Unlike the lines returned by :meth:`lines`,
            which are only the peaks, ``A`` holds the votes of every cell.
            The peaks of ``A`` and the lines from OpenCV can differ by a very
            few cells, where OpenCV's single-precision arithmetic rounds a
            :math:`\rho` value differently.  This is helpful for pedagogy but
            it is not the array that OpenCV used.

        .. versionchanged:: 2.5.0
            The accumulator is computed directly by voting, so ``A`` is the
            complete array and not just its peaks, and ``extent`` spans the
            whole range of :math:`\rho`.  Previously it was reconstructed
            from repeated calls of ``HoughLines``, which was slow.  The ``skip``
            argument has been removed since every threshold is now counted.

        .. seealso:: :meth:`plot_accumulator` :meth:`lines`
        """

        image = self.image
        height, width = image.shape[:2]
        ntheta = int(round(np.pi / self.dtheta))
        # the same number of bins as OpenCV: rho = 0 is in the middle row
        nrho = int(round(((width + height) * 2 + 1) / self.drho))
        rho0 = (nrho - 1) // 2

        v, u = np.nonzero(image)
        self.nz = len(u)

        A = np.zeros((nrho, ntheta), dtype=int)
        for n in range(ntheta):
            theta = n * self.dtheta
            rho = np.round((u * np.cos(theta) + v * np.sin(theta)) / self.drho)
            A[:, n] = np.bincount(rho.astype(int) + rho0, minlength=nrho)

        # a line is a cell above the threshold that is a local maximum in the
        # rho and theta directions, which is the rule OpenCV uses
        P = np.pad(A, 1)
        centre = P[1:-1, 1:-1]
        peak = (
            (centre > P[:-2, 1:-1])
            & (centre >= P[2:, 1:-1])
            & (centre > P[1:-1, :-2])
            & (centre >= P[1:-1, 2:])
        )
        counts = np.bincount(A[peak], minlength=1)
        # a peak of v votes is found at every threshold t < v
        peaks_above = counts.sum() - np.cumsum(counts)
        votes = [int(n) for n in peaks_above[: np.max(A, initial=0)]]

        self.A = A
        self.votes = votes
        self.t = len(votes) - 1
        self.extent = (
            -self.dtheta / 2,
            (ntheta - 0.5) * self.dtheta,
            (-rho0 - 0.5) * self.drho,
            (nrho - 1 - rho0 + 0.5) * self.drho,
        )

    def plot_accumulator(
        self, ax: Axes | None = None, title: str | None = None, **kwargs: Any
    ) -> None:
        r"""
        Plot the Hough accumulator array

        :param ax: Matplotlib axes to plot into, defaults to a new figure
        :type ax: matplotlib.axes.Axes, optional
        :param title: window title, defaults to "Hough Accumulator" (not applied
            when ``ax`` is given, unless a title is passed explicitly)
        :type title: str, optional
        :param kwargs: options passed to :func:`~matplotlib.pyplot.imshow`

        The Hough accumulator is computed, if not already existing, and the displayed
        as an image where brightness is proportional to the number of votes for
        that :math:`(\theta, \rho)` coordinate.

        .. seealso:: :meth:`accumulator`
        """
        if self.A is None:
            self.accumulator()
        assert self.A is not None and self.extent is not None

        if ax is None:
            _, ax = plt.subplots()
            if title is None:
                title = "Hough Accumulator"
        if title is not None:
            set_window_title(ax.figure, title)

        ax.imshow(
            self.A,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            extent=self.extent,
            **kwargs,
        )

        ax.set_xlabel(r"$\theta$ (radians)")
        ax.set_xlim(0, np.pi)
        ax.set_ylabel(r"$\rho$ (pixels)")
        ax.grid(True)


if __name__ == "__main__":
    from pathlib import Path

    import pytest

    pytest.main(
        [
            str(
                Path(__file__).parent.parent.parent
                / "tests"
                / "test_image_line_features.py"
            ),
            "-v",
        ]
    )
