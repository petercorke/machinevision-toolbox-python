import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import ScalarFormatter

import cv2 as cv
from spatialmath import base

class ImageLineFeaturesMixin:
    """
    Line features are common in in many human-built environments.

    """


    def Hough(self, **kwargs):
        """
        Find Hough line features

        :return: Hough lines
        :rtype: :class:`Hough`

        Compute the Hough transform of the image and return an object that 
        represents the lines found within the image.

        :seealso: :class:`Hough`
        """
        return Hough(self, **kwargs)

# --------------------- supporting classes -------------------------------- #

class Hough:

    def __init__(self, image, ntheta=180, drho=1):
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
        - line segments using the probabilistic Hough algorith  :meth:`lines_p`

        The Hough accumulator is a 2D array that counts votes for lines
        
        .. math:: u \cos \theta + v \sin \theta = \rho

        with quantized parameters :math:`\theta` and :math:`\rho`.  The parameter
        :math:`\theta` is quantized into ``ntheta`` steps spanning the interval
        :math:`[-\pi, \pi)`, while :math:`\rho` is quantized into steps of 
        ``drho`` spanning the vertical dimension of the image.

        .. note:: Lines are not detected until :meth:`lines` or  :meth:`lines_p` 
            is called.  This instance simply holds parameters.

        :reference:
            - Robotics, Vision & Control for Python, Section 12.2, P. Corke, 
              Springer 2023.

        :seealso: :meth:`lines` :meth:`lines_p`
        """

        self.image = image.to_int()
        self.dtheta = np.pi / ntheta
        self.drho = drho
        self.A = None

    def lines(self, minvotes):
        r"""
        Get Hough lines

        :param minvotes: only return lines with at least this many votes
        :type minvotes: int
        :return: Hough lines, one per row as :math:`(\theta, \rho)`
        :rtype: ndarray(n,2)

        Return a set of lines that have at least ``minvotes`` of support.  Each 
        line is described by :math:`(\theta, \rho)` such that

        .. math:: u \cos \theta + v \sin \theta = \rho

        :seealso: :meth:`plot_lines` :meth:`lines_p`
        """
        lines = cv.HoughLines(
                image=self.image,
                rho=self.drho,
                theta=self.dtheta,
                threshold=minvotes
                )
        if lines is None:
            return np.zeros((0, 2))
        else:
            return np.array((lines[:,0,1], lines[:,0,0])).T

    def lines_p(self, minvotes, minlinelength=30, maxlinegap=10, seed=None):
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

        :seealso: :meth:`plot_lines_p` :meth:`lines`
        """
        if seed is not None:
            cv.setRNGSeed(seed)
            
        lines = cv.HoughLinesP(
                image=self.image,
                rho=self.drho,
                theta=self.dtheta,
                threshold=minvotes,
                minLineLength=minlinelength,
                maxLineGap=maxlinegap
                )
        if lines is None:
            return np.zeros((0, 4))
        else:
            return lines[:,0,:]

    def plot_lines(self, lines, *args, **kwargs):
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

        :seealso: :meth:`lines` :meth:`lines_p`
        """
        if lines.shape[1] == 2:
            # Hough lines
            theta, rho = lines.T
            homlines = np.row_stack((np.cos(theta), np.sin(theta), -rho))
            base.plot_homline(homlines, *args, **kwargs)
        else:
            for line in lines:
                plt.plot(line[[0,2]], line[[1,3]], *args, **kwargs)

    def accumulator(self, skip=1):
        r"""
        Compute the Hough accumulator

        :param skip: increment for line strength threshold, defaults to 1
        :type skip: int, optional

        It creates two new attributes for the instance:

        - ``A`` which is the Hough "accumulator" array, rows represent :math:`\rho`
          and columns represent :math:`\theta`.
        - ``votes`` is a list of the number of lines found versus threshold, it
          can be used to select an optimal threshold.
        - ``extent`` is :math:`[\theta_{\mbox{min}}, \theta_{\mbox{max}}, 
          \rho_{\mbox{min}}, \rho_{\mbox{max}}]`.

        .. warning:: The OpenCV ``HoughLines`` function does not expose the
            accumulator array. This method "reverse engineers" the accumulator
            array through a costly process of computing the Hough transform 
            for all possible thresholds (increasing in steps of ``skip``). This
            is helpful for pedagogy but very inefficient in practice.

        :seealso: :meth:`plot_accumulator`
        """

        self.nz = np.sum(self.image > 0)
        t = 0
        theta = np.empty((0,))
        rho = np.empty((0,))
        votes = []
        while True:
            lines = cv.HoughLines(
                image=self.image,
                rho=self.drho,
                theta=self.dtheta,
                threshold=t
                )

            if lines is None:
                # no lines found at this threshold, bail out
                self.t = t - 1
                break

            # append the found lines and votes
            theta = np.concatenate((theta, lines[:,0,1].flatten()))
            rho = np.concatenate((rho, lines[:,0,0].flatten()))
            votes.append(lines.shape[0])

            t += skip  # increment the line strength threshold

        # now create the accumulator array
        theta_bins = np.arange(theta.min() - self.dtheta / 2, theta.max() + self.dtheta / 2, self.dtheta)
        rho_bins = np.arange(rho.min() - self.drho / 2, rho.max() + self.drho / 2, self.drho)

        self.extent = [theta_bins[0], theta_bins[-1], rho_bins[0], rho_bins[-1]]
        self.A = np.histogram2d(theta, rho, bins=(theta_bins, rho_bins))[0].T
        self.votes = votes

    def plot_accumulator(self, **kwargs):
        r"""
        Plot the Hough accumulator array

        :param kwargs: options passed to :func:`~matplotlib.pyplot.imshow`

        The Hough accumulator is computed, if not already existing, and the displayed
        as an image where brightness is proportional to the number of votes for
        that :math:`(\theta, \rho)` coordinate.

        :seealso: :meth:`accumulator`
        """
        if self.A is None:
            self.accumulator()

        plt.imshow(self.A, aspect='auto', interpolation='nearest', origin='lower', extent=(self.extent), **kwargs)

        plt.xlabel(r'$\theta$ (radians)')
        plt.xlim(0, np.pi)
        plt.ylabel(r'$\rho$ (pixels)')
        plt.grid(True)

if __name__ == "__main__":

    from machinevisiontoolbox import Image
    from math import pi

    square = Image.Squares(number=1, size=256, fg=128).rotate(0.3)
    edges = square.canny();
    h = edges.hough()
    print(h)
