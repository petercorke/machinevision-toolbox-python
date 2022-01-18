import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import ScalarFormatter

import cv2 as cv
from spatialmath import base

class ImageLineFeaturesMixin:

    def Hough(self, **kwargs):
        """[summary]

        :return: [description]
        :rtype: [type]
        """

        return Hough(self, **kwargs)

class Hough:

    def __init__(self, image, ntheta=180, drho=1):

        self.image = image.to_int()
        self.dtheta = np.pi / ntheta
        self.drho = drho
        self.A = None

    def lines(self, minvotes):
        """
        Get Hough lines

        :param minvotes: only return lines with at least this many votes
        :type t: int
        :return: Hough lines, one per row as :math:`(\theta, \rho)`
        :rtype: ndarray(n,2)

        :math:`\theta \in [0, \pi]` while :math:`\rho \in \mathbb{R}`
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

    def plot_lines(self, lines, *args, **kwargs):
        """
        Plot Hough lines

        :param lines: Hough lines, one per row as :math:`(\theta, \rho)`
        :type lines: ndarray(n,2)
        :param args: positional arguments passed to Matplotlib ``plot``
        :param kwargs: arguments passed to Matplotlib ``plot``

        Lines are clipped by the bounds of the current plot.

        :seealso: :meth:`lines`
        """
        theta, rho = lines.T
        homlines = np.row_stack((np.cos(theta), np.sin(theta), -rho))
        base.plot_homline(homlines, *args, **kwargs)

    def lines_p(self, minvotes, minlinelength=30, maxlinegap=10, seed=None):
        """
        Get probabilistic Hough lines 

        :param minvotes: only return lines with at least this many votes
        :type minvotes: int
        :param minlinelength: minimum line length. Line segments shorter than that are rejected.
        :type minlinelength: int
        :param maxlinegap: maximum allowed gap between points on the same line to link them.
        :type maxlinegap: int
        :return: Hough lines, one per row as :math:`(u_1, v_1, u_2, v_2))`
        :rtype: ndarray(n,4)

        :math:`\theta \in [0, \pi]` while :math:`\rho \in \mathbb{R}`
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

    def plot_lines_p(self, lines, *args, **kwargs):
        """
        Plot Hough lines

        :param lines: Hough lines, one per row as :math:`(u_1, v_1, u_2, v_2))`
        :type lines: ndarray(n,4)
        :param args: positional arguments passed to Matplotlib ``plot``
        :param kwargs: arguments passed to Matplotlib ``plot``

        Lines segments are drawn on the current plot.

        :seealso: :meth:`lines`
        """
        for line in lines:
            plt.plot(line[[0,2]], line[[1,3]], *args, **kwargs)

    def accumulator(self, skip=1):

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
                self.t = t - 1
                break
                # lines are (rho, theta)
                # plt.plot(lines[:,0,1], lines[:,0,0], '.b', markersize=4)
                # print(lines.shape)

            theta = np.concatenate((theta, lines[:,0,1].flatten()))
            rho = np.concatenate((rho, lines[:,0,0].flatten()))
            votes.append(lines.shape[0])

            t += skip

        theta_bins = np.arange(theta.min() - self.dtheta / 2, theta.max() + self.dtheta / 2, self.dtheta)
        rho_bins = np.arange(rho.min() - self.drho / 2, rho.max() + self.drho / 2, self.drho)

        self.extent = [theta_bins[0], theta_bins[-1], rho_bins[0], rho_bins[-1]]
        self.A = np.histogram2d(theta, rho, bins=(theta_bins, rho_bins))[0].T
        self.votes = votes

    def plot_accumulator(self, **kwargs):
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
