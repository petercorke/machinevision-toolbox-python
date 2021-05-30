import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import ScalarFormatter

import cv2 as cv
from spatialmath import base
from machinevisiontoolbox.base import plot_histogram

class ImageLineFeaturesMixin:

    def hough(self, **kwargs):
        """[summary]

        :return: [description]
        :rtype: [type]
        """

        return Hough(kwargs)

class Hough:

    def __init__(self, image, ntheta=180, drho=1):

        self.image = image.asint()
        self.dtheta = np.pi / ntheta
        self.drho = drho
        self.A = None

    def lines(self, t):
        lines = cv.HoughLines(
                image=self.image,
                rho=self.drho,
                theta=self.dtheta,
                threshold=t
                )
        if lines is None:
            return np.zeros((0, 2))
        else:
            return np.array((lines[:,0,1], lines[:,0,0])).T

    def plot_lines(self, lines, **kwargs):
        theta = lines[:, 0]
        rho = lines[:, 1]
        homlines = np.column_stack((np.cos(theta), np.sin(theta), -rho))
        base.plot_homline(homlines, **kwargs)

    def lines_p(self, t, minlinelength=30, maxlinegap=10):
        lines = cv.HoughLinesP(
                image=self.image,
                rho=self.drho,
                theta=self.dtheta,
                threshold=t,
                minLineLength=minlinelength,
                maxLineGap=maxlinegap
                )
        if lines is None:
            return np.zeros((0, 4))
        else:
            return lines[:,0,:]

    def plot_lines_p(self, lines, *args, **kwargs):
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

    def plot_accumulator(self):
        if self.A is None:
            self.accumulator()

        plt.imshow(self.A, aspect='auto', interpolation='none', origin='lower', extent=(self.extent))

        plt.xlabel(r'$\theta$ (radians)')
        plt.xlim(0, np.pi)
        plt.ylabel(r'$\rho$ (pixels)')
        plt.grid(True)



if __name__ == "__main__":

    from machinevisiontoolbox import Image
    from math import pi

    img = Image.Read('monalisa.png', dtype='float32', grey=False)
    print(img)
    # img.disp()

    h = img.hist()
    print(h)
    h.plot('frequency', style='overlay')
    plt.figure()
    h.plot('frequency', block=True)