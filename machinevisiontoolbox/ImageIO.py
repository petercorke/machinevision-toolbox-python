#!/usr/bin/env python

from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import interpolate
import cv2 as cv
from pathlib import Path
import os.path
from spatialmath.base import argcheck, getvector
from machinevisiontoolbox.base import iread, iwrite, colorname, int_image, float_image, idisp

class ImageIOMixin:

    # ======================= image i/io ================================== #

    @classmethod
    def Read(cls, filename, alpha=False, rgb=True, **kwargs):
        """
        Read image from file

        :param filename: image file name
        :type filename: str
        :param alpha: include alpha plane if present, defaults to False
        :type alpha: bool, optional
        :param kwargs: options applied to image frames, see :func:`~machinevisiontoolbox.base.imageio.convert`
        :raises ValueError: file not found
        :return: image from file
        :rtype: Image instance

        .. note::  If the path is not absolute it is first searched for relative
            to the current directory, and if not found, it is searched for in
            the ``images`` folder of the Toolbox installation.

        :seealso: :func:`machinevisiontoolbox.base.imageio.convert`,  `cv2.imread <https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56>`_
        """
        if not isinstance(filename, (str, Path)):
            raise ValueError('expecting a string or path')

        # read the image
        data = iread(filename, rgb=rgb, **kwargs)

        # result is a tuple(image, filename) or a list of tuples

        colororder = None
        if isinstance(data, tuple):
            # singleton image, make it a list
            image, name = data
            if not alpha and image.ndim == 3 and image.shape[2] == 4:
                image = image[:, :, :3]
            if image.ndim > 2:
                colororder = 'RGB' if rgb else 'BGR'
            return cls(image, name=name, colororder=colororder)  # OpenCV file read order)
        elif isinstance(data, list):
            raise ValueError('wildcard read not support, use FileCollection')

    def disp(self, title=None, **kwargs):
        """
        Display image via GUI

        :param title: display a title, this is the image ``name``
        :type title: bool
        :param kwargs: options, see :func:`~machinevisiontoolbox.base.idisp`

        :seealso: :func:`~machinevisiontoolbox.base.imageio.idisp`
        """
        if title is False:
            title = None
        elif title is None and self.name is not None:
            _, title = os.path.split(self.name)

        return idisp(self.A,
                title=title,
                bgr=self.isbgr,
                **kwargs)

    def write(self, filename, dtype='uint8', **kwargs):
        """
        Write image to file

        :param filename: filename to write to
        :type filename: str
        :param dtype: data type to convert to, before writing
        :type dtype: str
        :param kwargs: options for :func:`~machinevisiontoolbox.base.iwrite`

        :seealso: :func:`~machinevisiontoolbox.base.iwrite`, `cv2.imwrite <https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce>`_
        """

        # cv.imwrite can only save 8-bit single channel or 3-channel BGR images
        # with several specific exceptions
        # https://docs.opencv.org/4.4.0/d4/da8/group__imgcodecs.html
        # #gabbc7ef1aa2edfaa87772f1202d67e0ce
        # TODO imwrite has many quality/type flags

        ret = iwrite(self.image.astype(dtype), filename, **kwargs)

        return ret

    def EXIF(self):
        """
        Get image EXIF metadata

        :return: a dictionary of EXIF metadata
        :rtype: dict
        """
        try:
            import PIL
            from PIL.ExifTags import TAGS
        except ImportError:
            print('Pillow is required to read image file metadata\npip install pillow')

        image = PIL.Image.open(self.name)
        exif = {}

        # iterate over the EXIF tags 
        for tag, value in image._getexif().items():

            if tag in TAGS:
                # map tag number to tag name
                exif[TAGS[tag]] = value
        
        return exif

    def showpixels(self, textcolors=['yellow', 'blue'], fmt=None, ax=None, windowsize=0, **kwargs):

        if ax is None:
            ax = plt.gca()

        if self.isint:
            fmt = "{:d}"
            halfway = self.maxval / 2
        else:
            fmt = "{:.2f}"
            halfway = 0.5

        image = self.image
        for v in range(self.height):
            for u in range(self.width):
                if isinstance(textcolors, (list, tuple)):
                    if image[v, u] < halfway:
                        color = textcolors[0]
                    else:
                        color = textcolors[1]
                elif textcolors == 'grey':
                    if image[v, u] < halfway:
                        color = image[v, u] + 0.4 * np.r_[1,1,1]
                    else:
                        color = image[v,u] - 0.4 * np.r_[1,1,1]

                ax.text(u, v, fmt.format(image[v, u]), horizontalalignment='center', 
                    verticalalignment='center', color=color, **kwargs)

        ax.imshow(image, cmap='gray')
        ax.set_xlabel('u (pixels)')
        ax.set_ylabel('v (pixels)')

        plt.draw()

        class Window:
            def __init__(self, h=1, color='red', alpha=0.6, ax=None):
                self.h = h
                self.color = color
                self.alpha = alpha

                w = 2 * h + 1
                patch = plt.Rectangle((0, 0), w, w, color=color, alpha=alpha)
                if ax is None:
                    ax = plt.gca()

                ax.add_patch(patch)
                self.patch = patch

            def move(self, u, v, color=None, alpha=0.5):
                if color is not None:
                    self.color = color
                    self.patch.set_color(color)
                if alpha is not None:
                    self.alpha = alpha
                    self.patch.set_alpha(alpha)
                    
                self.patch.set_x(u - self.h - 0.5)
                self.patch.set_y(v - self.h - 0.5)

        if windowsize > 0:
            return Window(windowsize)

    # def ascvtype(self):
    #     if np.issubdtype(self.image.dtype, np.floating):
    #         return self.image.astype(np.float32)
    #     else:
    #         return self.image.astype(np.uint8)
    


    def anaglyph(self, right, colors='rc', disp=0):
        """
        Convert stereo images to an anaglyph image

        :param right: right image
        :type right: Image instance
        :param colors: lens colors (left, right), defaults to 'rc'
        :type colors: str, optional
        :param disp: disparity, defaults to 0
        :type disp: int, optional
        :return: anaglyph image
        :rtype: Image instance

        ``IM.analglyph(R)`` is an anaglyph image where the two images of
        a stereo pair are combined into a single image by coding them in two 
        different colors.  By default the left image is red, and the right 
        image is cyan.

        ``colors`` describes the lens color coding as a string with 2 letters,
        the first for left, the second for right, and each is one of:

            'r'   red
            'g'   green
            'b'   green
            'c'   cyan
            'm'   magenta

        If ``disp`` is positive the disparity is increased, if negative it
        is reduced.  These adjustments are achieved by trimming the images.  Use 
        this option to make the images more natural/comfortable to view, useful 
        if the images were captured with a stereo baseline significantly different
        the human eye separation (typically 65mm).

        :reference:
            - Robotics, Vision & Control, Section 14.3,
              P. Corke, Springer 2011.

        :seealso: :func:`Image.stdisp`
        """

        # ensure the images are greyscale
        width, height = self.size

        left = self.mono()
        right = right.mono()

        if disp > 0:
            left = left.trim(right=disp)
            right = right.trim(left=disp)

        elif disp < 0:
            disp = -disp
            left = left.trim(left=disp)
            right = right.trim(right=disp)

        colordict = {
            'r': (1, 0, 0),
            'g': (0, 1, 0),
            'b': (0, 0, 1),
            'c': (0, 1, 1),
            'm': (1, 0, 1),
            'o': (1, 1, 0),
        }
        return left.colorize(colordict[colors[0]]) \
             + right.colorize(colordict[colors[1]])

    def stdisp(self, right):
        class Cursor:
            """
            A cross hair cursor.
            """
            def __init__(self, ax, ax2):
                self.ax = ax
                self.ax2 = ax2
                self.horizontal_line = ax.axhline(color='k', lw=0.8)
                self.horizontal_line2 = ax2.axhline(color='k', lw=0.8)
                self.vertical_line = ax.axvline(color='k', lw=0.8)
                self.vertical_line2 = ax2.axvline(color='k', lw=0.8)
                self.vertical_line3 = ax2.axvline(color='k', lw=0.8, ls='--')
                self.leftclicked = False
                self.x_left = None

                # text location in axes coordinates
                self.text = self.ax2.text(0.05, 0.95, '', transform=ax2.transAxes,
                    backgroundcolor='w')

            def set_cross_hair_visible(self, visible):
                need_redraw = self.horizontal_line.get_visible() != visible
                self.horizontal_line.set_visible(visible)
                self.horizontal_line2.set_visible(visible)

                self.vertical_line.set_visible(visible)
                self.vertical_line2.set_visible(visible)
                self.vertical_line3.set_visible(visible)

                self.text.set_visible(visible)
                return need_redraw

            def on_mouse_move(self, event):
                if event.inaxes == self.ax2 and self.leftclicked:
                    x, y = event.xdata, event.ydata
                    # update the line positions
                    self.vertical_line3.set_xdata(x)
                    self.text.set_text('d={:.2f}'.format(self.x_left - x))
                    self.ax2.figure.canvas.draw()
                # if  event.inaxes:
                #     need_redraw = self.set_cross_hair_visible(False)
                #     if need_redraw:
                #         self.ax.figure.canvas.draw()
                # else:
                #     self.set_cross_hair_visible(True)
                #     x, y = event.xdata, event.ydata
                #     # update the line positions
                #     self.horizontal_line.set_ydata(y)
                #     self.vertical_line.set_xdata(x)
                #     # self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
                #     self.ax.figure.canvas.draw()

            def on_click(self, event):
                # if not event.inaxes:
                #     need_redraw = self.set_cross_hair_visible(False)
                #     if need_redraw:
                #         self.ax.figure.canvas.draw()
                # else:
                if event.inaxes == self.ax:
                    self.set_cross_hair_visible(True)
                    x, y = event.xdata, event.ydata
                    # update the line positions
                    self.horizontal_line.set_ydata(y)
                    self.vertical_line.set_xdata(x)
                    self.horizontal_line2.set_ydata(y)
                    self.vertical_line2.set_xdata(x)
                    self.ax.figure.canvas.draw()
                    self.leftclicked = True
                    self.x_left = x

        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

        self.disp(ax=ax1, grid=True)
        right.disp(ax=ax2, grid=True)

        cursor = Cursor(ax1, ax2)
        fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
        fig.canvas.mpl_connect('button_press_event', cursor.on_click)
        plt.show(block=True)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    import pathlib
    import os.path
    
    exec(open(pathlib.Path(__file__).parent.parent.absolute() / "tests" / "test_processing.py").read())  # pylint: disable=exec-used