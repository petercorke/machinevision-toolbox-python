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
        :type lut: array_like(256), ndarray(256,n)
        :param colororder: colororder for output image, optional
        :type colororder: str or dict
        :return: transformed image
        :rtype: Image instance

        For a greyscale image the LUT can be:

            - (256,)
            - (256,n) in which case the resulting image has ``n`` planes created
              my applying the n'th column of the LUT to the input image
  
        For a color image the LUT can be:

            - (256,) and applied to every plane, or
            - (256,n) where the LUT columns are applied to the ``n`` planes of
              the input image.

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
        :rtype: Image instance

        If ``vectorize`` is False the function is called with the underlying NumPy array as
        the argument, and it must return a NumPy array.  The array can have different 
        dimensions to its arguments.

        If ``vectorize`` is True the function is called for every pixel which is a 1d-array
        of length equal to the number of color planes.
        """
        if vectorize:
            func = np.vectorize(func)
        return self.__class__(func(self.A), colororder=self.colororder)

    def apply2(self, other, func, vectorize=False):
        """
        Apply a function to two images

        :param func: function to apply to image or pixel
        :type func: callable
        :return: transformed image
        :rtype: Image instance

        If ``vectorize`` is False the function is called with the underlying NumPy array as
        the argument, and it must return a NumPy array.  The array can have different 
        dimensions to its arguments.

        If ``vectorize`` is True the function is called for every pixel which is a 1d-array
        of length equal to the number of color planes.

        images must be same size, same number of color planes
        """
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
        :rtype: Image instance

        Pxiels in the returned image will have values in the range [``min``, ``max``]
        inclusive.
        """

        return self.__class__(np.clip(self.A, min, max), colororder=self.colororder)


    def roll(self, ru=0, rv=0):
        return self.__class__(np.roll(self.image, (ru, rv), (1, 0)))

    def normhist(self, nbins=256, opt=None):
        """
        Histogram normalisaton

        :param nbins: number of bins for histogram
        :type nbins: integer
        :param opt: histogram option
        :type opt: string
        :return nim: Image with normalised image
        :rtype nim: Image instance

        - ``IM.normhist()`` is a histogram normalized version of the image.

        Example:

        .. runblock:: pycon

        .. note::

            - Highlights image detail in dark areas of an image.
            - The histogram of the normalized image is approximately uniform,
              that is, all grey levels ae equally likely to occur.
            - Color images automatically converted to grayscale

        :seealso: `cv2.equalizeHist <https://docs.opencv.org/master/d6/dc7/group__imgproc__hist.html#ga7e54091f0c937d49bf84152a16f76d6e>`_
        """
        out = cv.equalizeHist(self.to_int())
        return self.__class__(self.like(out))

    def stretch(self, max=1, r=None):
        """
        Image normalisation

        :param max: M  pixels are mapped to the r 0 to M
        :type max: scalar integer or float
        :param r: r[0] is mapped to 0, r[1] is mapped to 1 (or max value)
        :type r: 2-tuple or numpy array (2,1)
        :return: Image with pixel values stretched to M across r
        :rtype: Image instance

        - ``IM.stretch()`` is a normalised image in which all pixel values lie
          in the r range of 0 to 1. That is, a linear mapping where the minimum
          value of ``im`` is mapped to 0 and the maximum value of ``im`` is
          mapped to 1.

        Example:

        .. runblock:: pycon

        .. note::

            - For an integer image the result is a float image in the range 0
              to max value

        :references:

            - Robotics, Vision & Control, Section 12.1, P. Corke,
              Springer 2011.
        """

        # TODO make all infinity values = None?

        im = self.A
        if r is None:
            mn = np.min(im)
            mx = np.max(im)
        else:
            r = argcheck.getvector(r)
            mn = r[0]
            mx = r[1]

        zs = (im - mn) / (mx - mn) * max

        if r is not None:
            zs = np.maximum(0, np.minimum(max, zs))
        return self.__class__(zs)

    def thresh(self, t=None, opt='binary'):
        """
        Image threshold

        :param t: threshold
        :type t: scalar
        :param opt: threshold option (see below)
        :type opt: string
        :return imt: Image thresholded binary image
        :rtype imt: Image instance
        :return: threshold if opt is otsu or triangle
        :rtype: list of scalars

        - ``IM.thresh()`` uses Otsu's method for thresholding a greyscale
          image.

        - ``IM.thresh(t)`` as above but the threshold ``t`` is specified.

        - ``IM.thresh(t, opt)`` as above but the threshold option is specified.
          See opencv threshold types for threshold options
          https://docs.opencv.org/4.2.0/d7/d1b/group__imgproc__
          misc.html#gaa9e58d2860d4afa658ef70a9b1115576

        Example:

        .. runblock:: pycon

        :options:

            - 'binary' # TODO consider the LaTeX formatting of equations
            - 'binary_inv'
            - 'trunc'
            - 'tozero'
            - 'tozero_inv'
            - 'otsu'
            - 'triangle'

        .. note::

            - Converts a color image to greyscale.
            - For a uint8 class image the slider range is 0 to 255.
            - For a floating point class image the slider range is 0 to 1.0
        """

        # dictionary of threshold options from OpenCV
        threshopt = {
            'binary': cv.THRESH_BINARY,
            'binary_inv': cv.THRESH_BINARY_INV,
            'trunc': cv.THRESH_TRUNC,
            'tozero': cv.THRESH_TOZERO,
            'tozero_inv': cv.THRESH_TOZERO_INV,
            'otsu': cv.THRESH_OTSU,
            'triangle': cv.THRESH_TRIANGLE
        }

        flag = threshopt[opt]
        if isinstance(t, str):
            # auto threshold requested
            flag |= threshopt[t]

            threshvalue, imt = cv.threshold(
                src=self.to_int(),
                thresh=0.0,
                maxval=self.maxval,
                type=flag)
            return self.__class__(self.like(imt)), self.like(int(threshvalue), max='uint8')

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

        # ACKNOWLEDGEMENT: https://matplotlib.org/devdocs/gallery/widgets/range_slider.html
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        from matplotlib import colors

        #N = 128
        Ncolors = 256
        img = self.image
        t = int((img.max() + img.min()) / 2)

        x = np.linspace(self.min, self.max, Ncolors)

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

        def update(val):
            # The val passed to a callback by the Slider

            # Update the image's colormap
            # im.norm.vmin = val
            # im.norm.vmax = val
            im.set_cmap(colormap(val))

            # Update the position of the vertical line
            lower_limit_line.set_xdata([val, val])

            # Redraw the figure to ensure it updates
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show(block=True)

    def ithresh2(self):

        # ACKNOWLEDGEMENT: https://matplotlib.org/devdocs/gallery/widgets/range_slider.html
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RangeSlider

        #N = 128
        img = self.image

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plt.subplots_adjust(bottom=0.25)

        im = axs[0].imshow(img)
        axs[1].hist(img.flatten(), bins='auto')
        axs[1].set_title('Histogram of pixel intensities')

        # Create the RangeSlider
        slider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
        slider = RangeSlider(slider_ax, "Threshold", img.min(), img.max())

        # Create the Vertical lines on the histogram
        lower_limit_line = axs[1].axvline(slider.val[0], color='k')
        upper_limit_line = axs[1].axvline(slider.val[1], color='k')


        def update(val):
            # The val passed to a callback by the RangeSlider will
            # be a tuple of (min, max)

            # Update the image's colormap
            im.norm.vmin = val[0]
            im.norm.vmax = val[1]

            # Update the position of the vertical lines
            lower_limit_line.set_xdata([val[0], val[0]])
            upper_limit_line.set_xdata([val[1], val[1]])

            # Redraw the figure to ensure it updates
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show(block=True)

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

        :return t: Otsu's threshold
        :rtype t: image type

        - ``otsu(im)`` is an optimal threshold for binarizing an image with a
          bimodal intensity histogram.  ``t`` is a scalar threshold that
          maximizes the variance between the classes of pixels below and above
          the thresold ``t``.

        Example::

        .. runblock:: pycon

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
        """

        # mvt-mat has options on levels and valleys, which Opencv does not have
        # TODO best option is likely just to code the function itself, with
        # default option of simply calling OpenCV's Otsu implementation

        _, t = self.thresh(t='otsu')
        return t

    def blend(self, image2, alpha, beta=None, gamma=0):
        r"""
        Image blending

        :param image2: second image
        :type image2: Image instance
        :param alpha: fraction of image
        :type alpha: float
        :param beta: fraction of ``image2``, defaults to 1-``alpha``
        :type beta: float, optional
        :param gamma: [description], defaults to 0
        :type gamma: int, optional
        :raises ValueError: images are not same size
        :raises ValueError: images are of different type
        :return: blended image
        :rtype: Image instance

        The resulting image is :math:`\alpha \mathbf{I}_1 + \beta \mathbf{I}_2`

        :seealso: `cv2.addWeighted <https://docs.opencv.org/master/d2/de8/group__core__array.html#gafafb2513349db3bcff51f54ee5592a19>`_
        """

        if self.shape != image2.shape:
            raise ValueError('images are not the same size')
        if self.isint != image2.isint:
            raise ValueError('images must be both int or both floating type')
            
        if beta is None:
            beta = 1 - alpha
        a = self.ascvtype()
        b = image2.ascvtype()
        out = cv.addWeighted(a, alpha, b, beta, gamma)
        return self.__class__(out, colororder=self.colororder)

    def choose(self, image2, mask):
        """
        Pixel-wise image merge

        :param mask: image mask
        :type mask: numpy array
        :param image2: second image
        :type image2: Image instance
        :return: merged images
        :rtype: Image instance

        - ``IM.pixelswitch(mask, im2)`` is an image where each pixel is
          selected from the corresponding pixel in self or ``image2`` according
          to the corresponding pixel values in ``mask``.  If the element of
          ``mask`` is zero/false self is selected, otherwise ``image2`` is selected.

        - ``im2`` can contain a color descriptor which is one of: a scalar
          value corresponding to a greyscale, a 3-vector corresponding to a
          color value, or a string containing the name of a color which is
          found using COLORNAME.

        .. note::

            - ``im2`` and ``mask`` must all have the same number of
              rows and columns (unless ``im1`` or ``im2`` are specifying a
              color)
            - If ``im1`` and ``im2`` are both greyscale then ``out`` is
              greyscale.
            - If either of ``im1`` or ``im2`` are color then ``out`` is color.
            - If either one image is double and one is integer then the integer
              image is first converted to a double image.
        
        :seealso: `cv2.bitwise_and <https://docs.opencv.org/master/d2/de8/group__core__array.html#ga60b4d04b251ba5eb1392c34425497e14>`_
        """
        im1 = self.image

        if isinstance(mask, self.__class__):
            mask = mask.A > 0
        elif not isinstance(mask, np.ndarray):
            raise ValueError('bad type for mask')

        mask = mask.astype(np.uint8)

        if isinstance(image2, self.__class__):
            # second image is Image type
            im2 = image2.image
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

    # ======================= interpolate ============================= #

    def meshgrid(self=None, width=None, height=None, step=1):
        """
        Domain matrices for image

        :param a1: array input 1
        :type a1: numpy array
        :param a2: array input 2
        :type a2: numpy array
        :return u: domain of image, horizontal
        :rtype u: numpy array
        :return v: domain of image, vertical
        :rtype v: numpy array

        - ``IM.imeshgrid()`` are matrices that describe the domain of image
          ``im (h,w)`` and are each ``(h,w)``. These matrices are used for the
          evaluation of functions over the image. The element ``u(r,c) = c``
          and ``v(r,c) = r``.

        - ``IM.imeshgrid(w, h)`` asv[] above but the domain is ``(w,h)``.

        - ``IM.imeshgrid(s)`` as above but the domain is described by ``s``
          which can be a scalar ``(s,s)`` or a 2-vector ``s=[w,h]``.

        U, V = image.meshgrid()
        U[v,u] -> u
        V[v,u] -> v

        Example:

        .. runblock:: pycon

        """
        if self is not None:
            u = self.uspan(step)
            v = self.vspan(step)
        else:                
            u = np.arange(0, width, step)
            v = np.arange(0, height, step)
        
        return np.meshgrid(u, v)

    def interp2d(self, Ui, Vi, U=None, V=None, **kwargs):

        if U is None and V is None:
            if self.domain is None:
                U, V = self.meshgrid()
            else:
                U, V = np.meshgrid(*self.domain)

        points = np.array((U.flatten(), V.flatten())).T
        values = self.image.flatten()
        xi = np.array((Ui.flatten(), Vi.flatten())).T
        Zi = sp.interpolate.griddata(points, values, xi)
        
        return self.__class__(Zi.reshape(Ui.shape), **kwargs)

    def rotate_spherical(self, pose):
        Phi, Theta = self.meshgrid(*self.domain)
        nPhi, nTheta = sphere_rotate(Phi, Theta, pose)

        # warp the image
        return self.interp2d(nPhi, nTheta, domain=self.domain)

    def paste(self,
              pattern,
              pt,
              method='set',
              position='topleft',
              zero=True):
        """
        Paste an image into an image

        INPLACE!

        :param pattern: sub-image super-imposed onto onto canvas
        :type pattern: numpy array
        :param pt: coordinates where pattern is pasted
        :type pt: 2-element vector of integer coordinates
        :param method: options for image merging
        :type method: string
        :param centre: True if pattern centered at pt, else topleft of pattern
        :type centre: boolean
        :param zero: zero-based coordinates (True) or 1-based coordinates
        :type zero: boolean
        :return out: Image with pasted image
        :rtype out: Image instance

        - ``IM.paste(pattern, pt)`` is the image canvas with the subimage
          ``pattern`` pasted in at the position ``pt=[U, V]``.

        - ``IM.paste(pattern, pt, centre)`` as above with centre specified. The
          pasted image is centred at ``pt``, otherwise ``pt`` is the top-left
          corner of the subimage in the image (default).

        - ``IM.paste(pattern, pt, zero)`` as above with zero specified. The
          coordinates of ``pt`` start at zero, by default (0, 0) is assumed.
          TODO shouldn't this be a point? like ``origin`` or something

        - ``IM.paste(pattern, pt, opt)`` as above with opt specified as below.

        :options:

            - 'set'       ``pattern`` overwrites the pixels in ``canvas``
              (default).
            - 'add'       ``pattern`` is added to the pixels in ``canvas``.
            - 'mean'      ``pattern`` is set to the mean of pixel values
              in ``canvas`` and ``pattern``.

        Example:

        .. runblock:: pycon

        .. note::

            - Pixels outside the pasted in region are unaffected.
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

        if position == 'centre':
            left = pt[0] - np.floor(pw / 2)
            top = pt[1] - np.floor(ph / 2)
        elif position == 'topleft':
            left = pt[0]  # x
            top = pt[1]  # y
        else:
            raise ValueError('bad position specified')

        left = int(left)
        top = int(top)

        if not zero:
            left += 1
            top += 1

        # indices must be integers
        top = np.int(top)
        left = np.int(left)

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

        return self.__class__(o)

    def invert(self):
        if self.isint:
            out = np.where(self.image == 0, self.like(self.maxval), self.like(self.minval))
        elif self.isfloat:
            out = np.where(self.image == 0, 1.0, 0.0)
        return self.__class__(out)

    def distance_transform(self, invert=False, norm="L2", maskSize=3):
        # OpenCV does distance to nearest zero pixel
        # this function does distance to nearest non-zero pixel by default,
        # and the OpenCV thing if invert=True
        if invert:
            # distance to nearest zero pixel
            im = self.to_int()
        else:
            # distance to nearest non-zero pixel, invert the image
            im = self.invert().to_int()

        normdict = {
            "L1": cv.DIST_L1,
            "L2": cv.DIST_L2,
        }

        out = cv.distanceTransform(im, distanceType=normdict[norm], maskSize=maskSize)
        return self.__class__(out)

    # ======================= labels ============================= #

    def labels_binary(self, connectivity=4):
        """
        Binary connectivity analysis

        :param connectivity: [description], defaults to 4
        :type connectivity: int, optional
        :return: [description]
        :rtype: [type]

        `labels, N = IM.labels_binary()` performs binary connectivity
        where ``labels`` is a label image, the same size as ``IM``, containing
        integer blobs labels.  The background has label 0.  ``N`` is the number
        of labels, so labels lie in the range [0, N-1].

        :seealso: `cv2.connectedComponents <https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gaedef8c7340499ca391d459122e51bef5>`_
        """
        retval, labels = cv.connectedComponents(
            image=self.to_int(),
            connectivity=connectivity,
            ltype=cv.CV_32S
        )
        return self.__class__(labels), retval

    def labels_MSER(self, **kwargs):

        mser = cv.MSER_create(**kwargs)
        regions, _ = mser.detectRegions(self.to_int())

        if len(regions) < 256:
            dtype = np.uint8
        else:
            dtype=np.uint32

        out = np.zeros(self.shape, dtype=dtype)

        for i, points in enumerate(regions):
            # print('region ', i, points.shape[0])
            out[points[:,1], points[:,0]] = i

        return self.__class__(out, dtype=dtype), len(regions)

    def labels_graphseg(self, sigma=0.5, k=2000, minsize=100):
        # P. Felzenszwalb, D. Huttenlocher: "Graph-Based Image Segmentation
        segmenter = cv.ximgproc.segmentation.createGraphSegmentation(
            sigma=0.5,
            k=2000,
            min_size=100)
        out = segmenter.processImage(self.to_int())

        return self.__class__(out), np.max(out)


    # ======================= stereo ================================== #

    def stereo_simple(self, right, hw, drange):

        def window_stack(image, hw):
            # convert an image to a stack where the planes represent shifted
            # version of image.  For a pixel at (u, v) ...

            stack = []
            w = 2 * hw + 1
            w1 = w - 1

            for upad in range(w):
                for vpad in range(w):
                    stack.append(
                        np.pad(image, ((vpad, w1 - vpad), (upad, w1- upad)),
                        mode='constant', constant_values=np.nan)
                    )
            return np.dstack(stack)

        if isinstance(drange, int):
            drange = (0, drange)

        # left = self.mono().image.astype(np.float32)
        # right = right.mono().image.astype(np.float32)
        left = self.mono().image
        right = right.mono().image

        # convert to window stacks
        left = window_stack(left, hw)
        right = window_stack(right, hw)

        # offset the mean value of each template
        left = left - left.mean(axis=2)[..., np.newaxis]
        right = right - right.mean(axis=2)[..., np.newaxis]

        # idisp(np.sum(left ** 2, axis=2))
        # idisp(np.sum(right ** 2, axis=2))

        # shift right image to the right
        right = right[:, :-drange[0], :]
        right = np.pad(right, ((0, 0), (drange[0], 0), (0, 0)),
            mode='constant', constant_values=np.nan)

        similarities = []

        # suppress divide by zero error messages
        # possible ZNCC values include:
        #  - NaN 0 / 0  invalid value encountered in true_divide
        # - inf  x / 0  divide by zero encountered in true_divide
        with np.errstate(divide='ignore', invalid='ignore'):

            for d in np.arange(drange[1] - drange[0]):

                # compute the ZNCC
                sumLL = np.sum(left ** 2, axis=2)
                sumRR = np.sum(right ** 2, axis=2)
                sumLR = np.sum(left * right, axis=2)

                denom = np.sqrt(sumLL * sumRR)
                # if (denom == 0).sum() > 0:
                #     print('divide by zero in ZNCC')

                similarity = sumLR / denom

                similarity = np.where(denom==0, np.nan, similarity)
                similarities.append(similarity)

                # shift right image 1 pixel to the right
                right = right[:, :-1, :]
                right = np.pad(right, ((0, 0), (1, 0), (0, 0)),
                    mode='constant', constant_values=np.nan)

        # stack the similarity images at each disparity into the 3D DSI
        dsi = np.dstack(similarities)
        
        # disparity is the index of the maxima in the disparity direction
        disparity = np.argmax(dsi, axis=2).astype(np.float32) + drange[0]

        # maxima is the maximum similarity in the disparity direction
        maxima = np.max(dsi, axis=2)

        # whereever maxima is nan set disparity to nan, similarity will be 
        # done for border regions
        disparity = np.where(np.isnan(maxima), np.nan, disparity)

        disparity[:, :drange[0]] = np.nan

        return self.__class__(disparity, dtype=np.float32), \
               self.__class__(maxima), \
               dsi

    @classmethod
    def DSI_refine(cls, DSI, drange=None):
        DSI_flat = DSI.reshape((-1,DSI.shape[2]))

        YP = []
        Y = []
        YN = []
        
        if drange is None:
            disparity = np.argmax(DSI, axis=2)
            drange = [disparity.min(), disparity.max()]
        for i, d in enumerate(np.argmax(DSI, axis=2).ravel()):
            if drange[0] < d < drange[1]:
                YP.append(DSI_flat[i, d-1])
                Y.append(DSI_flat[i, d])
                YN.append(DSI_flat[i, d+1])
            else:
                YP.append(np.nan)
                Y.append(np.nan)
                YN.append(np.nan)
        
        YP= np.array(YP).reshape(DSI.shape[:2])
        Y = np.array(Y).reshape(DSI.shape[:2])
        YN = np.array(YN).reshape(DSI.shape[:2])

        A = YP + YN - 2 * Y
        B = YN - YP

        d_subpix = disparity - B / (2 * A)

        return cls(d_subpix), cls(A)



    def stereo_BM(self, right, hw, drange, speckle=None):
        # https://docs.opencv.org/master/d9/dba/classcv_1_1StereoBM.html
        
        if isinstance(drange, int):
            drange = (0, drange)
        
        if hw < 2:
            raise ValueError('block size too small')

        # number of disparities must be multiple of 16
        ndisparities = drange[1] - drange[0]
        ndisparities = int(np.ceil(ndisparities // 16) * 16)

        # create the stereo matcher
        stereo = cv.StereoBM_create(
            numDisparities=ndisparities,
            blockSize=2*hw+1)
        stereo.setMinDisparity(drange[0])

        left = self.mono().image.astype(np.uint8)
        right = right.mono().image.astype(np.uint8)

        # set speckle filter
        # it seems to make very little difference
        # it's not clear if range is in the int16 units or not
        if speckle is None:
            speckle = (0, 0)

        stereo.setSpeckleWindowSize(speckle[0])
        stereo.setSpeckleRange(int(16 * speckle[1]))

        disparity = stereo.compute(
            left=left,
            right=right)

        return self.__class__(disparity / 16.0)

    def stereo_SGBM(self, right, hw, drange, speckle=None):
        # https://docs.opencv.org/master/d2/d85/classcv_1_1StereoSGBM.html#details
        
        if isinstance(drange, int):
            drange = (0, drange)
        
        if hw < 2:
            raise ValueError('block size too small')

        # number of disparities must be multiple of 16
        ndisparities = drange[1] - drange[0]
        ndisparities = int(np.ceil(ndisparities // 16) * 16)

        # create the stereo matcher
        stereo = cv.StereoSGBM_create(
            minDisparity=drange[0],
            numDisparities=ndisparities,
            blockSize=2*hw+1)


        left = self.mono().image.astype(np.uint8)
        right = right.mono().image.astype(np.uint8)

        # set speckle filter
        # it seems to make very little difference
        # it's not clear if range is in the int16 units or not
        if speckle is not None:
            stereo.setSpeckleWindowSize(speckle[0])
            stereo.setSpeckleRange(speckle[1])

        disparity = stereo.compute(
            left=left,
            right=right)

        return self.__class__(disparity / 16.0)

    def line(self, start, end, color):
        return self.__class__(cv.line(self.image, start, end, color))

    def warp_perspective(self, H, method='linear', inverse=False, tile=False, size=None):

        if not (isinstance(H, np.ndarray) and H.shape == (3,3)):
            raise TypeError('H must be a 3x3 NumPy array')
        if size is None:
            size = self.size
        
        if tile:
            corners = np.array([
                [0, size[0], size[0],  0],
                [0, 0,        size[1], size[1]]
            ])
            if inverse:
                # can't use WARP_INVERSE_MAP if we want to compute the output
                # tile
                H = np.linalg.inv(H)
                inverse = False
            wcorners = h2e(H @ e2h(corners))
            tl = np.floor(wcorners.min(axis=1)).astype(int)
            br = np.ceil(wcorners.max(axis=1)).astype(int)
            size = br - tl
            H = transl2(-tl)  @ H

        warp_dict = {
            'linear': cv.INTER_LINEAR,
            'nearest': cv.INTER_NEAREST
        }
        flags = warp_dict[method]
        if inverse:
            flags |= cv.WARP_INVERSE_MAP
        out = cv.warpPerspective(src=self.A, M=H, dsize=tuple(size), flags=flags)

        if tile:
            return self.__class__(out), tl, wcorners
        else:
            return self.__class__(out)

    def rectify_homographies(self, m, F):
        retval, H1, H2 = cv.stereoRectifyUncalibrated(m.inliers.p1, m.inliers.p2, F, self.size)
        return H1, H2

    def scalespace(self, n, sigma=1):

        im = self.copy()
        g = [im]
        scale = 0.5
        scales = [scale]
        lap = []

        for i in range(n-1):
            im = im.smooth(sigma)
            scale = np.sqrt(scale ** 2 + sigma ** 2)
            scales.append(scale)
            g.append(im)
            x = (g[-1] - g[-2]) * scale ** 2 
            lap.append(x)

        return g, lap, scales

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