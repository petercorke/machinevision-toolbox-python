import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import ScalarFormatter

import cv2 as cv
from spatialmath import base, SE3
from machinevisiontoolbox.base import findpeaks, findpeaks2d

class ImageWholeFeaturesMixin:
    
    def hist(self, nbins=256, opt=None):
        """
        Image histogram

        :param nbins: number of histogram bins, defaults to 256
        :type nbins: int, optional
        :param opt: histogram option
        :type opt: str
        :return: histogram of image
        :rtype: :class:`Histogram`

        Returns an object that summarizes the distribution of
        pixel values in each color plane.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('street.png')
            >>> type(hist)
            >>> hist = img.hist()
            >>> hist
            >>> hist.plot()

        .. note::

            - The bins spans the greylevel range 0-255.
            - For a floating point image the histogram spans the greylevel range
              0.0 to 1.0 with 256 bins.
            - For floating point images all NaN and Inf values are first
              removed.
            - Computed using OpenCV CalcHist. Only works on floats up to 32 bit,
              images are automatically converted from float64 to float32


        :references:
            - Robotics, Vision & Control for Python, Section 14.4.3, P. Corke, Springer 2023.

        :seealso: `opencv.calcHist <https://docs.opencv.org/3.4/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d>`_
        """

        # check inputs
        optHist = ['sorted']
        if opt is not None and opt not in optHist:
            raise ValueError(opt, 'opt is not a valid option')

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
        if self.A.dtype == np.dtype('float64'):
            implanes = cv.split(self.A.astype('float32'))
        else:
            implanes = cv.split(self.A)
            
        for i in range(self.nplanes):
            # bin coordinates
            x = np.linspace(*xrange, nbins, endpoint=True).T
            # h = cv.calcHist(implanes, [i], None, [nbins], [0, maxrange + 1])
            h = cv.calcHist(implanes, [i], None, [nbins], xrange)
            if i==0:
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


    # def sum(self):
    #     """
    #     Sum of all pixels

    #     :return: sum of all pixel values
    #     :rtype: float or ndarray(P)

    #     Computes the sum of pixels in the image:
        
    #     .. math::
        
    #         \sum_{uv} I_{uv}

    #     For a P-channel image the result is a P-element array.

    #     Example:

    #     .. runblock:: pycon

    #         >>> from machinevisiontoolbox import Image
    #         >>> img = Image.Read('street.png')
    #         >>> img.sum()

    #     :seealso: :meth:`mpq` :meth:`npq` :meth:`upq`
    #     """
    #     out = []
    #     for im in self:
    #         out.append(np.sum(im.A))

    #     if len(out) == 1:
    #         return out[0]
    #     else:
    #         return out        

    def mpq(self, p, q):
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
            - Robotics, Vision & Control for Python, Section 12.1.3.4, P. Corke, Springer 2023.

        :seealso: :meth:`sum` :meth:`npq` :meth:`upq`
        """

        if not isinstance(p, int) or not isinstance(q, int):
            raise TypeError(p, 'p, q must be an int')

        im = self.mono().A
        X, Y = self.meshgrid()
        return np.sum(im * (X ** p) * (Y ** q))

    def upq(self, p, q):
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
            - Robotics, Vision & Control for Python, Section 12.1.3.4, P. Corke, Springer 2023.

        :seealso: :meth:`sum` :meth:`mpq` :meth:`upq`
        """

        if not isinstance(p, int) or not isinstance(q, int):
            raise TypeError(p, 'p, q must be an int')

        m00 = self.mpq(0, 0)
        xc = self.mpq(1, 0) / m00
        yc = self.mpq(0, 1) / m00

        im = self.mono().A
        X, Y = self.meshgrid()

        return np.sum(im * ((X - xc) ** p) * ((Y - yc) ** q))


    def npq(self, p, q):
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
            - Robotics, Vision & Control for Python, Section 12.1.3.4, P. Corke, Springer 2023.

        :seealso: :meth:`sum` :meth:`mpq` :meth:`upq`
        """
        if not isinstance(p, int) or not isinstance(q, int):
            raise TypeError(p, 'p, q must be an int')
        if (p+q) < 2:
            raise ValueError(p+q, 'normalized moments only valid for p+q >= 2')

        g = (p + q) / 2 + 1

        return self.upq(p, q) / self.mpq(0, 0) ** g

    def moments(self, binary=False):
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
            - Robotics, Vision & Control for Python, Section 12.1.3.4, P. Corke, Springer 2023.

        :seealso: :meth:`mpq` :meth:`npq` :meth:`upq` `opencv.moments <https://docs.opencv.org/master/d8/d23/classcv_1_1Moments.html>`_
        """
        return cv.moments(self.mono().to_int(), binary)

    def humoments(self):
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
            - Robotics, Vision & Control for Python, Section 12.1.3.6, P. Corke, Springer 2023.

        :seealso: :func:`opencv.HuMoments <https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gab001db45c1f1af6cbdbe64df04c4e944>`_
        """

        # TODO check for binary image

        moments = cv.moments(self.A)
        hu = cv.HuMoments(moments)
        return hu.flatten()

    def nonzero(self):
        """
        Find non-zero pixel values

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
            - Robotics, Vision & Control for Python, Section 12.1.3.2, P. Corke, Springer 2023.

        :seealso: :meth:`flatnonzero`
        """
        v, u = np.nonzero(self.A)
        return np.vstack((u, v))


    def flatnonzero(self):
        """
        Find non-zero pixel values

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
        return np.flatnonzero(self.A)

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

        ret = findpeaks2d(self.A, npeaks=npeaks, scale=scale, interp=interp)
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
            h = h[:,0]

        self._h = h # histogram
        self._x = x.flatten()  # x value
        self.isfloat = isfloat
        # 'hist', 'h cdf normcdf x')

    def __str__(self):
        """
        Histogram summary as a string

        :return: concise summary of histogram
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Random()
            >>> h = im.hist(100)
            >>> print(h)
        """
        s = f"histogram with {len(self.x)} bins"
        if self.nplanes > 1:
            s += f" x {self.h.shape[1]} planes"
        s += f": xrange {self.x[0]} - {self.x[-1]}, yrange {np.min(self.h)} - {np.max(self.h)}"
        return s

    def __repr__(self):
        """
        Print histogram summary

        :return: print concise summary of histogram
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Random()
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

    def plot(self, type='frequency', block=False, bar=None, style='stack', alpha=0.5, **kwargs):
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

        if type == 'frequency':
            y = self.h
            maxy = np.max(y)
            ylabel1 = 'frequency'
            ylabel2 = 'frequency'
            if bar is not False:
                bar = True
        elif type in ('cdf', 'cumulative'):
            y = self.cdf
            maxy = np.max(y[-1, :])
            ylabel1 = 'cumulative frequency'
            ylabel2 = 'cumulative frequency'
        elif type in ('ncdf', 'normalized'):
            y = self.ncdf
            ylabel1 = 'norm. cumulative freq.'
            ylabel2 = 'normalized cumulative frequency'
            maxy = 1
        else:
            raise ValueError('unknown type')

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
            if style == 'overlay':
                raise ValueError('cannot use overlay style for monochrome image')

        if style == 'stack':
            for i in range(n):
                ax = plt.subplot(n, 1, i + 1)
                if bar:
                    ax.bar(x, y[:, i], width=x[1] - x[0], bottom=0, **kwargs)
                else:
                    ax.plot(x, y[:, i], **kwargs)
                ax.grid()
                if n == 1:
                    ax.set_ylabel(ylabel1)
                else:
                    ax.set_ylabel(ylabel1 + ' (' + colors[i] + ')')
                ax.set_xlim(*xrange)
                ax.set_ylim(0, maxy)
                ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
            ax.set_xlabel('pixel value')

        elif style == 'overlay':
            x = np.r_[0, x, 255]
            ax = plt.subplot(1, 1, 1)

            patchcolor = []
            goodcolors = [c for c in "rgbykcm"]
            for i, color in colors:
                if color.lower() in "rgbykcm":
                    patchcolor.append(color.lower())
                else:
                    patchcolor.append(goodcolors.pop(0))

            for i in range(n):
                yi = np.r_[0, y[:, i], 0]
                p1 = np.array([x, yi]).T
                poly1 = Polygon(p1, closed=True, facecolor=patchcolor[i], alpha=alpha, **kwargs)
                ax.add_patch(poly1)
            ax.set_xlim(*xrange)
            ax.set_ylim(0, maxy)
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))

            ax.set_xlabel('pixel value')
            ax.set_ylabel(ylabel2)

            ax.grid(True)
            plt.legend(colors)
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

        :seealso: :func:`findpeaks`
        """
        if self.nplanes == 1:
            # greyscale image
            x, _ = findpeaks(self.h, self.x, **kwargs)
            return x
        
        else:
            xp = []
            for i in range(self.nplanes):
                x, _ = findpeaks(self.h[:,i], self.x, **kwargs)
                xp.append(x)
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

    from machinevisiontoolbox import Image
    from math import pi

    # img = Image.Read('flowers1.png', dtype='float32', grey=True)
    # print(img)
    # # img.disp()

    # h = img.hist()
    # print(h)
    # with np.printoptions(precision=2, threshold=5):
    #     print(h.h)
    #     print(h.cdf)
    #     print(h.ncdf)
    # print(h.peaks(scale=0.2))
    # # h.plot('frequency', style='overlay')
    # # plt.figure()
    # h.plot('frequency', block=True)

    # print(img.moments())

    im = Image.Read('penguins.png')
    z = im.ocr(minconf=90)
    print(z)
