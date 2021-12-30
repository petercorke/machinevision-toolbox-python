import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import ScalarFormatter

import cv2 as cv
from spatialmath import base
from machinevisiontoolbox.base import plot_histogram

class ImageFeaturesMixin:
    
    def hist(self, nbins=256, opt=None):
        """
        Image histogram

        :param nbins: number of bins for histogram
        :type nbins: integer
        :param opt: histogram option
        :type opt: string
        :return hist: histogram h as a column vector, and corresponding bins x,
        cdf and normcdf
        :rtype hist: collections.namedtuple

        - ``IM.hist()`` is the histogram of intensities for image as a vector.
          For an image with  multiple planes, the histogram of each plane is
          given in a separate column. Additionally, the cumulative histogram
          and normalized cumulative histogram, whose maximum value is one, are
          computed.

        - ``IM.hist(nbins)`` as above with the number of bins specified

        - ``IM.hist(opt)`` as above with histogram options specified

        :options:

            - 'sorted' histogram but with occurrence sorted in descending
              magnitude order.  Bin coordinates X reflect this sorting.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('street.png')
            >>> hist = im.hist()
            >>> type(hist)
            >>> hist.plot()

        .. note::

            - The bins spans the greylevel range 0-255.
            - For a floating point image the histogram spans the greylevel
              range 0-1 with 256 bins.
            - For floating point images all NaN and Inf values are first
              removed.
            - OpenCV CalcHist only works on floats up to 32 bit, images are
              automatically converted from float64 to float32
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
        implanes = cv.split(self.A)
        for i in range(self.nplanes):
            # bin coordinates
            x = np.linspace(*xrange, nbins, endpoint=True).T
            # h = cv.calcHist(implanes, [i], None, [nbins], [0, maxrange + 1])
            h = cv.calcHist(implanes, [i], None, [nbins], xrange)
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


    def sum(self):
        out = []
        for im in self:
            out.append(np.sum(im.A))

        if len(out) == 1:
            return out[0]
        else:
            return out        

    def mpq(self, p, q):
        """
        Image moments

        :param p: p'th exponent
        :type p: integer
        :param q: q'th exponent
        :type q: integer
        :return: moment
        :type: list of scalars (same as image type)

        -``IM.mpq(p, q)`` is the pq'th moment of the image. That is, the sum of
        ``im(x,y) . x^p . y^q``

        Example:

        .. runblock:: pycon

        """

        if not isinstance(p, int) or not isinstance(q, int):
            raise TypeError(p, 'p, q must be an int')

        x, y = self.meshgrid(im.A)
        return np.sum(self.mono().A * (x ** p) * (y ** q))

    def upq(self, p, q):
        """
        Central image moments

        :param p: p'th exponent
        :type p: integer
        :param q: q'th exponent
        :type q: integer
        :return: moment
        :type: list of scalar (same as image type)

        - ``IM.upq(p, q)`` is the pq'th central moment of the image. That is,
          the sum of ``im(x,y) . (x - x0)^p . (y - y0)^q`` where (x0, y0) is
          the centroid

        Example:

        .. runblock:: pycon

        .. notes::

            - The central moments are invariant to translation

        """

        if not isinstance(p, int) or not isinstance(q, int):
            raise TypeError(p, 'p, q must be an int')

        x, y = self.imeshgrid(im.A)
        m00 = im.mpq(0, 0)
        xc = im.mpq(1, 0) / m00
        yc = im.mpq(0, 1) / m00
        return np.sum(im.A * ((x - xc) ** p) * ((y - yc) ** q))


    def npq(self, p, q):
        """
        Normalized central image moments

        :param p: p'th exponent
        :type p: integer
        :param q: q'th exponent
        :type q: integer
        :return: moment
        :type: list of scalar (same as image type)

        - ``IM.npq(p, q)`` is the pq'th normalized central moment of the image.
          That is, the sum of upq(im,p,q) / mpq(im,0,0)

        Example:

        .. runblock:: pycon

        .. notes::

            - The normalized central moments are invariant to translation and
              scale.

        """
        if not isinstance(p, int) or not isinstance(q, int):
            raise TypeError(p, 'p, q must be an int')
        if (p+q) < 2:
            raise ValueError(p+q, 'normalized moments only valid for p+q >= 2')

        g = (p + q) / 2 + 1

        return m.upq(p, q) / self.mpq(0, 0) ** g

    def moments(self, binary=False):
        """
        Image moments

        :param im: binary image
        :type im: numpy array
        :param binary: if True, all non-zero pixels are treated as 1's
        :type binary: bool
        :return: image moments
        :type: dictionary

        - ``IM.moments()`` are the image moments of the image, supplied as a
          dictionary.

        - ``IM.moments(binary)`` as above, but if True, all non-zero pixels are
          treated as 1's in the image.

        Example:

        .. runblock:: pycon

        .. note::

            - Converts a color image to greyscale.

        """
        return cv.moments(self.mono().asint(), binary)

    def nonzero(self):
        v, u = np.nonzero(self.A)
        return np.array((u, v))


    def peak2(self, npeaks=2, scale=1, interp=False, positive=True):
        """
        Find peaks in a matrix

        :param npeaks: number of peaks to return (default all)
        :type npeaks: scalar
        :param sc: scale of peaks to consider
        :type sc: float
        :param interp:  interpolation done on peaks
        :type interp: boolean
        :return: peaks, xy locations, ap? TODO
        :rtype: collections.namedtuple

        - ``IM.peak2()`` are the peak values in the 2-dimensional signal
          ``IM``. Also returns the indices of the maxima in the matrix ``IM``.
          Use SUB2IND to convert these to row and column.

        - ``IM.peak2(npeaks)`` as above with the number of peaks to return
          specifieid (default all).

        - ``IM.peak2(sc)`` as above with scale ``sc`` specified. Only consider
          as peaks the largest value in the horizontal and vertical range +/- S
          units.

        - ``IM.peak2(interp)`` as above with interp specified. Interpolate peak
          (default no peak interpolation).

        Example:

        .. runblock:: pycon

        .. note::

            - A maxima is defined as an element that larger than its eight
              neighbours. Edges elements will never be returned as maxima.
            - To find minima, use PEAK2(-V).
            - The interp options fits points in the neighbourhood about the
              peak with a paraboloid and its peak position is returned.  In
              this case IJ will be non-integer.

        """

        # TODO check valid input

        # create a neighbourhood mask for non-local maxima suppression

        # scale is taken as half-width of the window
        w = 2 * scale + 1
        M = np.ones((w, w), dtype='uint8')
        M[scale, scale] = 0  # set middle pixel to zero


        # compute the neighbourhood maximum
        # znh = self.window(self.float(z), M, 'max', 'wrap')
        # image = self.asint()
        # nh_max = cv.morphologyEx(image, cv.MORPH_DILATE, M)
        image = self.A
        nhood_max = sp.ndimage.maximum_filter(image, footprint=M)

        # find all pixels greater than their neighbourhood
        
        if positive:
            k = np.flatnonzero((image > nhood_max) & (image > 0))
        else:
            k = np.flatnonzero(image > nhood_max)

        # sort these local maxima into descending order
        image_flat = self.A.ravel()

        maxima = image_flat[k]

        ks = np.argsort(-maxima)
        k = k[ks]

        npks = min(len(k), npeaks)
        k = k[0:npks]

        x, y = np.unravel_index(k, image.shape)
        # xy = np.stack((y, x), axis=0)
        return np.column_stack((y, x, image_flat[k]))

        # interpolate peaks if required
        if interp:
            # TODO see peak2.m, line 87-131
            raise ValueError(interp, 'interp not yet supported')
        else:
            xyp = xy
            zp = image_flat[k]
            ap = []


    def ocr(self, minconf=50, plot=False):
        # https://github.com/madmaze/pytesseract
        try:
            import pytesseract
        except:
            print('you need to install pytesseract:')
            return

        ocr = pytesseract.A_to_data(
                self.A,
                output_type=pytesseract.Output.DICT)

        if plot:
            for i, conf in enumerate(ocr['conf']):
                if conf == '-1':  # I suspect this was not meant to be a string
                    continue
                if conf < minconf:
                    continue

                plot_labelbox(
                    ocr['text'][i],
                    tl=(ocr['left'][i], ocr['top'][i]),
                    wh=(ocr['width'][i], ocr['height'][i]),
                    color='y',
                    linestyle='--')

        return ocr
        
    def humoments(self):
        """
        Hu image moments
        :param im: image
        :type im: Image instance
        :return: Hu image moments
        :type: ndarray(7)

        - ``IM.humoments()`` are the Hu image moments of the imag as a
          dictionary.

        Example:

        .. runblock:: pycon

        .. note::

            - image is assumed to be a binary image of a single connected
              region

        :references:

            - M-K. Hu, Visual pattern recognition by moment invariants. IRE
              Trans. on Information Theory, IT-8:pp. 179-187, 1962.
        `cv.moments<https://docs.opencv.org/master/d8/d23/classcv_1_1Moments.html>`_
        `cv2.HuMoments<https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gab001db45c1f1af6cbdbe64df04c4e944>`_
        """

        # TODO check for binary image

        moments = cv.moments(self.A)
        hu = cv.HuMoments(moments)
        return hu.flatten()


    def harriscorner(self, nfeat=100, k=0.04, hw=2, scale=7):
        dst = cv.cornerHarris(self.asfloat(), 2, 2 * hw + 1, k)
        dst = self.__class__(dst)
        pks = dst.peak2(npeaks=nfeat, scale=scale, positive=True)

        return pks, dst

class Histogram:

    def __init__(self, h, x, isfloat):
        self._h = h # histogram
        self._x = x.flatten()  # x value
        self.isfloat = isfloat
        # 'hist', 'h cdf normcdf x')

    def __str__(self):
        s = f"histogram with {len(self.x)} bins"
        if self.h.shape[1] > 1:
            s += f" x {self.h.shape[1]} planes"
        s += f": xrange {self.x[0]} - {self.x[-1]}, yrange {np.min(self.h)} - {np.max(self.h)}"
        return s

    def __repr__(self):
        return str(self)

    @property
    def x(self):
        return self._x

    @property
    def h(self):
        return self._h

    @property
    def cdf(self):
        return np.cumsum(self._h, axis=0)

    @property
    def ncdf(self):
        y = np.cumsum(self._h, axis=0)
        y = y / y[-1, :]
        return y

    def plot(self, type='frequency', block=False, bar=None, style='stack', alpha=0.5, **kwargs):

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

        if self.isfloat:
            xrange = (0.0, 1.0)
        else:
            xrange = (0, 255)

        if self.colordict is not None:
            colors = list(self.colordict.keys())
            n = len(colors)
            ylabel1 += ' (' + colors[i] + ')'
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
                ax.set_ylabel(ylabel1)
                ax.set_xlim(*xrange)
                ax.set_ylim(0, maxy*2)
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

    def peak(self):
        pass

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

    img = Image.Read('monalisa.png', dtype='float32', grey=False)
    print(img)
    # img.disp()

    h = img.hist()
    print(h)
    h.plot('frequency', style='overlay')
    plt.figure()
    h.plot('frequency', block=True)