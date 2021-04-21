class FeatureMixin:
    
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

        .. note::

            - The bins spans the greylevel range 0-255.
            - For a floating point image the histogram spans the greylevel
              range 0-1.
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
            maxrange = np.iinfo(self.dtype).max
        else:
            # float image
            maxrange = 1.0

        out = []
        for im in self:
            # normal histogram case

            xc = []
            hc = []
            hcdf = []
            hnormcdf = []
            implanes = cv.split(im.image)
            for i in range(self.numchannels):
                # bin coordinates
                x = np.linspace(0, maxrange, nbins, endpoint=True).T
                h = cv.calcHist(implanes, [i], None, [nbins], [0, maxrange + 1])

                if opt == 'sorted':
                    h = np.sort(h, axis=0)
                    isort = np.argsort(h, axis=0)
                    x = x[isort]

                cdf = np.cumsum(h)
                normcdf = cdf / cdf[-1]

                xc.append(x)
                hc.append(h)
                hcdf.append(cdf)
                hnormcdf.append(normcdf)

            # stack into arrays
            xs = np.vstack(xc).T
            hs = np.hstack(hc)
            cs = np.vstack(hcdf).T
            ns = np.vstack(hnormcdf).T

            # TODO this seems too complex, why do we stack stuff as well
            # as have an array of hist tuples??

            hhhx = Histogram(hs, cs, ns, xs)
            out.append(hhhx)

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

        if not isinstance(p, int):
            raise TypeError(p, 'p must be an int')
        if not isinstance(q, int):
            raise TypeError(q, 'q must be an int')

        out = []
        for im in self:
            x, y = self.imeshgrid(im.image)
            out.append(np.sum(im.image * (x ** p) * (y ** q)))

        return out

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

        if not isinstance(p, int):
            raise TypeError(p, 'p must be an int')
        if not isinstance(q, int):
            raise TypeError(q, 'q must be an int')

        out = []
        for im in self:
            x, y = self.imeshgrid(im.image)
            m00 = im.mpq(0, 0)
            xc = im.mpq(1, 0) / m00
            yc = im.mpq(0, 1) / m00
            out.append(np.sum(im.image * ((x - xc) ** p) * ((y - yc) ** q)))

        return out

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
        if not isinstance(p, int):
            raise TypeError(p, 'p must be an int')
        if not isinstance(q, int):
            raise TypeError(q, 'q must be an int')
        if (p+q) < 2:
            raise ValueError(p+q, 'normalized moments only valid for p+q >= 2')

        g = (p + q) / 2 + 1

        out = []
        for im in self:
            out.append(im.upq(p, q) / im.mpq(0, 0) ** g)

        return out

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
        im = self.mono()

        out = []
        for im in self:
            out.append(cv.moments(im.image, binary))
        # TODO check binary is True/False, but also consider 1/0

        return out

    def humoments(self):
        """
        Hu image moments
        :param im: binary image
        :type im: numpy array
        :return: hu image moments
        :type: dictionary

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
        """

        # TODO check for binary image
        out = []
        for im in self:
            h = cv.moments(im.image)
            out.append(cv.HuMoments(h))
        return out


class Histogram:

    def __init__(self, hs, cs, ns, xs):
        self.hs = hs # histogram
        self.cs = cs # cumulative histogram
        self.ns = ns # normalized cumulative histogram
        self.xs = xs  # x value
        # 'hist', 'h cdf normcdf x')

    def __str__(self):
        return f"histogram with {len(self.cs)} bins"

    def plot(self, type='histogram', block=False, **kwargs):

        if type == 'histogram':
            plot_histogram(self.xs.flatten(), self.hs.flatten(), block=block, 
            xlabel='pixel value', ylabel='number of pixels', **kwargs)
        elif type == 'cumulative':
            plot_histogram(self.xs.flatten(), self.cs.flatten(), block=block, 
            xlabel='pixel value', ylabel='cumulative number of pixels', **kwargs)
        elif type == 'normalized':
            plot_histogram(self.xs.flatten(), self.ns.flatten(), block=block, 
            xlabel='pixel value', ylabel='normalized cumulative number of pixels', **kwargs)

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