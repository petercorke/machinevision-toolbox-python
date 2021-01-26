#!/usr/bin/env python

import numpy as np
import cv2 as cv
import time
import scipy as sp


class ImageProcessingMorphMixin:
    """
    Image processing morphological operations on the Image class
    """

    def _getse(self, se):
        """
        Get structuring element

        :param se: structuring element
        :type se: array (N,H)
        :return se: structuring element
        :rtype: Image instance (N,H) as uint8

        - ``IM.getse(se)`` converts matrix ``se`` into a uint8 numpy array for
          opencv, which only accepts kernels of type CV_8U
        """
        if se.min() < 0:
            raise ValueError('cannot convert array with negative values to a structuring element')
        return np.array(se).astype(np.uint8)

    def erode(self, se, n=1, opt='replicate', **kwargs):
        """
        Morphological erosion

        :param se: structuring element
        :type se: numpy array (S,T), where S < N and T < H
        :param n: number of times to apply the erosion
        :type n: integer
        :param opt: option specifying the type of erosion
        :type opt: string
        :return out: Image with eroded binary image pixel values
        :rtype: Image instance

        - ``IM.erode(se, opt)`` is the image after morphological erosion with
          structuring element ``se``.

        - ``IM.erode(se, n, opt)`` as above, but the structruring element
          ``se`` is applied ``n`` times, that is ``n`` erosions.

        :options:

            - 'replicate'     the border value is replicated (default)
            - 'none'          pixels beyond the border are not included in the
              window
            - 'trim'          output is not computed for pixels where the
              structuring element crosses the image border, hence output image
              has reduced dimensions TODO

        .. note::

            - Cheaper to apply a smaller structuring element multiple times
              than one large one, the effective structuing element is the
              Minkowski sum of the structuring element with itself N times.

        Example:

        .. runblock:: pycon

        :references:

            - Robotics, Vision & Control, Section 12.5, P. Corke,
              Springer 2011.
        """

        # check if valid input:
        se = self._getse(se)
        # TODO check if se is valid (odd number and less than im.shape)
        # consider cv.getStructuringElement?
        # eg, se = cv.getStructuringElement(cv.MORPH_RECT, (3,3))

        if not isinstance(n, int):
            n = int(n)
        if n <= 0:
            raise ValueError(n, 'n must be greater than 0')

        if not isinstance(opt, str):
            raise TypeError(opt, 'opt must be a string')

        cvopt = {
            'replicate': cv.BORDER_REPLICATE,
            'none': cv.BORDER_ISOLATED,
            # 'wrap': cv.BORDER_WRAP # BORDER_WRAP is not supported in OpenCV
        }

        if opt not in cvopt.keys():
            raise ValueError(opt, 'opt is not a valid option')
        out = []
        for im in self:
            out.append(cv.erode(im.image, se,
                                iterations=n,
                                borderType=cvopt[opt],
                                **kwargs))

        return self.__class__(out)

    def dilate(self, se, n=1, opt='replicate', **kwargs):
        """
        Morphological dilation

        :param se: structuring element
        :type se: numpy array (S,T), where S < N and T < H
        :param n: number of times to apply the dilation
        :type n: integer
        :param opt: option specifying the type of dilation
        :type opt: string :return
        out: Image with dilated binary image values
        :rtype: Image instance

        - ``IM.dilate(se, opt)`` is the image after morphological dilation with
          structuring element ``se``.

        - ``IM.dilate(se, n, opt)`` as above, but the structruring element
          ``se`` is applied ``n`` times, that is ``n`` dilations.

        :options::

            - 'replicate'     the border value is replicated (default)
            - 'none'          pixels beyond the border are not included in the
              window
            - 'trim'          output is not computed for pixels where the
              structuring element crosses the image border, hence output image
              has reduced dimensions TODO

        .. note::

            - Cheaper to apply a smaller structuring element multiple times
            than one large one, the effective structuing element is the
            Minkowski sum of the structuring element with itself N times.

        Example:

        .. runblock:: pycon

        :references:

            - Robotics, Vision & Control, Section 12.5, P. Corke,
              Springer 2011.
        """

        # check if valid input:
        se = self._getse(se)

        if not isinstance(n, int):
            n = int(n)
        if n <= 0:
            raise ValueError(n, 'n must be greater than 0')

        if not isinstance(opt, str):
            raise TypeError(opt, 'opt must be a string')

        # convert options TODO trim?
        cvopt = {
            'replicate': cv.BORDER_REPLICATE,
            'none': cv.BORDER_ISOLATED
        }

        if opt not in cvopt.keys():
            raise ValueError(opt, 'opt is not a valid option')

        out = []
        # for im in [img.image in self]: # then can use cv.dilate(im)
        for im in self:
            out.append(cv.dilate(im.image, se,
                       iterations=n,
                       borderType=cvopt[opt],
                       **kwargs))

        return self.__class__(out)

    def morph(self, se, oper, n=1, opt='replicate', **kwargs):
        """
        Morphological neighbourhood processing

        :param se: structuring element :type se: numpy array (S,T), where S < N
        and T < H :param oper: option specifying the type of morphological
        operation :type oper: string :param n: number of times to apply the
        operation :type n: integer :param opt: option specifying the border
        options :type opt: string :return out: Image with morphed pixel values
        :rtype: Image instance

        - ``IM.morph(se, opt)`` is the image after morphological operation with
          structuring element ``se``.

        - ``IM.morph(se, n, opt)`` as above, but the structruring element
          ``se`` is applied ``n`` times, that is ``n`` morphological
          operations.

        :operation options:

            - 'min'       minimum value over the structuring element
            - 'max'       maximum value over the structuring element
            - 'diff'      maximum - minimum value over the structuring element
            - 'plusmin'   the minimum of the pixel value and the pixelwise sum
            of the () structuring element and source neighbourhood. :TODO:

        :border options:

            - 'replicate'    the border value is replicated (default)
            - 'none'      pixels beyond the border not included in window
            - 'trim'      output is not computed for pixels where the se
            crosses the image border, hence output image has reduced dimensions

        .. note::

            - Cheaper to apply a smaller structuring element multiple times
              than one large one, the effective structuing element is the
              Minkowski sum of the structuring element with itself N times.
            - Performs greyscale morphology
            - The structuring element shoul dhave an odd side length.
            - For binary image, min = erosion, max = dilation.
            - The ``plusmin`` operation can be used to compute the distance
              transform.
            - The input can be logical, uint8, uint16, float or double.
            - The output is always double

        Example:

        .. runblock:: pycon

        :references:

            - Robotics, Vision & Control, Section 12.5, P. Corke,
              Springer 2011.
        """

        # check if valid input:
        # se = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
        se = self._getse(se)

        # TODO check if se is valid (odd number and less than im.shape),
        # can also be a scalar

        if not isinstance(oper, str):
            raise TypeError(oper, 'oper must be a string')

        if not isinstance(n, int):
            n = int(n)
        if n <= 0:
            raise ValueError(n, 'n must be greater than 0')

        if not isinstance(opt, str):
            raise TypeError(opt, 'opt must be a string')

        # convert options TODO trim?
        cvopt = {
            'replicate': cv.BORDER_REPLICATE,
            'none': cv.BORDER_ISOLATED
        }

        if opt not in cvopt.keys():
            raise ValueError(opt, 'opt is not a valid option')
        # note: since we are calling erode/dilate, we stick with opt. we use
        # cvopt[opt] only when calling the cv.erode/cv.dilate functions

        out = []
        for im in self:
            if oper == 'min':
                imo = self.erode(se, n=n, opt=opt, **kwargs)
            elif oper == 'max':
                imo = self.dilate(se, n=n, opt=opt, **kwargs)
            elif oper == 'diff':
                se = self.getse(se)
                imo = cv.morphologyEx(im.image,
                                      cv.MORPH_GRADIENT,
                                      se,
                                      iterations=n,
                                      borderType=cvopt[opt],
                                      **kwargs)
            elif oper == 'plusmin':
                # out = None  # TODO
                raise ValueError(oper, 'plusmin not supported yet')
            else:
                raise ValueError(oper, 'morph does not support oper')
            out.append(imo)

        return self.__class__(out)

    def hitormiss(self, s1, s2=None):
        """
        Hit or miss transform

        :param s1: structuring element 1
        :type s1: numpy array (S,T), where S < N and T < H
        :param s2: structuring element 2
        :type s2: numpy array (S,T), where S < N and T < H
        :return out: Image
        :rtype: Image instance

        - ``IM.hitormiss(s1, s2)`` is the image with the hit-or-miss transform
          of the binary image with the structuring element ``s1``. Unlike
          standard morphological operations, ``s1`` has three possible values:
          0, 1 and don't care (represented by nans).

        Example:

        .. runblock:: pycon

        :references:

            - Robotics, Vision & Control, Section 12.5, P. Corke,
              Springer 2011.
        """
        # check valid input
        # TODO also check if binary image?

        if s2 is None:
            s2 = np.float32(s1 == 0)
            s1 = np.float32(s1 == 1)

        out = []
        for im in self:
            imv = self.__class__(1 - im.image)
            imhm = im.morph(s1, 'min').image * imv.morph(s2, 'min').image
            out.append(imhm)
        return self.__class__(out)

    def endpoint(self):
        """
        Find end points on a binary skeleton image

        :return out: Image with endpoints
        :rtype: Image instance (N,H,3) or (N,H)

        - ``IM.endpoint()`` is the binary image where pixels are set if the
          corresponding pixel in the binary image ``im`` is the end point of a
          single-pixel wide line such as found in an image skeleton.  Computed
          using the hit-or-miss morphological operator.

        :references:

            - Robotics, Vision & Control, Section 12.5.3, P. Corke,
              Springer 2011.
        """

        se = np.zeros((3, 3, 8))
        se[:, :, 0] = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
        se[:, :, 1] = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]])
        se[:, :, 2] = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
        se[:, :, 3] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        se[:, :, 4] = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
        se[:, :, 5] = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        se[:, :, 6] = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
        se[:, :, 7] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])

        out = []
        for im in self:
            o = np.zeros(im.shape)
            for i in range(se.shape[2]):
                o = np.logical_or(o, im.hitormiss(se[:, :, i]).image)
            out.append(o)

        return self.__class__(out)

    def triplepoint(self):
        """
        Find triple points

        :return out: Image with triplepoints
        :rtype: Image instance (N,H,3) or (N,H)

        - ``IM.triplepoint()`` is the binary image where pixels are set if the
          corresponding pixel in the binary image  is a triple point, that is
          where three single-pixel wide line intersect. These are the Voronoi
          points in an image skeleton.  Computed using the hit-or-miss
          morphological operator.

        :references:

            - Robotics, Vision & Control, Section 12.5.3, P. Corke,
              Springer 2011.
        """

        se = np.zeros((3, 3, 16))
        se[:, :, 0] = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]])
        se[:, :, 1] = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
        se[:, :, 2] = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 0]])
        se[:, :, 3] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 1]])
        se[:, :, 4] = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]])
        se[:, :, 5] = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]])
        se[:, :, 6] = np.array([[0, 1, 0], [1, 1, 0], [0, 1, 0]])
        se[:, :, 7] = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]])
        se[:, :, 8] = np.array([[0, 1, 0], [0, 1, 1], [1, 0, 0]])
        se[:, :, 9] = np.array([[0, 0, 1], [1, 1, 0], [0, 0, 1]])
        se[:, :, 10] = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0]])
        se[:, :, 11] = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 1]])
        se[:, :, 12] = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0]])
        se[:, :, 13] = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 0]])
        se[:, :, 14] = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
        se[:, :, 15] = np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]])

        out = []
        for im in self:
            o = np.zeros(im.shape)
            for i in range(se.shape[2]):
                o = np.logical_or(o, im.hitormiss(se[:, :, i]).image)
            out.append(o)

        return self.__class__(out)

    def open(self, se, **kwargs):
        """
        Morphological opening

        :param se: structuring element
        :type se: numpy array (S,T), where S < N and T < H
        :param n: number of times to apply the dilation
        :type n: integer
        :param opt: option specifying the type of dilation
        :type opt: string
        :return out: Image
        :rtype: Image instance

        - ``IM.iopen(se, opt)`` is the image after morphological opening with
          structuring element ``se``. This is a morphological erosion followed
          by dilation.

        - ``IM.iopen(se, n, opt)`` as above, but the structruring element
          ``se`` is applied ``n`` times, that is ``n`` erosions followed by
          ``n`` dilations.

        :options:

            - 'border'    the border value is replicated (default)
            - 'none'      pixels beyond the border not included in the window
            - 'trim'      output is not computed for pixels where the
            structuring element crosses the image border, hence output
            image has reduced dimensions TODO

        .. note::

            - For binary image an opening operation can be used to eliminate
              small white noise regions.
            - Cheaper to apply a smaller structuring element multiple times
              than one large one, the effective structuing element is the
              Minkowski sum of the structuring element with itself N times.

        Example:

        .. runblock:: pycon

        :references:

            - Robotics, Vision & Control, Section 12.5, P. Corke,
              Springer 2011.
        """
        # probably cleanest approach:
        # out = [self.erode(se, **kwargs).dilate(se, **kwargs) for im in self]
        # return self.__class__(out)

        out = []
        for im in self:
            o = im.erode(se, **kwargs).dilate(se, **kwargs)
            out.append(o)
        return self.__class__(out)

    def close(self, se, **kwargs):
        """
        Morphological closing

        :param se: structuring element
        :type se: numpy array (S,T), where S < N and T < H
        :param n: number of times to apply the operation
        :type n: integer
        :param opt: option specifying the type of border behaviour
        :type opt: string
        :return out: Image
        :rtype: Image instance (N,H,3) or (N,H)

        - ``IM.iclose(se, opt)`` is the image after morphological closing with
          structuring element ``se``. This is a morphological dilation followed
          by erosion.

        - ``IM.iclose(se, n, opt)`` as above, but the structuring element
          ``se`` is applied ``n`` times, that is ``n`` dilations followed by
          ``n`` erosions.

        :options:

            - 'border'    the border value is replicated (default)
            - 'none'      pixels beyond the border not included in the window
            - 'trim'      output is not computed for pixels where the
            structuring element crosses the image border, hence output
            image has reduced dimensions TODO

        .. note::

            - For binary image an opening operation can be used to eliminate
              small white noise regions.
            - Cheaper to apply a smaller structuring element multiple times
              than one large one, the effective structuing element is the
              Minkowski sum of the structuring element with itself N times.

        Example:

        .. runblock:: pycon

        :references:

            - Robotics, Vision & Control, Section 12.5, P. Corke,
              Springer 2011.
        """
        out = []
        for im in self:
            o = im.dilate(se, **kwargs).erode(se, **kwargs)
            out.append(o)
        return self.__class__(out)

    def thin(self, delay=0.0):
        """
        Morphological skeletonization

        :param delay: seconds between each iteration of display
        :type delay: float
        :return out: Image
        :rtype: Image instance (N,H,3) or (N,H)

        - ``IM.thin()`` is the image as a binary skeleton of the binary image
          IM. Any non-zero region is replaced by a network of single-pixel wide
          lines.

        - ``IM.thin(delay)`` as above but graphically displays each iteration
          of the skeletonization algorithm with a pause of ``delay`` seconds
          between each iteration. TODO

        Example:

        .. runblock:: pycon

        :references:

            - Robotics, Vision & Control, Section 12.5, P. Corke,
              Springer 2011.
        """

        # create a binary image (True/False)
        # im = im > 0

        # create structuring elements
        sa = np.array([[0, 0, 0],
                       [np.nan, 1, np.nan],
                       [1, 1, 1]])
        sb = np.array([[np.nan, 0, 0],
                       [1, 1, 0],
                       [np.nan, 1, np.nan]])

        out = []
        for im in self:
            o = im
            while True:
                for i in range(4):
                    r = im.hitormiss(sa).image
                    # might also use the bitwise operator ^
                    im = self.__class__(np.logical_xor(im.image, r))
                    r = im.hitormiss(sb).image
                    im = self.__class__(np.logical_xor(im.image, r))
                    sa = np.rot90(sa)
                    sb = np.rot90(sb)
                if delay > 0.0:
                    im.disp()
                    # TODO add in delay timer for idisp
                    time.sleep(5)
                if np.all(o.image == im.image):
                    break
                o = im
            out.append(o)

        return self.__class__(out)

    def rank(self, se, rank=-1, opt='replicate'):
        """
        Rank filter

        :param se: structuring element
        :type se: numpy array
        :param rank: rank of filter
        :type rank: integer
        :param opt: border option
        :type opt: string
        :return out: Image  after rank filter applied to every pixel
        :rtype out: Image instance

        - ``IM.rank(se, rank)`` is a rank filtered version of image.  Only
          pixels corresponding to non-zero elements of the structuring element
          ``se`` are ranked and the ``rank``'ed value in rank becomes the
          corresponding output pixel value.  The highest rank, the maximum, is
          ``rank=-1``.

        - ``IM.rank(se, rank, opt)`` as above but the processing of edge pixels
          can be controlled.

        :options:

            - 'replicate'     the border value is replicated (default)
            - 'none'          pixels beyond the border are not included in
              the window
            - 'trim'          output is not computed for pixels where the
              structuring element crosses the image border, hence output image
              has reduced dimensions TODO

        Example:

        .. runblock:: pycon

        .. note::

            - The structuring element should have an odd side length.
            - The input can be logical, uint8, uint16, float or double, the
              output is always double
        """
        if not isinstance(rank, int):
            raise TypeError(rank, 'rank is not an int')

        # border options for rank_filter that are compatible with rank.m
        borderopt = {
            'replicate': 'nearest',
            'wrap': 'wrap'
        }

        if opt not in borderopt:
            raise ValueError(opt, 'opt is not a valid option')

        out = []
        for im in self:
            out.append(sp.ndimage.rank_filter(im.image,
                                              rank,
                                              footprint=se,
                                              mode=borderopt[opt]))
        return self.__class__(out)

    def label(self, conn=8, outtype='int32'):
        """
        Label an image

        :param conn: connectivity, 4 or 8
        :type conn: integer
        :param ltype: output image type
        :type ltype: string
        :return out_c: n_components
        :rtype out_c: int
        :return labels: labelled image
        :rtype labels: Image instance

        - ``IM.label()`` is a label image that indicates connected components
          within the image. Each pixel is an integer label that indicates which
          connected region the corresponding pixel in image belongs to.  Region
          labels are in the range 1 to ``n_components``.

        - ``IM.label(conn)`` as above, with the connectivity specified. 4 or 8.

        - ``IM.label(outtype)`` as above, with the output type specified as
          either int32 or uint16.

        Example:

        .. runblock:: pycon

        .. note::

            - Converts a color image to greyscale.
            - This algorithm is variously known as region labelling,
              connectivity analysis, connected component analysis,
              blob labelling.
            - All pixels within a region have the same value (or class).
            - The image can be binary or greyscale.
            - Connectivity is only performed in 2 dimensions.
            - Connectivity is performed using 8 nearest neighbours by default.
            - 8-way connectivity introduces ambiguities, a chequerboard is
              two blobs.
        """
        # NOTE cv.connectedComponents sees 0 background as one component
        # differs from ilabel.m, which sees the separated background as
        # different components

        # NOTE additionally, opencv's connected components does not give a
        # hierarchy! Only opencv's findcontours does.

        # NOTE possible solution: edge detection (eg Canny/findCOntours) on the
        # binary imaage then invert (bitwise negation) the edge image (or do
        # find contours and invert the countour image) limited to connectivity
        # of 4, since Canny is 8-connected though! Could dilate edge image to
        # accommodate 8-connectivity, but feels like a hack

        # TODO or, could just follow ilabel.m

        # NOTE consider scipy.ndimage.label
        # from scipy.ndimage import label, generate_binary_structure
        # s = generate_binary_structure(2,2) # 8-way connectivity
        # labels, n_components = label(im, structure=s), however, this has the
        # same behaviour as cv.connectedComponents

        # check valid input:
        # image must be uint8 - input image should actually be binary
        img = self.mono()

        # TODO input image must be 8-bit single-channel image
        if img.ndim > 2:
            raise ValueError(img, 'image must be single channel')

        if not (conn in [4, 8]):
            raise ValueError(conn, 'connectivity must be 4 or 8')

        # make labels uint32s, unique and never recycled?
        # set ltype to default to cv.CV_32S
        if outtype == 'int32':
            ltype = cv.CV_32S
            dtype = np.int32
        elif outtype == 'uint16':
            ltype = cv.CV_16U
            dtype = np.uint16
        else:
            raise TypeError(ltype, 'ltype must be either int32 or uint16')

        out_l = []
        out_c = []
        for im in img:
            labels = np.zeros((im.shape[0], im.shape[1]), dtype=dtype)

            # NOTE there is connectedComponentsWithAlgorithm, which grants
            # 1 other connected component algorithm
            # https://docs.opencv.org/4.5.0/d3/dc0/group__imgproc__shape.html
            # #gaedef8c7340499ca391d459122e51bef5

            n_components, labels = cv.connectedComponents(im.image,
                                                          labels,
                                                          connectivity=conn,
                                                          ltype=ltype)
            out_l.append(labels)
            out_c.append(n_components)

        return out_c, self.__class__(out_l)

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


# --------------------------------------------------------------------------#
if __name__ == '__main__':

    # test run ImageProcessingColor.py
    print('ImageProcessingMorph.py')
