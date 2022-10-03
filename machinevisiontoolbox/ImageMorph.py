#!/usr/bin/env python

import numpy as np
import cv2 as cv
import time
import scipy as sp


class ImageMorphMixin:
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
        se = np.array(se).astype(np.uint8)
        if se.min() < 0:
            raise ValueError('cannot convert array with negative values to a structuring element')

        return se

    def erode(self, se, n=1, border='replicate', bordervalue=0, **kwargs):
        """
        Morphological erosion

        :param se: structuring element
        :type se: ndarray(N,M)
        :param n: number of times to apply the erosion, defaults to 1
        :type n: int, optional
        :param border: option for boundary handling, see :meth:`~machinevisiontoolbox.ImageSpatial.convolve`, defaults to 'replicate'
        :type border: str, optional
        :param bordervalue: padding value, defaults to 0
        :type bordervalue: scalar, optional
        :param kwargs: addition options passed to :func:`opencv.erode`
        :return: eroded image
        :rtype: :class:`Image`

        Returns the image after morphological erosion with the structuring
        element ``se`` applied ``n`` times.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> img = Image.Squares(1,7)
            >>> img.print()
            >>> img.erode(np.ones((3,3))).print()

        .. note:: 
            - It is cheaper to apply a smaller structuring element multiple times
              than one large one, the effective structuing element is the
              Minkowski sum of the structuring element with itself N times.
            - The structuring element typically has odd side lengths.
            - For a greyscale image this is the maximum value over the 
              structuring element.

        :references:
            - Robotics, Vision & Control for Python, Section 11.6, P. Corke, Springer 2023.

        :seealso: :meth:`dilate` 
            `opencv.erode <https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb>`_
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

        out = cv.erode(self.to_int(), se,
                            iterations=n,
                            borderType=self._bordertype_cv(border, exclude=('wrap')),
                            borderValue=bordervalue,
                            **kwargs)

        return self.__class__(out)

    def dilate(self, se, n=1, border='replicate', bordervalue=0, **kwargs):
        """
        Morphological dilation

        :param se: structuring element
        :type se: ndarray(N,M)
        :param n: number of times to apply the dilation, defaults to 1
        :type n: int, optional
        :param border: option for boundary handling, see :meth:`~machinevisiontoolbox.ImageSpatial.convolve`, defaults to 'replicate'
        :type border: str, optional
        :param bordervalue: padding value, defaults to 0
        :type bordervalue: scalar, optional
        :param kwargs: addition options passed to :func:`opencv.dilate`
        :return: dilated image
        :rtype: :class:`Image`

        Returns the image after morphological dilation with the structuring
        element ``se`` applied ``n`` times.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> pixels = np.zeros((7,7)); pixels[3,3] = 1
            >>> img = Image(pixels)
            >>> img.print()
            >>> img.dilate(np.ones((3,3))).print()

        .. note:: 
            - It is cheaper to apply a smaller structuring element multiple times
              than one large one, the effective structuing element is the
              Minkowski sum of the structuring element with itself N times.
            - The structuring element typically has odd side lengths.
            - For a greyscale image this is the minimum value over the 
              structuring element.

        :references:
            - Robotics, Vision & Control for Python, Section 11.6, P. Corke, Springer 2023.

        :seealso: :meth:`erode` `opencv.dilate <https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c>`_
        """

        # check if valid input:
        se = self._getse(se)

        if not isinstance(n, int):
            n = int(n)
        if n <= 0:
            raise ValueError(n, 'n must be greater than 0')

        # for im in [img.image in self]: # then can use cv.dilate(im)
        out = cv.dilate(self.to_int(), se,
                    iterations=n,
                    borderType=self._bordertype_cv(border, exclude=('wrap')),
                    borderValue=bordervalue,
                    **kwargs)

        return self.__class__(out)

    def morph(self, se, op, n=1, border='replicate', bordervalue=0, **kwargs):
        """
        Morphological neighbourhood processing

        :param se: structuring element
        :type se: ndarray(N,M)
        :param op: morphological operation, one of: 'min', 'max', 'diff'
        :type op: str
        :param n: number of times to apply the operation, defaults to 1
        :type n: int, optional
        :param border: option for boundary handling, see :meth:`~machinevisiontoolbox.ImageSpatial.ve`, defaults to 'replicate'
        :type border: str, optional
        :param bordervalue: padding value, defaults to 0
        :type bordervalue: scalar, optional
        :param kwargs: addition options passed to ``opencv.morphologyEx``
        :return: morphologically transformed image
        :rtype: :class:`Image`

        Apply the morphological operation ``oper`` with structuring element ``se``
        to the image ``n`` times.

        =============  =======================================================
        ``'oper'``     description
        =============  =======================================================
        ``'min'``      minimum value over the structuring element
        ``'max'``      maximum value over the structuring element
        ``'diff'``     maximum - minimum value over the structuring element
        =============  =======================================================

        .. note::

            - It is cheaper to apply a smaller structuring element multiple times
              than one large one, the effective structuing element is the
              Minkowski sum of the structuring element with itself N times.
            - Performs greyscale morphology
            - The structuring element should have an odd side length.
            - For a binary image, min = erosion, max = dilation.

        :references:
            - Robotics, Vision & Control for Python, Section 11.6, P. Corke, Springer 2023.

        :seealso: :meth:`erode` :meth:`dilate` `opencv.morphologyEx <https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f>`_
        """

        # check if valid input:
        # se = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
        se = self._getse(se)

        # TODO check if se is valid (odd number and less than im.shape),
        # can also be a scalar

        if not isinstance(op, str):
            raise TypeError(op, 'oper must be a string')

        if not isinstance(n, int):
            n = int(n)
        if n <= 0:
            raise ValueError(n, 'n must be greater than 0')

        if self.isbool:
            image = self.to_int()
        else:
            image = self.A

        if op == 'min':
            out = cv.morphologyEx(image,
                                    cv.MORPH_ERODE,
                                    se,
                                    iterations=n,
                                    borderType=self._bordertype_cv(border),
                                    borderValue=bordervalue,
                                    **kwargs)
        elif op == 'max':
            out = cv.morphologyEx(self.A,
                                    cv.MORPH_DILATE,
                                    se,
                                    iterations=n,
                                    borderType=self._bordertype_cv(border),
                                    borderValue=bordervalue,
                                    **kwargs)
        elif op == 'diff':
            se = self.getse(se)
            out = cv.morphologyEx(self.A,
                                    cv.MORPH_GRADIENT,
                                    se,
                                    iterations=n,
                                    borderType=self._bordertype_cv(border),
                                    borderValue=bordervalue,
                                    **kwargs)
        else:
            raise ValueError('morph does not support oper')

        if self.isbool:
            out = out.astype(bool)
            
        return self.__class__(out)

    def open(self, se, n=1, border='replicate', bordervalue=0, **kwargs):
        """
        Morphological opening

        :param se: structuring element
        :type se: ndarray(N,M)
        :param n: number of times to apply the erosion then dilation, defauts to 1
        :type n: int, optional
        :param border: option for boundary handling, see :meth:`~machinevisiontoolbox.ImageSpatial.convolve`, defaults to 'replicate'
        :type border: str, optional
        :param bordervalue: padding value, defaults to 0
        :type bordervalue: scalar, optional
        :param kwargs: addition options passed to ``opencv.morphologyEx``
        :return: dilated image
        :rtype: :class:`Image`

        Returns the image after morphological opening with the structuring
        element ``se`` applied as ``n`` erosions followed by ``n`` dilations.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> img = Image.Read("eg-morph1.png")
            >>> img.print('{:1d}')
            >>> img.open(np.ones((5,5))).print('{:1d}')

        .. note::
            - For binary image an opening operation can be used to eliminate
              small white noise regions.
            - It is cheaper to apply a smaller structuring element multiple times
              than one large one, the effective structuing element is the
              Minkowski sum of the structuring element with itself N times.
            - The structuring element typically has odd side lengths.

        :references:
            - Robotics, Vision & Control for Python, Section 11.6, P. Corke, Springer 2023.

        :seealso: :meth:`close :meth:`morph` `opencv.morphologyEx <https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f>`_
        """

        # probably cleanest approach:
        # out = [self.erode(se, **kwargs).dilate(se, **kwargs) for im in self]
        # return self.__class__(out)

        out = cv.morphologyEx(self.to_int(),
                                cv.MORPH_OPEN,
                                se,
                                iterations=n,
                                borderType=self._bordertype_cv(border),
                                borderValue=bordervalue,
                                **kwargs)
        return self.__class__(out)


    def close(self, se, n=1, border='replicate', bordervalue=0, **kwargs):
        """
        Morphological closing

        :param se: structuring element
        :type se: ndarray(N,M)
        :param n: number of times to apply the dilation then erosion, defauts to 1
        :type n: int, optional
        :param border: option for boundary handling, see :meth:`~machinevisiontoolbox.ImageSpatial.convolve`, defaults to 'replicate'
        :type border: str, optional
        :param bordervalue: padding value, defaults to 0
        :type bordervalue: scalar, optional
        :param kwargs: addition options passed to ``opencv.morphologyEx``
        :return: dilated image
        :rtype: :class:`Image`

        Returns the image after morphological opening with the structuring
        element ``se`` applied as ``n`` dilations followed by ``n`` erosions.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> img = Image.Read("eg-morph2.png")
            >>> img.print('{:1d}')
            >>> img.close(np.ones((5,5))).print('{:1d}')

        .. note::
            - For binary image a closing operation can be used to eliminate
              joins between regions.
            - It is cheaper to apply a smaller structuring element multiple times
              than one large one, the effective structuing element is the
              Minkowski sum of the structuring element with itself N times.
            - The structuring element typically has odd side lengths.

        :references:
            - Robotics, Vision & Control for Python, Section 11.6, P. Corke, Springer 2023.

        :seealso: :meth:`open` :meth:`morph` `opencv.morphologyEx <https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f>`_
        """
        out = cv.morphologyEx(self.to_int(),
                                cv.MORPH_CLOSE,
                                se,
                                iterations=n,
                                borderType=self._bordertype_cv(border),
                                borderValue=bordervalue,
                                **kwargs)
        return self.__class__(out)

    def hitormiss(self, s1, s2=None, border='replicate', bordervalue=0, **kwargs):
        r"""
        Hit or miss transform

        :param s1: structuring element 1
        :type s1: ndarray(N,M)
        :param s2: structuring element 2
        :type s2: ndarray(N,M)
        :param kwargs: arguments passed to ``opencv.morphologyEx``
        :return: transformed image
        :rtype: :class:`Image`

        Return the hit-or-miss transform of the binary image which is defined by
        two structuring elements structuring elements
        
        .. math:: Y = (X \ominus S_1) \cap (X \ominus S_2)

        which is the logical-and of the binary image and its complement, eroded
        by two different structuring elements. This preserves pixels where ones
        in the window are consistent with :math:`S_1` and zeros in the window
        are consistent with :math:`S_2`.

        If only ``s1`` is provided it has three possible values:
            * 1, must match a non-zero value
            * -1, must match a zero value
            * 0, don't care, matches any value.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> pixels = np.array([[0,0,1,0,1,1],[1,1,1,1,0,1],[0,1,0,1,1,0],[1,1,1,1,0,0],[0,1,1,0,1,0]])
            >>> img = Image(pixels)
            >>> img.print()
            >>> se = np.array([[0,1,0],[1,-1,1],[0,1,0]])
            >>> se
            >>> img.hitormiss(se).print()

        .. note:: For the single argument case ``s1`` :math:`=S_1 - S_2`.

        :references:
            - Robotics, Vision & Control for Python, Section 11.6.3, P. Corke,
              Springer 2023.

        :seealso: :meth:`thin` :meth:`endpoint` :meth:`triplepoint`
            `opencv.morphologyEx
            <https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f>`_
        """
        # check valid input
        # TODO also check if binary image?

        if s2 is not None:
            s1 = s1 - s2

        out = cv.morphologyEx(self.A, cv.MORPH_HITMISS, s1)
        return self.__class__(out)


    def thin(self, **kwargs):
        """
        Morphological skeletonization

        :param kwargs: options passed to :meth:`hitormiss`
        :return: Image
        :rtype: :class:`Image` instance

        Return the thinned version (skeleton) of the binary image as another
        binary image. Any non-zero region is replaced by a network of
        single-pixel wide lines.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.String('000000|011110|011110|000000')
            >>> img.print()
            >>> img.thin().print()
            >>> img = Image.Read("shark2.png")
            >>> skeleton = img.thin()
    
        :references:
            - Robotics, Vision & Control for Python, Section 11.6.3, P. Corke, Springer 2023.
        
        :seealso: :meth:`thin_animate` :meth:`hitormiss` :meth:`endpoint` :meth:`triplepoint`
        """

        # create a binary image (True/False)
        # im = im > 0

        # create structuring elements
        sa = np.array([[-1, -1, -1],
                       [0, 1, 0],
                       [1, 1, 1]])
        sb = np.array([[0, -1, -1],
                       [1, 1, -1],
                       [0, 1, 0]])

        im = self.to('uint8')
        o = im
        while True:
            for i in range(4):
                r = im.hitormiss(sa)
                # might also use the bitwise operator ^
                im -= r
                r = im.hitormiss(sb)
                im -= r
                sa = np.rot90(sa)
                sb = np.rot90(sb)
            if np.all(o.A == im.A):
                break
            o = im

        return self.__class__(o)

    def thin_animate(self, delay=0.5, **kwargs):
        """
        Morphological skeletonization with animation

        :param delay: time in seconds between each iteration of display, default to 0.5
        :type delay: float, optional
        :param kwargs: options passed to :meth:`hitormiss`
        :return: Image
        :rtype: :class:`Image` instance

        Return the thinned version (skeleton) of the binary image as another
        binary image. Any non-zero region is replaced by a network of
        single-pixel wide lines.

        The algorithm is iterative, and the result of of each iteration is 
        displayed using Matplotlib.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("shark2.png")
            >>> img.thin_animate()

        :references:
        - Robotics, Vision & Control for Python, Section 11.6.3, P. Corke, Springer 2023.

        :seealso: :meth:`thin` :meth:`hitormiss` :meth:`endpoint` :meth:`triplepoint`
        """

        # create a binary image (True/False)
        # im = im > 0

        # create structuring elements
        sa = np.array([[-1, -1, -1],
                        [0, 1, 0],
                        [1, 1, 1]])
        sb = np.array([[0, -1, -1],
                        [1, 1, -1],
                        [0, 1, 0]])

        im = self.to('uint8')
        o = im
        h = im.disp()
        while True:
            for i in range(4):
                r = im.hitormiss(sa)
                # might also use the bitwise operator ^
                im -= r
                r = im.hitormiss(sb)
                im -= r
                sa = np.rot90(sa)
                sb = np.rot90(sb)
            if delay > 0:
                h.set_data(im.A)
                time.sleep(delay)
            if np.all(o.A == im.A):
                break
            o = im

        return self.__class__(o)

    def endpoint(self, **kwargs):
        """
        Find end points on a binary skeleton image

        :param kwargs: options passed to :meth:`hitormiss`
        :return: Image
        :rtype: :class:`Image` instance

        Return a binary image where pixels are True if the corresponding pixel
        in the binary image is the end point of a single-pixel wide line such as
        found in an image skeleton.
        
        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.String('000000|011110|000000')
            >>> img.print()
            >>> img.endpoint().print()

        .. note:: Computed using the hit-or-miss morphological operator.

        :references:
            - Robotics, Vision & Control for Python, Section 11.6.3, P. Corke, Springer 2023.

        :seealso: :meth:`hitormiss` :meth:`thin` :meth:`triplepoint`
        """

        se = np.zeros((3, 3, 8))
        se[:, :, 0] = np.array([[-1,  1, -1], [-1, 1, -1], [-1, -1, -1]])
        se[:, :, 1] = np.array([[-1, -1, 1], [-1, 1, -1], [-1, -1, -1]])
        se[:, :, 2] = np.array([[-1, -1, -1], [-1, 1, 1], [-1, -1, -1]])
        se[:, :, 3] = np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
        se[:, :, 4] = np.array([[-1, -1, -1], [-1, 1, -1], [-1, 1, -1]])
        se[:, :, 5] = np.array([[-1, -1, -1], [-1, 1, -1], [1, -1, -1]])
        se[:, :, 6] = np.array([[-1, -1, -1], [1, 1, -1], [-1, -1, -1]])
        se[:, :, 7] = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, -1]])

        out = np.zeros(self.shape)
        for i in range(se.shape[2]):
            out = np.logical_or(out, self.hitormiss(se[:, :, i]).A)

        return self.__class__(out)

    def triplepoint(self, **kwargs):
        """
        Find triple points

        :param kwargs: options passed to :meth:`hitormiss`
        :return: Image
        :rtype: :class:`Image` instance

        Return a binary image where pixels are True if the corresponding pixel
        in the binary image is a triple point, that is where three single-pixel
        wide line intersect. These are the Voronoi points in an image skeleton.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.String('000000|011110|001000|001000|000000')
            >>> img.print()
            >>> img.triplepoint().print()

        .. note:: Computed using the hit-or-miss morphological operator.

        :references:
            - Robotics, Vision & Control for Python, Section 11.6.3, P. Corke, Springer 2023.

        :seealso: :meth:`hitormiss` :meth:`thin` :meth:`endpoint`
        """

        se = np.zeros((3, 3, 16), dtype='int8')
        se[:, :, 0] = np.array([[-1, 1, -1], [1, 1, 1], [-1, -1, -1]])
        se[:, :, 1] = np.array([[1, -1, 1], [-1, 1, -1], [-1, -1, 1]])
        se[:, :, 2] = np.array([[-1, 1, -1], [-1, 1, 1], [-1, 1, -1]])
        se[:, :, 3] = np.array([[-1, -1, 1], [-1, 1, -1], [1, -1, 1]])
        se[:, :, 4] = np.array([[-1, -1, -1], [1, 1, 1], [-1, 1, -1]])
        se[:, :, 5] = np.array([[1, -1, -1], [-1, 1, -1], [1, -1, 1]])
        se[:, :, 6] = np.array([[-1, 1, -1], [1, 1, -1], [-1, 1, -1]])
        se[:, :, 7] = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, -1]])
        se[:, :, 8] = np.array([[-1, 1, -1], [-1, 1, 1], [1, -1, -1]])
        se[:, :, 9] = np.array([[-1, -1, 1], [1, 1, -1], [-1, -1, 1]])
        se[:, :, 10] = np.array([[1, -1, -1], [-1, 1, 1], [-1, 1, -1]])
        se[:, :, 11] = np.array([[-1, 1, -1], [-1, 1, -1], [1, -1, 1]])
        se[:, :, 12] = np.array([[-1, -1, 1], [1, 1, -1], [-1, 1, -1]])
        se[:, :, 13] = np.array([[1, -1, -1], [-1, 1, 1], [1, -1, -1]])
        se[:, :, 14] = np.array([[-1, 1, -1], [1, 1, -1], [-1, -1, 1]])
        se[:, :, 15] = np.array([[1, -1, 1], [-1, 1, -1], [-1, 1, -1]])

        out = np.zeros(self.shape, self.dtype)
        for i in range(se.shape[2]):
            out = np.bitwise_or(out, self.hitormiss(se[:, :, i]).A)

        return self.__class__(out)



# --------------------------------------------------------------------------#
if __name__ == '__main__':

    img = Image.Read("shark2.png")
    img.thin_animate()
    # test run ImageProcessingColor.py
    print('ImageProcessingMorph.py')
