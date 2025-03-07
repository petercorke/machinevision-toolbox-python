#!/usr/bin/env python
"""
SIFT feature class
@author: Dorian Tsai
@author: Peter Corke
"""

# https://docs.opencv.org/4.4.0/d7/d60/classcv_1_1SIFT.html

import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv

from ansitable import ANSITable, Column
from spatialmath import SE3

from machinevisiontoolbox.ImagePointFeatures import BaseFeature2D
from machinevisiontoolbox.decorators import array_result


class ImageRegionFeaturesMixin:
    def MSER(self, **kwargs):
        """
        Find MSER features in image

        :param kwargs: arguments passed to ``opencv.MSER_create``
        :return: set of MSER features
        :rtype: :class:`MSERFeature`

        Find all the maximally stable extremal regions in the image and
        return an object that represents the MSERs found. The object behaves
        like a list and can be indexed, sliced and used as an iterator in
        for loops and comprehensions.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("castle.png")
            >>> mser = img.MSER()
            >>> len(mser)  # number of features
            >>> mser[:5].bbox

        :references:
            - Robotics, Vision & Control for Python, Section 12.1.1.2, P. Corke, Springer 2023.

        :seealso: :class:`MSERFeature`, `cv2.MSER_create <https://docs.opencv.org/4.5.2/d3/d28/classcv_1_1MSER.html>`_
        """

        return MSERFeature(self, **kwargs)

    def ocr(self, minconf=50, plot=False):
        """
        Optical character recognition

        :param minconf: minimum confidence value for text to be returned or
            plotted (percentage), defaults to 50
        :type minconf: int, optional
        :param plot: overlay detected text on the current plot, assumed to be the
            image, defaults to False
        :type plot: bool, optional
        :return: detected strings and metadata
        :rtype: list of :class:`OCRWord`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('penguins.png')
            >>> for word in img.ocr(minconf=90):
            >>>     print(word)

        Each recognized text string is described by an :class:`OCRWord` instance
        that contains the string, confidence and bounding box within the image.

        .. warning:: `PyTessearct <https://github.com/madmaze/pytesseract>`_ must be installed.

        :references:
            - Robotics, Vision & Control for Python, Section 12.4.1, P. Corke, Springer 2023.

        :seealso: :class:`OCRWord`
        """
        try:
            import pytesseract
        except:
            print("you need to install pytesseract:")
            return

        ocr = pytesseract.image_to_data(self.A, output_type=pytesseract.Output.DICT)

        # create list of dicts, rather than dict of lists
        n = len(ocr["conf"])
        words = []
        for i in range(n):
            conf = ocr["conf"][i]
            if conf == "-1":  # I suspect this was not meant to be a string
                continue
            if conf < minconf:
                continue

            word = OCRWord(ocr, i)
            if plot:
                word.plot()
            words.append(word)
        return words


# --------------------- supporting classes -------------------------------- #


class MSERFeature:
    def __init__(self, image=None, **kwargs):
        """
        Find MSERs

        :param image: input image
        :type image: :class:`Image`
        :param kwargs: parameters passed to :func:`opencv.MSER_create`

        Find all the maximally stable extremal regions in the image and
        return an object that represents the MSERs found.
        This class behaves like a list and each MSER is an element of the list.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('shark2.png')
            >>> msers = img.MSER()
            >>> len(msers)
            >>> msers[0]
            >>> msers.bbox

        :references:
            - J. Matas, O. Chum, M. Urban, and T. Pajdla.
              "Robust wide baseline stereo from maximally stable extremal regions."
              Proc. of British Machine Vision Conference, pages 384-396, 2002.
            - Robotics, Vision & Control for Python, Section 12.1.2.2, P. Corke, Springer 2023.


        :seealso: :meth:`bbox` :meth:`points`
        """

        if image is not None:
            detector = cv.MSER_create(**kwargs)
            msers, bboxes = detector.detectRegions(image.A)

            # msers is a tuple of ndarray(M,2), each row is (u,v)
            # bbox is ndarray(N,4), each row is l, r, w, h
            # returns different things, msers is a list of points
            # u, v, point=centroid, scale=area
            # https://www.toptal.com/machine-learning/real-time-object-detection-using-mser-in-ios

            self._points = [
                mser.T for mser in msers
            ]  # transpose point arrays to be Nx2
            bboxes[:, 2:] = bboxes[:, 0:2] + bboxes[:, 2:]  # convert to lrtb
            self._bboxes = bboxes

    def __len__(self):
        """
        Number of MSER features

        :return: number of features
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("castle.png")
            >>> mser = img.MSER()
            >>> len(mser)  # number of features

        :seealso: :meth:`__getitem__`
        """
        return len(self._points)

    def __getitem__(self, i):
        """
        Get MSERs from MSER feature object

        :param i: index
        :type i: int or slice
        :raises IndexError: index out of range
        :return: subset of point features
        :rtype: :class:`MSERFeature` instance

        This method allows a ``MSERFeature`` object to be indexed, sliced or iterated.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("castle.png")
            >>> mser = img.MSER()
            >>> len(mser)  # number of features
            >>> mser[:5]   # first 5 MSER features
            >>> mser[::50]  # every 50th MSER feature

        :seealso: :meth:`len`
        """
        new = self.__class__()

        if isinstance(i, int):
            new._points = self._points[i]
            new._bboxes = self._bboxes[np.newaxis, i, :]  # result is 2D
        elif isinstance(i, slice):
            new._points = self._points[i]
            new._bboxes = self._bboxes[i, :]  # result is 2D
        elif isinstance(i, np.ndarray):
            if np.issubdtype(i.dtype, bool):
                new._points = [self._points[k] for k, true in enumerate(i) if true]
                new._bboxes = self._bboxes[i, :]
            elif np.issubdtype(i.dtype, np.integer):
                new._points = [self._points[k] for k in i]
                new._bboxes = self._bboxes[i, :]
        elif isinstance(i, (list, tuple)):
            new._points = [self._points[k] for k in i]
            new._bboxes = self._bboxes[i, :]

        return new

    def __str__(self):
        """
        String representation of MSER

        :return: Brief readable description of MSER
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("castle.png")
            >>> msers = img.MSER()
            >>> str(msers)
            >>> str(msers[0])
        """
        if len(self) > 1:
            return f"MSER features, {len(self)} regions"
        else:
            s = f"MSER feature: u: {self._bboxes[0,0]} - {self._bboxes[0,2]}, v: {self._bboxes[0,1]} - {self._bboxes[0,3]}"
            return s

    def __repr__(self):
        """
        Representation of MSER

        :return: Brief readable description of MSER
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("castle.png")
            >>> msers = img.MSER()
            >>> msers
            >>> msers[0]
        """
        return str(self)

    @property
    @array_result
    def points(self):
        """
        Points belonging to MSERs

        :return: Coordinates of points in (u,v) format that belong to MSER
        :rtype: ndarray(2,N), list of ndarray(2,N)

        If the object contains just one region the result is an array, otherwise
        it is a list of arrays (with different numbers of rows).

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> img = Image.Read("castle.png")
            >>> msers = img.MSER()
            >>> np.printoptions(threshold=10)
            >>> msers[0].points
            >>> msers[2:4].points

        :seealso: :meth:`bbox`
        """
        return self._points

    @property
    @array_result
    def bbox(self):
        """
        Bounding boxes of MSERs

        :return: Bounding box of MSER in [umin, vmin, umax, vmax] format
        :rtype: ndarray(4) or ndarray(N,4)

        If the object contains just one region the result is a 1D array,
        otherwise it is a 2D arrays with one row per bounding box.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("castle.png")
            >>> msers = img.MSER()
            >>> msers[0].bbox
            >>> msers[:4].bbox

        :seealso: :meth:`points`

        """
        return self._bboxes


class OCRWord:
    def __init__(self, ocr, i):
        """
        OCR word and metadata

        :param ocr: dict from Tesseract
        :type ocr: dict of lists
        :param i: index of word
        :type i: int
        :return: OCR data for word
        :rtype: :class:`OCRWord` instance

        Describes a word detected by OCR including its metadata which is available
        as a number of properties:

        ==========  =======================================================
        Property    Meaning
        ==========  =======================================================
        ``text``    recognized text
        ``conf``    confidence in text recognition (percentage)
        ``l``       left coordinate (umin) of rectangle containing the text
        ``t``       top coordinate (vmin) of rectangle containing the text
        ``w``       height of rectangle containing the text
        ``h``       height of rectangle containing the text
        ``ltrb``    bounding box [left, top, right, bottom]
        ==========  =======================================================

        :seealso: :meth:`~machinevisiontoolbox.ImageFeatures.ImageFeaturesMixin.ocr`
        """
        self.dict = {}
        for key in ocr.keys():
            self.dict[key] = ocr[key][i]

    def __str__(self):
        """
        String representation of MSER

        :return: Brief readable description of OCR word
        :rtype: str
        """
        return f"{self.dict['text']} ({self.dict['conf']}%)"

    def __repr__(self):
        return str(self)

    @property
    def l(self):
        """
        Left side of word bounding box

        :return: left side coordinate of bounding box in pixels
        :rtype: int

        :seealso: :meth:`t` :meth:`ltrb`
        """
        return self.dict["left"]

    @property
    def t(self):
        """
        Top side of word bounding box

        :return: top side coordinate of bounding box in pixels
        :rtype: int

        :seealso: :meth:`l` :meth:`ltrb`
        """
        return self.dict["top"]

    @property
    def w(self):
        """
        Width of word bounding box

        :return: width of bounding box in pixels
        :rtype: int

        :seealso: :meth:`h` :meth:`ltrb`
        """
        return self.dict["width"]

    @property
    def h(self):
        """
        Height of word bounding box

        :return: height of bounding box in pixels
        :rtype: int

        :seealso: :meth:`w` :meth:`ltrb`
        """
        return self.dict["height"]

    @property
    def ltrb(self):
        """
        Word bounding box

        :return: bounding box [left top right bottom] in pixels
        :rtype: list

        :seealso: :meth:`l` :meth:`t` :meth:`w` :meth:`h`
        """
        return [
            self.dict["left"],
            self.dict["top"],
            self.dict["left"] + self.dict["width"],
            self.dict["top"] + self.dict["height"],
        ]

    @property
    def conf(self):
        """
        Word confidence

        :return: confidence of word (percentage)
        :rtype: int

        :seealso: :meth:`text`
        """
        return self.dict["conf"]

    @property
    def text(self):
        """
        Word as a string

        :return: word
        :rtype: str

        :seealso: :meth:`conf`
        """
        return self.dict["text"]

    def plot(self):
        """
        Plot word and bounding box

        Plot a label box around the word in the image, and show the OCR string
        in the label field.

        :seealso: :func:`~machinevisiontoolbox.base.graphics.plot_labelbox`
        """
        plot_labelbox(
            self.text,
            tl=(self.l, self.t),
            wh=(self.w, self.h),
            color="y",
            linestyle="--",
        )


if __name__ == "__main__":
    from machinevisiontoolbox import Image

    im = Image.Read("castle.png")
    mser = im.MSER()
    print(len(mser))
    print(mser)
    m0 = mser[0]
    print(m0)
    print(m0.bbox.shape)
    print(m0.bbox)

    print(m0.points.shape)
    print(m0.points)

    mm = mser[:5]
    print(mm)
    print(mm.bbox.shape)
    print(mm.bbox)
    print(len(mm))
    print(mm.points)

    k = np.arange(len(mser)) < 5
    mm = mser[k]
    print(mm)
    print(mm.bbox.shape)
    print(mm.bbox)
    print(len(mm))
    print(mm.points)

    k = [0, 2, 1, 3, 4]
    mm = mser[k]
    print(mm)
    print(mm.bbox.shape)
    print(mm.bbox)
    print(len(mm))
    print(mm.points)

    k = np.array([0, 2, 1, 3, 4])
    mm = mser[k]
    print(mm)
    print(mm.bbox.shape)
    print(mm.bbox)
    print(len(mm))
    print(mm.points)
    pass
