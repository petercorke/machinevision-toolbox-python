#!/usr/bin/env python
"""
2D Blob feature class
@author: Dorian Tsai
@author: Peter Corke
"""
import copy
import numpy as np
from collections import namedtuple
import cv2 as cv
from numpy.lib.arraysetops import isin
from spatialmath import base
from ansitable import ANSITable, Column
from machinevisiontoolbox.base import color_bgr
from spatialmath.base.graphics import plot_box, plot_point
import scipy as sp
import tempfile
import subprocess
import webbrowser
import sys

# NOTE, might be better to use a matplotlib color cycler
import random as rng
rng.seed(13543)  # would this be called every time at Blobs init?
import matplotlib.pyplot as plt

class Blobs:
    """
    A 2D feature blob class
    """
    # list of attributes
    _area = []
    _uc = []  # centroid (uc, vc)
    _vc = []

    _umin = []  # bounding box
    _umax = []
    _vmin = []
    _vmax = []

    _class = []  # TODO check what the class of pixel is?
    _label = []  # TODO label assigned to this region (based on ilabel.m)
    _parent = []  # -1 if no parent, else index points to i'th parent contour
    _children = []  # list of children, -1 if no children
    # _edgepoint = []  # TODO (x,y) of a point on the perimeter
    # _edge = []  # list of edge points # replaced with _contours
    _perimeter = []  # length of edge
    _touch = []  # 0 if bbox doesn't touch the edge, 1 if it does

    _a = []  # major axis length # equivalent ellipse parameters
    _b = []  # minor axis length
    _orientation = []  # angle of major axis wrt the horizontal
    _aspect = []  # b/a < 1.0
    _circularity = []


    _moment_tuple = namedtuple('moments', 
                    ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 
                    'm30', 'm21', 'm12', 'm03', 'mu20', 'mu11', 'mu02', 
                    'mu30', 'mu21', 'mu12', 'mu03', 'nu20', 'nu11', 'nu02', 
                    'nu30', 'nu21', 'nu12', 'nu03'])
    _moments = []  # list of named tuple of m00, m01, m10, m02, m20, m11

    # note that RegionFeature.m has edge, edgepoint - these are the contours
    _contours = []
    
    _image = []  # keep image saved for each Blobs object
    # probably not necessary in the long run, but for now is useful
    # to retain for debugging purposes. Not practical if blob
    # accepts a large/long sequence of images
    _hierarchy = []

    def __init__(self, image=None):
        """
        Find blobs and compute their attributes

        :param image: image to use, defaults to None
        :type image: Image instance, optional

        Uses OpenCV functions ``findContours`` to find a hierarchy of regions
        represented by their contours, and ``boundingRect``, ``moments`` to
        compute moments, perimeters, centroids etc.

        .. note:: The image is internally converted to greyscale.

        :seealso:  `cv2.moments <https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga556a180f43cab22649c23ada36a8a139>`_,
            `cv2.boundingRect <https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga103fcbda2f540f3ef1c042d6a9b35ac7>`_,
            `cv2.findContours <https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0>`_
        """

        if image is None:
            # initialise empty Blobs
            # Blobs()
            self._area = None
            self._uc = None  # Two element array, empty? Nones? []?
            self._vc = None
            self._perimeter = None

            self._umin = None
            self._umax = None
            self._vmin = None
            self._vmax = None
            self._touch = None

            self._a = None
            self._b = None
            self._orientation = None
            self._aspect = None
            self._circularity = None
            self._moments = None

            self._contours = None
            self._hierarchy = None
            self._parent = None
            self._children = None

            self._image = None
            return

        # check if image is valid - it should be a binary image, or a
        # thresholded image ()
        # convert to grayscale/mono
        # ImgProc = mvt.ImageProcessing()
        # image = Image(image)
        # image = ImgProc.mono(image)

        self._image = image

        image = image.mono().to_int()
        # note: OpenCV doesn't have a binary image type, so it defaults to
        # uint8 0 vs 255
        # image = ImgProc.iint(image)

        # we found cv.simpleblobdetector too simple.
        # Cannot get pixel values/locations of blobs themselves
        # therefore, use cv.findContours approach
        contours, hierarchy = cv.findContours(image,
                                              mode=cv.RETR_TREE,
                                              method=cv.CHAIN_APPROX_NONE)

        # TODO contourpoint, or edgepoint: take first pixel of contours

        # change hierarchy from a (1,M,4) to (M,4)
        self._hierarchy = hierarchy[0,:,:]  # drop the first singleton dimension
        self._hierarchy_raw = hierarchy
        self._parent = self._hierarchy[:, 3]

        # change contours to list of 2xN arraay
        self._contours = [c[:,0,:].T for c in contours]
        self._contours_raw = contours
        self._children = self._getchildren()

        self._contourpoint = [c[:,0].flatten() for c in self._contours]

        # get moments as a dictionary for each contour
        mu = [cv.moments(contours[i]) for i in range(len(contours))]
        for m in mu:
            if m['m00'] == 0:
                m['m00'] = 1

        # recompute moments wrt hierarchy
        mf = self._hierarchicalmoments(mu)
        self._moments = mf

        # get mass centers/centroids:
        mc = np.array(self._computecentroids())
        self._uc = mc[:, 0]
        self._vc = mc[:, 1]

        # get areas:
        self._area = np.array(self._computearea())
        # TODO sort contours wrt area descreasing?

        # get perimeter:
        self._perimeter = np.array(self._computeperimeter())

        # get circularity
        self._circularity = np.array(self._computecircularity())

        # get bounding box:
        bbox = np.array(self._computeboundingbox())

        # bbox in [u0, v0, length, width]
        self._umax = bbox[:, 0] + bbox[:, 2]
        self._umin = bbox[:, 0]
        self._vmax = bbox[:, 1] + bbox[:, 3]
        self._vmin = bbox[:, 1]

        self._touch = np.r_[self._touchingborder(image.shape)]

        # equivalent ellipse from image moments
        a, b, orientation = self._computeequivalentellipse()
        self._a = np.array(a)
        self._b = np.array(b)
        self._orientation = np.array(orientation)
        self._aspect = self._b / self._a

    def __len__(self):
        """
        Number of blobs in blob object

        :return: Number of blobs in blob object
        :rtype: int

        A blob object contains information about multiple blobs.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('multiblobs.png')
            >>> blobs = im.blobs()
            >>> len(blobs)

        :seealso: :meth:`.__getitem__`
        """
        return len(self._contours)

    def __getitem__(self, i):
        """
        Get item from blob object

        :param i: index
        :type i: int or slice
        :raises IndexError: index out of range
        :return: subset of blobs
        :rtype: Blob instance

        This method allows a ``Blob`` object to be indexed, sliced or iterated.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('multiblobs.png')
            >>> blobs = im.blobs()
            >>> print(blobs)
            >>> print(blobs[0])
            >>> print(blobs[5:8])
            >>> [b.area for b in blobs]

        :seealso: :meth:`.__len__`
        """
        if isinstance(i, np.ndarray):
            if np.issubdtype(z.dtype, np.bool_):
                i = np.nonzero(i)[0]
        if isinstance(self._uc, np.ndarray):
            new = Blobs()

            new._area = self._area[i]
            new._uc = self._uc[i]
            new._vc = self._vc[i]
            new._perimeter = self._perimeter[i]
            new._contours = self._contours[i]

            if isinstance(i, int):
                new._contourpoint = [self._contourpoint[i]]
            else:
                new._contourpoint = self._contourpoint[i]

            new._umin = self._umin[i]
            new._umax = self._umax[i]
            new._vmin = self._vmin[i]
            new._vmax = self._vmax[i]

            new._a = self._a[i]
            new._b = self._b[i]
            new._aspect = self._aspect[i]
            new._orientation = self._orientation[i]
            new._circularity = self._circularity[i]
            new._touch = self._touch[i]
            new._parent = self._parent[i]

            new._moments = self._moments[i]

            new._children = self._children[i]
            new._image = self._image

            return new
        else:
            if i > 0:
                raise IndexError

            return self

    def __repr__(self):
        # s = "" for i, blob in enumerate(self): s += f"{i}:
        # area={blob.area:.1f} @ ({blob.uc:.1f}, {blob.vc:.1f}),
        # touch={blob.touch}, orient={blob.orientation * 180 / np.pi:.1f}°,
        # aspect={blob.aspect:.2f}, circularity={blob.circularity:.2f},
        # parent={blob._parent}\n"

        # return s

        table = ANSITable(
                    Column("id"),
                    Column("parent"),
                    Column("centroid"),
                    Column("area", fmt="{:.3g}"),
                    Column("touch"),
                    Column("perim", fmt="{:.1f}"),
                    Column("circularity", fmt="{:.3f}"),
                    Column("orient", fmt="{:.1f}°"),
                    Column("aspect", fmt="{:.3g}"),
                    border="thin"
        )
        for i, b in enumerate(self):
            table.row(i, b.parent, f"{b.u:.1f}, {b.v:.1f}",
                      b.area,
                      b.touch,
                      b.perimeter,
                      b.circularity,
                      b.orientation * 180 / np.pi,
                      b.aspect)

        return str(table)

    @property
    def area(self):
        """
        Area of the blob

        :return: area in pixels
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].area
            >>> blobs.area
        """
        return self._area

    @property
    def u(self):
        """
        u-coordinate of the blob centroid

        :return: u-coordinate (horizontal)
        :rtype: float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].u
            >>> blobs.u
        """
        return self._uc

    @property
    def v(self):
        """
        v-coordinate of the blob centroid

        :return: v-coordinate (vertical)
        :rtype: float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].v
            >>> blobs.v
        """
        return self._vc

    @property
    def centroid(self):
        """
        Centroid of blob

        :return: centroid of the blob
        :rtype: 2-tuple

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].bboxarea
            >>> blobs.bboxarea

        :seealso:  :meth:`uc`, :meth:`vc`
        """
        return (self._uc, self._vc)
        # TODO maybe ind for centroid: b.centroid[0]?

    @property
    def a(self):
        """
        Radius of equivalent ellipse

        :return: largest ellipse radius
        :rtype: float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].a
            >>> blobs.a

        :seealso: :meth:`b`, :meth:`aspect`
        """
        return self._a

    @property
    def b(self):
        """
        Radius of equivalent ellipse

        :return: smallest ellipse radius
        :rtype: float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].b
            >>> blobs.b

        :seealso: :meth:`a`, :meth:`aspect`
        """
        return self._b

    @property
    def aspect(self):
        r"""
        Blob aspect ratio

        :return: ratio of equivalent ellipse axes, :math:`<= 1`
        :rtype: float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].aspect
            >>> blobs.aspect

        :seealso: func:`a`, :meth:`b`
        """
        return self._aspect

    @property
    def orientation(self):
        """
        Blob orientation

        :return: Orientation of equivalent ellipse (in radians)
        :rtype: float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].orientation
            >>> blobs.orientation
        """
        return self._orientation

    @property
    def bbox(self):
        """
        Bounding box

        :return: bounding
        :rtype: ndarray(2,2)

        The bounding box is a 2x2 matrix  [u1, u2; v1, v2].  The rows are the
        u- and v-axis extent respectively.  The columns are the bottom-left
        and top-right corners of the bounding box.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].bbox
            >>> blobs.bbox

        .. note:: The bounding box is the smallest box with vertical and
            horizontal edges that fully encloses the blob.

        :seealso: :meth:`.umin`, :meth:`.vmin`, :meth:`.umax`, :meth:`.umax`,
        """
        return np.array([
            [self._umin, self._umax], 
            [self._vmin, self._vmax],
        ])

    @property
    def umin(self):
        """
        Minimum u-axis extent

        :return: maximum u-coordinate of the blob
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].umin
            >>> blobs.umin

        :seealso: :meth:`.umax`, :seealso: :meth:`.bbox`
        """
        return self._umin

    @property
    def umax(self):
        """
        Maximum u-axis extent

        :return: maximum u-coordinate of the blob
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].umin
            >>> blobs.umin

        :seealso: :meth:`.umin`, :seealso: :meth:`.bbox`
        """
        return self._umax

    @property
    def vmin(self):
        """
        Maximum u-axis extent

        :return: maximum v-coordinate of the blob
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].vmin
            >>> blobs.vmin

        :seealso: :meth:`.vmax`, :seealso: :meth:`.bbox`
        """
        return self._vmin

    @property
    def vmax(self):
        """
        Minimum b-axis extent

        :return: maximum v-coordinate of the blob
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].vmax
            >>> blobs.vmax

        :seealso: :meth:`.vmin`, :seealso: :meth:`.bbox`
        """
        return self._vmax


    @property
    def bboxarea(self):
        """
        Area of the bounding box

        :return: area of the bounding box in pixels
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].bboxarea
            >>> blobs.bboxarea

        .. note:: The bounding box is the smallest box with vertical and
            horizontal edges that fully encloses the blob.

        :seealso: :meth:`.bbox`
        """
        return [(b._umax - b._umin) * (b._vmax - b._vmin) for b in self]

    @property
    def humoments(self):
        hu = []
        for blob in self:
            m = blob.moments
            phi = np.empty((7,))
            phi[0] = m.nu20 + m.nu02
            phi[1] = (m.nu20 - m.nu02)**2 + 4*m.nu11**2
            phi[2] = (m.nu30 - 3*m.nu12)**2 + (3*m.nu21 - m.nu03)**2
            phi[3] = (m.nu30 + m.nu12)**2 + (m.nu21 + m.nu03)**2
            phi[4] = (m.nu30 - 3*m.nu12) \
                        * (m.nu30+m.nu12) \
                        * ((m.nu30 +m.nu12)**2 - 3*(m.nu21+m.nu03)**2) \
                    + \
                    (3*m.nu21 - m.nu03) \
                        * (m.nu21+m.nu03) \
                        * (3*(m.nu30+m.nu12)**2 - (m.nu21+m.nu03)**2)
            phi[5] = (m.nu20 - m.nu02)*((m.nu30 +m.nu12)**2 \
                    - (m.nu21+m.nu03)**2) \
                    + 4*m.nu11 *(m.nu30+m.nu12)*(m.nu21+m.nu03)
            phi[6] = (3*m.nu21 - m.nu03) \
                        * (m.nu30+m.nu12) \
                        * ((m.nu30 +m.nu12)**2 - 3*(m.nu21+m.nu03)**2) \
                    + \
                        (3*m.nu12 - m.nu30) \
                        * (m.nu21+m.nu03) \
                        * ( 3*(m.nu30+m.nu12)**2 - (m.nu21+m.nu03)**2)
            hu.append(phi)
        return np.array(hu)

    @property
    def perimeter_length(self):
        """
        Perimeter length of the blob

        :return: perimeter length in pixels
        :rtype: float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].perimeter
            >>> blobs.perimeter

        :seealso: :meth:`.contour`
        """
        return self._perimeter

    def perimeter(self, epsilon=None, closed=True):
        """
        Contour of the blob

        :param epsilon: maximum distance between the original curve and its approximation, default is exact contour
        :type epsilon: int
        :param closed: 	the approximated curve is closed (its first and last vertices are connected)
        :type closed: bool
        :return: Contour, one point per column
        :rtype: ndarray(2,N)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> c = blobs[0].contour()
            >>> c.shape

        :seealso: :meth:`.perimeter`, :meth:`.polar`, `cv2.approxPolyDP <https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c>`_
        """
        if epsilon is not None:
            c = cv.approxPolyDP(self._contours.T, epsilon=epsilon, closed=closed)
            return c[:,0,:].T
        else:
            return self._contours

    @property
    def color(self):
        col = []

        for p in self._contourpoint:
            col.append(self._image.A[p[1], p[0]])
        if len(col) == 1:
            return col[0]
        else:
            return col

    def polar(self, N=400):
        """
        Boundary in polar cooordinate form

        :param N: number of points around perimeter, defaults to 400
        :type N: int, optional
        :return: Contour, one point per column
        :rtype: ndarray(2,N)

        Returns a polar representation of the boundary with
        respect to the centroid.  Each boundary point is represented by a column
        :math:`(r, \theta)`.  The polar profile can be used for scale and
        orientation invariant matching of shapes.
        
        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> p = blobs[0].polar()
            >>> p.shape
        
        .. note:: The points are evenly spaced around the perimeter but are
            not evenly spaced in subtended angle.

        :seealso: :meth:`.polarmatch`, :meth:`.contour`
        """
        contour = np.array(self.perimeter()) - np.c_[self.centroid].T

        r = np.sqrt(np.sum(contour ** 2, axis=0))
        theta = -np.arctan2(contour[1, :], contour[0, :])

        s = np.linspace(0, 1, len(r))
        si = np.linspace(0, 1, N)

        f_r = sp.interpolate.interp1d(s, r)
        f_theta = sp.interpolate.interp1d(s, theta)

        return np.array((f_r(si), f_theta(si)))


    def polarmatch(self, target):
        """
        Compare polar profiles

        :param target: the blob to match against
        :type target: int
        :return: similarity and orientation offset
        :rtype: ndarray(N), ndarray(N)

        Performs cross correlation between the polar profiles of blobs.  All
        blobs are matched again blob ``target``.  Blob ``target`` is included
        in the results.
        
        There are two return values:

        1. Similarity is a vector, one entry per blob, where a value of one
           indicates maximum similarity irrespective of orientation and scale.
        2. Offset, one entry per blob, is the relative orientation of blobs with
           respect to the ``target`` blob.  The ``target`` blob has an
           orientation of 0.5. These values lie in the range [0, 1), equivalent
           to :math:`[0, 2\pi)` and wraps around.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs.polarmatch(1)

        Notes::
        - Can be considered as matching two functions defined over S(1).

        :seealso: :meth:`.polar`, :meth:`.contour`
        """

        # assert(numrows(r1) == numrows(r2), 'r1 and r2 must have same number of rows');

        R = []
        for i in range(len(self)):
            # get the radius profile
            r = self[i].polar()[0, :]
            # normalize to zero mean and unit variance
            r -= r.mean()
            r /= np.std(r)
            R.append(r)
        R = np.array(R)  # on row per blob boundary
        n =  R.shape[1]

        # get the target profile
        target = R[target, :]

        # cross correlate, with wrapping
        out = sp.ndimage.correlate1d(R, target, axis=1, mode='wrap') / n
        idx = np.argmax(out, axis=1)
        return [out[k, idx[k]] for k in range(len(self))], idx / n

    @property
    def touch(self):
        """
        Blob edge touch status

        :return: blob touches the edge of the image
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].touch
            >>> blobs.touch
        """
        return self._touch

    @property
    def circularity(self):
        r"""
        Blob circularity

        :return: circularity
        :rtype: float

        Computed as :math:`\rho = \frac{A}{4 \pi p^2}`.  Is one for a circular
        blob and < 1 for all other shapes, approaching zero for a line.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].circularity
            >>> blobs.circularity

        .. note::  Apply Kulpa's correction factor to account for edge
            discretization:

            - Area and perimeter measurement of blobs in discrete binary pictures.
              Z.Kulpa. Comput. Graph. Image Process., 6:434-451, 1977.
    
            - Methods to Estimate Areas and Perimeters of Blob-like Objects: a
              Comparison. Proc. IAPR Workshop on Machine Vision Applications.,
              December 13-15, 1994, Kawasaki, Japan
              L. Yang, F. Albregtsen, T. Loennestad, P. Groettum
        """
        return self._circularity

    @property
    def parent(self):
        """
        Parent blob

        :return: index of this blob's parent
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('multiblobs.png')
            >>> blobs = im.blobs()
            >>> print(blobs)
            >>> blobs[5].parent
            >>> blobs[6].parent

        A parent of -1 is the image background.
        """
        return self._parent

    @property
    def children(self):
        """
        Child blobs

        :return: list of indices of this blob's children
        :rtype: list of int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('multiblobs.png')
            >>> blobs = im.blobs()
            >>> blobs[5].children
        """
        return self._children


    def _computeboundingbox(self, epsilon=3, closed=True):
        # cpoly = [cv.approxPolyDP(c,
        #                          epsilon=epsilon,
        #                          closed=closed)
        #          for i, c in enumerate(self._contours)]
        bbox = [cv.boundingRect(contour.T) for contour in self._contours]
        return bbox

    def _computeequivalentellipse(self):
        nc = len(self)
        mf = self._moments
        mc = np.stack((self._uc, self._vc), axis=1)
        # w = [None] * nc
        # v = [None] * nc
        orientation = [None] * nc
        a = [None] * nc
        b = [None] * nc
        for i in range(nc):
            u20 = mf[i].m20 / mf[i].m00 - mc[i, 0]**2
            u02 = mf[i].m02 / mf[i].m00 - mc[i, 1]**2
            u11 = mf[i].m11 / mf[i].m00 - mc[i, 0]*mc[i, 1]

            cov = np.array([[u20, u11], [u02, u11]])
            w, v = np.linalg.eig(cov)  # w = eigenvalues, v = eigenvectors

            a[i] = 2.0 * np.sqrt(np.max(np.diag(v)) / mf[i].m00)
            b[i] = 2.0 * np.sqrt(np.min(np.diag(v)) / mf[i].m00)

            ev = v[:, -1]
            orientation[i] = np.arctan(ev[1] / ev[0])
        return a, b, orientation

    def _computecentroids(self):
        mf = self._moments
        mc = [(mf[i].m10 / (mf[i].m00), mf[i].m01 / (mf[i].m00))
              for i in range(len(mf))]
        return mc

    def _computearea(self):
        return [self._moments[i].m00 for i in range(len(self))]

    def _computecircularity(self):
        # apply Kulpa's correction factor when computing circularity
        # should have max 1 circularity for circle, < 1 for non-circles
        # Peter's reference:
        # Area and perimeter measurement of blobs in discrete binary pictures.
        # Z.Kulpa. Comput. Graph. Image Process., 6:434-451, 1977.
        # Another reference that Dorian found:
        # Methods to Estimate Areas and Perimeters of Blob-like Objects: a
        # Comparison. Proc. IAPR Workshop on Machine Vision Applications.,
        # December 13-15, 1994, Kawasaki, Japan
        # L. Yang, F. Albregtsen, T. Loennestad, P. Groettum
        kulpa = np.pi / 8.0 * (1.0 + np.sqrt(2.0))
        circularity = [((4.0 * np.pi * self._area[i]) /
                        ((self._perimeter[i] * kulpa) ** 2))
                       for i in range(len(self))]
        return circularity

    def _computeperimeter(self):
        nc = len(self)
        perimeter = []
        for i in range(nc):
            # edgelist[i] = np.vstack((self._contours[i][0:],
            #                          np.expand_dims(self._contours[i][0],
            #                                         axis=0)))
            edgediff = np.diff(self._contours[i], axis=1)
            edgenorm = np.linalg.norm(edgediff, axis=0)
            edgesum = np.sum(edgenorm)
            perimeter.append(edgesum)
        return perimeter

    def _touchingborder(self, imshape):
        touch = []
        for i in range(len(self)):
            t = self._umin[i] == 0 or \
                self._umax[i] == imshape[0] or \
                    self._vmin[i] == 0 or \
                     self._vmax[i] == imshape[1]
            touch.append(t)
        return touch

    def _hierarchicalmoments(self, mu):
        # for moments in a hierarchy, for any pq moment of a blob ignoring its
        # children you simply subtract the pq moment of each of its children.
        # That gives you the “proper” pq moment for the blob, which you then
        # use to compute area, centroid etc. for each contour
       
        new = []
        for i in range(len(self)):  # for each blob
            if len(self._children[i]):
                m = copy.copy(mu[i])  # copy current moment dictionary

                for c in self._children[i]:
                    # subtract moments of the child
                    values =[]
                    for field in self._moment_tuple._fields:
                        values.append(m[field] -  mu[c][field])
                    # m = {key: m[key] -  mu[c][key] for key in m}
            else:
                values =[]
                for field in self._moment_tuple._fields:
                    values.append(mu[i][field])

            new.append(self._moment_tuple._make(values))

        return new

    def _getchildren(self):
        # gets list of children for each contour based on hierarchy
        # follows similar for loop logic from _hierarchicalmoments, so
        # TODO use _getchildren to cut redundant code in _hierarchicalmoments

        children = [ [] for i in range(len(self))]

        for i in range(len(self)):

            if self._hierarchy[i, 3] != -1:
                # parent of contour i
                parent = self._hierarchy[i, 3]
                children[parent].append(i)
        return children


    def plot_box(self, **kwargs):
        """
        Plot a bounding box for the blob using matplotlib

        :param kwargs: arguments passed to ``plot_box``

        Plot a bounding box for every blob described by this object.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('multiblobs.png')
            >>> blobs = im.blobs()
            >>> blobs[5].plot_box('r') # red bounding box for blob 5
            >>> blobs.plot_box('g') # green bounding box for all blobs

        :seealso: :func:`~machinevisiontoolbox.base.graphics.plot_box`
        """

        for blob in self:
            plot_box(bbox=blob.bbox, **kwargs)


    def plot_labelbox(self, **kwargs):
        """
        Plot a labelled bounding box for the blob using matplotlib

        :param kwargs: arguments passed to ``plot_labelbox``

        Plot a labelled bounding box for every blob described by this object.
        The blobs are labeled by their blob index.

        :seealso: :func:`~machinevisiontoolbox.base.graphics.plot_labelbox`
        """

        for i, blob in enumerate(self):
            plot_labelbox(text=f"{i}", bbox=blob.bbox, **kwargs)

    def plot_centroid(self, label=False, **kwargs):
        """
        Draw the centroid of the blob using matplotlib

        :param label: add a sequential numeric label to each point, defaults to False
        :type label: bool
        :param kwargs: other arguments passed to ``plot_point``

        If no marker style is given then it will be an overlaid "o" and "x"
        in blue.

        :seealso: :func:`~machinevisiontoolbox.base.graphics.plot_point`
        """
        if label:
            text = f"{i}"
        else:
            text = None
        
        if 'marker' not in kwargs:
            kwargs['marker'] = ['bx', 'bo']
            kwargs['fillstyle'] = 'none'
        for i, blob in enumerate(self):
            plot_point(pos=blob.centroid, text=text, **kwargs)

    def plot_perimeter(self, **kwargs):
        for i in range(len(self)):
            xy = self._contours[i]
            plt.plot(xy[0, :], xy[1, :], **kwargs)

    def drawBlobs(self,
                  image,
                  drawing=None,
                  icont=None,
                  color=None,
                  contourthickness=cv.FILLED,
                  textthickness=2):
        """
        Draw the blob contour

        :param image: [description]
        :type image: [type]
        :param drawing: [description], defaults to None
        :type drawing: [type], optional
        :param icont: [description], defaults to None
        :type icont: [type], optional
        :param color: [description], defaults to None
        :type color: [type], optional
        :param contourthickness: [description], defaults to cv.FILLED
        :type contourthickness: [type], optional
        :param textthickness: [description], defaults to 2
        :type textthickness: int, optional
        :return: [description]
        :rtype: [type]

        :seealso: `cv2.drawContours <https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc>`_
        """
        # draw contours of blobs
        # contours - the contour list
        # icont - the index of the contour(s) to plot
        # drawing - the image to draw the contours on
        # colors - the colors for the icont contours to be plotted (3-tuple)
        # return - updated drawing

        # TODO split this up into drawBlobs and drawCentroids methods

        # image = Image(image)
        # image = self.__class__(image)  # assuming self is Image class
        # @# assume image is Image class

        if drawing is None:
            drawing = np.zeros(
                (image.shape[0], image.shape[1], 3), dtype=np.uint8)

        if icont is None:
            icont = np.arange(0, len(self))
        else:
            icont = np.array(icont, ndmin=1, copy=True)

        if color is None:
            # make colors a list of 3-tuples of random colors
            color = [None]*len(icont)

            for i in range(len(icont)):
                color[i] = (rng.randint(0, 256),
                            rng.randint(0, 256),
                            rng.randint(0, 256))
                # contourcolors[i] = np.round(colors[i]/2)
            # TODO make a color option, specified through text,
            # as all of a certain color (default white)

        # make contour colours slightly different but similar to the text color
        # (slightly dimmer)?
        cc = [np.uint8(np.array(color[i])/2) for i in range(len(icont))]
        contourcolors = [(int(cc[i][0]), int(cc[i][1]), int(cc[i][2]))
                         for i in range(len(icont))]

        # TODO check contours, icont, colors, etc are valid
        hierarchy = np.expand_dims(self._hierarchy, axis=0)
        # done because we squeezed hierarchy from a (1,M,4) to an (M,4) earlier

        for i in icont:
            # TODO figure out how to draw alpha/transparencies?
            cv.drawContours(drawing,
                            self._contours,
                            icont[i],
                            contourcolors[i],
                            thickness=contourthickness,
                            lineType=cv.LINE_8,
                            hierarchy=hierarchy)

        for i in icont:
            ic = icont[i]
            cv.putText(drawing,
                       str(ic),
                       (int(self._uc[ic]), int(self._vc[ic])),
                       fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=1,
                       color=color[i],
                       thickness=textthickness)

        return image.__class__(drawing)

    def label_image(self,
                  image,
                  drawing=None
                  ):

        # different label assignment compared to imageLabels()

        if drawing is None:
            drawing = np.zeros(image.shape, dtype=np.uint8)

        # TODO check contours, icont, colors, etc are valid
        # done because we squeezed hierarchy from a (1,M,4) to an (M,4) earlier

        for i in range(len(self)):
            # TODO figure out how to draw alpha/transparencies?
            cv.drawContours(image=drawing,
                            contours=self._contours_raw,
                            contourIdx=i,
                            color=i+1,
                            thickness=-1,  # fill the contour
                            hierarchy=self._hierarchy_raw)

        return image.__class__(drawing[:,:])



    def printBlobs(self):
        # TODO accept kwargs or args to show/filter relevant parameters

        # convenience function to plot
        for i in range(len(self)):
            print(str.format(r'({0})  area={1:.1f}, \
                  cent=({2:.1f}, {3:.1f}), \
                  orientation={4:.3f}, \
                  b/a={5:.3f}, \
                  touch={6:d}, \
                  parent={7}, \
                  children={8}',
                             i, self._area[i], self._uc[i], self._vc[i],
                             self._orientation[i], self._aspect[i],
                             self._touch[i], self._parent[i],
                             self._children[i]))

    def dotfile(self, filename=None, direction=None, show=False):
        """
        Create a GraphViz dot file

        :param filename: filename to save graph to, defaults to None
        :type filename: str, optional

        ``g.dotfile()`` creates the specified file which contains the
        GraphViz code to represent the embedded graph.  By default output
        is to the console

        .. note::

            - The graph is undirected if it is a subclass of ``UGraph``
            - The graph is directed if it is a subclass of ``DGraph``
            - Use ``neato`` rather than dot to get the embedded layout

        .. note:: If ``filename`` is a file object then the file will *not*
            be closed after the GraphViz model is written.
        """

        if show:
            # create the temporary dotfile
            filename = tempfile.TemporaryFile(mode="w")
        if filename is None:
            f = sys.stdout
        elif isinstance(filename, str):
            f = open(filename, "w")
        else:
            f = filename

        print("digraph {", file=f)

        if direction is not None:
            print(f"rankdir = {direction}", file=f)

        # add the nodes including name and position
        for id, blob in enumerate(self):
            print('  "{:d}"'.format(id), file=f)
            print('  "{:d}" -> "{:d}"'.format(blob.parent, id), file=f)

        print('}', file=f)

        if show:
            # rewind the dot file, create PDF file in the filesystem, run dot
            f.seek(0)
            pdffile = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            subprocess.run("dot -Tpdf", shell=True, stdin=filename, stdout=pdffile)

            # open the PDF file in browser (hopefully portable), then cleanup
            webbrowser.open(f"file://{pdffile.name}")
        else:
            if filename is None or isinstance(filename, str):
                f.close()  # noqa

    @property
    def moments(self):
        return self._moments

class ImageBlobsMixin:

    def blobs(self, **kwargs):
        """
        Compute blobs in image

        :return: blobs
        :rtype: Blob

        ``image.blobs()`` is a ``Blob`` object that contains information about
        all the blobs in the image.  It behaves like a list object so it can
        be indexed and sliced.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> type(blobs)
            >>> len(blobs)
            >>> print(blobs)
        """

        # TODO do the feature extraction here
        # each blob is a named tuple??
        # This could be applied to MSERs
        return Blobs(self, **kwargs)


if __name__ == "__main__":

    from machinevisiontoolbox import Image
    import matplotlib.pyplot as plt

    im = Image.Read('multiblobs.png')
    blobs = im.blobs()

    print(blobs.color)

    print(blobs[2].humoments)
    print(blobs.humoments)


    # print(blobs[3].moments)

    # blobs.dotfile(show=True)

    # from ansitable.table import _unicode
    # _unicode = False

    # print(blobs)
    # print(blobs.children)
    # print(blobs[5:8].children)
    # print(blobs[5].contour(epsilon=20))
    # im.disp()
    # blobs.plot_labelbox(filled=False, labelcolor='red', edgecolor='red')
    # blobs.plot_centroid()
    # print(blobs[0].children)
    # plt.show(block=True)

    


    # # read image
    # from machinevisiontoolbox import Image
    # im = Image(cv.imread('images/multiblobs.png', cv.IMREAD_GRAYSCALE))

    # # call Blobs class
    # b = Blob(image=im)

    # # plot image
    # # plot centroids of blobs
    # # label relevant centroids for the labelled blobs
    # # import random as rng  # for random colors of blobs
    # rng.seed(53467)

    # drawing = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
    # colors = [None]*len(b)
    # icont = [None]*len(b)
    # for i in range(len(b)):
    #     icont[i] = i
    #     colors[i] = (rng.randint(0, 256), rng.randint(
    #         0, 256), rng.randint(0, 256))

    #     cv.rectangle(drawing, (b[i].umin, b[i].vmin), (b[i].umax, b[i].vmax),
    #                  colors[i], thickness=2)
    #     # cv.putText(drawing, str(i), (int(b[i].uc), int(b[i].vc)),
    #     #           fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,
    #     #           color=colors, thickness=2)

    # drawing = b.drawBlobs(im, drawing, icont, colors,
    #                       contourthickness=cv.FILLED)
    # # mvt.idisp(drawing)

    # # import matplotlib.pyplot as plt
    # # plt.imshow(d2)
    # # plt.show()
    # # mvt.idisp(d2)
    # im2 = Image('images/multiblobs_edgecase.png')
    # im2.disp()

    # press Ctrl+D to exit and close the image at the end
    # import code
    # code.interact(local=dict(globals(), **locals()))
