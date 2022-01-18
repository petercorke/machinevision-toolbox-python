#!/usr/bin/env python
"""
2D Blob feature class
@author: Dorian Tsai
@author: Peter Corke
"""
import copy
import numpy as np
from collections import namedtuple, UserList
import cv2 as cv
from numpy.lib.arraysetops import isin
from spatialmath import base
from ansitable import ANSITable, Column
from machinevisiontoolbox.base import color_bgr
from spatialmath.base import plot_box, plot_point, isscalar
import scipy as sp
import tempfile
import subprocess
import webbrowser
import sys

# NOTE, might be better to use a matplotlib color cycler
import random as rng
rng.seed(13543)  # would this be called every time at Blobs init?
import matplotlib.pyplot as plt

def scalar_result(func):
    def inner(*args):
        out = func(*args)
        if len(out) == 1:
            return out[0]
        else:
            return np.array(out)
    return inner

def array_result(func):
    def inner(*args):
        out = func(*args)
        if len(out) == 1:
            return out[0]
        else:
            return out
    return inner

_moment_tuple = namedtuple('moments', 
    ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 
    'm30', 'm21', 'm12', 'm03', 'mu20', 'mu11', 'mu02', 
    'mu30', 'mu21', 'mu12', 'mu03', 'nu20', 'nu11', 'nu02', 
    'nu30', 'nu21', 'nu12', 'nu03'])
class Blob:
    id = None
    bbox = None
    moments = None
    touch = None
    perimeter = None
    a = None
    b = None
    orientation = None
    children = None
    parent = None
    uc = None
    vc = None
    level = None

    def __init__(self):
        return

    def __str__(self):
        l = [f"{key}: {value}" for key, value in self.__dict__.items()]
        return '\n'.join(l)

    def __repr__(self):
        return str(self)
class Blobs(UserList):
    """
    A 2D feature blob class
    """
    
    _image = []  # keep image saved for each Blobs object

    def __init__(self, image=None, **kwargs):
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
        super().__init__(self)

        if image is None:
            # initialise empty Blobs
            # Blobs()
            return

        self._image = image  # keep reference to original image

        image = image.mono()
        
        # get all the contours
        contours, hierarchy = cv.findContours(image.to_int(),
                                              mode=cv.RETR_TREE,
                                              method=cv.CHAIN_APPROX_NONE)


        self._hierarchy_raw = hierarchy
        self._contours_raw = contours

        # change hierarchy from a (1,M,4) to (M,4)
        # the elements of each row are:
        #   0: index of next contour at same level, 
        #   1: index of previous contour at same level, 
        #   2: index of first child, 
        #   3: index of parent
        hierarchy = hierarchy[0,:,:]  # drop the first singleton dimension
        parents = hierarchy[:, 3]

        # change contours to list of 2xN arraay
        contours = [c[:,0,:] for c in contours]

        ## first pass: moments, children, bbox

        for i, (contour, hier) in enumerate(zip(contours, hierarchy)):

            blob = Blob()
            blob.id = i

            ## bounding box: umin, vmin, width, height
            u1, v1, w, h = cv.boundingRect(contour)
            u2 = u1 + w
            v2 = v1 + h
            blob.bbox = np.r_[u1, u2, v1, v2]

            blob.touch = u1 == 0  or v1 == 0 or u2 == image.umax or v2 == image.vmax

            ## children

            # gets list of children for each contour based on hierarchy
            # follows similar for loop logic from _hierarchicalmoments, so
            # TODO use _getchildren to cut redundant code in _hierarchicalmoments

            blob.parent = hier[3]

            children = []
            child = hier[2]
            while child != -1:
                children.append(child)
                child = hierarchy[child, 0]
            blob.children = children

            ## moments

            # get moments as a dictionary for each contour
            blob.moments = cv.moments(contour)

            ## perimeter, the contour is not closed

            blob.perimeter = contour.T
            blob.perimeter_length = cv.arcLength(contour, closed=False)

            blob.contourpoint = blob.perimeter[:, 0]

            ## append the new Blob instance
            self.data.append(blob)

        ## second pass: equivalent ellipse

        for blob, contour in zip(self.data, contours):

            ## moment hierarchy

            # for moments in a hierarchy, for any pq moment of a blob ignoring its
            # children you simply subtract the pq moment of each of its children.
            # That gives you the “proper” pq moment for the blob, which you then
            # use to compute area, centroid etc. for each contour
       
            # TODO: this should recurse all the way down
            M = blob.moments
            for child in blob.children:
                # subtract moments of the child
                M = {key: M[key] -  self.data[child].moments[key] for key in M}

            # convert dict to named tuple, easier to access using dot notation
            M = _moment_tuple._make([M[field] for field in _moment_tuple._fields])
            blob.moments = M

            ## centroid
            blob.uc = M.m10 / M.m00
            blob.vc = M.m01 / M.m00

            ## equivalent ellipse
            J = np.array([[M.mu20, M.mu11], [M.mu11, M.mu02]])
            e, X = np.linalg.eig(J)

            blob.a = 2.0 * np.sqrt(e.max() / M.m00)
            blob.b = 2.0 * np.sqrt(e.min() / M.m00)

            # find eigenvector for largest eigenvalue
            k = np.argmax(e)
            x = X[:, k]
            blob.orientation = np.arctan2(x[1], x[0])

            ## circularity

            # apply Kulpa's correction factor when computing circularity
            # should have max 1 circularity for circle, < 1 for non-circles
            # * Area and perimeter measurement of blobs in discrete binary pictures.
            #   Z.Kulpa. Comput. Graph. Image Process., 6:434-451, 1977.
            # * Methods to Estimate Areas and Perimeters of Blob-like Objects: a
            #   Comparison. Proc. IAPR Workshop on Machine Vision Applications.,
            #   December 13-15, 1994, Kawasaki, Japan
            #   L. Yang, F. Albregtsen, T. Loennestad, P. Groettum
            kulpa = np.pi / 8.0 * (1.0 + np.sqrt(2.0))
            blob.circularity = (4.0 * np.pi * M.m00) / (blob.perimeter_length * kulpa) ** 2

        ## third pass, region tree coloring to determine vertex depth
        while any([b.level is None for b in self.data]):  # while some uncolored
            for blob in self.data:
                if blob.level is None:
                    if blob.parent == -1:
                        blob.level = 0 # root level
                    elif self.data[blob.parent].level is not None:
                        # one higher than parent's depth
                        blob.level = self.data[blob.parent].level + 1

        self.filter(**kwargs)

        return

    def filter(self, area=None, circularity=None, color=None, touch=None, aspect=None):
        mask = []

        if area is not None:
            _area = self.area
            if isscalar(area):
                mask.append(_area >= area)
            elif len(area) == 2:
                mask.append(_area >= area[0])
                mask.append(_area <= area[1])

        if circularity is not None:
            _circularity = self.circularity
            if isscalar(circularity):
                mask.append(_circularity >= circularity)
            elif len(circularity) == 2:
                mask.append(_circularity >= circularity[0])
                mask.append(_circularity <= circularity[1])

        if aspect is not None:
            _aspect = self.aspect
            if isscalar(aspect):
                mask.append(_aspect >= aspect)
            elif len(circularity) == 2:
                mask.append(_aspect >= aspect[0])
                mask.append(_aspect <= aspect[1])

        if color is not None:
            _color = self.color
            mask.append(_color == _color)

        if touch is not None:
            _touch = self.touch
            mask.append(_touch == touch)

        m = np.array(mask).all(axis=0)

        return self[m]

    def sort(self, by="area", reverse=False):
        if by == "area":
            k = np.argsort(self.area)
        elif by == "circularity":
            k = np.argsort(self.circularity)
        elif by == "perimeter":
            k = np.argsort(self.perimeter_length)
        elif by == "aspect":
            k = np.argsort(self.aspect)
        elif by == "touch":
            k = np.argsort(self.touch)
        
        if reverse:
            k = k[::-1]
        
        return self[k]

    def __getitem__(self, i):
        new = Blobs()
        new._image = self._image

        if isinstance(i, (int, slice)):
            data = self.data[i]
            if not isinstance(data, list):
                data = [data]
            new.data = data

        elif isinstance(i, (list, tuple)):
            new.data = [self.data[k] for k in i]

        elif isinstance(i, np.ndarray):
            # numpy thing
            if np.issubdtype(i.dtype, np.integer):
                new.data = [self.data[k] for k in i]
            elif np.issubdtype(i.dtype, np.bool_) and len(i) == len(self):
                new.data = [self.data[k] for k in range(len(i)) if i[k]]
        return new

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
                    Column("circul", fmt="{:.3f}"),
                    Column("orient", fmt="{:.1f}°"),
                    Column("aspect", fmt="{:.3g}"),
                    border="thin"
        )
        for b in self.data:
            table.row(b.id, b.parent, f"{b.uc:.1f}, {b.vc:.1f}",
                      b.moments.m00,
                      b.touch,
                      b.perimeter_length,
                      b.circularity,
                      np.rad2deg(b.orientation),
                      b.b / b.a)

        return str(table)

    @property
    @scalar_result
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
        return [b.moments.m00 for b in self.data]

    @property
    @scalar_result
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
        return [b.uc for b in self.data]

    @property
    @scalar_result
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
        return [b.vc for b in self.data]

    @property
    @scalar_result
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
        return [b.a for b in self.data]

    @property
    @scalar_result
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
        return [b.b for b in self.data]

    @property
    @scalar_result
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
        return [b.b / b.a for b in self.data]

    @property
    @scalar_result
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
        return [b.orientation for b in self.data]

    @property
    @scalar_result
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
        return [b.bbox[0] for b in self.data]

    @property
    @scalar_result
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
        return [b.bbox[0] + b.bbox[2] for b in self.data]

    @property
    @scalar_result
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
        return [b.bbox[0] for b in self.data]

    @property
    @scalar_result
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
        return [b.bbox[1] + b.bbox[3] for b in self.data]


    @property
    @scalar_result
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
        return [b.bbox[2] * b.bbox[3] for b in self.data]

    @property
    @scalar_result
    def fillfactor(self):
        """
        Fill factor, ratio of area to bounding box area

        :return: fill factor
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].fillfactor
            >>> blobs.fillfactor

        .. note:: The bounding box is the smallest box with vertical and
            horizontal edges that fully encloses the blob.

        :seealso: :meth:`.bbox`
        """
        return [b.moments.m00 / (b.bbox[2] * b.bbox[3]) for b in self.data]

    @property
    @scalar_result
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
        return [b.perimeter_length for b in self.data]



    @property
    @scalar_result
    def level(self):
        """
        Blob level in hierarchy

        :return: blob level in hierarchy
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('multiblobs.png')
            >>> blobs = im.blobs()
            >>> blobs[2].level
            >>> blobs.level
        """
        return [b.level for b in self.data]

    @property
    @scalar_result
    def color(self):
        """
        Blob color

        :return: blob color
        :rtype: int

        Blob color in a binary image.  This is inferred from the level in
        the blob hierarchy.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('multiblobs.png')
            >>> blobs = im.blobs()
            >>> blobs[2].color
            >>> blobs.color
        """
        return [b.level & 1 for b in self.data]

    @property
    @scalar_result
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
        return [b.touch for b in self.data]

    @property
    @scalar_result
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
        return [b.circularity for b in self.data]

    @property
    @scalar_result
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
        return [b.parent for b in self.data]

    @property
    @array_result
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
        return [(b.uc, b.vc) for b in self.data]

    @property
    @array_result
    def p(self):
        """
        Centroid point of blob

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
        return [(b.uc, b.vc) for b in self.data]

    @property
    @array_result
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
        return [b.bbox for b in self.data]

    @property
    @array_result
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
        return [b.children for b in self.data]

    @property
    @array_result
    def moments(self):
        return [b.moments for b in self.data]

    @property
    @array_result
    def humoments(self):
        def hu(b):
            m = b.moments
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
            return phi

        return [hu(b) for b in self.data]

    @property
    @array_result
    def perimeter(self):
        """
        Perimeter of the blob

        :return: Perimeter, one point per column
        :rtype: ndarray(2,N)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> c = blobs[0].contour()
            >>> c.shape

        NOT CLOSED!

        :seealso: :meth:`.perimeter`, :meth:`.polar`, `cv2.approxPolyDP <https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c>`_
        """
        return [b.perimeter for b in self.data]

    @array_result
    def perimeter_approx(self, epsilon=None):
        """
        Approximate perimeter of the blob

        :param epsilon: maximum distance between the original curve and its approximation, default is exact contour
        :type epsilon: int
        :return: Perimeter, one point per column
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
        return [cv.approxPolyDP(b.perimeter.T,  epsilon=epsilon, closed=False) for b in self.data]


    @array_result
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
        def polarfunc(b):

            contour = np.array(b.perimeter) - np.c_[b.p].T

            r = np.sqrt(np.sum(contour ** 2, axis=0))
            theta = -np.arctan2(contour[1, :], contour[0, :])

            s = np.linspace(0, 1, len(r))
            si = np.linspace(0, 1, N)

            f_r = sp.interpolate.interp1d(s, r)
            f_theta = sp.interpolate.interp1d(s, theta)

            return np.array((f_r(si), f_theta(si)))

        return [polarfunc(b) for b in self]


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
            plot_box(lrbt=blob.bbox, **kwargs)


    def plot_labelbox(self, **kwargs):
        """
        Plot a labelled bounding box for the blob using matplotlib

        :param kwargs: arguments passed to ``plot_labelbox``

        Plot a labelled bounding box for every blob described by this object.
        The blobs are labeled by their blob index.

        :seealso: :func:`~machinevisiontoolbox.base.graphics.plot_labelbox`
        """

        for i, blob in enumerate(self):
            plot_labelbox(text=f"{i}", lrbt=blob.bbox, **kwargs)

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
        for blob in self:
            x, y = blob.perimeter
            plt.plot(x, y, **kwargs)

    def label_image(self,
                  image=None,
                  drawing=None
                  ):

        if image is None:
            image = self._image
            
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
    im = Image.Read('sharks.png')


    im.disp()
    blobs=im.blobs()
    print(blobs)

    blobs.plot_box(color="red")


    # blobs = Blobs()
    # print(len(blobs))

    blobs = im.blobs()
    print(len(blobs))
    print(blobs[0].area)
    print(blobs.area)

    print(blobs)

    print(blobs.level)
    print(blobs.color)
    print(blobs[1].children)
    print(blobs.p)
    print(blobs.moments)
    print(blobs.humoments)

    print(blobs[(3,2,1)])
    print(blobs[np.r_[3,2,1]])
    print(blobs[blobs.circularity > 0.8])

    print(blobs.sortby())
    print(blobs.sortby(reverse=True))
    print(blobs.sortby(by="circularity"))

    print(blobs.filter(circularity=0.8))

    # print(blobs.color)

    # print(blobs[2].humoments)
    # print(blobs.humoments)


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
