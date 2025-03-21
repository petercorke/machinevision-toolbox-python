#!/usr/bin/env python
"""
2D Blob feature class
@author: Dorian Tsai
@author: Peter Corke
"""
import sys
import copy
from collections import namedtuple, UserList
import tempfile
import subprocess
import webbrowser

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import cv2 as cv
from ansitable import ANSITable, Column
from machinevisiontoolbox.base import color_bgr, plot_labelbox
from spatialmath.base import plot_box, plot_point, isscalar
from spatialmath import SE2, base
from machinevisiontoolbox.decorators import scalar_result, array_result

"""
NOTES

Defines two key classes:

- ``Blob`` is a simple container for the parameters of a single blob.
- ``Blobs`` is a ``UserList`` of ``Blob`` instances.  It behaves like a list
    and each element of the list is a ``Blob`` instance.  A single element ``Blobs``
    instance represents a single blob.  A ``Blobs`` instance has many additional
    attributes compared to a ``Blob`` instance, and these are derived from the 
    ``Blob`` instances in the list.
"""


_moment_tuple = namedtuple(
    "moments",
    [
        "m00",
        "m10",
        "m01",
        "m20",
        "m11",
        "m02",
        "m30",
        "m21",
        "m12",
        "m03",
        "mu20",
        "mu11",
        "mu02",
        "mu30",
        "mu21",
        "mu12",
        "mu03",
        "nu20",
        "nu11",
        "nu02",
        "nu30",
        "nu21",
        "nu12",
        "nu03",
    ],
)


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
        """Constructor for Blob class

        A :class:`Blob` instance is a simple container for the parameters
        of a single blob.

        A set of blobs is represented by a :class:`Blobs` instance which acts
        like a list of :class:`Blob` instance.
        """
        return

    def __str__(self):
        """Create a compact string representation of the Blob object

        :return: compact string representation
        :rtype: str
        """
        return f"Blob[{self.id}](area={self._moments.m00:.2g}, color={self.color}, parent={self.parent.id if self.parent else None}"

    def __repr__(self):
        return str(self)

    def print(self):
        """Create a detailed string representation of the Blob object

        :return: detailed string representation
        :rtype: str

        The string representation includes all the attributes of the Blob object.
        """
        l = [f"{key}: {value}" for key, value in self.__dict__.items()]
        return "\n".join(l)


class Blobs(UserList):  # lgtm[py/missing-equals]
    _image = []  # keep image saved for each Blobs object

    def __init__(self, image=None, kulpa=True, **kwargs):
        """
        Find blobs and compute their attributes

        :param image: image to use, defaults to None
        :type image: :class:`Image`, optional
        :param kulpa: apply Kulpa's correction factor to circularity, defaults to True
        :type kulpa: bool, optional

        Uses OpenCV functions ``findContours`` to find a hierarchy of regions
        represented by their contours, and ``boundingRect``, ``moments`` to
        compute moments, perimeters, centroids etc.

        This class behaves like a list and each element of the list is a blob
        represented by a :class:`Blob` instance

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('sharks.png')
            >>> blobs = img.blobs()
            >>> len(blobs)
            >>> blobs[0]
            >>> blobs.area

        The list can be indexed, sliced or used as an iterator in a for loop
        or comprehension, for example::

            >>> for blob in blobs:
            >>>   # do a thing
            >>> areas = [blob.area for blob in blobs]

        However the last line can also be written as::

            >>> areas = blobs.area

        since all methods return a scalar if applied to a single blob::

            >>> blobs[1].area

        or a list if applied to multiple blobs::

            >>> blobs.area

        A blob has many attributes:

        .. list-table::
           :header-rows: 1

           * - Attribute
             - Description
           * - :meth:`area`
             - The area of the blob.
           * - :meth:`u`, :meth:`v`
             - The centroid (center of mass) of the blob.
           * - :meth:`bbox`
             - The bounding box of the blob.
           * - :meth:`color`
             - The vaue of pixels within the blob.
           * - :meth:`touch`
             - True if the blob touches the border.
           * - :meth:`contour_point`
             - A point on the contour of the blob.
           * - :meth:`perimeter`
             - A 2xN array of points on the perimeter of the blob.
           * - :meth:`perimeter_length`
             - The perimeter length of the blob.
           * - :meth:`circularity`
             - The circularity of the blob.
           * - :meth:`moments`
             - The moments of the blob including central, normalized upto 3rd order.
           * - :meth:`orientation`
             - Orientation of the equivalent ellipse.
           * - :meth:`a, b`
             - The equivalent ellipse radii.
           * - :meth:`children`
             - A list of references to child :class:`Blob` instances.
           * - :meth:`parent`
             - A reference to the parent :class:`Blob` instance, or None if no parent.
           * - :meth:`level`
             - The depth of the blob in the region tree.

        :note: A color image is internally converted to greyscale.

        :note: ``findContours`` can give surprising results for small images:

            - The perimeter length is computed between the mid points of the pixels, and
              the OpenCV function ``arcLength`` seems to underestimate the perimeter
              even more.  The perimeter length is not the same as the number of pixels
              in the contour.
            - The area will be less than the number of pixels in the blob, because the
              area is computed from the moments of the blob, which are computed from the
              contour, see above.

        :references:
            - Robotics, Vision & Control for Python, Section 12.1.2.1, P. Corke, Springer 2023.

        :seealso: :meth:`filter` :meth:`sort`
            `opencv.moments <https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga556a180f43cab22649c23ada36a8a139>`_,
            `opencv.boundingRect <https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga103fcbda2f540f3ef1c042d6a9b35ac7>`_,
            `opencv.findContours <https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0>`_
        """
        super().__init__(self)

        if image is None:
            # initialise empty Blobs
            # Blobs()
            return

        self._image = image  # keep reference to original image

        image = image.mono()

        # get all the contours
        contours, hierarchy = cv.findContours(
            image.to_int(), mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE
        )

        # for N blobs, each with a perimeter of P_i points (i=0...N-1)
        # - contours is a tuple of N ndarrays of shape (P_i,1,2)
        # - hierarchy is a (1,N,4) array

        self._contours_raw = contours  # save original contours from OpenCV
        # change contours to list of 2xN arraay
        contours = [c[:, 0, :] for c in contours]

        self._hierarchy_raw = hierarchy  # save original hierarchy from OpenCV
        # change hierarchy from a (1,N,4) to (N,4)
        # the elements of each row are:
        #   0: index of next contour at same level,
        #   1: index of previous contour at same level,
        #   2: index of first child,
        #   3: index of parent
        hierarchy = hierarchy[0, :, :]  # drop the first singleton dimension
        parents = hierarchy[:, 3]

        ## first pass: moments, children, bbox

        runts = 0
        allblobs = []
        for i, (contour, hier) in enumerate(zip(contours, hierarchy)):
            blob = Blob()
            blob.id = i

            ## bounding box: umin, vmin, width, height
            u1, v1, w, h = cv.boundingRect(contour)
            u2 = u1 + w - 1
            v2 = v1 + h - 1
            blob.bbox = np.r_[u1, u2, v1, v2]

            blob.touch = u1 == 0 or v1 == 0 or u2 == image.umax or v2 == image.vmax

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

            pp = contour[0, :]
            blob.color = image.A[pp[1], pp[0]]

            ## moments

            # get moments as a dictionary for each contour
            blob._moments = cv.moments(contour)

            ## perimeter, the contour is not closed

            blob.perimeter = contour.T
            blob.perimeter_length = cv.arcLength(contour, closed=False)

            blob.contourpoint = blob.perimeter[:, 0]

            ## For a single set pixel OpenCV returns all moments as zero, skip such blobs
            ## TODO handle this situation by setting m00=1, m10=x, m01=y etc.
            if blob._moments["m00"] == 0:
                runts += 1
            else:
                self.data.append(blob)
            allblobs.append(blob)

        ## second pass: equivalent ellipse

        for blob, contour in zip(self.data, contours):
            ## moment hierarchy

            # for moments in a hierarchy, for any pq moment of a blob ignoring its
            # children you simply subtract the pq moment of each of its children.
            # That gives you the “proper” pq moment for the blob, which you then
            # use to compute area, centroid etc. for each contour

            # TODO: this should recurse all the way down
            M = blob._moments
            for child in blob.children:
                # subtract moments of the child
                M = {key: M[key] - allblobs[child]._moments[key] for key in M}

            # convert dict to named tuple, easier to access using dot notation
            M = _moment_tuple._make([M[field] for field in _moment_tuple._fields])
            blob._moments = M

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
            if kulpa is True:
                kfactor = np.pi / 8.0 * (1.0 + np.sqrt(2.0))
            elif isinstance(kulpa, (int, float)):
                kfactor = kulpa
            else:
                kfactor = 1.0
            blob.circularity = (4.0 * np.pi * M.m00) / (
                blob.perimeter_length * kfactor
            ) ** 2

        ## third pass, region tree coloring to determine vertex depth
        while any([b.level is None for b in self.data]):  # while some uncolored
            for blob in self.data:
                if blob.level is None:
                    if blob.parent == -1:
                        blob.level = 0  # root level
                    elif allblobs[blob.parent].level is not None:  ##
                        # one higher than parent's depth
                        blob.level = allblobs[blob.parent].level + 1  ##

        self.filter(**kwargs)

        for blob in self.data:
            if blob.parent != -1:
                blob.parent = allblobs[blob.parent]
            else:
                blob.parent = None
            if len(blob.children) > 0:
                blob.children = [allblobs[i] for i in blob.children]  ##

        if runts > 0:
            print(f"blobs: found {runts} runt blob{'s' if runts > 1 else ''}")

        return

    def filter(self, area=None, circularity=None, color=None, touch=None, aspect=None):
        """
        Filter blobs

        :param area: area minimum or range, defaults to None
        :type area: scalar or array_like(2), optional
        :param circularity: circularity minimum or range, defaults to None
        :type circularity: scalar or array_like(2), optional
        :param color: color/polarity to accept, defaults to None
        :type color: bool, optional
        :param touch: blob touch status to accept, defaults to None
        :type touch: bool, optional
        :param aspect: aspect ratio minimum or range, defaults to None
        :type aspect: scalar or array_like(2), optional
        :return: set of filtered blobs
        :rtype: :class:`Blobs`

        Return a set of blobs that match the filter criteria.

        =================   =========================================
        Parameter           Description
        =================   =========================================
        ``"area"``          Blob area
        ``"circularity"``   Blob circularity
        ``"aspect"``        Aspect ratio of equivalent ellipse
        ``"touch"``         Blob edge touch status
        =================   =========================================

        The filter parameter arguments are:

        - a scalar, representing the minimum acceptable value
        - a array_like(2), representing minimum and maximum acceptable value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('sharks.png')
            >>> blobs = img.blobs()
            >>> blobs
            >>> blobs.filter(area=10_000)
            >>> blobs.filter(area=10_000, circularity=0.3)

        .. warning:: Filtering can destroy the hierarchy of the blobs, deleting
            parents and children in the blob tree.  A blob may have references
            to parents and children that are not in the filtered set.

        :references:
            - Robotics, Vision & Control for Python, Section 12.1.2.1, P. Corke, Springer 2023.

        :seealso: :meth:`sort`
        """
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
            mask.append(_color == color)

        if touch is not None:
            _touch = self.touch
            mask.append(_touch == touch)

        m = np.array(mask).all(axis=0)

        return self[m]

    def sort(self, by="area", reverse=False):
        """
        Sort blobs

        :param by: parameter to sort on, defaults to "area"
        :type by: str, optional
        :param reverse: sort in ascending order, defaults to False
        :type reverse: bool, optional
        :return: set of sorted blobs
        :rtype: :class:`Blobs`

        Return a blobs object where the blobs are sorted according to the
        sort parameter:

        =================   =========================================
        Parameter           Description
        =================   =========================================
        ``"area"``          Blob area
        ``"circularity"``   Blob circularity
        ``"perimeter"``     Blob external perimeter length
        ``"aspect"``        Aspect ratio of equivalent ellipse
        ``"touch"``         Blob edge touch status
        =================   =========================================

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('sharks.png')
            >>> blobs = img.blobs()
            >>> blobs.sort()

        :references:
            - Robotics, Vision & Control for Python, Section 12.1.2.1, P. Corke, Springer 2023.

        :seealso: :meth:`filter`
        """
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
            elif np.issubdtype(i.dtype, bool) and len(i) == len(self):
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
            border="thin",
        )
        for b in self.data:
            table.row(
                b.id,
                b.parent.id if b.parent else -1,
                f"{b.uc:.1f}, {b.vc:.1f}",
                b._moments.m00,
                b.touch,
                b.perimeter_length,
                b.circularity,
                np.rad2deg(b.orientation),
                b.b / b.a,
            )

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
        return [b._moments.m00 for b in self.data]

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

        :seealso:  :meth:`v` :meth:`centroid`
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

        :seealso:  :meth:`u` :meth:`centroid`
        """
        return [b.vc for b in self.data]

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

        :seealso:  :meth:`u` :meth:`v` :meth:`moments`
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

        :seealso:  :meth:`u` :meth:`v`
        """
        return [(b.uc, b.vc) for b in self.data]

    @property
    @array_result
    def bbox(self):
        """
        Bounding box

        :return: bounding
        :rtype: ndarray(4)

        The bounding box is a 1D array [umin, umax, vmin, vmax].

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].bbox
            >>> blobs.bbox

        :note: The bounding box is the smallest box with vertical and
            horizontal edges that fully encloses the blob.

        :seealso: :meth:`umin` :meth:`vmin` :meth:`umax` :meth:`umax`,
        """
        return [b.bbox for b in self.data]

    @property
    @scalar_result
    def umin(self):
        """
        Minimum u-axis extent

        :return: maximum u-coordinate of the blob
        :rtype: int

        Returns the u-coordinate of the left side of the bounding box.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].umin
            >>> blobs.umin

        :seealso: :meth:`umax` :meth:`bbox`
        """
        return [b.bbox[0] for b in self.data]

    @property
    @scalar_result
    def umax(self):
        """
        Maximum u-axis extent

        :return: maximum u-coordinate of the blob
        :rtype: int

        Returns the u-coordinate of the right side of the bounding box.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].umin
            >>> blobs.umin

        :seealso: :meth:`umin` :meth:`bbox`
        """
        return [b.bbox[0] + b.bbox[2] for b in self.data]

    @property
    @scalar_result
    def vmin(self):
        """
        Maximum v-axis extent

        :return: maximum v-coordinate of the blob
        :rtype: int

        Returns the v-coordinate of the top side of the bounding box.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].vmin
            >>> blobs.vmin

        :seealso: :meth:`vmax` :meth:`bbox`
        """
        return [b.bbox[0] for b in self.data]

    @property
    @scalar_result
    def vmax(self):
        """
        Minimum v-axis extent

        :return: maximum v-coordinate of the blob
        :rtype: int

        Returns the v-coordinate of the bottom side of the bounding box.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].vmax
            >>> blobs.vmax

        :seealso: :meth:`vmin` :meth:`bbox`
        """
        return [b.bbox[1] + b.bbox[3] for b in self.data]

    @property
    @scalar_result
    def bboxarea(self):
        """
        Area of the bounding box

        :return: area of the bounding box in pixels
        :rtype: int

        Return the area of the bounding box which is invariant to blob
        position.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].bboxarea
            >>> blobs.bboxarea

        :note: The bounding box is the smallest box with vertical and
            horizontal edges that fully encloses the blob.

        :seealso: :meth:`bbox` :meth:`area` :meth:`fillfactor`
        """
        return [b.bbox[2] * b.bbox[3] for b in self.data]

    @property
    @scalar_result
    def fillfactor(self):
        r"""
        Fill factor, ratio of area to bounding box area

        :return: fill factor
        :rtype: int

        Return the ratio, :math:`\le 1`, of the blob area to the area of the
        bounding box. This is a simple shape metric which is invariant to blob
        position and scale.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].fillfactor
            >>> blobs.fillfactor

        :note: The bounding box is the smallest box with vertical and
            horizontal edges that fully encloses the blob.

        :seealso: :meth:`bbox`
        """
        return [b._moments.m00 / (b.bbox[2] * b.bbox[3]) for b in self.data]

    @property
    @scalar_result
    def a(self):
        """
        Radius of equivalent ellipse

        :return: largest ellipse radius
        :rtype: float

        Returns the major axis length which is invariant to blob position
        and orientation.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].a
            >>> blobs.a

        :seealso: :meth:`b` :meth:`aspect`
        """
        return [b.a for b in self.data]

    @property
    @scalar_result
    def b(self):
        """
        Radius of equivalent ellipse

        :return: smallest ellipse radius
        :rtype: float

        Returns the minor axis length which is invariant to blob position
        and orientation.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].b
            >>> blobs.b

        :seealso: :meth:`a` :meth:`aspect`
        """
        return [b.b for b in self.data]

    @property
    @scalar_result
    def aspect(self):
        r"""
        Blob aspect ratio

        :return: ratio of equivalent ellipse axes
        :rtype: float

        Returns the ratio of equivalent ellipse axis lengths, :math:`<1`, which
        is invariant to blob position, orientation and scale.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].aspect
            >>> blobs.aspect

        :seealso: :func:`a` :meth:`b`
        """
        return [b.b / b.a for b in self.data]

    @property
    @scalar_result
    def orientation(self):
        """
        Blob orientation

        :return: Orientation of equivalent ellipse (in radians)
        :rtype: float

        Returns the orientation of equivalent ellipse major axis with respect to
        the horizontal axis, which is invariant to blob position and scale.

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
    def touch(self):
        """
        Blob edge touch status

        :return: blob touches the edge of the image
        :rtype: bool

        Returns true if the blob touches the edge of the image.

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

        :seealso: :meth:`color` :meth:`parent` :meth:`children` :meth:`dotfile`
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
        the blob hierarchy. The background blob is black (0), the first-level
        child blobs are white (1), etc.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('multiblobs.png')
            >>> blobs = im.blobs()
            >>> blobs[2].color
            >>> blobs.color

        :seealso: :meth:`level` :meth:`parent` :meth:`children`
        """
        return [b.level & 1 for b in self.data]

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

        :seealso: :meth:`children` :meth:`level` :meth:`dotfile`
        """
        return [b.parent for b in self.data]

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

        :seealso: :meth:`parent` :meth:`level` :meth:`dotfile`
        """
        return [b.children for b in self.data]

    @property
    @array_result
    def moments(self):
        """
        Moments of blobs

        :return: moments of blobs
        :rtype: named tuple or list of named tuples

        Compute multiple moments of each blob and return them as a named tuple
        with attributes

        ==========================  ===============================================================================
        Moment type                 attribute name
        ==========================  ===============================================================================
        moments                     ``m00`` ``m10`` ``m01`` ``m20`` ``m11`` ``m02`` ``m30`` ``m21`` ``m12`` ``m03``
        central moments             ``mu20`` ``mu11`` ``mu02`` ``mu30`` ``mu21`` ``mu12`` ``mu03`` |
        normalized central moments  ``nu20`` ``nu11`` ``nu02`` ``nu30`` ``nu21`` ``nu12`` ``nu03`` |
        ==========================  ===============================================================================

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].moments.m00
            >>> blobs[0].moments.m10

        :seealso: :meth:`centroid` :meth:`humoments`
        """
        return [b._moments for b in self.data]

    @array_result
    def humoments(self):
        """
        Hu image moment invariants of blobs

        :return: Hu image moments
        :rtype: ndarray(7) or ndarray(N,7)

        Computes the seven Hu image moment invariants of the image.  These
        are a robust shape descriptor that is invariant to position, orientation
        and scale.

        Example:

        .. runblock:: pycon
            :precision: 4

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].humoments()

        :seealso: :meth:`moments`
        """

        def hu(b):
            m = b._moments
            phi = np.empty((7,))
            phi[0] = m.nu20 + m.nu02
            phi[1] = (m.nu20 - m.nu02) ** 2 + 4 * m.nu11**2
            phi[2] = (m.nu30 - 3 * m.nu12) ** 2 + (3 * m.nu21 - m.nu03) ** 2
            phi[3] = (m.nu30 + m.nu12) ** 2 + (m.nu21 + m.nu03) ** 2
            phi[4] = (m.nu30 - 3 * m.nu12) * (m.nu30 + m.nu12) * (
                (m.nu30 + m.nu12) ** 2 - 3 * (m.nu21 + m.nu03) ** 2
            ) + (3 * m.nu21 - m.nu03) * (m.nu21 + m.nu03) * (
                3 * (m.nu30 + m.nu12) ** 2 - (m.nu21 + m.nu03) ** 2
            )
            phi[5] = (m.nu20 - m.nu02) * (
                (m.nu30 + m.nu12) ** 2 - (m.nu21 + m.nu03) ** 2
            ) + 4 * m.nu11 * (m.nu30 + m.nu12) * (m.nu21 + m.nu03)
            phi[6] = (3 * m.nu21 - m.nu03) * (m.nu30 + m.nu12) * (
                (m.nu30 + m.nu12) ** 2 - 3 * (m.nu21 + m.nu03) ** 2
            ) + (3 * m.nu12 - m.nu30) * (m.nu21 + m.nu03) * (
                3 * (m.nu30 + m.nu12) ** 2 - (m.nu21 + m.nu03) ** 2
            )
            return phi

        return np.array([hu(b) for b in self.data])

    @property
    @scalar_result
    def perimeter_length(self):
        """
        Perimeter length of the blob

        :return: perimeter length in pixels
        :rtype: float

        Return the length of the blob's external perimeter.  This is an 8-way
        connected chain of edge pixels.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].perimeter_length
            >>> blobs.perimeter_length

        :note: The length of the internal perimeter is found from summing
            the external perimeter of each child blob.

        :seealso: :meth:`perimeter` :meth:`children`
        """
        return [b.perimeter_length for b in self.data]

    @property
    @scalar_result
    def circularity(self):
        r"""
        Blob circularity

        :return: circularity
        :rtype: float

        Circularity, computed as :math:`\rho = \frac{A}{4 \pi p^2} \le 1`.
        Circularity is one for a circular blob and < 1 for all other shapes,
        approaching zero for a line.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].circularity
            >>> blobs.circularity

        :note:  Kulpa's correction factor is applied to account for edge
            discretization:

            - Area and perimeter measurement of blobs in discrete binary pictures.
              Z.Kulpa. Comput. Graph. Image Process., 6:434-451, 1977.

            - Methods to Estimate Areas and Perimeters of Blob-like Objects: a
              Comparison. Proc. IAPR Workshop on Machine Vision Applications.,
              December 13-15, 1994, Kawasaki, Japan
              L. Yang, F. Albregtsen, T. Loennestad, P. Groettum

        :seealso: :meth:`area` :meth:`perimeter_length`
        """
        return [b.circularity for b in self.data]

    @property
    @array_result
    def perimeter(self):
        """
        Perimeter of the blob

        :return: Perimeter, one point per column
        :rtype: ndarray(2,N)

        Return the coordinates of the pixels that form the blob's external
        perimeter.  This is an 8-way connected chain of edge pixels.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].perimeter.shape
            >>> np.set_printoptions(threshold=10)
            >>> blobs[0].perimeter
            >>> blobs.perimeter

        .. plot::

            from machinevisiontoolbox import Image

            im = Image.Read("shark2.png")
            im.disp(darken=True)
            blobs = im.blobs()
            for blob in blobs:
                plt.plot(blob.perimeter[0], blob.perimeter[1], 'y-', linewidth=2)

        :note: The perimeter is not closed, that is, the first and last point
            are not the same.

        :seealso: :meth:`perimeter_approx` :meth:`perimeter_hull` :meth:`plot_perimeter` :meth:`polar`
        """
        return [b.perimeter for b in self.data]

    @array_result
    def perimeter_approx(self, epsilon=None):
        """
        Approximate perimeter of blob

        :param epsilon: maximum distance between the original curve and its approximation, default is exact contour
        :type epsilon: int
        :return: Perimeter, one point per column
        :rtype: ndarray(2,N) or list of ndarray(2,N)

        The result is a low-order polygonal approximation to the original
        perimeter.  Increasing ``epsilon`` reduces the number of perimeter points.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].perimeter.shape
            >>> blobs[0].perimeter_approx(5).shape
            >>> np.set_printoptions(threshold=10)
            >>> blobs[0].perimeter_approx(5)

        which in this case has reduced the number of perimeter points from
        471 to 15.

        To compute parameters of the area enclosed by the approximated perimeter we can
        first convert it to a :class:`~spatialmath.geom2d.Polygon2` object:

        .. runblock:: pycon
            :exclude: 1-3

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> from spatialmath import Polygon2
            >>> poly = Polygon2(blobs[0].perimeter_approx(5), close=True)
            >>> poly.area()
            >>> poly.moment(1, 0)  # first moment

        .. plot::

            from machinevisiontoolbox import Image

            im = Image.Read("shark2.png")
            im.disp(darken=True)
            blobs = im.blobs()
            for blob in blobs:
                perim = blob.perimeter_approx(5)
                plt.plot(perim[0], perim[1], 'y.-')

        :note: The perimeter is not closed, that is, the first and last point
            are not the same.

        :seealso: :meth:`plot_perimeter` :meth:`perimeter` :meth:`perimeter_hull` :meth:`polar` `cv2.approxPolyDP <https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c>`_
        """
        perimeters = []
        for b in self.data:
            perimeter = cv.approxPolyDP(b.perimeter.T, epsilon=epsilon, closed=False)
            # result is Nx1x2
            perimeters.append(np.squeeze(perimeter).T)

        return perimeters

    @array_result
    def perimeter_hull(self, clockwise=True):
        """
        Convex hull of blob's perimeter

        :param clockwise: direction of travel for computing the hull, defaults to clockwise
        :type clockwise: bool
        :return: Perimeter, one point per column
        :rtype: ndarray(2,N) or list of ndarray(2,N)

        The result is a convex perimeter that minimally contains the blob.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs[0].perimeter.shape
            >>> blobs[0].perimeter_hull(5).shape
            >>> np.set_printoptions(threshold=10)
            >>> blobs[0].perimeter_hull()

        To compute parameters of the area enclosed by the convex hull we can first
        convert it to a :class:`~spatialmath.geom2d.Polygon2` object:

        .. runblock:: pycon
            :exclude: 1-3

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> from spatialmath import Polygon2
            >>> poly = Polygon2(blobs[0].perimeter_hull(), close=True)
            >>> poly.area()
            >>> poly.moment(1, 0)  # first moment

        .. plot::

            from machinevisiontoolbox import Image

            im = Image.Read("shark2.png")
            im.disp(darken=True)
            blobs = im.blobs()
            for blob in blobs:
                perim = blob.perimeter_hull()
                plt.plot(perim[0], perim[1], 'y.-')

        :note: The perimeter is not closed, that is, the first and last point
            are not the same.

        :seealso: :meth:`plot_perimeter` :meth:`perimeter` :meth:`perimeter_approx` :meth:`perimeter_approx` :meth:`polar` `cv2.convexHull <https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga014b28e56cb8854c0de4a211cb2be656>`_
        """
        perimeters = []
        for b in self.data:
            perimeter = cv.convexHull(
                self.perimeter.T, returnPoints=True, clockwise=clockwise
            )
            perimeters.append(np.squeeze(perimeter).T)
        return perimeters

    @array_result
    def polar(self, N=400):
        r"""
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

        :note: The points are evenly spaced around the perimeter but are
            not evenly spaced in subtended angle.

        :seealso: :meth:`polarmatch` :meth:`perimeter`
        """

        def polarfunc(b):
            contour = np.array(b.perimeter) - np.c_[b.p].T

            r = np.sqrt(np.sum(contour**2, axis=0))
            theta = -np.arctan2(contour[1, :], contour[0, :])

            s = np.linspace(0, 1, len(r))
            si = np.linspace(0, 1, N)

            f_r = sp.interpolate.interp1d(s, r)
            f_theta = sp.interpolate.interp1d(s, theta)

            return np.array((f_r(si), f_theta(si)))

        return [polarfunc(b) for b in self]

    def polarmatch(self, target):
        r"""
        Compare polar profiles

        :param target: the blob index to match against
        :type target: int
        :return: similarity and orientation offset
        :rtype: ndarray(N), ndarray(N)

        Performs cross correlation between the polar profiles of blobs.  All
        blobs are matched against blob index ``target``.  Blob index ``target``
        is included in the results.

        There are two return values:

        1. Similarity is a 1D array, one entry per blob, where a value of one
           indicates maximum similarity irrespective of orientation and scale.
        2. Orientation offset is a 1D array, one entry per blob, is the relative
           orientation of blobs with respect to the ``target`` blob.  The
           ``target`` blob has an orientation offset of 0.5. These values lie in
           the range [0, 1), equivalent to :math:`[0, 2\pi)` and wraps around.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> blobs.polarmatch(1)

        :note:
            - Can be considered as matching two functions defined over :math:`S^1`.
            - Orientation is obtained by cross-correlation of the polar-angle
              profile.

        :seealso: :meth:`polar` :meth:`contour`
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
        n = R.shape[1]

        # get the target profile
        target = R[target, :]

        # cross correlate, with wrapping
        out = sp.ndimage.correlate1d(R, target, axis=1, mode="wrap") / n
        idx = np.argmax(out, axis=1)
        return [out[k, idx[k]] for k in range(len(self))], idx / n

    def plot_box(self, **kwargs):
        """
        Plot a bounding box for the blob using Matplotlib

        :param kwargs: arguments passed to ``plot_box``

        Plot the bounding box of a blob or blobs on the current Matplotlib axes.

        Example:

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read("sharks.png")
            >>> blobs = im.blobs()
            >>> blobs[:3].plot_box(color="g")
            >>> blobs[3].plot_box(color="r", linewidth=4)

        .. plot::

            from machinevisiontoolbox import Image

            im = Image.Read("sharks.png")
            im.disp()
            blobs = im.blobs()
            blobs[:3].plot_box(color='g')
            blobs[3].plot_box(color='r', linewidth=4)

        :seealso: :meth:`plot_labelbox` :meth:`plot_centroid` :meth:`plot_perimeter` :func:`~machinevisiontoolbox.base.graphics.plot_box`
        """

        for blob in self:
            plot_box(lrbt=blob.bbox, **kwargs)

    def plot_labelbox(self, label=None, **kwargs):
        """
        Plot a labelled bounding box of blobs using Matplotlib

        :param label: label to be displayed on the bounding box, defaults to blob id
        :type label: str, optional
        :param kwargs: arguments passed to ``plot_labelbox``

        Plot a labelled bounding box for every blob described by this object.

        By default, blobs are labeled by their blob id.

        Example:

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read("sharks.png")
            >>> blobs = im.blobs()
            >>> blobs[:3].plot_labelbox(color="yellow")
            >>> blobs[3].plot_labelbox(color="lightblue", linewidth=2, label="3")

        .. plot::

            from machinevisiontoolbox import Image

            im = Image.Read("sharks.png")
            im.disp()
            blobs = im.blobs()
            blobs[:3].plot_labelbox(color="yellow")
            blobs[3].plot_labelbox(color="lightblue", linewidth=2, label="3")

        :seealso: :meth:`plot_box` :meth:`plot_centroid` :meth:`plot_perimeter` :func:`~machinevisiontoolbox.base.graphics.plot_labelbox`
        """

        for blob in enumerate(self):
            if label is None:
                label = f"{blob.id}"
            plot_labelbox(text=label, lrbt=blob.bbox, **kwargs)

    def plot_centroid(self, label=False, **kwargs):
        """
        Plot the centroid of blobs using Matplotlib

        :param label: add a sequential numeric label to each point, defaults to False
        :type label: bool
        :param kwargs: other arguments passed to ``plot_point``

                Plot the major and minor axes of a blob or blobs on the current Matplotlib axes.

        If no marker style is given then it will be an overlaid "o" and "x"
        in blue.

        Example:

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read("sharks.png")
            >>> blobs = im.blobs()
            >>> blobs[:3].plot_centroid()
            >>> blobs[3].plot_centroid(marker="P", markeredgecolor="lightsteelblue", markerfacecolor="w", fillstyle="full")

        .. plot::

            from machinevisiontoolbox import Image

            im = Image.Read("sharks.png")
            im.disp()
            blobs = im.blobs()
            blobs[:3].plot_centroid()
            blobs[3].plot_centroid(marker="P", markeredgecolor="red", markerfacecolor="w", fillstyle="full")

        :seealso: :meth:`plot_box` :meth:`plot_perimeter` :func:`~machinevisiontoolbox.base.graphics.plot_point`
        """
        if label:
            text = "{:d}"
        else:
            text = ""

        if "marker" not in kwargs:
            kwargs["marker"] = ["bx", "bo"]
            kwargs["fillstyle"] = "none"
        for i, blob in enumerate(self):
            plot_point(pos=blob.centroid, text=text.format(i), **kwargs)

    def plot_perimeter(
        self, show: str = "full", epsilon=None, clockwise=True, **kwargs
    ):
        """
        Plot the perimeter of blobs using Matplotlib

        :param show: type of perimeter to plot, "full" (default), "approx" or "hull"
        :type show: str
        :param epsilon: maximum distance between the original curve and its approximation, default is exact contour
        :type epsilon: int  (only for ``show="approx"``)
        :param clockwise: direction of travel for computing the hull, defaults to clockwise
        :type clockwise: bool (only for ``show="hull"``)
        :param kwargs: line style parameters passed to ``plot``

        Plots the perimeter of blob or blobs on the current Matplotlib axes.

        Example:

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read("sharks.png")
            >>> blobs = im.blobs()
            >>> blobs[:3].plot_perimeter(color="red")
            >>> blobs[3].plot_perimeter(show="hull", color="orange", linewidth=3)

        .. plot::

            from machinevisiontoolbox import Image

            im = Image.Read("sharks.png")
            im.disp()
            blobs = im.blobs()
            blobs[:3].plot_perimeter(color="red")
            blobs[3].plot_perimeter(show="hull", color="orange", linewidth=3)

        :seealso: :meth:`perimeter` :meth:`perimeter_approx` :meth:`perimeter_hull` :meth:`plot_box` :meth:`plot_centroid`
        """
        if show == "full":
            perims = self.perimeter
        elif show == "approx":
            perims = self.perimeter_approx(epsilon=epsilon)
        elif show == "hull":
            perims = self.perimeter_hull(clockwise=clockwise)
        else:
            raise ValueError("unknown perimeter type")

        if not isinstance(perims, list):
            perims = [perims]
        for perim in perims:
            plt.plot(perim[0], perim[1], **kwargs)

    def plot_ellipse(self, **kwargs):
        """
        Plot the equivalent ellipses of blobs using Matplotlib

        :param kwargs: line style parameters passed to ``plot``

        Plots the equivalent ellipses of blob or blobs on the current
        Matplotlib axes.

        Example:

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read("sharks.png")
            >>> blobs = im.blobs()
            >>> blobs[:3].plot_ellipse(color="yellow")
            >>> blobs[3].plot_ellipse(color="green", linestyle="--", linewidth=3)

        .. plot::

            from machinevisiontoolbox import Image

            im = Image.Read("sharks.png")
            im.disp()
            blobs = im.blobs()
            blobs[:3].plot_ellipse(color="yellow")
            blobs[3].plot_ellipse(color="green", linestyle="--", linewidth=3)

        :seealso: :meth:`plot_axes` :meth:`plot_box` :meth:`plot_centroid` :func:`~spatialmath.base.plot_ellipse`
        """
        for blob in self:
            m = blob._moments
            # fmt: off
            J = np.array([
                [m.mu20, m.mu11], 
                [m.mu11, m.mu02]])
            # fmt: on
            base.plot_ellipse(
                4 * J / m.m00, centre=blob.centroid, inverted=True, **kwargs
            )

    def blob_frame(self):
        """
        Transformation from blob coordinate frame to image frame

        :return: Homogeneous transformation
        :rtype: :class:`~spatialmath.SE2`

        Returns the SE(2) transformation that maps point coordinates in the blob
        coordinate frame (origin at the centroid, x- and y-axes aligned with the major
        and minor ellipse axes) to their coordinate in the image coordinate frame.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('sharks.png')
            >>> blobs = im.blobs()
            >>> blobs.blob_frame()

        :seealso: :meth:`centroid` :meth:`orientation`
        """
        frames = SE2.Empty()
        for blob in self:
            frames.append(SE2(*blob.centroid, blob.orientation))
        return frames

    def plot_axes(self, **kwargs):
        """
        Plot equivalent ellipse axes of blobs using Matplotlib

        :param kwargs: line style parameters passed to ``plot``

        Plot the major and minor axes of a blob or blobs on the current Matplotlib axes.
        These are the axes of the equivalent ellipse and the intersection point is
        the blob's centroid.

        Example:

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read("sharks.png")
            >>> blobs = im.blobs()
            >>> blobs[:3].plot_axes(color="blue")
            >>> blobs[3].plot_axes(color="green", linewidth=3)

        .. plot::

            from machinevisiontoolbox import Image

            im = Image.Read("sharks.png")
            im.disp()
            blobs = im.blobs()
            blobs[:3].plot_axes(color="blue")
            blobs[3].plot_axes(color="green", linewidth=3)

        :seealso: :meth:`plot_ellipse` :meth:`plot_box` :meth:`plot_centroid`
        """
        for blob in self:
            T = SE2(*blob.centroid, blob.orientation)
            # fmt: off
            a_axis = np.array(  # major axis is parallel to x-axis
                [
                    [blob.a, -blob.a],
                    [0,       0],
                ]
            )
            b_axis = np.array(  # minor axis is parallel to y-axis
                [
                    [0,       0],
                    [blob.b, -blob.b],
                ]
            )
            # fmt: on
            p = T * a_axis
            plt.plot(p[0, :], p[1, :], **kwargs)

            p = T * b_axis
            plt.plot(p[0, :], p[1, :], **kwargs)

    @array_result
    def aligned_box(self):
        """
        Compute rectangle aligned with ellipse axes for blobs

        :return: tuple of area, centroid, vertices of the aligned box
        :rtype: tuple or list of tuples

        Compute the minimal enclosing box whose sides are parallel to the axes of the
        equivalent ellipse.  Return a list of vertices (not closed) and a list
        of box centroids.

        Example:

        .. runblock:: pycon
            :precision: 4

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read("sharks.png")
            >>> blobs = im.blobs()
            >>> blobs[0].aligned_box()  # downward shark

        .. plot::

            from machinevisiontoolbox import Image

            im = Image.Read("sharks.png")
            im.disp()

        :seealso: :meth:`plot_aligned_box` :meth:`plot_axes` :meth:`plot_box` :meth:`plot_centroid`
        """
        boxes = []
        for blob in self:
            T = SE2(*blob.centroid, blob.orientation)

            # transform perimeter to centroid coordinate frame
            p = T.inv() * blob.perimeter
            xmin = p[0, :].min()
            xmax = p[0, :].max()
            ymin = p[1, :].min()
            ymax = p[1, :].max()
            v = np.array(
                [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]
            ).T

            boxes.append(
                (
                    (xmax - xmin) * (ymax - ymin),
                    T * np.array([(xmin + xmax) / 2, (ymin + ymax) / 2]),
                    T * v,
                )
            )

        return boxes

    def plot_aligned_box(self, **kwargs):
        """
        Plot aligned rectangles of blobs using Matplotlib

        :param kwargs: line style parameters passed to ``plot``

                Compute the minimal enclosing box whose sides are parallel to the axes of the
        equivalent ellipse.  Return a list of vertices (not closed) and a list
        of box centroids.
        Highlights the perimeter of a blob or blobs on the current plot.

        Example:

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read("sharks.png")
            >>> blobs = im.blobs()
            >>> blobs[:3].plot_aligned_box(color="red")
            >>> blobs[3].plot_aligned_box(color="yellow", linestyle="--", linewidth=3)

        .. plot::

            from machinevisiontoolbox import Image

            im = Image.Read("sharks.png")
            im.disp()
            blobs = im.blobs()
            blobs[:3].plot_aligned_box(color="red")
            blobs[3].plot_aligned_box(color="yellow", linestyle="--", linewidth=3)

        :seealso: :meth:`plot_box` :meth:`plot_centroid`
        """
        boxes = self.aligned_box()
        for box in boxes:
            base.plot_polygon(box[2], close=True, **kwargs)

    def label_image(self, image=None):
        """
        Create label image from blobs

        :param image: image to draw into, defaults to new image
        :type image: :class:`Image`, optional
        :return: greyscale label image
        :rtype: :class:`Image`

        The perimeter information from the blobs is used to generate a greyscale
        label image where the greyvalue of each region corresponds to the blob
        index.

            Example:

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read("sharks.png")
            >>> blobs = im.blobs()
            >>> labels = blobs.label_image()
            >>> labels.disp(colorbar=True)

            .. plot::

                from machinevisiontoolbox import Image

                im = Image.Read("sharks.png")
                im.disp()
                blobs = im.blobs()
                labels = blobs.label_image()
                labels.disp(colorbar=True)

        .. note:: The label image is reconstituted from the OpenCV contours that are
            saved within the :class:`Blobs` object.

        :seealso: :meth:`~machinevisiontoolbox.ImageSpatial.labels_binary`
        """

        if image is None:
            image = self._image

        # TODO check contours, icont, colors, etc are valid
        # done because we squeezed hierarchy from a (1,M,4) to an (M,4) earlier

        labels = np.zeros(image.shape[:2], dtype=np.uint8)
        for i in range(len(self)):
            # TODO figure out how to draw alpha/transparencies?
            cv.drawContours(
                image=labels,
                contours=self._contours_raw,
                contourIdx=i,
                color=i + 1,
                thickness=-1,  # fill the contour
                hierarchy=self._hierarchy_raw,
            )

        return image.__class__(labels)

    def dotfile(self, filename=None, direction=None, show=False):
        """
        Create a GraphViz dot file

        :param filename: filename to save graph to, defaults to None
        :type filename: str, optional
        :param direction: graph drawing direction, defaults to top to bottom
        :type direction: str, optional
        :param show: compile the graph and display in browser tab, defaults to False
        :type show: bool, optional

        Creates the specified file which contains the `GraphViz
        <https://graphviz.org>`_ code to represent the blob hierarchy as a
        directed graph.  By default output is to the console.

        :note: If ``filename`` is a file object then the file will *not*
            be closed after the GraphViz model is written.

        :seealso: :meth:`child` :meth:`parent` :meth:`level`
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
            print(
                '  "{:d}" -> "{:d}"'.format(blob.parent.id if blob.parent else -1, id),
                file=f,
            )

        print("}", file=f)

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
        Find and describe blobs in image

        :return: blobs in the image
        :rtype: :class:`Blobs`

        Find all blobs in the image and return an object that contains geometric
        information about them. The object behaves like a list so it can
        be indexed and sliced.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('shark2.png')
            >>> blobs = im.blobs()
            >>> type(blobs)
            >>> len(blobs)
            >>> print(blobs)

        :references:
            - Robotics, Vision & Control for Python, Section 12.1.2.1, P. Corke, Springer 2023.
        """

        # TODO do the feature extraction here
        # each blob is a named tuple??
        # This could be applied to MSERs
        return Blobs(self, **kwargs)


if __name__ == "__main__":
    from machinevisiontoolbox import Image
    import matplotlib.pyplot as plt

    im = Image.Read("sharks.png")
    blobs = im.blobs()
    # frames = SE2.Empty()
    # for blob in blobs:
    #     frames.append(SE2(*blob.centroid, blob.orientation))
    frames = blobs.blob_frame()
    print(frames)
    print(blobs[1].moments.m00)
    print(blobs.humoments())

    # im.disp()
    # blobs = im.blobs()
    # blobs[:3].plot_perimeter(color="red")
    # blobs[3].plot_perimeter(which="hull", color="orange", linewidth=3)
    # plt.show(block=True)

    # from machinevisiontoolbox import Image

    # im = Image.Read("sharks.png")
    # im.disp()
    # blobs = im.blobs()
    # blobs.plot_centroid()
    # blobs[3].plot_centroid(
    #     marker="D",
    #     markeredgecolor="lightsteelblue",
    #     markerfacecolor="w",
    #     fillstyle="full",
    # )

    # im = Image.Read("multiblobs.png")

    # f = im.blobs()
    # # z = f.label_image()

    # labels = f.label_image()
    # labels.disp(
    #     colormap="viridis",
    #     ncolors=10,
    #     colorbar=dict(shrink=0.8, aspect=20 * 0.8),
    #     block=True,
    # )
    # pass

    # im = Image.Read('sharks.png')

    # im.disp()
    # blobs=im.blobs()
    # print(blobs)

    # blobs.plot_box(color="red")

    # # blobs = Blobs()
    # # print(len(blobs))

    # blobs = im.blobs()
    # print(len(blobs))
    # print(blobs[0].area)
    # print(blobs.area)

    # print(blobs)

    # print(blobs.level)
    # print(blobs.color)
    # print(blobs[1].children)
    # print(blobs.p)
    # print(blobs.moments)
    # print(blobs.humoments)

    # print(blobs[(3,2,1)])
    # print(blobs[np.r_[3,2,1]])
    # print(blobs[blobs.circularity > 0.8])

    # print(blobs.sortby())
    # print(blobs.sortby(reverse=True))
    # print(blobs.sortby(by="circularity"))

    # print(blobs.filter(circularity=0.8))

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
