#!/usr/bin/env python
"""
SIFT feature class
@author: Dorian Tsai
@author: Peter Corke
"""

# https://docs.opencv.org/4.4.0/d7/d60/classcv_1_1SIFT.html


import numpy as np
import math

import cv2 as cv
import matplotlib.pyplot as plt
from ansitable import ANSITable, Column
import spatialmath.base as smb
from machinevisiontoolbox.base import (
    findpeaks2d,
    draw_circle,
    draw_line,
    draw_point,
    color_bgr,
)

# from machinevisiontoolbox.classes import Image

# from machinevisiontoolbox.Image import *
# from machinevisiontoolbox.Image import Image


# TODO, either subclass SIFTFeature(BaseFeature2D) or just use BaseFeature2D
# directly

# decorators
def scalar_result(func):
    def innerfunc(*args):
        out = func(*args)
        if len(out) == 1:
            return out[0]
        else:
            return np.array(out)

    inner = innerfunc
    inner.__doc__ = func.__doc__  # pass through the doc string
    return inner


def array_result(func):
    def innerfunc(*args):
        out = func(*args)
        if len(out) == 1:
            return out[0]
        else:
            return np.array(out)

    inner = innerfunc
    inner.__doc__ = func.__doc__  # pass through the doc string
    return inner


def array_result2(func):
    def innerfunc(*args):
        out = func(*args)
        if len(out) == 1:
            return out[0].flatten()
        else:
            return np.squeeze(np.array(out)).T

    inner = innerfunc
    inner.__doc__ = func.__doc__  # pass through the doc string
    return inner


class BaseFeature2D:
    """
    A 2D point feature class
    """

    def __init__(
        self,
        kp=None,
        des=None,
        scale=False,
        orient=False,
        image=None,
    ):
        """
        Create set of 2D point features

        :param kp: list of :obj:`opencv.KeyPoint` objects, one per feature, defaults to None
        :type kp: list of N elements, optional
        :param des: Feature descriptor, each is an M-vector, defaults to None
        :type des: ndarray(N,M), optional
        :param scale: features have an inherent scale, defaults to False
        :type scale: bool, optional
        :param orient: features have an inherent orientation, defaults to False
        :type orient: bool, optional

        A :class:`~machinevisiontoolbox.ImagePointFeatures.BaseFeature2D` object:

            - has a length, the number of feature points it contains
            - can be sliced to extract a subset of features

        This object behaves like a list, allowing indexing, slicing and
        iteration over individual features.  It also supports a number of
        convenience methods.

        :note: OpenCV consider feature points as :obj:`opencv.KeyPoint` objects and the
            descriptors as a multirow NumPy array.  This class provides a more
            convenient abstraction.
        """

        # TODO flesh out sortby option, it can be by strength or scale
        # TODO what does nfeatures option to SIFT do? seemingly nothing

        self._has_scale = scale
        self._has_orient = orient
        self._image = image

        if kp is None:
            # initialise empty feature object
            self._feature_type = None
            self._kp = None
            self._descriptor = None

        else:
            self._kp = kp
            self._descriptor = des

    def __len__(self):
        """
        Number of features (base method)

        :return: number of features
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> orb = img.ORB()
            >>> len(orb)  # number of features

        :seealso: :meth:`.__getitem__`
        """
        return len(self._kp)

    def __getitem__(self, i):
        """
        Get item from point feature object (base method)

        :param i: index
        :type i: int or slice
        :raises IndexError: index out of range
        :return: subset of point features
        :rtype: BaseFeature2D instance

        This method allows a ``BaseFeature2D`` object to be indexed, sliced or iterated.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> orb = img.ORB()
            >>> print(orb[:5])  # first 5 ORB features
            >>> print(orb[::50])  # every 50th ORB feature

        :seealso: :meth:`.__len__`
        """
        new = self.__class__()

        new._has_scale = self._has_scale
        new._has_orient = self._has_orient

        # index or slice the keypoint list
        if isinstance(i, int):
            new._kp = [self._kp[i]]
        elif isinstance(i, slice):
            new._kp = self._kp[i]
        elif isinstance(i, np.ndarray):
            if np.issubdtype(i.dtype, bool):
                new._kp = [self._kp[k] for k, true in enumerate(i) if true]
            elif np.issubdtype(i.dtype, np.integer):
                new._kp = [self._kp[k] for k in i]
        elif isinstance(i, (list, tuple)):
            new._kp = [self._kp[k] for k in i]

        # index or slice the descriptor array
        if len(self._descriptor.shape) == 1:
            new._descriptor = self._descriptor
        else:
            new._descriptor = self._descriptor[i, :]

        return new

    def __str__(self):
        """
        String representation of feature (base method)

        :return: string representation
        :rtype: str

        For a feature object of length one display the feature type, position,
        strength and id.  For a feature object with multiple features display
        the feature type and number of features.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> orb = img.BRISK()
            >>> orb
            >>> orb[0]  # feature 0
        """
        if len(self) > 1:
            return f"{self.__class__.__name__} features, {len(self)} points"
        else:
            s = (
                f"{self.__class__.__name__}: ({self.u:.1f}, {self.v:.1f}),"
                f" strength={self.strength:.2f}"
            )
            if self._has_scale:
                s += f", scale={self.scale:.1f}"
            if self._has_orient:
                s += f", orient={self.orientation:.1f}°"
            s += f", id={self.id}"
            return s

    def __repr__(self):
        """
        Display features in readable form

        :return: string representation
        :rtype: str

        :seealso: :meth:`str`
        """
        return str(self)

    def list(self):
        """
        List matches

        Print the features in a simple format, one line per feature.

        :seealso: :meth:`table`
        """
        for i, f in enumerate(self):
            s = (
                f"{self._feature_type} feature {i}: ({f.u:.1f}, {f.v:.1f}),"
                f" strength={f.strength:.2f}"
            )
            if f._has_scale:
                s += f", scale={f.scale:.1f}"
            if f._has_orient:
                s += f", orient={f.orientation:.1f}°"
            s += f", id={f.id}"
            print(s)

    def table(self):
        """
        Print features in tabular form

        Each row is in the table includes: the index in the feature vector,
        centroid coordinate, feature strength, feature scale and image id.

        :seealso: :meth:`str`
        """
        columns = [Column("#"), Column("centroid"), Column("strength", fmt="{:.3g}")]
        if self._has_scale:
            columns.append(Column("scale", fmt="{:.3g}"))
        if self._has_orient:
            columns.append(Column("orient", fmt="{:.3g}°"))
        columns.append(Column("id", fmt="{:d}"))
        table = ANSITable(*columns, border="thin")
        for i, f in enumerate(self):
            values = [f.strength]
            if self._has_scale:
                values.append(f.scale)
            if self._has_orient:
                values.append(f.orientation)

            table.row(i, f"{f.u:.1f}, {f.v:.1f}", *values, f.id)
        table.print()

    def gridify(self, nbins, nfeat):
        """
        Sort features into grid

        :param nfeat: maximum number of features per grid cell
        :type nfeat: int
        :param nbins: number of grid cells horizontally and vertically
        :type nbins: int
        :return: set of gridded features
        :rtype: :class:`BaseFeature2D` instance

        Select features such that no more than ``nfeat`` features fall into each
        grid cell.  The image is divided into an ``nbins`` x ``nbins`` grid.

        .. warning:: Takes the first ``nfeat`` features in each grid cell, not the
            ``nfeat`` strongest.  Sort the features by strength to achieve this.

        :seealso: :meth:`sort`
        """

        try:
            nw, nh = nbins
        except:
            nw = nbins
            nh = nbins

        image = self._image
        binwidth = image.width // nw
        binheight = image.height // nh

        keep = []
        bins = np.zeros((nh, nw), dtype="int")

        for f in self.features:
            ix = f.p[0] // binwidth
            iy = f.p[1] // binheight

            if bins[iy, ix] < nfeat:
                keep.append(f)
                bins[iy, ix] += 1

        return self.__class__(keep)

    def __add__(self, other):
        """
        Add feature sets

        :param other: set of features
        :type other: :class:`BaseFeature2D`
        :raises TypeError: _description_
        :return: set of features
        :rtype: :class:`BaseFeature2D` instance

        Add two feature sets to form a new feature sets.  If ``other`` is
        equal to ``None`` or ``[]`` it is interpretted as an empty feature
        set, this is useful in a loop for aggregating feature sets.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img1 = Image.Read("eiffel-1.png")
            >>> img2 = Image.Read("eiffel-2.png")
            >>> orb = img1.ORB() + img2.ORB()
            >>> orb
            >>> orb = []
            >>> orb = img1.ORB() + orb
            >>> orb = img2.ORB() + orb
            >>> orb

        :seealso: :meth:`__radd__`
        """
        if isinstance(other, list) and len(other) == 0 or other is None:
            return self

        if self._feature_type != other._feature_type:
            raise TypeError(
                "cant add different feature types:",
                self._feature_type,
                other._feature_type,
            )
        new = self.__class__()
        new._feature_type = self._feature_type

        new._kp = self._kp + other._kp
        new._descriptor = np.vstack((self._descriptor, other._descriptor))

        return new

    def __radd__(self, other):
        """
        Add feature sets

        :param other: set of features
        :type other: :class:`BaseFeature2D`
        :raises TypeError: _description_
        :return: set of features
        :rtype: :class:`BaseFeature2D`

        Add two feature sets to form a new feature sets.  If ``other`` is
        equal to ``None`` or ``[]`` it is interpretted as an empty feature
        set, this is useful in a loop for aggregating feature sets.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img1 = Image.Read("eiffel-1.png")
            >>> img2 = Image.Read("eiffel-2.png")
            >>> orb = img1.ORB() + img2.ORB()
            >>> orb
            >>> orb = []
            >>> orb += img1.ORB()
            >>> orb += img2.ORB()
            >>> orb

        :seealso: :meth:`__add__`
        """
        if isinstance(other, list) and len(other) == 0:
            return self
        else:
            raise ValueError("bad")

    @property
    @scalar_result
    def u(self):
        """
        Horizontal coordinate of feature point

        :return: Horizontal coordinate
        :rtype: float or list of float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> orb = img.ORB()
            >>> orb[0].u
            >>> orb[:5].u

        """
        return [kp.pt[0] for kp in self._kp]

    @property
    @scalar_result
    def v(self):
        """
        Vertical coordinate of feature point

        :return: Vertical coordinate
        :rtype: float or list of float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> orb = img.ORB()
            >>> orb[0].v
            >>> orb[:5].v
        """
        return [kp.pt[1] for kp in self._kp]

    @property
    @scalar_result
    def id(self):
        """
        Image id for feature point

        :return: image id
        :rtype: int or list of int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> orb = img.ORB()
            >>> orb[0].id
            >>> orb[:5].id

        :note: Defined by the ``id`` attribute of the image passed to the
            feature detector
        """
        return [kp.class_id for kp in self._kp]

    @property
    @scalar_result
    def orientation(self):
        """
        Orientation of feature

        :return: Orientation in radians
        :rtype: float or list of float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> orb = img.ORB()
            >>> orb[0].orientation
            >>> orb[:5].orientation
        """
        # TODO should be in radians
        return [np.radians(kp.angle) for kp in self._kp]

    @property
    @scalar_result
    def scale(self):
        """
        Scale of feature

        :return: Scale
        :rtype: float or list of float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> orb = img.ORB()
            >>> orb[0].scale
            >>> orb[:5].scale
        """
        return [kp.size for kp in self._kp]

    @property
    @scalar_result
    def strength(self):
        """
        Strength of feature

        :return: Strength
        :rtype: float or list of float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> orb = img.ORB()
            >>> orb[0].strength
            >>> orb[:5].strength
        """
        return [kp.response for kp in self._kp]

    @property
    @scalar_result
    def octave(self):
        """
        Octave of feature

        :return: scale space octave containing the feature
        :rtype: float or list of float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> orb = img.ORB()
            >>> orb[0].octave
            >>> orb[:5].octave
        """
        return [kp.octave for kp in self._kp]

    @property
    @array_result
    def descriptor(self):
        """
        Descriptor of feature

        :return: Descriptor
        :rtype: ndarray(N,M)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> orb = img.ORB()
            >>> orb[0].descriptor.shape
            >>> orb[0].descriptor
            >>> orb[:5].descriptor.shape

        :note: For single feature return a 1D array vector, for multiple features return a set of column vectors.
        """
        return self._descriptor

    @property
    @array_result
    def p(self):
        """
        Feature coordinates

        :return: Feature centroids as matrix columns
        :rtype: ndarray(2,N)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> orb = img.ORB()
            >>> orb[0].p
            >>> orb[:5].p
        """
        return np.vstack([kp.pt for kp in self._kp]).T

    #         DEFAULT
    # Output image matrix will be created (Mat::create), i.e. existing memory of output image may be reused. Two source image, matches and single keypoints will be drawn. For each keypoint only the center point will be drawn (without the circle around keypoint with keypoint size and orientation).
    # DRAW_OVER_OUTIMG
    # Output image matrix will not be created (Mat::create). Matches will be drawn on existing content of output image.
    # NOT_DRAW_SINGLE_POINTS
    # Single keypoints will not be drawn.
    # DRAW_RICH_KEYPOINTS
    # For each keypoint the circle around keypoint with keypoint size and orientation will be drawn.

    # TODO def draw descriptors? (eg vl_feat, though mvt-mat doesn't have this)
    # TODO descriptor distance
    # TODO descriptor similarity
    # TODO display/print/char function?

    def distance(self, other, metric="L2"):
        """
        Distance between feature sets

        :param other: second set of features
        :type other: :class:`BaseFeature2D`
        :param metric: feature distance metric, one of "ncc", "L1", "L2" [default]
        :type metric: str, optional
        :return: distance between features
        :rtype: ndarray(N1, N2)

        Compute the distance matrix between two sets of feature. If the first set
        of features has length N1 and the ``other`` is of length N2, then
        compute an :math:`N_1 \times N_2` matrix where element
        :math:`D_{ij}` is the distance between feature :math:`i` in the
        first set and feature :math:`j` in the other set. The position of
        the closest match in row :math:`i` is the best matching feature to feature
        :math:`i`.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> orb1 = Image.Read("eiffel-1.png").ORB()
            >>> orb2 = Image.Read("eiffel-2.png").ORB()
            >>> dist = orb1.distance(orb2)
            >>> dist.shape

        :note:
            - The matrix is symmetric.
            - For the metric "L1" and "L2" the best match is the smallest distance
            - For the metric "ncc" the best match is the largest distance.  A value over
              0.8 is often considered to be a good match.

        :seealso: :meth:`match`
        """
        metric_dict = {"L1": 1, "L2": 2}

        n1 = len(self)
        n2 = len(other)
        D = np.empty((n1, n2))
        if n1 == 1:
            des1 = self._descriptor[np.newaxis, :]
        else:
            des1 = self._descriptor
        if n2 == 1:
            des2 = other._descriptor[np.newaxis, :]
        else:
            des2 = other._descriptor

        for i in range(n1):
            for j in range(n2):
                if metric == "ncc":
                    d = np.dot(des1[i, :], des2[j, :])
                else:
                    d = np.linalg.norm(des1[i, :] - des2[j, :], ord=metric_dict[metric])
                D[i, j] = d
                D[j, i] = d
        return D

    def match(
        self,
        other,
        ratio=0.75,
        crosscheck=False,
        metric="L2",
        sort=True,
        top=None,
        thresh=None,
    ):
        """
        Match point features

        :param other: set of feature points
        :type other: BaseFeature2D
        :param ratio: parameter for Lowe's ratio test, defaults to 0.75
        :type ratio: float, optional
        :param crosscheck: perform left-right cross check, defaults to False
        :type crosscheck: bool, optional
        :param metric: distance metric, one of: 'L1', 'L2' [default], 'hamming', 'hamming2'
        :type metric: str, optional
        :param sort: sort features by strength, defaults to True
        :type sort: bool, optional
        :raises ValueError: bad metric name provided
        :return: set of candidate matches
        :rtype: :class:`FeatureMatch` instance

        Return a match object that contains pairs of putative corresponding points.
        If ``crosscheck`` is True the ratio test is disabled

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> orb1 = Image.Read("eiffel-1.png").ORB()
            >>> orb2 = Image.Read("eiffel-2.png").ORB()
            >>> m = orb1.match(orb2)
            >>> len(m)

        :seealso: :class:`FeatureMatch` :meth:`distance`
        """

        # TODO: implement thresh

        # m = []

        # TODO check valid input
        # d1 and d2 must be numpy arrays
        # d1 and d2 must have equal (128 for SIFT) rows
        # d1 and d2 must have greater than 1 columns

        # do matching
        # sorting
        # return

        metricdict = {
            "L1": cv.NORM_L1,
            "L2": cv.NORM_L2,
            "hamming": cv.NORM_HAMMING,
            "hamming2": cv.NORM_HAMMING2,
        }
        if metric not in metricdict:
            raise ValueError("bad metric name")

        # create BFMatcher (brute force matcher) object
        # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        bf = cv.BFMatcher_create(metricdict[metric], crossCheck=crosscheck)

        # Match descriptors.
        # matches0 = bf.match(d1, d2)
        # there is also:
        if crosscheck:
            k = 1
        else:
            k = 2

        matches0 = bf.knnMatch(self.descriptor, other.descriptor, k=k)

        # the elements of matches are:
        #  queryIdx: first feature set (self)
        #  trainingIdx: second feature set (other)

        if not crosscheck:
            # apply ratio test
            good = []
            for m, n in matches0:
                if m.distance < ratio * n.distance:
                    good.append(m)
        else:
            # squeeze out the crosscheck failed matches
            good = [m[0] for m in matches0 if len(m) > 0]

        # Sort them in the order of increasing distance, best to worst match
        if sort or top is not None:
            good.sort(key=lambda x: x.distance)

        if top is not None:
            good = good[:top]

        # cv2.drawMatchesKnn expects list of lists as matches.
        # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)

        # Draw first 10 matches.
        # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)

        # opencv documentation for the descriptor matches
        # https://docs.opencv.org/4.4.0/d4/de0/classcv_1_1DMatch.html

        return FeatureMatch(
            [(m.queryIdx, m.trainIdx, m.distance) for m in good], self, other
        )

    def subset(self, N=100):
        """
        Select subset of features

        :param N: the number of features to select, defaults to 100
        :type N: int, optional
        :return: subset of features
        :rtype: :class:`BaseFeature2D` instance

        Return ``N`` features selected in constant steps from the input feature
        vector, ie. feature 0, s, 2s, etc.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> orb = Image.Read("eiffel-1.png").ORB()
            >>> len(orb)
            >>> orb2 = orb.subset(50)
            >>> len(orb2)
        """
        step = max(1, len(self) // N)
        k = list(range(0, len(self), step))
        k = k[:N]
        new = self[k]
        new._feature_type = self._feature_type
        return new

    def sort(self, by="strength", descending=True, inplace=False):
        """
        Sort features

        :param by: sort by ``'strength'`` [default] or ``'scale'``
        :type by: str, optional
        :param descending: sort in descending order, defaults to True
        :type descending: bool, optional
        :return: sorted features
        :rtype: :class:`BaseFeature2D` instance

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> orb = Image.Read("eiffel-1.png").ORB()
            >>> orb2 = orb.sort('strength')
            >>> orb2[:5].strength
        """
        # if by == 'strength':
        #     s = sorted(self, key=lambda f: f.strength, reverse=descending)
        # elif by == 'scale':
        #     s = sorted(self, key=lambda f: f.scale, reverse=descending)
        # else:
        #     raise ValueError('bad sort method', by)
        if by == "strength":
            key = self.strength
        elif by == "scale":
            key = self.scale
        else:
            raise ValueError("bad sort method", by)
        key = np.array(key)
        if descending:
            key = -key

        index = np.argsort(key)

        if inplace:
            self._kp = [self._kp[i] for i in index]
            self._descriptor = self._descriptor[index, :]
        else:
            new = self.__class__()
            new._kp = [self._kp[i] for i in index]
            new._descriptor = self._descriptor[index, :]
            new._feature_type = self._feature_type
            return new

    def support(self, images, N=50):
        """
        Find support region

        :param images: the image from which the feature was extracted
        :type images: :class:`Image` or list of :class:`Image`
        :param N: size of square window, defaults to 50
        :type N: int, optional
        :return: support region
        :rtype: :class:`Image` instance

        The support region about the feature's centroid is extracted,
        rotated and scaled.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> orb = img.ORB()
            >>> support = orb[0].support(img)
            >>> support

        :note: If the features come from multiple images then the feature's
            ``id`` attribute is used to index into ``images`` which must be a
            list of Image objects.
        """

        from machinevisiontoolbox import Image

        if len(self) > 1:
            raise ValueError("can only compute support region for single feature")

        if isinstance(images, Image):
            image = images.A
        else:
            # list or iterable
            image = images[self.id].A

        # M = smb.transl2(N/2, N/2) @ smb.trot2(self.orientation) @ smb.transl2(-self.u, -self.v)
        # M = M[:2, :] / self.scale * N / 2
        # translate to origin and rotate
        M = smb.trot2(self.orientation) @ smb.transl2(-self.u, -self.v)

        # scale it to fill the window
        M *= N / 2 / self.scale
        M[2, 2] = 1

        # translate to centre of window
        M = smb.transl2(N / 2, N / 2) @ M

        out = cv.warpAffine(src=image, M=M[:2, :], dsize=(N, N), flags=cv.INTER_LINEAR)
        return Image(out)

    def filter(self, **kwargs):
        """
        Filter features

        :param kwargs: the filter parameters
        :return: sorted features
        :rtype: :class:`BaseFeature2D` instance

        The filter is defined by arguments:

        ===============  ==================  ===================================
        argument         value               select if
        ===============  ==================  ===================================
        scale            (minimum, maximum)  minimum <= scale <= maximum
        minscale         minimum             minimum <= scale
        maxscale         maximum             scale <= maximum
        strength         (minimum, maximum)  minimum <= strength <= maximum
        minstrength      minimum             minimum <= strength
        percentstrength  percent             strength >= percent * max(strength)
        nstrongest       N                   strength
        ===============  ==================  ===================================

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> orb = Image.Read("eiffel-1.png").ORB()
            >>> len(orb)
            >>> orb2 = orb.filter(minstrength=0.001)
            >>> len(orb2)

        :note: If ``value`` is a range the ``numpy.Inf`` or ``-numpy.Inf``
            can be used as values.
        """

        features = self

        for filter, limits in kwargs.items():
            if filter == "scale":
                v = features.scale
                k = (limits[0] <= v) & (v <= limits[1])
            elif filter == "minscale":
                v = features.scale
                k = v >= limits
            elif filter == "maxscale":
                v = features.scale
                k = v <= limits
            elif filter == "strength":
                v = features.strength
                k = (limits[0] <= v) & (v <= limits[1])
            elif filter == "minstrength":
                v = features.strength
                k = limits >= v
            elif filter == "percentstrength":
                v = features.strength
                vmax = v.max()
                k = v >= vmax * limits / 100
            else:
                raise ValueError("unknown filter key", filter)

            features = features[k]

        return features

    def drawKeypoints(
        self,
        image,
        drawing=None,
        isift=None,
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        **kwargs,
    ):
        """
        Render keypoints into image

        :param image: original image
        :type image: :class:`Image`
        :param drawing: _description_, defaults to None
        :type drawing: _type_, optional
        :param isift: _description_, defaults to None
        :type isift: _type_, optional
        :param flags: _description_, defaults to cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        :type flags: _type_, optional
        :return: image with rendered keypoints
        :rtype: :class:`Image` instance

        If ``image`` is None then the keypoints are rendered over a black background.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> orb = img.ORB()
            >>> orb[0].p
            >>> orb[:5].p

        """
        # draw sift features on image using cv.drawKeypoints

        # check valid imagesource
        # TODO if max(self._u) or max(self._v) are greater than image width,
        # height, respectively, then raise ValueError

        # TODO check flags, setup dictionary or string for plot options

        if drawing is None:
            drawing = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        kp = self._kp

        if isift is None:
            isift = np.arange(0, len(self._kp))  # might need a +1 here
        else:
            isift = np.array(isift, ndmin=1, copy=True)

        # TODO should check that isift is consistent with kp (min value is 0,
        # max value is <= len(kp))
        cv.drawKeypoints(
            image.image,  # image, source image
            # kp[isift],
            kp,
            drawing,  # outimage
            flags=flags,
            **kwargs,
        )

        return image.__class__(drawing)

    def drawMatches(self, im1, sift1, im2, sift2, matches, **kwargs):
        # TODO should I just have input two SIFT objects,
        # or in this case just another SIFT object?

        # draw_params = dict(matchColor=(0, 255, 0),
        #                   singlePointColor=(255, 0, 0),
        #                   matchesMask=matches,
        #                   flags=0)

        out = cv.drawMatchesKnn(
            im1.image, sift1._kp, im2.image, sift2._kp, matches, None, **kwargs
        )

        return im1.__class__(out)

    def plot(
        self,
        *args,
        ax=None,
        filled=False,
        color="blue",
        alpha=1,
        hand=False,
        handcolor="blue",
        handthickness=1,
        handalpha=1,
        **kwargs,
    ):
        """
        Plot features using Matplotlib

        :param ax: axes to plot onto, defaults to None
        :type ax: axes, optional
        :param filled: shapes are filled, defaults to False
        :type filled: bool, optional
        :param hand: draw clock hand to indicate orientation, defaults to False
        :type hand: bool, optional
        :param handcolor: color of clock hand, defaults to 'blue'
        :type handcolor: str, optional
        :param handthickness: thickness of clock hand in pixels, defaults to 1
        :type handthickness: int, optional
        :param handalpha: transparency of clock hand, defaults to 1
        :type handalpha: int, optional
        :param kwargs: options passed to :obj:`matplotlib.Circle` such as color,
            alpha, edgecolor, etc.
        :type kwargs: dict

        Plot circles to represent the position and scale of features on a Matplotlib axis.
        Orientation, if applicable, is indicated by a radial line from the circle centre
        to the circumference, like a clock hand.

        """
        ax = smb.axes_logic(ax, 2)

        if filled:
            for kp in self:
                centre = kp.p.flatten()
                c = plt.Circle(
                    centre,
                    radius=kp.scale,
                    clip_on=True,
                    color=color,
                    alpha=alpha,
                    **kwargs,
                )
                ax.add_patch(c)
                if hand:
                    circum = (
                        centre
                        + kp.scale
                        * np.r_[math.cos(kp.orientation), math.sin(kp.orientation)]
                    )
                    l = plt.Line2D(
                        (centre[0], circum[0]),
                        (centre[1], circum[1]),
                        color=handcolor,
                        linewidth=handthickness,
                        alpha=handalpha,
                    )
                    ax.add_line(l)
        else:
            if len(args) == 0 and len(kwargs) == 0:
                kwargs = dict(marker="+y", markerfacecolor="none")
            smb.plot_point(self.p, *args, **kwargs)

    #     plt.draw()

    def draw(
        self,
        image,
        *args,
        ax=None,
        filled=False,
        color="blue",
        alpha=1,
        hand=False,
        handcolor="blue",
        handthickness=1,
        handalpha=1,
        **kwargs,
    ):
        """
        Draw features into image

        :param ax: axes to plot onto, defaults to None
        :type ax: axes, optional
        :param filled: shapes are filled, defaults to False
        :type filled: bool, optional
        :param hand: draw clock hand to indicate orientation, defaults to False
        :type hand: bool, optional
        :param handcolor: color of clock hand, defaults to 'blue'
        :type handcolor: str, optional
        :param handthickness: thickness of clock hand in pixels, defaults to 1
        :type handthickness: int, optional
        :param handalpha: transparency of clock hand, defaults to 1
        :type handalpha: int, optional
        :param kwargs: options passed to :obj:`matplotlib.Circle` such as color,
            alpha, edgecolor, etc.
        :type kwargs: dict

        Plot circles to represent the position and scale of features on a Matplotlib axis.
        Orientation, if applicable, is indicated by a radial line from the circle centre
        to the circumference, like a clock hand.

        """
        img = image.image
        if filled:
            for kp in self:
                centre = kp.p.flatten()

                draw_circle(
                    img,
                    centre,
                    radius=kp.scale,
                    clip_on=True,
                    color=color,
                    alpha=alpha,
                    **kwargs,
                )
                # draw_circle(img, centre, radius=kp.scale, clip_on=True, color=color, alpha=alpha, **kwargs)

                if hand:
                    circum = (
                        centre
                        + kp.scale
                        * np.r_[math.cos(kp.orientation), math.sin(kp.orientation)]
                    )
                    draw_line(
                        img,
                        (centre[0], circum[0]),
                        (centre[1], circum[1]),
                        color=handcolor,
                        thickness=handthickness,
                        alpha=handalpha,
                    )
        else:
            if len(args) == 0 and len(kwargs) == 0:
                kwargs = dict(marker="+y")
            draw_point(img, self.p, *args, fontsize=0.6, **kwargs)

    def draw2(self, image, color="y", type="point"):
        img = image.image
        if isinstance(color, str):
            color = color_bgr(color)

        options = {
            "rich": cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            "point": cv.DRAW_MATCHES_FLAGS_DEFAULT,
            "not": cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
        }

        cv.drawKeypoints(
            img,  # image, source image
            self._kp,
            img,  # outimage
            color=color,
            flags=options[type] + cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
        )

        return image.__class__(img)


class FeatureMatch:
    def __init__(self, m, fv1, fv2, inliers=None):
        """
        Create feature match object

        :param m: a list of match tuples (id1, id2, distance)
        :type m: list of tuples (int, int, float)
        :param fv1: first set of features
        :type fv1: :class:`BaseFeature2D`
        :param fv2: second set of features
        :type fv2: class:`BaseFeature2D`
        :param inliers: inlier status
        :type inliers: array_like of bool

        A :class:`FeatureMatch` object describes a set of correspondences
        between two feature sets. The object is constructed from two feature
        sets and a list of tuples ``(id1, id2, distance)`` where ``id1`` and
        ``id2`` are indices into the first and second feature sets. ``distance``
        is the distance between the feature's descriptors.

        A :class:`FeatureMatch` object:

            - has a length, the number of matches it contains
            - can be sliced to extract a subset of matches
            - inlier/outlier status of matches

        :note: This constructor would not be called directly, it is used by the
            ``match`` method of the :class:`BaseFeature2D` subclass.

        :seealso: :obj:`BaseFeature2D.match` `cv2.KeyPoint <https://docs.opencv.org/4.5.2/d2/d29/classcv_1_1KeyPoint.html#a507d41b54805e9ee5042b922e68e4372>`_
        """
        self._matches = m
        self._kp1 = fv1
        self._kp2 = fv2
        self._inliers = inliers
        self._inverse_dict1 = None
        self._inverse_dict2 = None

    def __getitem__(self, i):
        """
        Get matches from feature match object

        :param i: match subset
        :type i: int or Slice
        :raises IndexError: index out of range
        :return: subset of matches
        :rtype: Match instance

        Allow indexing, slicing or iterating over the set of matches

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> orb1 = Image.Read("eiffel-1.png").ORB()
            >>> orb2 = Image.Read("eiffel-2.png").ORB()
            >>> matches = orb1.match(orb2)
            >>> matches[:5]  # first 5 matches
            >>> matches[0]   # first match

        :seealso: :meth:`.__len__`
        """
        inliers = None
        if isinstance(i, int):
            matches = [self._matches[i]]
            if self._inliers is not None:
                inliers = self._inliers[i]
        elif isinstance(i, slice):
            matches = self._matches[i]
            if self._inliers is not None:
                inliers = self._inliers[i]
        elif isinstance(i, np.ndarray):
            if np.issubdtype(i.dtype, bool):
                matches = [m for m, g in zip(self._matches, i) if g]
                if self._inliers is not None:
                    inliers = [m for m, g in zip(self._inliers, i) if g]
            elif np.issubdtype(i.dtype, np.integer):
                matches = [self._matches[k] for k in i]
                if self._inliers is not None:
                    inliers = [self._inliers[k] for k in i]
        else:
            raise ValueError("bad index")
        return FeatureMatch(matches, self._kp1, self._kp2, inliers)

    def __len__(self):
        """
        Number of matches

        :return: number of matches
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> orb1 = Image.Read("eiffel-1.png").ORB()
            >>> orb2 = Image.Read("eiffel-2.png").ORB()
            >>> matches = orb1.match(orb2)
            >>> len(matches)

        :seealso: :meth:`.__getitem__`
        """
        return len(self._matches)

    def correspondence(self):
        """
        Feture correspondences

        :return: feature correspondences as array columns
        :rtype: ndarray(2,N)

        Return the correspondences as an array where each column contains
        the index into the first and second feature sets.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> orb1 = Image.Read("eiffel-1.png").ORB()
            >>> orb2 = Image.Read("eiffel-2.png").ORB()
            >>> matches = orb1.match(orb2)
            >>> matches.correspondence()
        """
        return np.array([m[:2] for m in self._matches]).T

    def by_id1(self, id):
        """
        Find match by feature id in first set

        :param id: id of feature in the first feature set
        :type id: int
        :return: match that includes feature ``id`` or None
        :rtype: :class:`FeatureMatch` instance containing one correspondence

        A :class:`FeatureMatch` object can contains multiple correspondences
        which are essentially tuples (id1, id2) where id1 and id2 are indices
        into the first and second feature sets that were matched. Each feature
        has a position, strength, scale and id.

        This method returns the match that contains the feature in the first
        feature set with specific ``id``. If no such match exists it returns
        None.

        :note:
            - For efficient lookup, on the first call a dict is built that maps
              feature id to index in the feature set.
            - Useful when features in the sets come from multiple images and
              ``id`` is used to indicate the source image.

        :seealso: :class:`BaseFeature2D` :obj:`BaseFeature2D.id` :meth:`by_id2`
        """
        if self._inverse_dict1 is None:
            # first call, build a dict for efficient mapping
            d = {}
            for k, m in enumerate(self._matches):
                d[m[0]] = k
            self._inverse_dict1 = d
        else:
            try:
                return self[self._inverse_dict1[id]]
            except KeyError:
                return None

    def by_id2(self, i):
        """
        Find match by feature id in second set

        :param id: id of feature in the second feature set
        :type id: int
        :return: match that includes feature ``id`` or None
        :rtype: :class:`FeatureMatch` instance containing one correspondence

        A :class:`FeatureMatch` object can contains multiple correspondences
        which are essentially tuples (id1, id2) where id1 and id2 are indices
        into the first and second feature sets that were matched. Each feature
        has a position, strength, scale and id.

        This method returns the match that contains the feature in the second
        feature set with specific ``id``. If no such match exists it returns
        None.

        :note:
            - For efficient lookup, on the first call a dict is built that maps
              feature id to index in the feature set.
            - Useful when features in the sets come from multiple images and
              ``id`` is used to indicate the source image.

        :seealso: :class:`BaseFeature2D` :obj:`BaseFeature2D.id` :meth:`by_id1`
        """
        if self._inverse_dict2 is None:
            # first call, build a dict for efficient mapping
            d = {}
            for k, m in enumerate(self._matches):
                d[m[0]] = k
            self._inverse_dict2 = d
        else:
            try:
                return self[self._inverse_dict2[i]]
            except KeyError:
                return None

    def __str__(self):
        """
        String representation of matches

        :return: string representation
        :rtype: str

        If the object contains a single correspondence, show the feature
        indices and distance metric.  For multiple correspondences, show
        summary data.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> orb1 = Image.Read("eiffel-1.png").ORB()
            >>> orb2 = Image.Read("eiffel-2.png").ORB()
            >>> matches = orb1.match(orb2)
            >>> str(matches)
            >>> str(matches[0])
        """

        if len(self) == 1:
            return (
                f"{self.status} {self.distance:6.2f}: ({self.p1[0, 0]:.1f},"
                f" {self.p1[1, 0]:.1f}) <--> ({self.p2[0, 0]:.1f}, {self.p2[1, 0]:.1f})"
            )
        else:
            s = f"{len(self)} matches"
            if self._inliers is not None:
                ninlier = sum(self._inliers)
                s += f", with {ninlier} ({ninlier/len(self)*100:.1f}%) inliers"
            return s

    def __repr__(self):
        """
        String representation of matches

        :return: string representation
        :rtype: str

        If the object contains a single correspondence, show the feature
        indices and distance metric.  For multiple correspondences, show
        summary data.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> orb1 = Image.Read("eiffel-1.png").ORB()
            >>> orb2 = Image.Read("eiffel-2.png").ORB()
            >>> matches = orb1.match(orb2)
            >>> matches
            >>> matches[0]
        """
        return str(self)

    @property
    @scalar_result
    def status(self):
        """
        Inlier status of matches

        :return: inlier status of matches
        :rtype: bool
        """
        if self._inliers is not None:
            return "+" if self._inliers else "-"
        else:
            return ""

    def list(self):
        """
        List matches

        Print the matches in a simple format, one line per match.

        :seealso: :meth:`table`
        """
        for i, m in enumerate(self._matches):
            # TODO shouldnt have to flatten
            p1 = self._kp1[m[0]].p.flatten()
            p2 = self._kp2[m[1]].p.flatten()
            if self._inliers is not None:
                status = "+" if self._inliers[i] else "-"
            else:
                status = ""
            s = (
                f"{i:3d}:  {status} {m[2]:6.2f} ({p1[0]:.1f}, {p1[1]:.1f}) <-->"
                f" ({p2[0]:.1f}, {p2[1]:.1f})"
            )
            print(s)

    def table(self):
        """
        Print matches in tabular form

        Each row in the table includes: the index of the match, inlier/outlier
        status, match strength, feature coordinates.

        :seealso: :meth:`__str__`
        """
        columns = [
            Column("#"),
            Column("inlier"),
            Column("strength", fmt="{:.3g}"),
            Column("p1", colalign="<", fmt="{:s}"),
            Column("p2", colalign="<", fmt="{:s}"),
        ]
        table = ANSITable(*columns, border="thin")

        for i, m in enumerate(self._matches):
            # TODO shouldnt have to flatten
            p1 = self._kp1[m[0]].p.flatten()
            p2 = self._kp2[m[1]].p.flatten()
            if self._inliers is not None:
                status = "+" if self._inliers[i] else "-"
            else:
                status = ""
            table.row(
                i,
                status,
                m[2],
                f"({p1[0]:.1f}, {p1[1]:.1f})",
                f"({p2[0]:.1f}, {p2[1]:.1f})",
            )
        table.print()

    @property
    def inliers(self):
        """
        Extract inlier matches

        :return: new match object containing only the inliers
        :rtype: :class:`FeatureMatch` instance

        :note: Inlier/outlier status is typically set by some RANSAC-based
            algorithm that applies a geometric constraint to the sets of
            putative matches.

        :seealso: :obj:`CentralCamera.points2F`
        """
        return self[self._inliers]

    @property
    def outliers(self):
        """
        Extract outlier matches

        :return: new match object containing only the outliers
        :rtype: :class:`FeatureMatch` instance

        .. note:: Inlier/outlier status is typically set by some RANSAC-based
            algorithm that applies a geometric constraint to the sets of
            putative matches.

        :seealso: :obj:`entralCamera.points2F`
        """
        return self[~self._inliers]

    def subset(self, N=100):
        """
        Select subset of features

        :param N: the number of features to select, defaults to 10
        :type N: int, optional
        :return: feature vector
        :rtype: BaseFeature2D

        Return ``N`` features selected in constant steps from the input feature
        vector, ie. feature 0, s, 2s, etc.
        """
        if len(self) < N:
            # fewer than N features, return them all
            return self
        else:
            # choose N, approximately evenly spaced
            k = np.round(np.linspace(0, len(self) - 1, N)).astype(int)
            return self[k]

    def plot(self, *pos, darken=False, ax=None, width=None, block=False, **kwargs):
        """
        Plot matches

        :param darken: darken the underlying , defaults to False
        :type darken: bool, optional
        :param width: figure width in millimetres, defaults to Matplotlib default
        :type width: float, optional
        :param block: Matplotlib figure blocks until window closed, defaults to False
        :type block: bool, optional

        Displays the original pair of images side by side, as greyscale images,
        and overlays the matches.

        """

        kp1 = self._kp1
        kp2 = self._kp2
        im1 = kp1._image
        im2 = kp2._image

        combo, u = im1.__class__.Hstack((im1.mono(), im2.mono()), return_offsets=True)
        if ax is None:
            combo.disp(darken=darken, width=width, block=None)
        else:
            plt.sca(ax)

        # for m in self:
        #     p1 = m.pt1
        #     p2 = m.pt2
        #     plt.plot((p1[0], p2[0] + u[1]), (p1[1], p2[1]), *pos, **kwargs)
        p1 = self.p1
        p2 = self.p2
        plt.plot((p1[0, :], p2[0, :] + u[1]), (p1[1, :], p2[1, :]), *pos, **kwargs)
        if plt.isinteractive():
            plt.show(block=block)

    def plot_correspondence(self, *arg, offset=(0, 0), **kwargs):
        p1 = self.p1
        p2 = self.p2
        plt.plot(
            (p1[0, :], p2[0, :] + offset[0]),
            (p1[1, :], p2[1, :] + offset[1]),
            *arg,
            **kwargs,
        )
        plt.draw()

    def estimate(self, func, method="ransac", **args):

        solution = func(self.p1, self.p2, method=method, **args)
        self._inliers = solution[-1]

        return solution[:-1]

    @property
    @scalar_result
    def distance(self):
        """
        Distance between corresponding features

        :return: _description_
        :rtype: float or ndarray(N)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> orb1 = Image.Read("eiffel-1.png").ORB()
            >>> orb2 = Image.Read("eiffel-2.png").ORB()
            >>> matches = orb1.match(orb2)
            >>> matches.distance
            >>> matches[0].distance
        """
        return [m[2] for m in self._matches]

    @property
    @array_result2
    def p1(self):
        """
        Feature coordinate in first image

        :return: feature coordinate
        :rtype: ndarray(2) or ndarray(2,N)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> orb1 = Image.Read("eiffel-1.png").ORB()
            >>> orb2 = Image.Read("eiffel-2.png").ORB()
            >>> matches = orb1.match(orb2)
            >>> matches.p1
            >>> matches[0].p1
        """
        return [self._kp1[m[0]].p for m in self._matches]

    @property
    @array_result2
    def p2(self):
        """
        Feature coordinate in second image

        :return: feature coordinate
        :rtype: ndarray(2) or ndarray(2,N)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> orb1 = Image.Read("eiffel-1.png").ORB()
            >>> orb2 = Image.Read("eiffel-2.png").ORB()
            >>> matches = orb1.match(orb2)
            >>> matches.p2
            >>> matches[0].p2
        """
        return [self._kp2[m[1]].p for m in self._matches]

    @property
    @array_result
    def descriptor1(self):
        """
        Feature descriptor in first image

        :return: feature descriptor
        :rtype: ndarray(M) or ndarray(N,M)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> orb1 = Image.Read("eiffel-1.png").ORB()
            >>> orb2 = Image.Read("eiffel-2.png").ORB()
            >>> matches = orb1.match(orb2)
            >>> matches.descriptor1
            >>> matches[0].descriptor1
        """
        return [self._kp1[m[0]].descriptor for m in self._matches]

    @property
    @array_result
    def descriptor2(self):
        """
        Feature descriptor in second image

        :return: feature descriptor
        :rtype: ndarray(M) or ndarray(N,M)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> orb1 = Image.Read("eiffel-1.png").ORB()
            >>> orb2 = Image.Read("eiffel-2.png").ORB()
            >>> matches = orb1.match(orb2)
            >>> matches.descriptor2
            >>> matches[0].descriptor2
        """
        return [self._kp2[m[1]].descriptor for m in self._matches]

    @property
    @scalar_result
    def id1(self):
        """
        Feature id in first image

        :return: feature id
        :rtype: int or ndarray(N)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> orb1 = Image.Read("eiffel-1.png").ORB()
            >>> orb2 = Image.Read("eiffel-2.png").ORB()
            >>> matches = orb1.match(orb2)
            >>> matches.id1
            >>> matches[0].id1
        """
        return [m[0] for m in self._matches]

    @property
    @scalar_result
    def id2(self):
        """
        Feature id in second image

        :return: feature id
        :rtype: int or ndarray(N)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> orb1 = Image.Read("eiffel-1.png").ORB()
            >>> orb2 = Image.Read("eiffel-2.png").ORB()
            >>> matches = orb1.match(orb2)
            >>> matches.id2
            >>> matches[0].id2
        """
        return [m[1] for m in self._matches]


# -------------------- subclasses of BaseFeature2D -------------------------- #
class SIFTFeature(BaseFeature2D):
    """
    Create set of SIFT point features

    .. inheritance-diagram:: machinevisiontoolbox.ImagePointFeatures.SIFTFeature
        :top-classes: machinevisiontoolbox.ImagePointFeatures.BaseFeature2D
        :parts: 1
    """


class ORBFeature(BaseFeature2D):
    """
    Create set of ORB point features

    .. inheritance-diagram:: machinevisiontoolbox.ImagePointFeatures.ORBFeature
        :top-classes: machinevisiontoolbox.ImagePointFeatures.BaseFeature2D
        :parts: 1
    """

    pass


class BRISKFeature(BaseFeature2D):
    """
    Create set of BRISK point features

    .. inheritance-diagram:: machinevisiontoolbox.ImagePointFeatures.BRISKFeature
        :top-classes: machinevisiontoolbox.ImagePointFeatures.BaseFeature2D
        :parts: 1
    """

    pass


class AKAZEFeature(BaseFeature2D):
    """
    Create set of AKAZE point features

    .. inheritance-diagram:: machinevisiontoolbox.ImagePointFeatures.AKAZEFeature
        :top-classes: machinevisiontoolbox.ImagePointFeatures.BaseFeature2D
        :parts: 1
    """

    pass


class HarrisFeature(BaseFeature2D):
    """
    Create set of Harris corner features

    .. inheritance-diagram:: machinevisiontoolbox.ImagePointFeatures.HarrisFeature
        :top-classes: machinevisiontoolbox.ImagePointFeatures.BaseFeature2D
        :parts: 1
    """

    pass


# pure feature descriptors


class FREAKFeature(BaseFeature2D):
    """
    Create set of FREAK point features

    .. inheritance-diagram:: machinevisiontoolbox.ImagePointFeatures.FREAKFeature
        :top-classes: machinevisiontoolbox.ImagePointFeatures.BaseFeature2D
        :parts: 1
    """

    pass


class BOOSTFeature(BaseFeature2D):
    """
    Create set of BOOST point features

    .. inheritance-diagram:: machinevisiontoolbox.ImagePointFeatures.BOOSTFeature
        :top-classes: machinevisiontoolbox.ImagePointFeatures.BaseFeature2D
        :parts: 1
    """

    pass


class BRIEFFeature(BaseFeature2D):
    """
    Create set of BRIEF point features

    .. inheritance-diagram:: machinevisiontoolbox.ImagePointFeatures.BRIEFFeature
        :top-classes: machinevisiontoolbox.ImagePointFeatures.BaseFeature2D
        :parts: 1
    """

    pass


class DAISYFeature(BaseFeature2D):
    """
    Create set of DAISY point features

    .. inheritance-diagram:: machinevisiontoolbox.ImagePointFeatures.DAISYFeature
        :top-classes: machinevisiontoolbox.ImagePointFeatures.BaseFeature2D
        :parts: 1
    """

    pass


class LATCHFeature(BaseFeature2D):
    """
    Create set of LATCH point features

    .. inheritance-diagram:: machinevisiontoolbox.ImagePointFeatures.LATCHFeature
        :top-classes: machinevisiontoolbox.ImagePointFeatures.BaseFeature2D
        :parts: 1
    """

    pass


class LUCIDFeature(BaseFeature2D):
    """
    Create set of LUCID point features

    .. inheritance-diagram:: machinevisiontoolbox.ImagePointFeatures.LUCIDFeature
        :top-classes: machinevisiontoolbox.ImagePointFeatures.BaseFeature2D
        :parts: 1
    """

    pass


class ImagePointFeaturesMixin:
    def _image2feature(
        self,
        cls,
        sortby=None,
        nfeat=None,
        id="image",
        scale=False,
        orient=False,
        **kwargs,
    ):
        # https://datascience.stackexchange.com/questions/43213/freak-feature-extraction-opencv
        algorithms = {
            "SIFT": cv.SIFT_create,
            "ORB": cv.ORB_create,
            "Harris": _Harris_create,
            "BRISK": cv.BRISK_create,
            "AKAZE": cv.AKAZE_create,
            # 'FREAK': (cv.FREAK_create, FREAKFeature),
            # 'DAISY': (cv.DAISY_create, DAISYFeature),
        }

        # check if image is valid
        # TODO, MSER can handle color
        image = self.mono()

        # get a reference to the appropriate detector
        algorithm = cls.__name__.replace("Feature", "")
        try:
            detector = algorithms[algorithm](**kwargs)
        except KeyError:
            raise ValueError("bad algorithm specified")

        kp, des = detector.detectAndCompute(image.A, mask=None)

        # kp is a list of N KeyPoint objects
        # des is NxM ndarray of keypoint descriptors

        if id == "image":
            if image.id is not None:
                # copy image id into the keypoints
                for k in kp:
                    k.class_id = image.id
        elif id == "index":
            for i, k in enumerate(kp):
                k.class_id = i
        elif isinstance(id, int):
            for k in kp:
                k.class_id = id
        else:
            raise ValueError("bad id")

        # do sorting in here

        if nfeat is not None:
            kp = kp[:nfeat]
            des = des[:nfeat, :]

        # construct a new Feature2DBase subclass
        features = cls(kp, des, scale=scale, orient=orient)

        # add attributes
        features._feature_type = algorithm
        features._image = self

        return features

    def SIFT(self, **kwargs):
        """
        Find SIFT features in image

        :param kwargs: arguments passed to OpenCV
        :return: set of 2D point features
        :rtype: :class:`SIFTFeature`

        .. inheritance-diagram:: machinevisiontoolbox.ImagePointFeatures.SIFTFeature
            :top-classes: machinevisiontoolbox.ImagePointFeatures.BaseFeature2D
            :parts: 1

        Returns an iterable and sliceable object that contains SIFT features and
        descriptors.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> sift = img.SIFT()
            >>> len(sift)  # number of features
            >>> print(sift[:5])

        :references:
            - Distinctive image features from scale-invariant keypoints.
              David G. Lowe
              Int. J. Comput. Vision, 60(2):91–110, November 2004.
            - Robotics, Vision & Control for Python, Section 14.1,
              P. Corke, Springer 2023.

        :seealso: :class:`SIFTFeature` `cv2.SIFT_create <https://docs.opencv.org/4.5.2/d7/d60/classcv_1_1SIFT.html>`_
        """

        return self._image2feature(SIFTFeature, scale=True, orient=True, **kwargs)

    def ORB(self, scoreType="harris", **kwargs):
        """
        Find ORB features in image

        :param kwargs: arguments passed to OpenCV
        :return: set of 2D point features
        :rtype: :class:`ORBFeature`

        .. inheritance-diagram:: machinevisiontoolbox.ImagePointFeatures.ORBFeature
            :top-classes: machinevisiontoolbox.ImagePointFeatures.BaseFeature2D
            :parts: 1

        Returns an iterable and sliceable object that contains 2D features with
        properties.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> orb = img.ORB()
            >>> len(orb)  # number of features
            >>> print(orb[:5])

        :seealso: :class:ORBFeature`, `cv2.ORB_create <https://docs.opencv.org/4.5.2/db/d95/classcv_1_1ORB.html>`_
        """

        scoreoptions = {"harris": cv.ORB_HARRIS_SCORE, "fast": cv.ORB_FAST_SCORE}
        return self._image2feature(
            ORBFeature, scoreType=scoreoptions[scoreType], **kwargs
        )

    def BRISK(self, **kwargs):
        """
        Find BRISK features in image

        .. inheritance-diagram:: machinevisiontoolbox.ImagePointFeatures.BRISKFeature
            :top-classes: machinevisiontoolbox.ImagePointFeatures.BaseFeature2D
            :parts: 1

        :param kwargs: arguments passed to OpenCV
        :return: set of 2D point features
        :rtype: :class:`BRISKFeature`

        Returns an iterable and sliceable object that contains BRISK features and
        descriptors.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> brisk = img.BRISK()
            >>> len(brisk)  # number of features
            >>> print(brisk[:5])

        :references:
            - Brisk: Binary robust invariant scalable keypoints.
              Stefan Leutenegger, Margarita Chli, and Roland Yves Siegwart.
              In Computer Vision (ICCV), 2011 IEEE International Conference on,
              pages 2548–2555. IEEE, 2011.

        :seealso: :class:`BRISKFeature` `cv2.BRISK_create <https://docs.opencv.org/4.5.2/d7/d60/classcv_1_1BRISK.html>`_
        """
        return self._image2feature(BRISKFeature, **kwargs)

    def AKAZE(self, **kwargs):
        """
        Find AKAZE features in image

        .. inheritance-diagram:: machinevisiontoolbox.ImagePointFeatures.AKAZEFeature
            :top-classes: machinevisiontoolbox.ImagePointFeatures.BaseFeature2D
            :parts: 1

        :param kwargs: arguments passed to OpenCV
        :return: set of 2D point features
        :rtype: :class:`AKAZEFeature`

        Returns an iterable and sliceable object that contains AKAZE features and
        descriptors.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> akaze = img.AKAZE()
            >>> len(akaze)  # number of features
            >>> print(akaze[:5])

        :references:
            - Fast explicit diffusion for accelerated features in nonlinear scale spaces.
              Pablo F Alcantarilla, Jesús Nuevo, and Adrien Bartoli.
              Trans. Pattern Anal. Machine Intell, 34(7):1281–1298, 2011.

        :seealso:
            :class:`~machinevisiontoolbox.ImagePointFeatures.AKAZEFeature`
            `cv2.AKAZE <https://docs.opencv.org/4.5.2/d7/d60/classcv_1_1AKAZE.html>`_
        """
        return self._image2feature(AKAZEFeature, **kwargs)

    def Harris(self, **kwargs):
        r"""
        Find Harris features in image

        .. inheritance-diagram:: machinevisiontoolbox.ImagePointFeatures.HarrisFeature
            :top-classes: machinevisiontoolbox.ImagePointFeatures.BaseFeature2D
            :parts: 1

        :param nfeat: maximum number of features to return, defaults to 250
        :type nfeat: int, optional
        :param k: Harris constant, defaults to 0.04
        :type k: float, optional
        :param scale: nonlocal minima suppression distance, defaults to 7
        :type scale: int, optional
        :param hw: half width of kernel, defaults to 2
        :type hw: int, optional
        :param patch: patch half width, defaults to 5
        :type patch: int, optional
        :return: set of 2D point features
        :rtype: :class:`HarrisFeature`

        Harris features are detected as non-local maxima in the Harris corner
        strength image.  The descriptor is a unit-normalized vector image
        elements in a :math:`w_p \times w_p` patch around the detected feature,
        where :math:`w_p = 2\mathtt{patch}+1`.

        Returns an iterable and sliceable object that contains Harris features and
        descriptors.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> harris = img.Harris()
            >>> len(harris)  # number of features
            >>> print(harris[:5])

        .. note:: The Harris corner detector and descriptor is not part of
            OpenCV and has been custom written for pedagogical purposes.

        :references:
            - A combined corner and edge detector.
              CG Harris, MJ Stephens
              Proceedings of the Fourth Alvey Vision Conference, 1988
              Manchester, pp 147–151
            - Robotics, Vision & Control for Python, Section 12.3.1,
                P. Corke, Springer 2023.

        :seealso: :class:`HarrisFeature`
        """
        return self._image2feature(HarrisFeature, **kwargs)

    def ComboFeature(self, detector, descriptor, det_opts, des_opts):
        """
        Combination feature detector and descriptor

        :param detector: detector name
        :type detector: str
        :param descriptor: descriptor name
        :type descriptor: str
        :param det_opts: options for detector
        :type det_opts: dict
        :param des_opts: options for descriptor
        :return: set of 2D point features
        :rtype: :class:`BaseFeature2D` subclass

        Detect corner features using the specified detector ``detector`` and
        describe them using the specified descriptor ``descriptor``.  A large
        number of possible combinations are possible.

        .. warning:: Incomplete

        :seealso: :class:`BOOSTFeature` :class:`BRIEFFeature` :class:`DAISYFeature` :class:`FREAKFeature` :class:`LATCHFeature` :class:`LUCIDFeature`
        """

        # WORK IN PROGRESS

        detectors = {
            "AGAST": cv.AgastFeatureDetector_create,
            "FAST": cv.FastFeatureDetector_create,
            "GoodFeaturesToTrack": cv.GFTTDetector_create,
        }

        descriptors = {
            "BOOST": (cv.xfeatures2d.BoostDesc_create, BOOSTFeature),
            "BRIEF": (cv.xfeatures2d.BriefDescriptorExtractor_create, BRIEFFeature),
            "DAISY": (cv.xfeatures2d.BriefDescriptorExtractor_create, DAISYFeature),
            "FREAK": (cv.xfeatures2d.BriefDescriptorExtractor_create, FREAKFeature),
            "LATCH": (cv.xfeatures2d.BriefDescriptorExtractor_create, LATCHFeature),
            "LUCID": (cv.xfeatures2d.BriefDescriptorExtractor_create, LUCIDFeature),
        }
        # eg. Feature2D('FAST', 'FREAK')
        if detector in detectors:
            # call it
            kp = detectors[detector](self.image.A, **det_opts)
        elif iscallable(detector):
            # call it
            kp = detector(self.image.A, **det_opts)
        else:
            raise ValueError("unknown detector")


class _Harris_create:
    def __init__(self, nfeat=250, k=0.04, scale=7, hw=2, patch=5):

        self.nfeat = nfeat
        self.k = k
        self.hw = hw
        self.peakscale = scale
        self.patch = patch
        self.scale = None

    def detectAndCompute(self, image, mask=None):
        # features are peaks in the Harris corner strength image
        dst = cv.cornerHarris(image, 2, 2 * self.hw + 1, self.k)
        peaks = findpeaks2d(dst, npeaks=None, scale=self.peakscale, positive=True)
        kp = []
        des = []
        w = 2 * self.patch + 1
        w2 = w**2
        for peak in peaks:
            x = int(round(peak[0]))
            y = int(round(peak[1]))
            try:
                W = image[
                    y - self.patch : y + self.patch + 1,
                    x - self.patch : x + self.patch + 1,
                ]
                v = W.flatten()
                if W.size > 0:
                    if len(v) != w2:
                        # handle case where last subscript is outside image bound
                        continue

                    des.append(smb.unitvec(v))
                    kp.append(cv.KeyPoint(x, y, 0, 0, peak[2]))
            except IndexError:
                # handle the case where the descriptor window falls off the edge
                pass

        for i, d in enumerate(des):
            if d.shape != des[1].shape:
                print(i, d.shape)

        return kp, np.array(des)


# ------------------------------------------------------------------------- #
if __name__ == "__main__":

    # step 1: familiarisation with open cv's sift

    # im = cv.imread('images/test/longquechen-moon.png')
    # im = cv.imread('images/monalisa.png')
    # imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    # sift = cv.SIFT_create()
    # kp = sift.detect(imgray, None)
    # kp, des = sift.detectAndCompute(imgray, None)
    # https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html
    # #aea339bc868102430087b659cd0709c11
    # kp[i].pt = (u,v)
    # kp[i].angle = orientation [deg?]
    # kp[i].class_id? unclear
    # kp[i].size = scale
    # kp[i].response = strength of keypoint
    # kp[i].octave - need to double check, but seems like a really large number

    # img = cv.drawKeypoints(imgray, kp, im,
    #                       flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # mvt.idisp(img, title='sift_keypoints')

    # sf = Sift(imgray)
    # sf.u

    # sf0 = sf[0:3]
    # sf0.u

    # drawing = sf.drawSiftKeypoints(imgray)

    # TODO would be nice to make a root-sift descriptor method, as it is a
    # simple addition to the SIFT descriptor

    # test matching

    # import code
    # code.interact(local=dict(globals(), **locals()))

    from machinevisiontoolbox import Image, ImageCollection

    # kp1 = Image.Read('eiffel-1.png').SIFT()
    # kp2 = Image.Read('eiffel-2.png').SIFT()

    # kp1 = Image.Read('eiffel2-1.png').Harris()

    # # d = kp1[0].distance(kp1[1])
    # # print(d)
    # d = kp1[0].distance(kp1[30])
    # print(d)

    # matches = kp1.match(kp2)
    # matches.subset(10).table()
    # matches.subset(100).plot(linewidth=0.7, darken=False, color="yellow")

    # c = matches.correspondences()

    # # im = Image('eiffel2-1.png')
    # # ax = im.disp()

    # # # sort into descending order
    # # ks = kp1.sort()
    # # print(len(kp1), len(ks))
    # # print(kp1[0]._descriptor)
    # # print(ks[0]._descriptor)

    # # kp1.plot(hand=True, handalpha=0.2)
    # from machinevisiontoolbox import Image

    # matches[:10].plot('b', alpha=0.6)

    # plt.show(block=True)

    # im1 = Image.Read("eiffel2-1.png", grey=True)
    # im2 = Image.Read("eiffel2-2.png", grey=True)
    # hf = im1.Harris()
    # hf = im1.Harris(nfeat=200)

    # im1.disp(darken=True); hf.plot("gs")

    # hf[0].distance(hf[1], metric="ncc")

    # images = ImageCollection("campus/*.png", mono=True);

    # features = [];

    # for image in images:
    #     features += image.SIFT()

    # # features.sort(by="scale", inplace=True);

    # len(features)
    # 42194
    # features[:10].table()

    im1 = Image.Read("eiffel-1.png", grey=True)
    # im2 = Image.Read("eiffel-2.png", grey=True)

    sf1 = im1.SIFT(nfeat=200)

    # im1.disp()
    # sf1.plot()
    # plt.show(block=True)

    im2 = im1.colorize()
    sf1.draw2(im2, color="r", type="rich")
    im2.disp()
    plt.show(block=True)

    # im3 = im1.colorize()
    # z = sf1.drawKeypoints(im3)
    # im3.disp(block=True)

    plt.show(block=True)

    # hf1 = im1.Harris(nfeat=250, scale=10)
    # print(hf1[5])
    # hf1[:5].table()
    # hf1[:5].list()

    # sf1 = []
    # sf1 += im1.SIFT(nfeat=250)
    # sf1 += im1.SIFT(nfeat=250)

    # print(sf1[5])
    # sf1[:5].table()
    # sf1[:5].list()
    # sf2 = im2.SIFT();

    # print(len(sf1))
    # print(len(sf2))
    # sf = sf1 + sf2
    # print(len(sf))

    # sf = [] + sf1
    # print(len(sf))
    # sf = sf1 + []
    # print(len(sf))

    sf1 = im1.BRISK()
    sf2 = im2.AKAZE()
    hf = im1.Harris()
    of = im1.ORB()

    # drawKeypoints(self,
    #                   image,
    #                   drawing=None,
    #                   isift=None,
    #                   flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    #                   **kwargs):

    z = of.drawKeypoints(im1)
    z.disp(block=True)
    # mm = sf1.match(sf2, thresh=20)

    # print(mm)
    # print(mm[3])
    # print(mm[:5])
    # mm.list()
    # mm.table()

    # from machinevisiontoolbox import CentralCamera
    # F, resid = mm.estimate(CentralCamera.points2F, method="ransac", confidence=0.99)
    # mm[:10].list()

    # mm = sf1.match(sf2, sort=True)[:10];

    # mm = sf1.match(sf2, ratio=0.8, crosscheck=True);
