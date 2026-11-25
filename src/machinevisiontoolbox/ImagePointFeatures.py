"""
Detection and matching of keypoint and descriptor features (SIFT, ORB, etc.) in images.
"""

import builtins
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Iterator, Self

import cv2
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import spatialmath.base as smb
from ansitable import ANSITable, Column

from machinevisiontoolbox._image_typing import ImageMixinBase
from machinevisiontoolbox.base.color import name2color
from machinevisiontoolbox.base.findpeaks import findpeaks2d
from machinevisiontoolbox.base.graphics import (
    _require_c_contiguous,
    draw_circle,
    draw_line,
    draw_point,
)
from machinevisiontoolbox.base.imageio import safe_plt_show
from machinevisiontoolbox.decorators import (
    array_result,
    array_result2,
    scalar_result,
)

if TYPE_CHECKING:
    pass

# TODO, either subclass SIFTFeature(BaseFeature2D) or just use BaseFeature2D
# directly


class BaseFeature2D:
    """
    A 2D point feature class
    """

    def __init__(
        self,
        kp: list | np.ndarray | None = None,
        des: np.ndarray | None = None,
        scale: bool = False,
        orient: bool = False,
        image: np.ndarray | None = None,
        strength: bool = True,
    ) -> None:
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
        :param strength: features have a strength (detector response), defaults to True
        :type strength: bool, optional

        The :attr:`scale`, :attr:`orientation` and :attr:`strength` properties raise
        ``ValueError`` for a set of features that does not have that quantity, for example
        :attr:`orientation` of FAST features.

        A :class:`~machinevisiontoolbox.ImagePointFeatures.BaseFeature2D` object:

            - has a length, the number of feature points it contains
            - can be sliced to extract a subset of features

        This object behaves like a list, allowing indexing, slicing and
        iteration over individual features.  It also supports a number of
        convenience methods.

        .. note:: OpenCV consider feature points as :obj:`opencv.KeyPoint` objects and the
            descriptors as a multirow NumPy array.  This class provides a more
            convenient abstraction.
        """

        # TODO flesh out sortby option, it can be by strength or scale
        # TODO what does nfeatures option to SIFT do? seemingly nothing

        self._has_scale: bool = scale
        self._has_orient: bool = orient
        self._has_strength: bool = strength
        self._image: Any = image
        self._feature_type: str | None = None
        if kp is None:
            self._kp: list[cv2.KeyPoint] = []
        else:
            self._kp = list(kp)
        self._descriptor: np.ndarray | None = des

    def _require(self, flag: bool, what: str) -> None:
        """Raise ValueError if these features do not have ``what``"""
        if not flag:
            kind = self._feature_type or self.__class__.__name__
            raise ValueError(f"{kind} features have no {what}")

    def _like(self) -> Self:
        """A new empty set of the same kind, with the metadata of this one

        The flags that say which of scale, orientation and strength the features
        have, the feature type and the image they came from are copied, so a set made
        by slicing, adding or sorting keeps them.
        """
        new = self.__class__()
        new._has_scale = self._has_scale
        new._has_orient = self._has_orient
        new._has_strength = self._has_strength
        new._feature_type = self._feature_type
        new._image = self._image
        return new

    def __len__(self) -> int:
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
        return len(self._kp)  # Number of features (base method)

    def __getitem__(
        self, i: int | slice | Sequence[int] | np.ndarray
    ) -> "BaseFeature2D":
        """
        Get item from point feature object (base method)

        :param i: index
        :type i: int, slice, sequence of int, or ndarray of int or bool
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
        new = self._like()  # Get item from point feature object (base method)
        # compute canonical integer index list for both _kp and descriptor
        n = len(self)
        if isinstance(i, int):
            idx: list[int] = [i if i >= 0 else n + i]
            new._kp = [self._kp[idx[0]]]
        elif isinstance(i, slice):
            idx = list(range(*i.indices(n)))
            new._kp = [self._kp[k] for k in idx]
        elif isinstance(i, np.ndarray) and np.issubdtype(i.dtype, bool):
            idx = [k for k, flag in enumerate(i.tolist()) if flag]
            new._kp = [self._kp[k] for k in idx]
        elif isinstance(i, np.ndarray):
            idx = [int(k) for k in i.flatten().tolist()]
            new._kp = [self._kp[k] for k in idx]
        else:
            idx = [int(k) for k in i]
            new._kp = [self._kp[k] for k in idx]
        # index the descriptor rows using np.take (works with list[int])
        if self._descriptor is not None and self._descriptor.size > 0:
            if self._descriptor.ndim == 1:
                new._descriptor = self._descriptor
            else:
                new._descriptor = np.take(self._descriptor, idx, axis=0)
        else:
            new._descriptor = None
        return new

    def __str__(self) -> str:
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
        if len(self) > 1:  # String representation of feature (base method)
            return f"{self.__class__.__name__} features, {len(self)} points"
        else:
            s = f"{self.__class__.__name__}: ({self.u:.1f}, {self.v:.1f}),"
            if self._has_strength:
                s += f" strength={self.strength:.2f}"
            if self._has_scale:
                s += f", scale={self.scale:.1f}"
            if self._has_orient:
                s += f", orient={np.degrees(self.orientation):.1f}°"
            s += f", id={self.id}"
            return s

    def __repr__(self) -> str:
        """
        Display features in readable form

        :return: string representation
        :rtype: str

        :seealso: :meth:`str`
        """
        return str(self)  # Display features in readable form

    def __iter__(self) -> "Iterator[BaseFeature2D]":
        for i in range(len(self)):
            yield self[i]

    def list(self) -> None:
        """
        List matches

        Print the features in a simple format, one line per feature.

        :seealso: :meth:`table`
        """
        for i, f in enumerate(self):  # List matches
            s = f"{self._feature_type} feature {i}: ({f.u:.1f}, {f.v:.1f}),"
            if f._has_strength:
                s += f" strength={f.strength:.2f}"
            if f._has_scale:
                s += f", scale={f.scale:.1f}"
            if f._has_orient:
                s += f", orient={np.degrees(f.orientation):.1f}°"
            s += f", id={f.id}"
            print(s)

    def table(self) -> None:
        """
        Print features in tabular form

        Each row is in the table includes: the index in the feature vector,
        centroid coordinate, feature strength, feature scale and image id.

        :seealso: ``str``
        """
        columns = [
            Column("#"),
            Column("centroid"),
        ]  # Print features in tabular form
        if self._has_strength:
            columns.append(Column("strength", fmt="{:.3g}"))
        if self._has_scale:
            columns.append(Column("scale", fmt="{:.3g}"))
        if self._has_orient:
            columns.append(Column("orient", fmt="{:.3g}°"))
        columns.append(Column("id", fmt="{:d}"))
        table = ANSITable(*columns, border="thin")
        for i, f in enumerate(self):
            values = []
            if self._has_strength:
                values.append(f.strength)
            if self._has_scale:
                values.append(f.scale)
            if self._has_orient:
                values.append(np.degrees(f.orientation))

            table.row(i, f"{f.u:.1f}, {f.v:.1f}", *values, f.id)
        table.print()

    def gridify(self, nbins: int | tuple[int, int], nfeat: int) -> "BaseFeature2D":
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

        .. versionchanged:: 2.5.0
            The result keeps the descriptors and the scale, orientation and strength
            of the features, and a feature at the right or bottom edge of the image
            no longer raises ``IndexError``.

        :seealso: :meth:`sort`
        """

        try:  # Sort features into grid
            nw, nh = nbins  # type: ignore[misc]
        except (TypeError, ValueError):
            nw = nbins
            nh = nbins

        image = self._image
        if image is None:
            raise ValueError("no image associated with features")

        keep: list[int] = []
        bins = np.zeros((nh, nw), dtype="int")

        for i, f in enumerate(self):
            # the last cell takes the pixels left over when the size is not a
            # multiple of the number of cells
            ix = min(int(f.p[0].item() * nw // image.width), nw - 1)
            iy = min(int(f.p[1].item() * nh // image.height), nh - 1)

            if bins[iy, ix] < nfeat:
                keep.append(i)
                bins[iy, ix] += 1

        # indexing keeps the descriptors and the flags, as slicing does
        return self[keep]

    def __add__(self, other: "BaseFeature2D | builtins.list | None") -> "BaseFeature2D":
        """
        Add feature sets

        :param other: set of features
        :type other: :class:`BaseFeature2D`
        :raises TypeError: ``other`` is not a feature set, ``[]`` or ``None``, or
            the two sets are of different feature types
        :return: set of features
        :rtype: :class:`BaseFeature2D` instance

        Add two feature sets to form a new feature sets.  If ``other`` is
        equal to ``None`` or ``[]`` it is interpreted as an empty feature
        set, this is useful in a loop for aggregating feature sets.

        .. versionchanged:: 2.5.0
            Adding something that is not a feature set, ``[]`` or ``None``
            raises ``TypeError``.  It used to raise ``AssertionError``.

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
        if other is None or (
            isinstance(other, list) and len(other) == 0
        ):  # Add feature sets
            return self
        if not isinstance(other, BaseFeature2D):
            raise TypeError(
                f"cannot add {type(other).__name__} to a feature set, "
                "only another feature set, [] or None"
            )
        if self._feature_type != other._feature_type:
            raise TypeError(
                "can't add different feature types:",
                self._feature_type,
                other._feature_type,
            )
        new = self._like()
        new._kp = self._kp + other._kp
        if self._descriptor is not None and other._descriptor is not None:
            new._descriptor = np.vstack((self._descriptor, other._descriptor))
        return new

    def __radd__(
        self, other: "BaseFeature2D | builtins.list | int | None"
    ) -> "BaseFeature2D":
        """
        Add feature sets

        :param other: set of features
        :type other: :class:`BaseFeature2D`
        :raises TypeError: ``other`` is not a feature set, ``[]``, ``None`` or ``0``
        :return: set of features
        :rtype: :class:`BaseFeature2D`

        Add two feature sets to form a new feature sets.  If ``other`` is
        equal to ``None``, ``[]`` or ``0`` it is interpreted as an empty feature
        set, this is useful in a loop for aggregating feature sets.  The ``0``
        is the starting value of the builtin :func:`sum`, so a list of
        feature sets can be combined by ``sum(features)``.

        .. versionchanged:: 2.5.0
            ``None`` and ``0`` are accepted, and anything else raises
            ``TypeError`` with a message.  It used to raise ``ValueError("bad")``,
            and ``sum()`` of feature sets did not work.

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
            >>> orb = sum([img1.ORB(), img2.ORB()])
            >>> orb

        :seealso: :meth:`__add__`
        """
        # an empty set, or the 0 that sum() starts with
        if (
            other is None
            or (isinstance(other, list) and len(other) == 0)
            or (isinstance(other, int) and not isinstance(other, bool) and other == 0)
        ):
            return self
        raise TypeError(
            f"cannot add a feature set to {type(other).__name__}, "
            "only to [], None or 0 (as the start of sum)"
        )

    @property
    @scalar_result
    def u(self) -> "builtins.list[float]":  # Horizontal coordinate of feature point
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
    def v(self) -> "builtins.list[float]":  # Vertical coordinate of feature point
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
    def id(self) -> "builtins.list[int]":  # Image id for feature point
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

        .. note:: Defined by the ``id`` attribute of the image passed to the
            feature detector
        """
        return [kp.class_id for kp in self._kp]

    @property
    @scalar_result
    def orientation(self) -> "builtins.list[float]":  # Orientation of feature
        """
        Orientation of feature

        :return: Orientation in radians
        :rtype: float or list of float
        :raises ValueError: if the features have no orientation, such as FAST or Harris
            features

        .. versionchanged:: 2.5.0
            Raises ``ValueError`` if the features have no orientation. It used to return
            a constant for a detector that has no orientation.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> orb = img.ORB()
            >>> orb[0].orientation
            >>> orb[:5].orientation
        """
        self._require(self._has_orient, "orientation")
        return [np.radians(kp.angle) for kp in self._kp]

    @property
    @scalar_result
    def scale(self) -> "builtins.list[float]":  # Scale of feature
        """
        Scale of feature

        :return: Scale
        :rtype: float or list of float
        :raises ValueError: if the features have no scale, such as FAST or Harris features

        .. versionchanged:: 2.5.0
            Raises ``ValueError`` if the features have no scale. It used to return the
            keypoint size, which is a constant for a detector that has no scale.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> orb = img.ORB()
            >>> orb[0].scale
            >>> orb[:5].scale
        """
        self._require(self._has_scale, "scale")
        return [kp.size for kp in self._kp]

    @property
    @scalar_result
    def strength(self) -> "builtins.list[float]":  # Strength of feature
        """
        Strength of feature

        :return: Strength
        :rtype: float or list of float
        :raises ValueError: if the features have no strength

        .. versionchanged:: 2.5.0
            Raises ``ValueError`` if the features have no strength.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> orb = img.ORB()
            >>> orb[0].strength
            >>> orb[:5].strength
        """
        self._require(self._has_strength, "strength")
        return [kp.response for kp in self._kp]

    @property
    @scalar_result
    def octave(self) -> "builtins.list[int]":  # Octave of feature
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
    def descriptor(self) -> np.ndarray | None:  # Descriptor of feature
        """
        Descriptor of feature

        :return: Descriptor, or None if the features have no descriptors
        :rtype: ndarray(N,M) or None

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> orb = img.ORB()
            >>> orb[0].descriptor.shape
            >>> orb[0].descriptor
            >>> orb[:5].descriptor.shape

        .. note:: For single feature return a 1D array vector, for multiple features return a set of column vectors.

        .. versionchanged:: 2.5.0
            Returns None, instead of raising ``TypeError``, if the features have
            no descriptors.
        """
        des = self._descriptor
        if des is None:
            return None
        # a single feature gives a 1D vector, as array_result does for the
        # other properties
        return des[0] if len(des) == 1 else des

    @property
    @array_result
    def p(self) -> np.ndarray:  # Feature coordinates
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

    def distance(self, other: "BaseFeature2D", metric: str = "L2") -> np.ndarray:
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

        .. note::
            - The matrix is symmetric.
            - For the metric "L1" and "L2" the best match is the smallest distance
            - For the metric "ncc" the best match is the largest distance.  A value over
              0.8 is often considered to be a good match.

        :seealso: :meth:`match`
        """
        metric_dict = {"L1": 1, "L2": 2}  # Distance between feature sets

        if self._descriptor is None or other._descriptor is None:
            raise ValueError("features have no descriptors")
        n1 = len(self)
        n2 = len(other)
        D = np.empty((n1, n2))
        if n1 == 1:
            des1 = self._descriptor.reshape(1, -1)
        else:
            des1 = self._descriptor
        if n2 == 1:
            des2 = other._descriptor.reshape(1, -1)
        else:
            des2 = other._descriptor

        for i in range(n1):
            for j in range(n2):
                if metric == "ncc":
                    d = np.dot(des1[i, :], des2[j, :])
                else:
                    d = np.linalg.norm(des1[i, :] - des2[j, :], ord=metric_dict[metric])
                D[i, j] = d
        return D

    def match(
        self,
        other: "BaseFeature2D",
        ratio: float = 0.75,
        crosscheck: bool = False,
        metric: str | None = None,
        sort: bool = True,
        top: int | None = None,
        thresh: float | None = None,
    ) -> "FeatureMatch":
        """
        Match point features

        :param other: set of feature points
        :type other: BaseFeature2D
        :param ratio: parameter for Lowe's ratio test, defaults to 0.75
        :type ratio: float, optional
        :param crosscheck: perform left-right cross check, defaults to False
        :type crosscheck: bool, optional
        :param metric: distance metric, one of: 'L1', 'L2', 'hamming', 'hamming2'.
            Defaults to ``None``, which auto-selects 'hamming' for binary
            descriptors (eg. ORB, BRISK, AKAZE) and 'L2' otherwise (eg. SIFT).
        :type metric: str, optional
        :param sort: sort features by strength, defaults to True
        :type sort: bool, optional
        :raises ValueError: bad metric name provided, or the two sets have different
            kinds of descriptor, for example SIFT and ORB
        :return: set of candidate matches
        :rtype: :class:`FeatureMatch` instance

        Return a match object that contains pairs of putative corresponding points.
        If ``crosscheck`` is True the ratio test is disabled

        .. note:: Binary descriptors (eg. ORB, BRISK, AKAZE) are ``uint8``
            arrays, for which 'L2' distance is close to meaningless -- it
            will silently return very few matches rather than raising an
            error, which is why this is auto-selected rather than left for
            the caller to always remember.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> orb1 = Image.Read("eiffel-1.png").ORB()
            >>> orb2 = Image.Read("eiffel-2.png").ORB()
            >>> m = orb1.match(orb2)
            >>> len(m)

        :seealso: :class:`FeatureMatch` :meth:`distance`
        """

        # TODO: implement thresh  # Match point features

        # m = []

        # TODO check valid input
        # d1 and d2 must be numpy arrays
        # d1 and d2 must have equal (128 for SIFT) rows
        # d1 and d2 must have greater than 1 columns

        # do matching
        # sorting
        # return

        des1 = self.descriptor
        des2 = other.descriptor
        if des1 is None or des2 is None:
            raise ValueError("features have no descriptors")
        if des1.dtype != des2.dtype or des1.shape[1] != des2.shape[1]:
            raise ValueError(
                "cannot match features with different kinds of descriptor: "
                f"{self._feature_type} ({des1.shape[1]} {des1.dtype}) and "
                f"{other._feature_type} ({des2.shape[1]} {des2.dtype})"
            )

        if metric is None:
            metric = "hamming" if des1.dtype == np.uint8 else "L2"

        metricdict = {
            "L1": cv2.NORM_L1,
            "L2": cv2.NORM_L2,
            "hamming": cv2.NORM_HAMMING,
            "hamming2": cv2.NORM_HAMMING2,
        }
        if metric not in metricdict:
            raise ValueError("bad metric name")

        # create BFMatcher (brute force matcher) object
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        bf = cv2.BFMatcher(normType=metricdict[metric], crossCheck=crosscheck)

        # Match descriptors.
        # matches0 = bf.match(d1, d2)
        # there is also:
        if crosscheck:
            k = 1
        else:
            k = 2

        matches0 = bf.knnMatch(des1, des2, k=k)

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

    def subset(self, N: int = 100) -> "BaseFeature2D":
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
        step = max(1, len(self) // N)  # Select subset of features
        k = list(range(0, len(self), step))
        k = k[:N]
        new = self[k]
        new._feature_type = self._feature_type
        return new

    def sort(
        self, by: str = "strength", descending: bool = True, inplace: bool = False
    ) -> "BaseFeature2D | None":
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
        if by == "strength":  # Sort features
            key = self.strength
        elif by == "scale":
            key = self.scale
        else:
            raise ValueError("bad sort method", by)
        key = np.array(key)
        if descending:
            key = -key

        index = np.argsort(key)

        sorted_index = index.tolist()
        if inplace:
            self._kp = [self._kp[i] for i in sorted_index]
            if self._descriptor is not None:
                self._descriptor = np.take(self._descriptor, sorted_index, axis=0)
        else:
            new = self._like()
            new._kp = [self._kp[i] for i in sorted_index]
            if self._descriptor is not None:
                new._descriptor = np.take(self._descriptor, sorted_index, axis=0)
            return new
        return None

    def support(self, images: Any, N: int = 50) -> Any:
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

        .. note:: If the features come from multiple images then the feature's
            ``id`` attribute is used to index into ``images`` which must be a
            list of Image objects.
        """

        from machinevisiontoolbox import Image

        if len(self) > 1:  # Find support region
            raise ValueError("can only compute support region for single feature")

        if isinstance(images, Image):
            image = images._A
        else:
            # list or iterable
            image = images[self.id]._A

        # M = smb.transl2(N/2, N/2) @ smb.trot2(self.orientation) @ smb.transl2(-self.u, -self.v)
        # M = M[:2, :] / self.scale * N / 2
        # translate to origin and rotate
        M = smb.trot2(self.orientation) @ smb.transl2(-self.u, -self.v)

        # scale it to fill the window
        M *= N / 2 / self.scale
        M[2, 2] = 1

        # translate to centre of window
        M = smb.transl2(N / 2, N / 2) @ M

        out = cv2.warpAffine(
            src=image, M=M[:2, :], dsize=(N, N), flags=cv2.INTER_LINEAR
        )
        return Image(out)

    def filter(self, **kwargs: Any) -> "BaseFeature2D":
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

        .. note:: If ``value`` is a range the ``numpy.Inf`` or ``-numpy.Inf``
            can be used as values.
        """

        features = self  # Filter features

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
                k = v >= limits
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
        image: Any,
        drawing: np.ndarray | None = None,
        isift: Any = None,
        color: tuple[int, int, int] = (0, 255, 0),
        flags: int = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        **kwargs: Any,
    ) -> Any:
        """
        Render keypoints into image

        :param image: original image
        :type image: :class:`Image`
        :param drawing: _description_, defaults to None
        :type drawing: _type_, optional
        :param isift: _description_, defaults to None
        :type isift: _type_, optional
        :param flags: _description_, defaults to cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
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
        # draw sift features on image using cv2.drawKeypoints

        # check valid imagesource
        # TODO if max(self._u) or max(self._v) are greater than image width,
        # height, respectively, then raise ValueError

        # TODO check flags, setup dictionary or string for plot options

        if drawing is None:  # Render keypoints into image
            drawing = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        kp = self._kp

        if isift is None:
            isift = np.arange(0, len(self._kp))  # might need a +1 here
        else:
            isift = np.array(isift, ndmin=1, copy=True)

        # TODO should check that isift is consistent with kp (min value is 0,
        # max value is <= len(kp))
        cv2.drawKeypoints(
            image=image._A,  # image, source image
            # kp[isift],
            keypoints=kp,
            outImage=drawing,  # outimage
            color=color,
            flags=flags,
        )

        return image.__class__(drawing)

    def drawMatches(
        self,
        im1: Any,
        sift1: "BaseFeature2D",
        im2: Any,
        sift2: "BaseFeature2D",
        matches: Any,
        **kwargs: Any,
    ) -> Any:
        # TODO should I just have input two SIFT objects,
        # or in this case just another SIFT object?

        # draw_params = dict(matchColor=(0, 255, 0),
        #                   singlePointColor=(255, 0, 0),
        #                   matchesMask=matches,
        #                   flags=0)

        h = max(im1._A.shape[0], im2._A.shape[0])  # Draw matches
        w = im1._A.shape[1] + im2._A.shape[1]
        nplanes = im1._A.shape[2] if im1._A.ndim == 3 else 1
        blank = np.zeros((h, w, max(nplanes, 3)), dtype=np.uint8)
        out = cv2.drawMatchesKnn(
            img1=im1._A,
            keypoints1=sift1._kp,
            img2=im2._A,
            keypoints2=sift2._kp,
            matches1to2=matches,
            outImg=blank,
        )

        return im1.__class__(out)

    def plot(
        self,
        *args: Any,
        ax: Any = None,
        filled: bool = False,
        color: Any = "blue",
        alpha: float = 1,
        hand: bool = False,
        handcolor: Any = "blue",
        handthickness: int = 1,
        handalpha: float = 1,
        **kwargs: Any,
    ) -> None:
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
        :param kwargs: options passed to :class:`matplotlib.patches.Circle` such as color,
            alpha, edgecolor, etc.
        :type kwargs: dict

        Plot circles to represent the position and scale of features on a Matplotlib axis.
        Orientation, if applicable, is indicated by a radial line from the circle centre
        to the circumference, like a clock hand.

        """
        ax = smb.axes_logic(ax, 2)  # Plot features using Matplotlib

        if filled:
            for kp in self:
                centre = kp.p.flatten()
                c = mpatches.Circle(
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
                    l = mlines.Line2D(
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
        image: Any,
        *args: Any,
        ax: Any = None,
        filled: bool = False,
        color: Any = "blue",
        alpha: float = 1,
        hand: bool = False,
        handcolor: Any = "blue",
        handthickness: int = 1,
        handalpha: float = 1,
        **kwargs: Any,
    ) -> Any:
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
        :param kwargs: options passed to :class:`matplotlib.patches.Circle` such as color,
            alpha, edgecolor, etc.
        :type kwargs: dict

        Plot circles to represent the position and scale of features on a Matplotlib axis.
        Orientation, if applicable, is indicated by a radial line from the circle centre
        to the circumference, like a clock hand.

        .. warning:: The image is drawn into in place.  OpenCV requires that its pixel
            data is C-contiguous, otherwise a ``ValueError`` is raised.  See
            :class:`~machinevisiontoolbox.Image` for how to create such an image.
        """
        img = image._A  # Draw features into image
        _require_c_contiguous(img)
        if filled:
            for kp in self:
                centre = kp.p.flatten()

                draw_circle(
                    img,
                    centre,
                    radius=kp.scale,
                    color=color,
                    **kwargs,
                )

                if hand:
                    circum = (
                        centre
                        + kp.scale
                        * np.r_[math.cos(kp.orientation), math.sin(kp.orientation)]
                    )
                    draw_line(
                        img,
                        centre,
                        circum,
                        color=handcolor,
                        thickness=handthickness,
                    )
        else:
            if len(args) == 0 and len(kwargs) == 0:
                kwargs = dict(marker="+y")
            kwargs.setdefault("fontsize", 0.6)
            draw_point(img, self.p, *args, **kwargs)

    def draw2(self, image: Any, color: Any = "y", type: str = "point") -> Any:
        """
        Draw features into image using OpenCV keypoint drawing

        :param image: image to draw into
        :type image: :class:`~machinevisiontoolbox.Image`
        :param color: color of the features, defaults to 'y'
        :type color: str, array_like or tuple, optional
        :param type: style, one of 'point' [default], 'rich' (size and orientation) or
            'not' (omit single points)
        :type type: str, optional
        :raises ValueError: image pixel data is not C-contiguous
        :return: the image drawn into
        :rtype: :class:`~machinevisiontoolbox.Image`

        .. warning:: The image is drawn into in place.  OpenCV requires that its pixel
            data is C-contiguous, otherwise a ``ValueError`` is raised.  See
            :class:`~machinevisiontoolbox.Image` for how to create such an image.
        """
        img = image._A  # Draw features into image
        _require_c_contiguous(img)
        if isinstance(color, str):
            color_ndarray: Any = name2color(
                color, dtype=image.dtype, colororder=image.colororder
            )
            color_scalar: tuple[float, float, float, float] = (
                float(color_ndarray.flat[0]),
                float(color_ndarray.flat[1]) if color_ndarray.size > 1 else 0.0,
                float(color_ndarray.flat[2]) if color_ndarray.size > 2 else 0.0,
                0.0,
            )
        elif isinstance(color, np.ndarray):
            color_scalar = (
                float(color.flat[0]),
                float(color.flat[1]) if color.size > 1 else 0.0,
                float(color.flat[2]) if color.size > 2 else 0.0,
                0.0,
            )
        elif isinstance(color, (list, tuple)):
            vals = list(color) + [0.0, 0.0, 0.0, 0.0]
            color_scalar = (
                float(vals[0]),
                float(vals[1]),
                float(vals[2]),
                float(vals[3]),
            )
        else:
            color_scalar = (0.0, 255.0, 0.0, 0.0)

        options = {
            "rich": cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            "point": cv2.DRAW_MATCHES_FLAGS_DEFAULT,
            "not": cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
        }

        cv2.drawKeypoints(
            image=img,  # image, source image
            keypoints=self._kp,
            outImage=img,  # outimage
            color=color_scalar,
            flags=options[type] + cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
        )

        return image.__class__(img)


class FeatureMatch:
    """
    A set of matches between two sets of point features

    Returned by :meth:`BaseFeature2D.match`.  It can be indexed, sliced and
    iterated over, and has the corresponding points ``p1`` and ``p2`` and the
    match ``distance``.  Inlier and outlier status is set by :meth:`estimate`.
    """

    def __init__(
        self,
        m: list[tuple[int, int, float]],
        fv1: BaseFeature2D,
        fv2: BaseFeature2D,
        inliers: Sequence[bool] | np.ndarray | None = None,
    ) -> None:
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

        .. note:: This constructor would not be called directly, it is used by the
            ``match`` method of the :class:`BaseFeature2D` subclass.

        :seealso: :obj:`BaseFeature2D.match` `cv2.KeyPoint <https://docs.opencv.org/4.5.2/d2/d29/classcv_1_1KeyPoint.html#a507d41b54805e9ee5042b922e68e4372>`_
        """
        self._matches = m
        self._kp1 = fv1
        self._kp2 = fv2
        # always an ndarray of bool, or None if the status has not been set
        self._inliers: np.ndarray | None = (
            None if inliers is None else np.asarray(inliers, dtype=bool)
        )
        self._inverse_dict1: dict[int, int] | None = None
        self._inverse_dict2: dict[int, int] | None = None

    def __getitem__(
        self, i: int | slice | Sequence[int] | np.ndarray
    ) -> "FeatureMatch":
        """
        Get matches from feature match object

        :param i: match subset
        :type i: int, slice, sequence of int, or ndarray of int or bool
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
        if not isinstance(i, (int, slice, np.ndarray)):
            if isinstance(i, (str, bytes)) or not isinstance(i, Sequence):
                raise ValueError(
                    f"index must be an int, slice, sequence or array, got {type(i).__name__}"
                )
            # a list or tuple of ints or bools
            i = np.asarray(i)
            if i.size == 0:
                i = i.astype(int)

        if isinstance(i, int):
            matches = [self._matches[i]]
            select: Any = [i]
        elif isinstance(i, slice):
            matches = self._matches[i]
            select = i
        elif np.issubdtype(i.dtype, bool):
            matches = [m for m, g in zip(self._matches, i) if g]
            select = i
        elif np.issubdtype(i.dtype, np.integer):
            matches = [self._matches[k] for k in i]
            select = i
        else:
            raise ValueError(f"index array must be integer or bool, got {i.dtype}")

        # indexing the array keeps the inlier status an ndarray, whatever the index
        inliers = None if self._inliers is None else self._inliers[select]
        return FeatureMatch(matches, self._kp1, self._kp2, inliers)

    def __len__(self) -> int:
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

    def correspondence(self) -> np.ndarray:
        """
        Feature correspondences

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

    def by_id1(self, id: int) -> "FeatureMatch | None":
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

        .. note::
            - For efficient lookup, on the first call a dict is built that maps
              feature id to index in the feature set.
            - Useful when features in the sets come from multiple images and
              ``id`` is used to indicate the source image.

        :seealso: :class:`BaseFeature2D` :obj:`BaseFeature2D.id` :meth:`by_id2`
        """
        if self._inverse_dict1 is None:
            # first call, build a dict for efficient mapping
            d: dict[int, int] = {}
            for k, m in enumerate(self._matches):
                d[m[0]] = k
            self._inverse_dict1 = d
        try:
            return self[self._inverse_dict1[id]]
        except KeyError:
            return None

    def by_id2(self, i: int) -> "FeatureMatch | None":
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

        .. note::
            - For efficient lookup, on the first call a dict is built that maps
              feature id to index in the feature set.
            - Useful when features in the sets come from multiple images and
              ``id`` is used to indicate the source image.

        :seealso: :class:`BaseFeature2D` :obj:`BaseFeature2D.id` :meth:`by_id1`
        """
        if self._inverse_dict2 is None:
            # first call, build a dict for efficient mapping
            d: dict[int, int] = {}
            for k, m in enumerate(self._matches):
                d[m[1]] = k
            self._inverse_dict2 = d
        try:
            return self[self._inverse_dict2[i]]
        except KeyError:
            return None

    def __str__(self) -> str:
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
            # p1 and p2 are 1D arrays when there is a single match
            p1 = np.ravel(self.p1)
            p2 = np.ravel(self.p2)
            return (
                f"{self.status} {self.distance:6.2f}: ({p1[0]:.1f},"
                f" {p1[1]:.1f}) <--> ({p2[0]:.1f}, {p2[1]:.1f})"
            )
        else:
            s = f"{len(self)} matches"
            if self._inliers is not None:
                ninlier = sum(self._inliers)
                s += f", with {ninlier} ({ninlier / len(self) * 100:.1f}%) inliers"
            return s

    def __repr__(self) -> str:
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
    def status(self) -> "builtins.list[str]":
        """
        Inlier status of matches

        :return: ``"+"`` for an inlier and ``"-"`` for an outlier, or ``""`` if the
            status has not been set
        :rtype: str or ndarray(N) of str

        A single string for one match, otherwise an array with one string per match.

        .. versionchanged:: 2.5.0
            Works for more than one match.  It used to raise ``ValueError``.
        """
        if self._inliers is not None:
            return ["+" if inlier else "-" for inlier in self._inliers]
        else:
            return [""] * len(self)

    def list(self) -> None:
        """
        List matches

        Print the matches in a simple format, one line per match.

        :seealso: :meth:`table`
        """
        for i, m in enumerate(self._matches):
            # TODO shouldn't have to flatten
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

    def table(self) -> None:
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
            # TODO shouldn't have to flatten
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
    def inliers(self) -> "FeatureMatch":
        """
        Extract inlier matches

        :return: new match object containing only the inliers
        :rtype: :class:`FeatureMatch` instance

        .. note:: Inlier/outlier status is typically set by some RANSAC-based
            algorithm that applies a geometric constraint to the sets of
            putative matches.

        :seealso: :obj:`CentralCamera.points2F`
        """
        if self._inliers is None:
            raise ValueError("inlier status has not been set")
        return self[self._inliers]

    @property
    def outliers(self) -> "FeatureMatch":
        """
        Extract outlier matches

        :return: new match object containing only the outliers
        :rtype: :class:`FeatureMatch` instance

        .. note:: Inlier/outlier status is typically set by some RANSAC-based
            algorithm that applies a geometric constraint to the sets of
            putative matches.

        :seealso: :obj:`entralCamera.points2F`
        """
        if self._inliers is None:
            raise ValueError("inlier status has not been set")
        return self[~self._inliers]

    def subset(self, N: int = 100) -> "FeatureMatch":
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

    def plot(
        self,
        *pos: Any,
        darken: bool = False,
        ax: Any = None,
        width: Any = None,
        block: bool = False,
        title: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot matches

        :param darken: darken the underlying , defaults to False
        :type darken: bool, optional
        :param width: figure width in millimetres, defaults to Matplotlib default
        :type width: float, optional
        :param block: Matplotlib figure blocks until window closed, defaults to False
        :type block: bool, optional
        :param title: window title, passed to :meth:`Image.disp`, defaults to
            "Machine Vision Toolbox for Python"
        :type title: str, optional

        Displays the original pair of images side by side, as greyscale images,
        and overlays the matches.

        """

        kp1 = self._kp1
        kp2 = self._kp2
        im1 = kp1._image
        im2 = kp2._image

        combo, u = im1.__class__.Hstack((im1.mono(), im2.mono()), return_offsets=True)
        if ax is None:
            combo.disp(darken=darken, width=width, block=None, title=title)
        else:
            plt.sca(ax)

        # for m in self:
        #     p1 = m.pt1
        #     p2 = m.pt2
        #     plt.plot((p1[0], p2[0] + u[1]), (p1[1], p2[1]), *pos, **kwargs)
        p1 = self.p1
        p2 = self.p2
        plt.plot((p1[0, :], p2[0, :] + u[1]), (p1[1, :], p2[1, :]), *pos, **kwargs)
        safe_plt_show(block=block)

    def plot_correspondence(
        self, *arg: Any, offset: tuple[int | float, int | float] = (0, 0), **kwargs: Any
    ) -> None:
        """
        Plot lines joining corresponding points

        :param arg: line style passed to Matplotlib, for example ``"g"``
        :param offset: offset (u,v) added to the points of the second set, for example
            the width of the first image when the two are shown side by side,
            defaults to (0, 0)
        :param kwargs: options passed to :func:`matplotlib.pyplot.plot`

        Draws a line from each point in the first set to its match in the second
        set, in the current figure.
        """
        p1 = self.p1
        p2 = self.p2
        plt.plot(
            (p1[0, :], p2[0, :] + offset[0]),
            (p1[1, :], p2[1, :] + offset[1]),
            *arg,
            **kwargs,
        )
        plt.draw()

    def estimate(
        self, func: Any, method: str = "ransac", **args: Any
    ) -> tuple[Any, ...]:
        """
        Estimate a geometric relationship from the matches

        :param func: estimation function, for example
            :meth:`~machinevisiontoolbox.Camera.CentralCamera.points2F` or
            :meth:`~machinevisiontoolbox.Camera.CentralCamera.points2H`
        :param method: robust estimation method passed to ``func``, defaults to ``"ransac"``
        :param args: further options passed to ``func``
        :return: the estimate and any other values that ``func`` returns, but not
            the inlier mask if there is one

        Calls ``func(p1, p2, method=method, **args)`` with the corresponding points.
        If the last value that ``func`` returns is an inlier mask, a boolean array
        with one element per match, which the robust methods ``"ransac"`` and
        ``"lmeds"`` return, it is kept as the inlier status of the matches (see
        :attr:`inliers` and :attr:`outliers`) and is not returned. Otherwise, for
        example with ``method="8p"``, everything that ``func`` returns is returned and
        the inlier status is cleared.

        .. versionchanged:: 2.5.0
            A non-robust method used to lose its last return value (the residual, or
            the estimate itself if there was no residual) and leave it as the inlier
            status.

        Example:

        .. code-block:: python

            F, resid = matches.estimate(CentralCamera.points2F, method="ransac", confidence=0.95)
        """
        solution = func(self.p1, self.p2, method=method, **args)
        if not isinstance(solution, (tuple, list)):
            solution = (solution,)  # a single value

        last = solution[-1]
        if (
            isinstance(last, np.ndarray)
            and last.dtype == bool
            and last.shape == (len(self),)
        ):
            self._inliers = last
            return tuple(solution[:-1])
        else:
            self._inliers = None  # no mask, so any earlier one no longer applies
            return tuple(solution)

    @property
    @scalar_result
    def distance(self) -> "builtins.list[float]":
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
    def p1(self) -> "builtins.list[np.ndarray]":
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
    def p2(self) -> "builtins.list[np.ndarray]":
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
    def descriptor1(self) -> "builtins.list[np.ndarray | None]":
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
    def descriptor2(self) -> "builtins.list[np.ndarray | None]":
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
    def id1(self) -> "builtins.list[int]":
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
    def id2(self) -> "builtins.list[int]":
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


class SURFFeature(BaseFeature2D):
    """
    Create set of SURF point features

    .. inheritance-diagram:: machinevisiontoolbox.ImagePointFeatures.SURFFeature
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


# --- OpenCV 4/5 compat -------------------------------------------------
# OpenCV 5 dropped BRISK/AKAZE/KAZE from the main cv2 namespace; they now
# live under cv2.xfeatures2d (verified empirically 2026-08-02, opencv-
# contrib-python 5.0.0.93 vs 4.13.0.92 -- no OPENCV_ENABLE_NONFREE build
# flag needed for these, unlike SURF). SIFT/ORB are unaffected, stay in
# main cv2 in both versions. Resolve by attribute existence, not a
# version-string check, so this keeps working if a future OpenCV release
# moves things back or a build has both.
def _resolve_feature_create(name: str) -> Any:
    if hasattr(cv2, f"{name}_create"):
        return getattr(cv2, f"{name}_create")
    if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, f"{name}_create"):
        return getattr(cv2.xfeatures2d, f"{name}_create")
    raise AttributeError(f"{name}_create not found in cv2 or cv2.xfeatures2d")


def _SURF_create(**kwargs: Any) -> Any:
    """Create an OpenCV SURF detector, or explain why it is not available.

    SURF is patented.  It is in the contrib modules (``cv2.xfeatures2d``), and even
    then only when OpenCV was built with ``OPENCV_ENABLE_NONFREE``, which the
    ``opencv-contrib-python`` wheels on PyPI are not.
    """
    try:
        # without the contrib modules, this raises ContribNotAvailableError,
        # whose message explains what to install
        return cv2.xfeatures2d.SURF_create(**kwargs)  # type: ignore[attr-defined]
    except cv2.error as e:
        raise RuntimeError(
            "SURF is not available in this OpenCV: it is patented and is left out of "
            "standard builds, including the opencv-contrib-python wheels on PyPI.  "
            "Use a build of OpenCV with its contrib modules and the CMake option "
            "OPENCV_ENABLE_NONFREE=ON, or use another detector such as SIFT, ORB, "
            "BRISK or AKAZE."
        ) from e


# -------------------------------------------------------------------------


def _set_keypoint_ids(kp: list, image: Any, id: str | int) -> None:
    """Set the ``class_id`` of each keypoint, as the ``id`` argument of the features says"""
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
        raise ValueError(f"id must be 'image', 'index' or an int, got {id!r}")


class ImagePointFeaturesMixin(ImageMixinBase):
    """
    Methods of :class:`Image` that find point features, such as SIFT, ORB and Harris
    """

    def _image2feature(
        self,
        cls: type[BaseFeature2D],
        sortby: str | None = None,
        nfeat: int | None = None,
        id: str | int = "image",
        **kwargs: Any,
    ) -> Any:
        # check if image is valid
        # TODO, MSER can handle color
        image = self.mono()

        # get a reference to the appropriate detector
        algorithm = cls.__name__.replace("Feature", "")
        try:
            factory, has_scale, has_orient = _DETECTORS[algorithm]
        except KeyError:
            raise ValueError("bad algorithm specified") from None

        # every keyword argument belongs to the detector, none is claimed here
        detector = factory(**kwargs)

        kp, des = detector.detectAndCompute(image._A, mask=None)

        # kp is a list of N KeyPoint objects
        # des is NxM ndarray of keypoint descriptors

        _set_keypoint_ids(kp, image, id)

        # do sorting in here

        if nfeat is not None:
            kp = kp[:nfeat]
            des = des[:nfeat, :]

        # construct a new Feature2DBase subclass
        features = cls(kp, des, scale=has_scale, orient=has_orient)

        # add attributes
        features._feature_type = algorithm
        features._image = self

        return features

    def SIFT(self, **kwargs: Any) -> "SIFTFeature":
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
            - |RVC3|, Section 14.1.


        :seealso: :class:`SIFTFeature` `cv2.SIFT_create <https://docs.opencv.org/4.5.2/d7/d60/classcv_1_1SIFT.html>`_
        """

        return self._image2feature(SIFTFeature, **kwargs)

    def SURF(self, **kwargs: Any) -> "SURFFeature":
        """
        Find SURF features in image

        .. inheritance-diagram:: machinevisiontoolbox.ImagePointFeatures.SURFFeature
            :top-classes: machinevisiontoolbox.ImagePointFeatures.BaseFeature2D
            :parts: 1

        :param kwargs: arguments passed to OpenCV, for example ``hessianThreshold``
        :return: set of 2D point features
        :rtype: :class:`SURFFeature`
        :raises RuntimeError: SURF is not available in this OpenCV
        :raises AttributeError: OpenCV's contrib modules are not installed

        Returns an iterable and sliceable object that contains SURF features and
        descriptors.

        .. warning:: SURF is patented, so it is excluded from standard OpenCV builds.
            It needs OpenCV's contrib modules and a build with the CMake option
            ``OPENCV_ENABLE_NONFREE=ON``, which the ``opencv-contrib-python`` wheels on
            PyPI are not.  If it is not available this method raises an exception that
            says so.  :meth:`SIFT` is a freely available alternative.

        Example:

        .. code-block:: python

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> surf = img.SURF()
            >>> len(surf)  # number of features

        :references:
            - SURF: Speeded up robust features.
              Herbert Bay, Tinne Tuytelaars, and Luc Van Gool.
              In European Conference on Computer Vision (ECCV), 2006.
            - |RVC3|, Section 14.1.

        :seealso: :class:`SURFFeature` :meth:`SIFT` `cv2.xfeatures2d.SURF_create <https://docs.opencv.org/4.x/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html>`_

        .. versionadded:: 2.5.0
        """
        return self._image2feature(SURFFeature, **kwargs)

    def ORB(self, scoreType: str = "harris", **kwargs: Any) -> "ORBFeature":
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

        scoreoptions = {"harris": cv2.ORB_HARRIS_SCORE, "fast": cv2.ORB_FAST_SCORE}
        return self._image2feature(
            ORBFeature,
            scoreType=scoreoptions[scoreType],
            **kwargs,
        )

    def BRISK(self, **kwargs: Any) -> "BRISKFeature":
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

    def AKAZE(self, **kwargs: Any) -> "AKAZEFeature":
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

    def Harris(self, **kwargs: Any) -> "HarrisFeature":
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
        strength image.  The descriptor is a unit-normalized vector of image
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
            - |RVC3|, Section 12.3.1.

        :seealso: :class:`HarrisFeature`
        """
        return self._image2feature(HarrisFeature, **kwargs)

    def ComboFeature(
        self,
        detector: str,
        descriptor: str,
        det_opts: dict[str, Any] | None = None,
        des_opts: dict[str, Any] | None = None,
        nfeat: int | None = None,
        id: str | int = "image",
    ) -> BaseFeature2D:
        """
        Find features using one algorithm to detect and another to describe

        :param detector: name of the detector, one of ``"FAST"``, ``"AGAST"``,
            ``"GFTT"``, ``"Harris"``, ``"SIFT"``, ``"ORB"``, ``"BRISK"`` or
            ``"AKAZE"``
        :param descriptor: name of the descriptor, one of ``"SIFT"``, ``"ORB"``,
            ``"BRISK"``, ``"AKAZE"``, ``"FREAK"``, ``"BRIEF"``, ``"DAISY"``,
            ``"LATCH"``, ``"LUCID"`` or ``"BOOST"``
        :param det_opts: options passed to the detector, defaults to None
        :param des_opts: options passed to the descriptor, defaults to None
        :param nfeat: keep at most this many of the strongest features, defaults to
            None, which keeps all of them.  A few can be lost, because a descriptor
            needs a margin of image around a point
        :param id: what to put in each feature's ``id``: ``"image"`` (the image id),
            ``"index"`` (0, 1, ...) or an int, defaults to ``"image"``
        :raises ValueError: unknown detector or descriptor, a pair that cannot be
            combined, or a descriptor that needs a color image given a greyscale one
        :return: set of 2D point features
        :rtype: :class:`BaseFeature2D` subclass

        Detect features with ``detector`` and describe each one with ``descriptor``.
        Most detectors, such as FAST, find corners but give no description of them,
        and most descriptors, such as BRIEF, describe the image around a point
        but do not find one.  This method joins any detector to any descriptor.

        The result is a set of the class of the descriptor, for example
        :class:`BRIEFFeature`, and can be matched with another set made with the
        same descriptor.  The detector can differ, and the descriptors of the two
        sets must be of the same kind.  A feature set has scale and orientation only
        if the detector provides them (SIFT, ORB, BRISK, AKAZE), or the descriptor
        computes an orientation (BRISK, FREAK).

        Two combinations cannot be made.  The AKAZE descriptor uses information that
        only the AKAZE detector records, and the ORB descriptor cannot use the
        keypoints of the SIFT detector.  The LUCID descriptor describes a color
        patch, so it needs a color image.

        .. rubric:: Examples

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("eiffel-1.png")
            >>> features = img.ComboFeature("FAST", "BRIEF", nfeat=500)
            >>> len(features)
            >>> features.descriptor.shape

        .. versionadded:: 2.5.0

        .. seealso:: :class:`BOOSTFeature` :class:`BRIEFFeature` :class:`DAISYFeature` :class:`FREAKFeature` :class:`LATCHFeature` :class:`LUCIDFeature`
        """
        try:
            det_factory, has_scale, has_orient = _COMBO_DETECTORS[detector]
        except KeyError:
            raise ValueError(
                f"unknown detector {detector!r}, must be one of "
                f"{', '.join(_COMBO_DETECTORS)}"
            ) from None
        try:
            des_factory, cls, needs_color, finds_orient = _DESCRIPTORS[descriptor]
        except KeyError:
            raise ValueError(
                f"unknown descriptor {descriptor!r}, must be one of "
                f"{', '.join(_DESCRIPTORS)}"
            ) from None
        reason = _unsupported_pair(detector, descriptor)
        if reason is not None:
            raise ValueError(f"cannot combine {detector} with {descriptor}: {reason}")
        if needs_color and self.nplanes != 3:
            raise ValueError(
                f"{descriptor} describes a color patch, so needs a color image"
            )

        mono = self.mono()
        kp = list(det_factory(**(det_opts or {})).detect(mono._A, None))
        if nfeat is not None:
            kp = sorted(kp, key=lambda k: k.response, reverse=True)[:nfeat]

        # the descriptor may drop keypoints too close to the edge
        kp, des = des_factory(**(des_opts or {})).compute(
            self._A if needs_color else mono._A, kp
        )

        # after description, since the AKAZE descriptor uses class_id
        _set_keypoint_ids(kp, mono, id)

        features = cls(kp, des, scale=has_scale, orient=has_orient or finds_orient)
        features._feature_type = descriptor
        features._image = self
        return features


class _Harris_create:
    def __init__(
        self,
        nfeat: int = 250,
        k: float = 0.04,
        scale: int = 7,
        hw: int = 2,
        patch: int = 5,
    ) -> None:
        self.nfeat = nfeat
        self.k = k
        self.hw = hw
        self.peakscale = scale
        self.patch = patch
        self.scale = None

    def detect(
        self, image: np.ndarray, mask: np.ndarray | None = None
    ) -> list[cv2.KeyPoint]:
        return self.detectAndCompute(image, mask)[0]

    def detectAndCompute(
        self, image: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        # features are peaks in the Harris corner strength image
        dst = cv2.cornerHarris(src=image, blockSize=2, ksize=2 * self.hw + 1, k=self.k)
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
                    kp.append(cv2.KeyPoint(x, y, 0, 0, peak[2]))
            except IndexError:
                # handle the case where the descriptor window falls off the edge
                pass

        for i, d in enumerate(des):
            if d.shape != des[1].shape:
                print(i, d.shape)

        return kp, np.array(des)


if __name__ == "__main__":
    from pathlib import Path

    import pytest

    pytest.main(
        [
            str(
                Path(__file__).parent.parent.parent
                / "tests"
                / "test_image_point_features.py::TestImagePointFeatures"
            ),
            "-v",
        ]
    )


# What each detector behind Image.SIFT(), Image.ORB() etc. is built from, and what it
# provides.  The flags are facts about the algorithm, so they live here rather than
# being passed in by each caller (where a keyword such as ``scale`` would collide with
# an option of the detector itself, as Harris has one).  The factories are resolved
# only when used, so that a detector missing from this OpenCV (OpenCV 5 without the
# contrib modules) does not break the others.
#
#   name: (factory, has scale, has orientation)
_DETECTORS: dict[str, tuple[Any, bool, bool]] = {
    "SIFT": (lambda **kw: getattr(cv2, "SIFT_create")(**kw), True, True),
    "SURF": (_SURF_create, True, True),
    "ORB": (lambda **kw: getattr(cv2, "ORB_create")(**kw), True, True),
    "BRISK": (lambda **kw: _resolve_feature_create("BRISK")(**kw), True, True),
    "AKAZE": (lambda **kw: _resolve_feature_create("AKAZE")(**kw), True, True),
    "Harris": (_Harris_create, False, False),
}


def _contrib_create(name: str) -> Any:
    """A factory for an OpenCV algorithm that lives in the contrib modules in OpenCV 5"""

    def create(**kwargs: Any) -> Any:
        try:
            return _resolve_feature_create(name)(**kwargs)
        except AttributeError:
            raise RuntimeError(
                f"{name} is not available in this OpenCV, which needs its contrib "
                "modules: pip install opencv-contrib-python"
            ) from None

    return create


# Detectors that Image.ComboFeature can join to a descriptor, each of which has a
# ``detect`` method, with what the keypoints it finds provide.
#
#   name: (factory, has scale, has orientation)
_COMBO_DETECTORS: dict[str, tuple[Any, bool, bool]] = {
    "FAST": (
        lambda **kw: getattr(cv2, "FastFeatureDetector_create")(**kw),
        False,
        False,
    ),
    "AGAST": (_contrib_create("AgastFeatureDetector"), False, False),
    "GFTT": (lambda **kw: getattr(cv2, "GFTTDetector_create")(**kw), False, False),
    "Harris": (_Harris_create, False, False),
    "SIFT": _DETECTORS["SIFT"],
    "ORB": _DETECTORS["ORB"],
    "BRISK": _DETECTORS["BRISK"],
    "AKAZE": _DETECTORS["AKAZE"],
}

# Descriptors that Image.ComboFeature can join to a detector, each of which has a
# ``compute`` method.
#
#   name: (factory, class of the result, needs a color image, finds an orientation)
_DESCRIPTORS: dict[str, tuple[Any, type[BaseFeature2D], bool, bool]] = {
    "SIFT": (_DETECTORS["SIFT"][0], SIFTFeature, False, False),
    "ORB": (_DETECTORS["ORB"][0], ORBFeature, False, False),
    "BRISK": (_DETECTORS["BRISK"][0], BRISKFeature, False, True),
    "AKAZE": (_DETECTORS["AKAZE"][0], AKAZEFeature, False, False),
    "FREAK": (_contrib_create("FREAK"), FREAKFeature, False, True),
    "BRIEF": (_contrib_create("BriefDescriptorExtractor"), BRIEFFeature, False, False),
    "DAISY": (_contrib_create("DAISY"), DAISYFeature, False, False),
    "LATCH": (_contrib_create("LATCH"), LATCHFeature, False, False),
    "LUCID": (_contrib_create("LUCID"), LUCIDFeature, True, False),
    "BOOST": (_contrib_create("BoostDesc"), BOOSTFeature, False, False),
}


def _unsupported_pair(detector: str, descriptor: str) -> str | None:
    """Why this detector cannot be joined to this descriptor, or None if it can"""
    if descriptor == "AKAZE" and detector != "AKAZE":
        return (
            "the AKAZE descriptor needs the scale space index that only the AKAZE "
            "detector records"
        )
    if descriptor == "ORB" and detector == "SIFT":
        return "the ORB descriptor cannot use the keypoints of the SIFT detector"
    return None
