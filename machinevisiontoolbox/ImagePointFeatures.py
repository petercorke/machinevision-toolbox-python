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
from machinevisiontoolbox.base import peak2
# from machinevisiontoolbox.classes import Image

# from machinevisiontoolbox.Image import *
# from machinevisiontoolbox.Image import Image


# TODO, either subclass SIFTFeature(BaseFeature2D) or just use BaseFeature2D
# directly

class BaseFeature2D:
    """
    A 2D point feature class
    """
    # # list of attributes
    # _u = []             # horizontal image coordinate
    # _v = []             # vertical image coordinate
    # _strength = []      # feature strength
    # _orientation = []   # feature orientation [rad]
    # _scale = []         # feature scale
    # _octave = []        # octave pyramid octave in which keypoint was detected
    # # TODO not sure if this is entirely useful for the user
    # _descriptor = []    # feature desciptor vector



    # _siftparameters = []  # dictionary for parameters and values used for sift
    # # feature extraction

    # _kp = []  # keypoints of sift (for interfacing with opencv functions)

    # _image = None

    def __init__(self, arg=None, detector=None, sortby=None, nfeat=None, id='image', **kwargs):

        # TODO flesh out sortby option, it can be by strength or scale
        # TODO what does nfeatures option to SIFT do? seemingly nothing

        self._has_scale = False
        self._has_orient = False

        if arg is None:
            # initialise empty Sift
            self._feature_type = None
            self._kp = None
            self._descriptor = None

        elif isinstance(arg, list):
            # TODO not sure what this is for
            self._feature_type = None
            self._kp = [f._kp[0] for f in arg]
            self._descriptor = np.array([f._descriptor for f in arg])

        elif type(arg).__name__ == 'Image': 
            detectors = {
                'SIFT': cv.SIFT_create,
                'ORB': cv.ORB_create,
                'MSER': cv.MSER_create,
                'Harris': _Harris_create
            }
            # check if image is valid
            # TODO, MSER can handle color
            image = arg.mono()
            self._feature_type = detector
            # get a reference to the appropriate detector
            # make it case insensitive
            try:
                self._detector = detectors[detector](**kwargs)
            except KeyError:
                raise ValueError('bad detector specified')
            
            self._image = image

            if detector == "mser":
                msers, bboxes = self._detector.detectRegions(image.A)
                # returns different things, msers is a list of points
                # u, v, point=centroid, scale=area
                # https://www.toptal.com/machine-learning/real-time-object-detection-using-mser-in-ios
            else:
                kp, des = self._detector.detectAndCompute(image.A, mask=None)
            if id == 'image':
                if arg.id is not None:
                    # copy image id into the keypoints
                    for k in kp:
                        k.class_id = arg.id
            elif id == 'index':
                for i, k in enumerate(kp):
                        k.class_id = i
            elif isinstance(id, int):
                for k in kp:
                        k.class_id = id
            else:
                raise ValueError('bad id')

            # do sorting in here
            
            if nfeat is not None:
                kp = kp[:nfeat]
                des = des[:nfeat, :]

            self._kp = kp
            self._descriptor = des


        else:
            raise TypeError('bad argument')

    def __len__(self):
        """
        Number of features

        :return: number of features
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read("eiffel2-1.png")
            >>> orb = im.ORB()
            >>> len(orb)  # number of features

        :seealso: :meth:`.__getitem__`
        """
        return len(self._kp)

    def __getitem__(self, i):
        """
        Get item from point feature object

        :param i: index
        :type i: int or slice
        :raises IndexError: index out of range
        :return: subset of point features
        :rtype: BaseFeature2D instance

        This method allows a ``BaseFeature2D`` object to be indexed, sliced or iterated.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read("eiffel2-1.png")
            >>> orb = im.ORB()
            >>> print(orb[:5])  # first 5 ORB features
            >>> print(orb[::50])  # every 50th ORB feature

        :seealso: :meth:`.__len__`
        """
        new = self.__class__()
        new._feature_type = self._feature_type
        if isinstance(i, int):
            new._kp = [self._kp[i]]
        elif isinstance(i, slice):
            new._kp = self._kp[i]
        elif isinstance(i, np.ndarray):
            if np.issubdtype(i.dtype, np.bool):
                new._kp = [self._kp[k] for k, true in enumerate(i) if true]
            elif np.issubdtype(i, np.integer):
                new._kp = [self._kp[k] for k in i]
        elif isinstance(i, (list, tuple)):
            new._kp = [self._kp[k] for k in i]
        if len(self._descriptor.shape) == 1:
            new._descriptor = self._descriptor
        else:
            new._descriptor = self._descriptor[i, :]
        new._has_scale = self._has_scale
        new._has_orient = self._has_orient

        return new

    def __str__(self):
        if len(self) > 1:
            return f"{self._feature_type} features, {len(self)} points"
        else:
            s = f"{self._feature_type} feature: ({self.u:.1f}, {self.v:.1f}), strength={self.strength:.2f}"
            if self._has_scale:
                s += f", scale={self.scale:.1f}"
            if self._has_orient:
                s += f", orient={self.orientation:.1f}°"
            s += f", id={self.id}"
            return s
    def __repr__(self):
        return str(self)

    def list(self):
        for i, f in enumerate(self):
            s = f"{self._feature_type} feature {i}: ({f.u:.1f}, {f.v:.1f}), strength={f.strength:.2f}"
            if f._has_scale:
                s += f", scale={f.scale:.1f}"
            if f._has_orient:
                s += f", orient={f.orientation:.1f}°"
            s += f", id={f.id}"
            print(s)
            
    def table(self):
        """
        Print features in tabular form

        Each row is:
            - the index in the feature vector
            - centroid
            - strength
            - scale
            - image id
        """
        columns = [
                Column("#"),
                Column("centroid"),
                Column("strength", fmt="{:.3g}")
        ]
        if self._has_scale:
            columns.append(
                    Column("scale", fmt="{:.3g}")
            )
        if self._has_orient:
            columns.append(
                    Column("orient", fmt="{:.3g}°")
            )
        columns.append(Column("id", fmt="{:d}"))
        table = ANSITable(*columns, border="thin")
        for i, f in enumerate(self):
            values = [f.strength]
            if self._has_scale:
                values.append(f.scale)
            if self._has_orient:
                values.append(f.orientation)

            table.row(i, f"{f.u:.1f}, {f.v:.1f}",
                    *values,
                    f.id)
        table.print()

    def gridify(self, nbins, nfeat):

        try:
            nw, nh = nbins
        except:
            nw = nbins
            nh = nbins

        image = self._image
        binwidth = image.width // nw
        binheight = image.height // nh

        keep = []
        bins = np.zeros((nh, nw), dtype='int')

        for f in self.features:
            ix = f.p[0] // binwidth
            iy = f.p[1] // binheight

            if bins[iy, ix] < nfeat:
                keep.append(f)
                bins[iy, ix] += 1

        return self.__class__(keep)

    def __add__(self, other):
        if isinstance(other, list) and len(other) == 0:
            return self

        if self._feature_type != other._feature_type:
            raise TypeError('cant add different feature types:', self._feature_type, other._feature_type)
        new = self.__class__()
        new._feature_type = self._feature_type

        new._kp = self._kp + other._kp
        new._descriptor = np.vstack((self._descriptor, other._descriptor))

        return new

    def __radd__(self, other):

        if isinstance(other, list) and len(other) == 0:
            return self
        else:
            raise ValueError('bad')

    def distance(self, other, metric="L2"):
        metric_dict = {'L1': 1, 'L2': 2}

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
            des2 =  other._descriptor

        for i in range(n1):
            for j in range(n2):
                if metric == 'ncc':
                    d = np.dot(des1[i, :], des2[j, :])
                else:
                    d = np.linalg.norm(des1[i, :] - des2[j, :],
                        ord=metric_dict[metric])
                D[i, j] = d
                D[j, i] = d
        return D

    @property
    def u(self):
        """
        Horizontal coordinate of feature point

        :return: Horizontal coordinate
        :rtype: float or list of float

        .. note:: For multiple features return a list
        """
        u = [kp.pt[0] for kp in self._kp]
        if len(u) == 1:
            return u[0]
        else:
            return u

    @property
    def v(self):
        """
        Vertical coordinate of feature point

        :return: Vertical coordinate
        :rtype: float or list of float

        .. note:: For multiple features return a list
        """
        v = [kp.pt[1] for kp in self._kp]
        if len(v) == 1:
            return v[0]
        else:
            return v

    @property
    def id(self):
        """
        Image id for feature point

        :return: image id
        :rtype: int

        .. note:: Defined by the ``id`` attribute of the image passed to the
            feature detector
        """
        id = [kp.class_id for kp in self._kp]
        if len(id) == 1:
            return id[0]
        else:
            return id

    @property
    def orientation(self):
        """
        Orientation of feature

        :return: Orientation in degrees
        :rtype: float or list of float

        .. note:: For multiple features return a list
        """
        # TODO should be in radians
        angle = [kp.angle for kp in self._kp]
        if len(angle) == 1:
            return angle[0]
        else:
            return np.radians(angle)

    @property
    def scale(self):
        """
        Scale of feature

        :return: Scale
        :rtype: float or list of float

        .. note:: For multiple features return a list
        """
        scale = [kp.size for kp in self._kp]
        if len(scale) == 1:
            return scale[0]
        else:
            return scale

    @property
    def strength(self):
        """
        Strength of feature

        :return: Strength
        :rtype: float or list of float

        .. note:: For multiple features return a list
        """
        strength = [kp.response for kp in self._kp]
        if len(strength) == 1:
            return strength[0]
        else:
            return strength

    @property
    def octave(self):
        octave = [kp.octave for kp in self._kp]
        if len(octave) == 1:
            return octave[0]
        else:
            return octave

    @property
    def descriptor(self):
        """
        Descriptor of feature

        :return: Descriptor
        :rtype: ndarray(m,n)

        .. note:: For single feature return a column vector, for multiple features return a set of column vectors.
        """
        return self._descriptor

    @property
    def p(self):
        """
        Feature coordinates

        :return: Feature centroids as matrix columns
        :rtype: ndarray(2,N)
        """
        return np.vstack([kp.pt for kp in self._kp]).T


    def drawKeypoints(self,
                      image,
                      kp=None,
                      drawing=None,
                      isift=None,
                      flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                      **kwargs):
        # draw sift features on image using cv.drawKeypoints

        # check valid imagesource
        # TODO if max(self._u) or max(self._v) are greater than image width,
        # height, respectively, then raise ValueError

        # TODO check flags, setup dictionary or string for plot options

        if drawing is None:
            drawing = np.zeros((image.shape[0], image.shape[1], 3),
                               dtype=np.uint8)

        if kp is None:
            kp = self._kp

        if isift is None:
            isift = np.arange(0, len(self._kp))  # might need a +1 here
        else:
            isift = np.array(isift, ndmin=1, copy=True)

        # TODO should check that isift is consistent with kp (min value is 0,
        # max value is <= len(kp))
        cv.drawKeypoints(image.image,
                         kp[isift],
                         drawing,
                         flags=flags,
                         **kwargs)

        return image.__class__(drawing)

    # TODO def draw descriptors? (eg vl_feat, though mvt-mat doesn't have this)
    # TODO descriptor distance
    # TODO descriptor similarity
    # TODO display/print/char function?

    def match(self, other, ratio=0.75, crosscheck=False, metric='L2', sort=True, top=None, thresh=None):
        """
        Match point features

        :param other: set of feature points
        :type other: BaseFeature2D
        :param ratio: parameter for Lowe's ratio test, defaults to 0.75
        :type ratio: float, optional
        :param crosscheck: perform left-right cross check, defaults to False
        :type crosscheck: bool, optional
        :param metric: distance metric, 'L1', 'L2' [default], 'hamming', 'hamming2'
        :type metric: str, optional
        :param sort: sort features by strength, defaults to True
        :type sort: bool, optional
        :raises ValueError: bad metric name provided
        :return: set of candidate matches
        :rtype: Match instance

        If crosscheck is True the ratio test is disabled

        ``f1.match(f2)`` is a match object 
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
            'L1': cv.NORM_L1,
            'L2': cv.NORM_L2,
            'hamming': cv.NORM_HAMMING,
            'hamming2': cv.NORM_HAMMING2,
        }
        if metric not in metricdict:
            raise ValueError('bad metric name')

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

        return FeatureMatch([(m.queryIdx, m.trainIdx, m.distance) for m in good], self, other)

    def drawMatches(self,
                    im1,
                    sift1,
                    im2,
                    sift2,
                    matches,
                    **kwargs):
        # TODO should I just have input two SIFT objects,
        # or in this case just another SIFT object?

        # draw_params = dict(matchColor=(0, 255, 0),
        #                   singlePointColor=(255, 0, 0),
        #                   matchesMask=matches,
        #                   flags=0)

        out = cv.drawMatchesKnn(im1.image,
                                sift1._kp,
                                im2.image,
                                sift2._kp,
                                matches,
                                None,
                                **kwargs)

        return im1.__class__(out)
    
    def plot(self, *args, ax=None, filled=False,
        hand=False, handcolor='blue', handthickness=1, handalpha=1, **kwargs):

        ax = smb.axes_logic(ax, 2)

        if filled:
            for kp in self:
                centre = kp.p.flatten()
                c = plt.Circle(centre, radius=kp.scale, clip_on=True, **kwargs)
                ax.add_patch(c)
                if hand:
                    circum = centre + kp.scale * np.r_[math.cos(kp.orientation), math.sin(kp.orientation)]
                    l = plt.Line2D((centre[0], circum[0]), (centre[1], circum[1]), color=handcolor, linewidth=handthickness, alpha=handalpha)
                    ax.add_line(l)
        else:
            if len(args) == 0 and len(kwargs) == 0:
                kwargs = dict(marker='+y', markerfacecolor='none')
            smb.plot_point(self.p, *args, **kwargs)

    #     plt.draw()

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
        step = max(1, len(self)  // N)
        k = list(range(0, len(self), step))
        k = k[:N]
        new = self[k]
        new._feature_type = self._feature_type
        return new
        
    def sort(self, by='strength', descending=True, inplace=False):
        """
        Sort features

        :param by: sort by ``'strength'`` [default] or ``'scale'``
        :type by: str, optional
        :param descending: sort in descending order, defaults to True
        :type descending: bool, optional
        :return: sorted features
        :rtype: BaseFeature2D
        """
        # if by == 'strength':
        #     s = sorted(self, key=lambda f: f.strength, reverse=descending)
        # elif by == 'scale':
        #     s = sorted(self, key=lambda f: f.scale, reverse=descending)
        # else:
        #     raise ValueError('bad sort method', by)
        if by == 'strength':
            key = self.strength
        elif by == 'scale':
            key = self.scale
        else:
            raise ValueError('bad sort method', by)
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
        :type images: Image or list
        :param N: size of square window, defaults to 50
        :type N: int, optional
        :return: support region
        :rtype: Image

        The support region about the feature's centroid is extracted, 
        rotated and scaled.

        .. note:: If the features come from multiple images then the feature's
            ``id`` attribute is used to index into ``images`` which must be a
            list of Image objects.
        """

        from machinevisiontoolbox.classes import Image

        if len(self) > 1:
            raise ValueError('can only compute support region for single feature')
        
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
        M[2,2] = 1

        # translate to centre of window
        M = smb.transl2(N/2, N/2) @ M

        out = cv.warpAffine(src=image, M=M[:2,:], dsize=(N, N), flags=cv.INTER_LINEAR)
        return Image(out)

    def filter(self, **kwargs):
        """
        Filter features

        :param kwargs: the filter parameters
        :return: sorted features
        :rtype: BaseFeature2D

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

        """

        features = self

        for filter, limits in kwargs.items():
            if filter == 'scale':
                v = np.r_[features.scale]
                k = (limits[0] <= v) & (v <= limits[1])
            elif filter == 'minscale':
                v = np.r_[features.scale]
                k = v >= limits
            elif filter == 'maxscale':
                v = np.r_[features.scale]
                k = v <= limits
            elif filter == 'strength':
                v = np.r_[features.strength]
                k = (limits[0] <= v) & (v <= limits[1])
            elif filter == 'minstrength':
                v = np.r_[features.strength]
                k = limits[0] >= v
            elif filter == 'percentstrength':
                v = np.r_[features.strength]
                vmax = v.max()
                k = v >= vmax * limits / 100
            else:
                raise ValueError('unknown filter key', filter)

            features = features[k]
        
        return features

class FeatureMatch:

    def __init__(self, m, kp1, kp2, inliers=None):
        """
        Create feature match object

        :param m: a list of match tuples (id1, id2, distance)
        :type m: list of tuples (int, int, float)
        :param kp1: first set of feature keypoints
        :type kp1: BaseFeature2D
        :param kp2: second set of feature keypoints
        :type kp2: BaseFeature2D
        :param distance: [description], defaults to None
        :type distance: [type], optional

        A Match object can contains multiple correspondences which are
        essentially tuples (id1, id2, distance) where id1 and id2 are indices
        into the first and second feature sets that were matched. distance is
        the distance between the feature's descriptors, a measure of feature
        dissimilarity.

        ``kp1`` and ``kp2`` are arrays of OpenCV ``KeyPoint`` objects which have attributes
            - position (``pt``)
            - scale (``size``)
            - strength (``response``)

        A Match object:
            - has a length, the number of matches it contains
            - can be sliced to extract a subset of matches
            - can contain a mask vector indicating which matches are inliers
        
        Each feature has a
        position, strength, scale and id.

        :seealso: `cv2.KeyPoint <https://docs.opencv.org/4.5.2/d2/d29/classcv_1_1KeyPoint.html#a507d41b54805e9ee5042b922e68e4372>`_
        """
        self._matches = m
        self._kp1 = kp1
        self._kp2 = kp2
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
            >>> im = Image.Read("eiffel2-1.png")
            >>> orb = im.ORB()
            >>> print(orb[:5])  # first 5 ORB features
            >>> print(orb[::50])  # every 50th ORB feature

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
            if np.issubdtype(i.dtype, np.bool):
                matches = [m for m, g in zip(self._matches, i) if g]
                if self._inliers is not None:
                    inliers = [m for m, g in zip(self._inliers, i) if g]
            elif np.issubdtype(i.dtype, np.integer):
                matches = [self._matches[k] for k in i]
                if self._inliers is not None:
                    inliers = [self._inliers[k] for k in i]
        else:
            raise ValueError('bad index')
        return FeatureMatch(matches, self._kp1, self._kp2, inliers)

    def __len__(self):
        """
        Number of matches

        :return: number of matches
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read("eiffel2-1.png")
            >>> orb = im.ORB()
            >>> len(orb)  # number of features

        :seealso: :meth:`.__getitem__`
        """
        return len(self._matches)

    def correspondence(self):
        return np.array([m[:2] for m in self._matches]).T

    def by_id1(self, id):
        """
        Find match by feature id

        :param i: feature id
        :type i: int
        :return: match that includes feature ``id`` or None
        :rtype: Match object containing one correspondence

        A Match object can contains multiple correspondences which are
        essentially tuples (id1, id2) where id1 and id2 are indices into the
        first and second feature sets that were matched. Each feature has a
        position, strength, scale and id.

        This method returns the match that contains the feature in the first
        feature set with specific ``id``. If no such match exists it returns
        None.

        For efficient lookup, on the first call a dict is built that maps
        feature id to index in the feature set.

        :seealso: :meth:`by_id2`
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
        Find match by feature id

        :param i: feature id
        :type i: int
        :return: match that includes feature ``id`` or None
        :rtype: Match object containing one correspondence

        A Match object can contains multiple correspondences which are
        essentially tuples (id1, id2) where id1 and id2 are indices into the
        first and second feature sets that were matched. Each feature has a
        position, strength, scale and id.

        This method returns the match that contains the feature in the second
        feature set with specific ``id``. If no such match exists it returns
        None.

        For efficient lookup, on the first call a dict is built that maps
        feature id to index in the feature set.

        :seealso: :meth:`by_id1`
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
        if len(self) == 1:
            return f"{self.status} {self.distance:6.2f}: ({self.p1[0, 0]:.1f}, {self.p1[1, 0]:.1f}) <--> ({self.p2[0, 0]:.1f}, {self.p2[1, 0]:.1f})"
        else:
            s = f"{len(self)} matches"
            if self._inliers is not None:
                ninlier = sum(self._inliers)
                s += f", with {ninlier} ({ninlier/len(self)*100:.1f}%) inliers"
            return s

    def __repr__(self):
        return str(self)

    @property
    def status(self):
        if self._inliers is not None:
            return '+' if self._inliers else '-'
        else:
            return ''

    def list(self):
        for i, m in enumerate(self._matches):
            # TODO shouldnt have to flatten
            p1 = self._kp1[m[0]].p.flatten()
            p2 = self._kp2[m[1]].p.flatten()
            if self._inliers is not None:
                status = '+' if self._inliers[i] else '-'
            else:
                status = ''
            s = f"{i:3d}:  {status} {m[2]:6.2f} ({p1[0]:.1f}, {p1[1]:.1f}) <--> ({p2[0]:.1f}, {p2[1]:.1f})"
            print(s)
            
    def table(self):
        """
        Print matches in tabular form

        Each row is:
            - the index of the match
            - inlier/outlier status
            - strength
            - p1
            - p2
        """
        columns = [
                Column("#"),
                Column("inlier"),
                Column("strength", fmt="{:.3g}"),
                Column("p1", colalign="<", fmt="{:s}"),
                Column("p2", colalign="<", fmt="{:s}")
        ]
        table = ANSITable(*columns, border="thin")

        for i, m in enumerate(self._matches):
            # TODO shouldnt have to flatten
            p1 = self._kp1[m[0]].p.flatten()
            p2 = self._kp2[m[1]].p.flatten()
            if self._inliers is not None:
                status = '+' if self._inliers[i] else '-'
            else:
                status = ''
            table.row(i, status, m[2], f"({p1[0]:.1f}, {p1[1]:.1f})",
                f"({p2[0]:.1f}, {p2[1]:.1f})")
        table.print()


    @property
    def inliers(self):
        return self[self._inliers]

    @property
    def outliers(self):
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

    def plot(self, *pos, darken=True, width=None, **kwargs):
        """
        Plot matches

        Displays the original pair of images side by side and overlays the
        matches.
        """
        kp1 = self._kp1
        kp2 = self._kp2
        im1 = kp1._image
        im2 = kp2._image

        combo, u = im1.__class__.hcat(im1, im2, return_offsets=True)
        combo.disp(darken=darken, width=width)

        # for m in self:
        #     p1 = m.pt1
        #     p2 = m.pt2
        #     plt.plot((p1[0], p2[0] + u[1]), (p1[1], p2[1]), *pos, **kwargs)
        p1 = self.p1
        p2 = self.p2
        plt.plot((p1[0, :], p2[0, :] + u[1]), (p1[1, :], p2[1, :]), *pos, **kwargs)
        plt.draw()

    def plot_correspondence(self, *arg, offset=(0,0), **kwargs):
        p1 = self.p1
        p2 = self.p2
        plt.plot((p1[0, :], p2[0, :] + offset[0]), (p1[1, :], p2[1, :] + offset[1]), *arg, **kwargs)
        plt.draw()

    def estimate(self, func, method='ransac', **args):

        solution = func(self.p1, self.p2, method=method, **args)
        self._inliers = solution[-1]

        return solution[:-1]

    @property
    def distance(self):
        out = [m[2] for m in self._matches]
        if len(self) == 1:
            return out[0]
        else:
            return np.array(out)

    @property
    def p1(self):
        """
        Feature coordinate in first image

        :return: feature coordinate
        :rtype: ndarray(2,N)
        """
        out = [self._kp1[m[0]].p for m in self._matches]
        return np.hstack(out)

    @property
    def p2(self):
        """
        Feature coordinate in second image

        :return: feature coordinate
        :rtype: ndarray(2,N)
        """
        out = [self._kp2[m[1]].p for m in self._matches]
        return np.hstack(out)

    @property
    def descriptor1(self):
        """
        Feature coordinate in first image

        :return: feature coordinate
        :rtype: ndarray(2,N)
        """
        out = [self._kp1[m[0]] for m in self._matches]
        if len(out) == 1:
            return out[0]
        else:
            return out

    @property
    def descriptor2(self):
        """
        Feature coordinate in second image

        :return: feature coordinate
        :rtype: ndarray(2,N)
        """
        out = [self._kp2[m[1]] for m in self._matches]
        if len(out):
            return out[0]
        else:
            return out


class SIFTFeature(BaseFeature2D):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._has_scale = True
        self._has_orient = True

class ORBFeature(BaseFeature2D):
    pass

class MSERFeature(BaseFeature2D):
    pass

class HarrisFeature(BaseFeature2D):
    pass

class ImagePointFeaturesMixin:

    def SIFT(self,
             **kwargs):
        """
        Find SIFT features in image

        :param kwargs: arguments passed to `cv2.SIFT_create <https://docs.opencv.org/4.5.2/d7/d60/classcv_1_1SIFT.html>`_
        :return: set of 2D point features
        :rtype: BaseFeature2D

        Returns an iterable and sliceable object that contains 2D features with
        properties.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read("eiffel2-1.png")
            >>> sift = im.SIFT()
            >>> len(sift)  # number of features
            >>> print(sift[:5])

        :seealso: :class:`BaseFeature2D` `cv2.SIFT_create <https://docs.opencv.org/4.5.2/d7/d60/classcv_1_1SIFT.html>`_
        """

        return SIFTFeature(self,
                              detector="SIFT",
                              **kwargs)

    def ORB(self,
            scoreType='harris',
            **kwargs):
        """
        Find ORB features in image

        :param kwargs: arguments passed to `cv2.ORB_create <https://docs.opencv.org/4.5.2/db/d95/classcv_1_1ORB.html>`_
        :return: set of 2D point features
        :rtype: BaseFeature2D

        Returns an iterable and sliceable object that contains 2D features with
        properties.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read("eiffel2-1.png")
            >>> orb = im.ORB()
            >>> len(orb)  # number of features
            >>> print(orb[:5])

        :seealso: :func:`BaseFeature2D`, `cv2.ORB_create <https://docs.opencv.org/4.5.2/db/d95/classcv_1_1ORB.html>`_
        """

        scoreoptions = {'harris': cv.ORB_HARRIS_SCORE,
                        'fast': cv.ORB_FAST_SCORE}

        return ORBFeature(self,
                              detector="ORB",
                              scoreType=scoreoptions[scoreType],
                              **kwargs)

    def MSER(self, **kwargs):
        """
        Find MSER features in image

        :param kwargs: arguments passed to `cv2.MSER_create <https://docs.opencv.org/4.5.2/d3/d28/classcv_1_1MSER.html>`_
        :return: set of 2D point features
        :rtype: BaseFeature2D

        Returns an iterable and sliceable object that contains 2D features with
        properties.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read("eiffel2-1.png")
            >>> mser = im.MSER()
            >>> len(mser)  # number of features
            >>> print(mser[:5])

        :seealso: :func:`BaseFeature2D`, `cv2.MSER_create <https://docs.opencv.org/4.5.2/d3/d28/classcv_1_1MSER.html>`_

        """

        return MSERFeature(self,
                              detector='MSER',
                              **kwargs)

    # each detector should explitly list (&document) all its parameters

    def Harris(self, **kwargs):

        return HarrisFeature(self,
                              detector='Harris',
                              **kwargs)

    def Harris_corner_strength(self, k=0.04, hw=2):
        dst = cv.cornerHarris(self.mono().image, 2, 2 * hw + 1, k)
        return self.__class__(dst)
class _Harris_create:

    def __init__(self, nfeat=250, k=0.04, scale=7, hw=2, patch=5):
        self.nfeat = nfeat
        self.k = k
        self.hw = hw
        self.peakscale = scale
        self.patch = patch
        self.scale = None
        

    def detectAndCompute(self, image, mask=None):
        dst = cv.cornerHarris(image, 2, 2 * self.hw + 1, self.k)
        peaks = peak2(dst, npeaks=None, scale=self.peakscale, positive=True)
        kp = []
        des = []
        w = 2 * self.patch + 1
        w2 = w**2
        for peak in peaks:
            x  = int(round(peak[0]))
            y  = int(round(peak[1]))
            try:
                W = image[y-self.patch:y+self.patch+1, x-self.patch:x+self.patch+1]
                v = W.flatten()
                if W.size > 0:
                    if len(v) != w2:
                        # handle case where last subscript is outside image bound
                        continue
                    
                    des.append(smb.unitvec(v))
                    kp.append(cv.KeyPoint(x, y, self.scale, 0, peak[2]))
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

    # kp1 = Image.Read('eiffel2-1.png').SIFT()
    # kp2 = Image.Read('eiffel2-2.png').SIFT()

    # kp1 = Image.Read('eiffel2-1.png').Harris()

    # # d = kp1[0].distance(kp1[1])
    # # print(d)
    # d = kp1[0].distance(kp1[30])
    # print(d)

    # matches = kp1.match(kp2)

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

    im1 = Image.Read("eiffel2-1.png", grey=True)
    im2 = Image.Read("eiffel2-2.png", grey=True)

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

    sf1 = im1.SIFT()
    sf2 = im2.SIFT()
    mm = sf1.match(sf2, thresh=20)

    print(mm)
    print(mm[3])
    print(mm[:5])
    mm.list()
    mm.table()

    from machinevisiontoolbox import CentralCamera
    F, resid = mm.estimate(CentralCamera.points2F, method="ransac", confidence=0.99)
    mm[:10].list()

    # mm = sf1.match(sf2, sort=True)[:10];

    # mm = sf1.match(sf2, ratio=0.8, crosscheck=True);

    pass