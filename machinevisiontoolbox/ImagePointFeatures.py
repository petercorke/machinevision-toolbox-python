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

    def __init__(self, arg=None, detector=None, sortby=None, **kwargs):

        # TODO flesh out sortby option, it can be by strength or scale
        # TODO what does nfeatures option to SIFT do? seemingly nothing

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
                'MSER': cv.MSER_create
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

            if detector == "mser":
                msers, bboxes = self._detector.detectRegions(image.A)
                # returns different things, msers is a list of points
                # u, v, point=centroid, scale=area
                # https://www.toptal.com/machine-learning/real-time-object-detection-using-mser-in-ios
            else:
                kp, des = self._detector.detectAndCompute(image.A, mask=None)
            if arg.id is not None:
                # copy image id into the keypoints
                for k in kp:
                    k.class_id = arg.id
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

        return new

    def __str__(self):
        if len(self) > 1:
            return f"{self._feature_type} feature vector with {len(self)} features"
        else:
            return f"{self._feature_type} feature: ({self.u:.1f}, {self.v:.1f}), strength={self.strength:.2f}, scale={self.scale:.1f}, id={self.id}"

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        if self._feature_type != other._feature_type:
            raise TypeError('cant add different feature types:', self._feature_type, other._feature_type)
        new = self.__class__()
        new._feature_type = self._feature_type

        new._kp = self._kp + other._kp
        new._descriptor = np.vstack((self._descriptor, other._descriptor))

        return new

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
        table = ANSITable(
                    Column("#"),
                    Column("centroid"),
                    Column("strength", fmt="{:.3g}"),
                    Column("scale", fmt="{:.3g}"),
                    Column("id", fmt="{:d}"),
                    border="thin"
        )
        for i, f in enumerate(self):
            table.row(i, f"{f.u:.1f}, {f.v:.1f}",
                      f.strength,
                      f.scale,
                      f.id)
        table.print()

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

    def match(self, other, ratio=0.75, crosscheck=False, metric='L2', sort=True):
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

        ``f1.match(f2)`` is a match object 
        """

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
        bf = cv.BFMatcher(metricdict[metric], crossCheck=crosscheck)

        # Match descriptors.
        # matches0 = bf.match(d1, d2)
        # there is also:
        matches0 = bf.knnMatch(self.descriptor, other.descriptor, k=2)

        # Apply ratio test
        good = []
        for m, n in matches0:
            if m.distance < ratio * n.distance:
                good.append(m)

        # Sort them in the order of increasing distance, best to worst match
        if sort:
            good.sort(key=lambda x: x.distance)

        # cv2.drawMatchesKnn expects list of lists as matches.
        # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)

        # Draw first 10 matches.
        # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)

        # opencv documentation for the descriptor matches
        # https://docs.opencv.org/4.4.0/d4/de0/classcv_1_1DMatch.html
        # i - likely the index for the matches
        # a - likely the index for the match pairs?
        # matches[i][a].distance
        # matches[i][a].imgIdx - which image it refers to
        # matches[i][a].queryIdx - which feature it is looking at I assume
        # matches[i][a].trainIdx?

        return Match([(m.queryIdx, m.trainIdx, m.distance) for m in good], self, other)

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
    
    def plot(self, *fmt, ax=None, filled=False,
        hand=False, handcolor='blue', handthickness=None, handalpha=1, **kwargs):

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
            ax.plot(self.u, self.v, *fmt, **kwargs)

        plt.draw()

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
        step = len(self)  // N
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

class Match:

    def __init__(self, m, kp1=None, kp2=None, distance=None):
        self._matches = m
        self._kp1 = kp1
        self._kp2 = kp2

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

        if isinstance(i, int):
            matches = [self._matches[i]]
        elif isinstance(i, slice):
            matches = self._matches[i]
        elif isinstance(i, np.ndarray):
            if np.issubdtype(i.dtype, np.bool):
                matches = [m for m, g in zip(self._matches, i) if g]
            elif np.issubdtype(good.dtype, np.integer):
                matches = [self._matches[k] for k in i]
        else:
            raise ValueError('bad index')
        return Match(matches, self._kp1, self._kp2)

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

    def __repr__(self):
        s = ''
        for i, m in enumerate(self._matches):
            # TODO shouldnt have to flatten
            p1 = self._kp1[m[0]].p.flatten()
            p2 = self._kp2[m[1]].p.flatten()
            s += f"{i:3d}: ({p1[0]:.1f}, {p1[1]:.1f}) <--> ({p2[0]:.1f}, {p2[1]:.1f})\n"
        return s

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

        combo, u = im1.__class__.hcat(im1, im2)
        combo.disp(darken=darken, width=width)

        # for m in self:
        #     p1 = m.pt1
        #     p2 = m.pt2
        #     plt.plot((p1[0], p2[0] + u[1]), (p1[1], p2[1]), *pos, **kwargs)
        p1 = self.pt1
        p2 = self.pt2
        plt.plot((p1[0, :], p2[0, :] + u[1]), (p1[1, :], p2[1, :]), *pos, **kwargs)
        plt.draw()

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

        return BaseFeature2D(self,
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

        return BaseFeature2D(self,
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

        return BaseFeature2D(self,
                              detector='MSER',
                              **kwargs)

    # each detector should explitly list (&document) all its parameters


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

    from machinevisiontoolbox import Image

    kp1 = Image('eiffel2-1.png').SIFT()
    kp2 = Image('eiffel2-2.png').SIFT()

    matches = kp1.match(kp2)


    # im = Image('eiffel2-1.png')
    # ax = im.disp()

    # # sort into descending order
    # ks = kp1.sort()
    # print(len(kp1), len(ks))
    # print(kp1[0]._descriptor)
    # print(ks[0]._descriptor)
    
    # kp1.plot(hand=True, handalpha=0.2)
    from machinevisiontoolbox import Image


    matches[:10].plot('b', alpha=0.6)

    plt.show(block=True)
    