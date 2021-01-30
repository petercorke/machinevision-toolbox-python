#!/usr/bin/env python
"""
SIFT feature class
@author: Dorian Tsai
@author: Peter Corke
"""

# https://docs.opencv.org/4.4.0/d7/d60/classcv_1_1SIFT.html

# from abc import ABC
import numpy as np
import math
import cv2 as cv
from ansitable import ANSITable, Column
import matplotlib.pyplot as plt
from machinevisiontoolbox.IImage import IImage

# from machinevisiontoolbox.Image import *
# from machinevisiontoolbox.Image import Image


# TODO, either subclass SIFTFeature(SuperFeature2D) or just use SuperFeature2D
# directly

class SuperFeature2D:
    """
    A 2D point feature class
    """
    # list of attributes
    _u = []             # horizontal image coordinate
    _v = []             # vertical image coordinate
    _strength = []      # feature strength
    _orientation = []   # feature orientation [rad]
    _scale = []         # feature scale
    _octave = []        # octave pyramid octave in which keypoint was detected
    # TODO not sure if this is entirely useful for the user
    _descriptor = []    # feature desciptor vector
    # length of feature descriptor vector (might be useful
    _descriptorlength = []
    # when dealing with other feature descriptors) _image_id = []      # index
    # of image containing feature (? or image name?)

    _siftparameters = []  # dictionary for parameters and values used for sift
    # feature extraction

    _kp = []  # keypoints of sift (for interfacing with opencv functions)

    _image = None

    def __init__(self, arg=None, detector=None, sortby=None, **kwargs):

        # TODO flesh out sortby option, it can be by strength or scale
        # TODO what does nfeatures option to SIFT do? seemingly nothing

        if arg is None:
            # initialise empty Sift

            self._image_id = None
            self._kp = None
            self._descriptor = None
            self._image = None

        elif isinstance(arg, list):
            self._image_id = None
            self._kp = [f._kp[0] for f in arg]
            self._descriptor = np.array([f._descriptor for f in arg])
            self._image = arg[0]._image

        elif isinstance(arg, IImage): 

            detectors = {
                'sift': cv.SIFT_create,
                'orb': cv.ORB_create,
                'mser': cv.MSER_create
            }
            # check if image is valid
            image = arg.mono()

            self._image_id = image.filename

            # TODO for each image in imagesequence, (input could be image
            # sequence)
            # do SIFT on each image channel

            # get a reference to the appropriate detector
            # make it case insensitive
            try:
                # keyword args not being passed yet
                self._detector = detectors[detector.lower()]()

            except KeyError:
                raise ValueError('bad detector specified')

            kp, des = self._detector.detectAndCompute(image.image, mask=None)
            self._kp = kp
            self._descriptor = des
            self._image = image
        else:
            raise TypeError('bad argument')

    def __len__(self):
        return len(self._kp)

    def __getitem__(self, ind):
        new = self.__class__()
        new._image_id = self._image_id  # TODO should be list of all imageids
        if isinstance(ind, int):
            new._kp = [self._kp[ind]]
        else:
            new._kp = self._kp[ind]

        if len(self._descriptor.shape) == 1:
            new._descriptor = self._descriptor
        else:
            new._descriptor = self._descriptor[ind, :]

        return new

    def __repr__(self):
        table = ANSITable(
                    Column("id"),
                    Column("centroid"),
                    Column("strength", fmt="{:.3g}"),
                    Column("scale", fmt="{:.3g}"),
                    border="thin"
        )
        for i, f in enumerate(self):
            table.row(i, f"{f.u:.1f}, {f.v:.1f}",
                      f.strength,
                      f.scale)
        return str(table)

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
    def pt(self):
        """
        Feature coordinate as NumPy array

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
        Match SIFT point features

        :param other: set of feature points
        :type other: SuperFeature2D
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
    
    def plot(self, N=500, ax=None,
        edgecolor='none', facecolor='yellow', alpha=0.5, 
        hand=False, handcolor='blue', handthickness=None, handalpha=1):

        if ax is None:
            ax = plt.gca()

        for kp in self[:N]:
            centre = kp.pt.flatten()
            c = plt.Circle(centre, radius=kp.scale, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, clip_on=True)
            ax.add_patch(c)

            if hand:
                circum = centre + kp.scale * np.r_[math.cos(kp.orientation), math.sin(kp.orientation)]
                l = plt.Line2D((centre[0], circum[0]), (centre[1], circum[1]), color=handcolor, linewidth=handthickness, alpha=handalpha)
                ax.add_line(l)

        plt.draw()

    def sort(self, by='strength', reverse=True):
        if by == 'strength':
            s = sorted(self, key=lambda f: f.strength, reverse=reverse)
        elif by == 'scale':
            s = sorted(self, key=lambda f: f.scale, reverse=reverse)

        return self.__class__(s)

class Match:

    def __init__(self, m, kp1=None, kp2=None):
        self._matches = m
        self._kp1 = kp1
        self._kp2 = kp2

    def __getitem__(self, i):
        """
        Index operator

        :param i: match subset
        :type i: int or Slice
        :return: subset of matches
        :rtype: Match instance

        Allow indexing or slicing of the set of matches
        """

        return Match(self._matches[i], self._kp1, self._kp2)

    def __len__(self):
        return len(self._matches)

    def __repr__(self):
        s = ''
        for i, m in enumerate(self._matches):
            p1 = kp1[m[0]].pt
            p2 = kp2[m[1]].pt
            s += f"{i:3d}: ({p1[0]:.1f}, {p1[1]:.1f}) <--> ({p1[0]:.1f}, {p1[1]:.1f})\n"
        return s

    def plot(self, *pos, **kwargs):
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
        combo.disp()

        for m in self._matches:
            p1 = kp1[m[0]].pt
            p2 = kp2[m[1]].pt
            plt.plot((p1[0], p2[0] + u[1]), (p1[1], p2[1]), *pos, **kwargs)
        plt.draw()

class Features2DMixin:
    """
    Class adding blob capability to Image

    It's methods become methods of Image

    """

    def SIFT(self,
             nfeatures=0,
             nOctaveLayers=3,
             contrastThreshold=0.04,
             edgeThreshold=10,
             sigma=1.6,
             **kwargs):
        """
        Detect SIFT features in image

        :param nfeatures: [description], defaults to 0
        :type nfeatures: int, optional
        :param nOctaveLayers: [description], defaults to 3
        :type nOctaveLayers: int, optional
        :param contrastThreshold: [description], defaults to 0.04
        :type contrastThreshold: float, optional
        :param edgeThreshold: [description], defaults to 10
        :type edgeThreshold: int, optional
        :param sigma: [description], defaults to 1.6
        :type sigma: float, optional
        :param kwargs: [description], defaults to 1.6
        :type kwargs: float, optional
        :return: set of 2D point features
        :rtype: SuperFeature2D

        ``IM.SIFT()`` is an iterable and sliceable object that contains 2D
        features with properties:


        Example:

        .. autorun:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image("eiffel2-1.png")
            >>> sift = im.SIFT()
            >>> len(sift)  # number of 
            >>> print(sift[0:5])

        :seealso: :func:`SuperFeature2D`
        """

        return SuperFeature2D(self,
                              detector="sift",
                              nfeatures=nfeatures,
                              nOctaveLayers=nOctaveLayers,
                              contrastThreshold=contrastThreshold,
                              edgeThreshold=edgeThreshold,
                              sigma=sigma)

    def ORB(self,
            nfeatures=0,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            nPointsBriefDescriptor=2,
            scoreType='harris',
            patchSize=31,
            fastThreshold=20,
            **kwargs):
        """
        Detect ORB features in image

        :param nfeatures: [description], defaults to 0
        :type nfeatures: int, optional
        :param nOctaveLayers: [description], defaults to 3
        :type nOctaveLayers: int, optional
        :param contrastThreshold: [description], defaults to 0.04
        :type contrastThreshold: float, optional
        :param edgeThreshold: [description], defaults to 10
        :type edgeThreshold: int, optional
        :param sigma: [description], defaults to 1.6
        :type sigma: float, optional
        :param kwargs: additional arguments to 
        :type kwargs: float, optional
        :return: set of 2D point features
        :rtype: SuperFeature2D

        ``IM.ORB()`` is an iterable and sliceable object that contains 2D
        features with properties

        :seealso: :func:`SuperFeature2D`


        """

        scoreoptions = {'harris': cv.ORB_HARRIS_SCORE,
                        'fast': cv.ORB_FAST_SCORE}

        score = scoreoptions[scoreType]

        return SuperFeature2D(self,
                              detector="orb",
                              nfeatures=nfeatures,
                              scaleFactor=scaleFactor,
                              nlevels=nlevels,
                              edgeThreshold=edgeThreshold,
                              firstLevel=firstLevel,
                              WTA_K=nPointsBriefDescriptor,
                              scoreType=score,
                              patchSize=patchSize,
                              fastThreshold=fastThreshold)

    def MSER(self,
             delta=5,
             minarea=60,
             maxarea=14400,
             maxvariation=0.25,
             mindiversity=0.2,
             maxevolution=200,
             areathreshold=1.01,
             minmargin=0.003,
             edgeblur=5,
             **kwargs):
        """
        Detect MSER features in image

        :param nfeatures: [description], defaults to 0
        :type nfeatures: int, optional
        :param nOctaveLayers: [description], defaults to 3
        :type nOctaveLayers: int, optional
        :param contrastThreshold: [description], defaults to 0.04
        :type contrastThreshold: float, optional
        :param edgeThreshold: [description], defaults to 10
        :type edgeThreshold: int, optional
        :param sigma: [description], defaults to 1.6
        :type sigma: float, optional
        :param kwargs: [description], defaults to 1.6
        :type kwargs: float, optional
        :return: set of 2D point features
        :rtype: SuperFeature2D

        ``IM.MSER()`` is an iterable and sliceable object that contains 2D
        features with properties:

        :seealso: :func:`SuperFeature2D`

        """
        return SuperFeature2D(self,
                              detector='mser',
                              _delta=delta,
                              _min_area=minarea,
                              _max_area=maxarea,
                              _max_variation=maxvariation,
                              _max_evolution=maxevolution,
                              _area_threshold=areathreshold,
                              _min_margin=minmargin,
                              _edge_blur_size=edgeblur)

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
    