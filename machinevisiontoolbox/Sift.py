#!/usr/bin/env python
"""
SIFT feature class
@author: Dorian Tsai
@author: Peter Corke
"""

# https://docs.opencv.org/4.4.0/d7/d60/classcv_1_1SIFT.html

import numpy as np
import cv2 as cv
# import spatialmath.base.argcheck as argcheck
import machinevisiontoolbox as mvt
from machinevisiontoolbox.Image import Image


class Sift:
    """
    A SIFT feature class
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
    # when dealing with other feature descriptors)
    # _image_id = []      # index of image containing feature (? or image name?)

    _siftparameters = []  # dictionary for parameters and values used for sift
    # feature extraction

    _kp = []  # keypoints of sift (for interfacing with opencv functions)

    def __init__(self, image=None, siftparameters=None):
        if image is None:
            # initialise empty Sift
            self._u = None
            self._v = None
            self._strength = None
            self._orientation = None
            self._octave = None
            self._scale = None
            self._descriptor = None
            self._descriptorlength = 128
            self._image_id = None
            self._kp = None

            # default sift parameters from OpenCV/ the D. Lowe paper
            if siftparameters is None:
                self._siftparameters = {'nfeatures': 0,
                                        'nOctaveLayers': 3,
                                        'contrastThreshold': 0.04,
                                        'edgeThreshold': 10,
                                        'sigma': 1.6
                                        }
            else:
                # TODO check if siftparameters are valid
                self._siftparameters = siftparameters

        else:
            # check if image is valid
            image = Image(image)
            ImgProc = mvt.ImageProcessing()
            image = ImgProc.mono(image)

            self._image_id = image.filename

            # TODO for each image in imagesequence, (input could be image
            # sequence)
            # do SIFT on each image channel

            # call OpenCV sift detect and compute
            sift = cv.SIFT_create()
            kp, des = sift.detectAndCompute(image.image, mask=None)
            self._kp = kp
            # get all sift feature attributes

            # get u,v
            u = [kp[i].pt[0] for i in range(len(kp))]
            v = [kp[i].pt[1] for i in range(len(kp))]
            self._u = np.array(u)
            self._v = np.array(v)

            # get orientation
            orientation = [kp[i].angle for i in range(len(kp))]
            self._orientation = np.array(orientation)

            # get scale
            scale = [kp[i].size for i in range(len(kp))]
            self._scale = np.array(scale)

            # get strength
            strength = [kp[i].response for i in range(len(kp))]
            self._strength = np.array(strength)

            # get octave
            octave = [kp[i].octave for i in range(len(kp))]
            self._octave = np.array(octave)

            # get descriptors
            self._descriptor = des

    def __len__(self):
        return len(self._u)

    def __getitem__(self, ind):
        new = Sift()
        new._image_id = self._image_id  # TODO should be list of all imageids
        new._u = self._u[ind]
        new._v = self._v[ind]
        new._orientation = self._orientation[ind]
        new._scale = self._scale[ind]
        new._strength = self._strength[ind]
        new._octave = self._octave[ind]
        new._descriptor = self._descriptor[ind, 0:]
        new._descriptorlength = self._descriptorlength

        new._siftparameters = self._siftparameters
        # TODO may have to invoke similar imlist function in Image.py if ind is
        # a slice object
        new._kp = [self._kp[i] for i in ind]

        return new

    @property
    def u(self):
        return self._u

    @property
    def v(self):
        return self._v

    @property
    def orientation(self):
        return self._orientation

    @property
    def scale(self):
        return self._scale

    @property
    def strength(self):
        return self._strength

    @property
    def octave(self):
        return self._octave

    @property
    def kp(self):
        return self._kp

    @property
    def descriptor(self):
        return self._descriptor

    @property
    def feature(self):
        # order: u, v, strength, scale, theta
        # each column of f represents a different SIFT feature
        f = np.vstack((self._u,
                       self._v,
                       self._strength,
                       self._scale,
                       self._orientation))
        return f

    @property
    def pt(self):
        return np.vstack((self._u, self._v))

    def drawSiftKeypoints(self,
                          image,
                          kp=None,
                          drawing=None,
                          isift=None,
                          flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                          **kwargs):
        # draw sift features on image using cv.drawKeypoints

        # check valid imagesource
        image = Image(image)
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
                         kp,
                         drawing,
                         flags=flags,
                         **kwargs)

        return Image(drawing)

    # TODO def draw descriptors? (eg vl_feat, though mvt-mat doesn't have this)
    # TODO descriptor distance
    # TODO descriptor similarity
    # TODO display/print/char function?

    def match(self, d1, d2):
        """
        Match SIFT point features

        :param d1: descriptor matrix for feature set 1
        :type d1: numpy array
        :param d2: descriptor matrix for feature set 2
        :type d2: numpy array
        :return: matches m
        :rtype: numpy array
        """
        m = []

        ratio = 0.75  # TODO set as input parameter

        # TODO check valid input
        # d1 and d2 must be numpy arrays
        # d1 and d2 must have equal (128 for SIFT) rows
        # d1 and d2 must have greater than 1 columns

        # do matching
        # sorting
        # return

        # create BFMatcher object
        # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        bf = cv.BFMatcher()

        # Match descriptors.
        # matches0 = bf.match(d1, d2)
        # there is also:
        matches0 = bf.knnMatch(d1, d2, k=2)

        # Apply ratio test
        good = []
        for m, n in matches0:
            if m.distance < ratio*n.distance:
                good.append([m])

        # Sort them in the order of their distance.
        # matches = sorted(good, key=lambda x: x.distance)

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

        return good

    def drawSiftMatches(self, im1, sift1, im2, sift2, matches,
                        **kwargs):
        # TODO should I just have input two SIFT objects,
        # or in this case just another SIFT object?

        #draw_params = dict(matchColor=(0, 255, 0),
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

        return Image(out)


if __name__ == "__main__":
    # step 1: familiarisation with open cv's sift

    #im = cv.imread('images/test/longquechen-moon.png')
    # im = cv.imread('images/monalisa.png')
    #imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    #sift = cv.SIFT_create()
    # kp = sift.detect(imgray, None)
    #kp, des = sift.detectAndCompute(imgray, None)
    # https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html#aea339bc868102430087b659cd0709c11
    # kp[i].pt = (u,v)
    # kp[i].angle = orientation [deg?]
    # kp[i].class_id? unclear
    # kp[i].size = scale
    # kp[i].response = strength of keypoint
    # kp[i].octave - need to double check, but seems like a really large number

    #img = cv.drawKeypoints(imgray, kp, im, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #mvt.idisp(img, title='sift_keypoints')

    #sf = Sift(imgray)
    # sf.u

    #sf0 = sf[0:3]
    # sf0.u

    # drawing = sf.drawSiftKeypoints(imgray)

    # TODO would be nice to make a root-sift descriptor method, as it is a simple
    # addition to the SIFT descriptor

    # test matching



    #import code
    #code.interact(local=dict(globals(), **locals()))
    print(True)
