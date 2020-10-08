#!/usr/bin/env python
"""
2D Blob feature class
@author: Dorian Tsai
@author: Peter Corke
"""

import numpy as np
import cv2 as cv
import spatialmath.base.argcheck as argcheck
import machinevisiontoolbox as mvt

from collections import namedtuple


class Blobs:
    """
    A 2D feature blob class
    """
    # list of properties
    area = []
    centroid = []

    def __init__(self, image=None):

        if image is None:
            # initialise empty Blobs
            # Blobs()
            self.area = None
            self.centroid = [None, None]  # Two element array, empty? Nones? []?

        else:
            # check if image is valid
            # convert to grayscale/mono
            image = mvt.getimage(image)
            image = mvt.imono(image)

            # detect and compute keypoints and descriptors using opencv
            # TODO pass in parameters as an option?
            params = cv.SimpleBlobDetector_Params()

            params.minThreshold = 100
            params.maxThreshold = 255  # TODO check if image must be uint8?

            params.filterByArea = True
            params.minArea = 60
            params.maxArea = 100

            params.filterByColor = False  # this feature might be broken
            params.blobColor = 1  # 1 - 255, dark vs light

            params.filterByCircularity = False
            params.minCircularity = 0.1  # 0-1, how circular (1) vs line(0)

            params.filterByConvexity = False
            params.minConvexity = 0.87  # 0-1, convexity - area of blob/area of convex hull, convex hull being tightest convex shape that encloses the blob

            params.filterByInertia = False
            params.minInertiaRatio = 0.01  # 0-1, how elongated (circle = 1, line = 0)

            d = cv.SimpleBlobDetector_create(params)
            keypts = d.detect(image)

            # set properties as a list for every single blob
            self.area = [keypts[k].size for k, val in enumerate(keypts)]
            self.centroid = [keypts[k].pt for k, val in enumerate(keypts)]  # pt is a tuple

    def __getitem__(self, ind):
        new = Blobs()
        new.area = self.area[ind]
        new.centroid = self.centroid[ind]
        return new


if __name__ == "__main__":

    # read image
    # im = cv.imread('images/test/longquechen-moon.png', cv.IMREAD_GRAYSCALE)
    im = cv.imread('images/test/BlobTest.jpg', cv.IMREAD_GRAYSCALE)

    # call Blobs class
    b = Blobs(image=im)

    b.area
    b.centroid



    # draw detected blobs as red circles
    # DRAW_MATCHES_FLAGS... makes size of circle correspond to size of blob
    # im_kp = cv.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # show keypoints
    #cv.imshow('blob keypoints', im_kp)
    # cv.waitKey(1000)

    # press Ctrl+D to exit and close the image at the end
    import code
    code.interact(local=dict(globals(), **locals()))

    b0 = b[0]