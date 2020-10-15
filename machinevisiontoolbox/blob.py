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
import random as rng

import pdb

rng.seed(13543)  # would this be called every time at Blobs init?

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
    _label = []  # label assigned to this region
    _parent = []  # TODO need to discuss with Peter how to do this?
    _children = []  # TODO
    _edgepoint = []  # (x,y) of a point on the perimeter
    _edge = []  # list of edge points
    _perimeter = []  # length of edge
    _touch = []  # 0 if it doesn't touch the edge, 1 if it does TODO what is "it"?

    _a = []  # major axis length # equivalent ellipse parameters
    _b = []  # minor axis length
    _theta = []  # angle of major axis wrt the horizontal
    _aspect = []  # b/a < 1.0
    _circularity = []

    _moments = []  # named tuple of m00, m01, m10, m02, m20, m11

    # note that RegionFeature.m has edge, edgepoint - these are the contours
    _contours = []
    _image = []
    _hierarchy = []

    _perimeter = []

    def __init__(self, image=None):

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

            self._a = None
            self._b = None
            self._theta = None
            self._aspect = None
            self._circularity = None
            self._moments = None

            self._contours = None
            self._hierarchy = None
            self._image = None

        else:
            # check if image is valid - it should be a binary image, or a
            # thresholded image ()
            # convert to grayscale/mono
            image = mvt.getimage(image)
            image = mvt.mono(image)
            # TODO OpenCV doesn't have a binary image type, so it defaults to uint8 0 vs 255
            image = mvt.iint(image)

            self._image = image

            # I believe this screws up the image moment calculations though,
            # which are expecting a binary 0 or 1 image

            # detect and compute keypoints and descriptors using opencv
            # TODO pass in parameters as an option?
            # TODO simpleblob detector becomes backbone of ilabels?
            """
            params = cv.SimpleBlobDetector_Params()

            params.minThreshold = 0
            params.maxThreshold = 255  # TODO check if image must be uint8?

            params.filterByArea = False
            params.minArea = 60
            params.maxArea = 100

            params.filterByColor = False  # this feature might be broken
            params.blobColor = 1  # 1 - 255, dark vs light

            params.filterByCircularity = False
            params.minCircularity = 0.1  # 0-1, how circular (1) vs line(0)

            params.filterByConvexity = False
            # 0-1, convexity - area of blob/area of convex hull, convex hull being tightest convex shape that encloses the blob
            params.minConvexity = 0.87

            params.filterByInertia = False
            # 0-1, how elongated (circle = 1, line = 0)
            params.minInertiaRatio = 0.01

            d = cv.SimpleBlobDetector_create(params)

            keypts = d.detect(image)

            # set properties as a list for every single blob
            self._area = np.array([keypts[k].size for k, val in enumerate(keypts)])
            centroid = np.array([keypts[k].pt for k, val in enumerate(keypts)])  # pt is a tuple
            self._uc = np.array([centroid[k][0] for k, val in enumerate(centroid)])
            self._vc = np.array([centroid[k][1] for k, val in
            enumerate(centroid)])
            """
            # simpleblobdetector - too simple. Cannot get pixel values/locations of blobs themselves
            # findcontours approach
            contours, hierarchy = cv.findContours(
                image, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

            self._contours = contours
            nc = len(self._contours)

            self._hierarchy = np.squeeze(hierarchy)  # change hierarchy from a (1,M,4) to (M,4)

            # get moments as a dictionary for each contour
            mu = [cv.moments(self._contours[i])
                  for i in range(nc)]

            mf = self._hierarchicalmoments(mu)

            self._moments = mf

            # TODO for moments in a hierarchy, for any pq moment of a blob ignoring its
            # children you simply subtract the pq moment of each of its children.
            # That gives you the “proper” pq moment for the blob, which you then use
            # to compute area, centroid etc.
            # for each contour
            #   find all children (row i to hierarchy[0,i,0]-1, if same then no
            #   children)
            #   recompute all moments

            # get mass centers:
            mc = [(mf[i]['m10'] / (mf[i]['m00']), mf[i]['m01'] / (mf[i]['m00']))
                for i in range(nc)]
            mc = np.array(mc)

            self._uc = mc[:, 0]
            self._vc = mc[:, 1]

            # get areas:
            area = [mf[i]['m00'] for i in range(nc)]
            self._area = np.array(area)

            # TODO sort contours wrt area descreasing

            # get perimeters:
            # pdb.set_trace()
            perimeter = [np.sum(len(self._contours[i])) for i in range(nc)]
            self._perimeter = np.array(perimeter)

            # get bounding box:
            cpoly = [cv.approxPolyDP(c, epsilon=3, closed=True)
                     for i, c in enumerate(self._contours)]
            bbox = [cv.boundingRect(cpoly[i]) for i in range(len(cpoly))]
            bbox = np.array(bbox)

            # bbox in [u0, v0, length, width]
            self._umax = bbox[:, 0] + bbox[:, 2]
            self._umin = bbox[:, 0]
            self._vmax = bbox[:, 1] + bbox[:, 3]
            self._vmin = bbox[:, 1]

            # TODO could do these in list comprehensions, but then much harder
            # to read?
            # equivalent ellipse from image moments
            w = [None] * nc
            v = [None] * nc
            theta = [None] * nc
            a = [None] * nc
            b = [None] * nc

            for i in range(nc):
                u20 = mf[i]['m20'] / mf[i]['m00'] - mc[i, 0]**2
                u02 = mf[i]['m02'] / mf[i]['m00'] - mc[i, 1]**2
                u11 = mf[i]['m11'] / mf[i]['m00'] - mc[i, 0]*mc[i, 1]

                cov = np.array([[u20, u11], [u02, u11]])
                w, v = np.linalg.eig(cov)  # w = eigenvalues, v = eigenvectors

                a[i] = 2.0 * np.sqrt(np.max(np.diag(v)) / mf[i]['m00'])
                b[i] = 2.0 * np.sqrt(np.min(np.diag(v)) / mf[i]['m00'])

                ev = v[:, -1]
                theta[i] = np.arctan(ev[1] / ev[0])

            self._a = np.array(a)
            self._b = np.array(b)
            self._theta = np.array(theta)
            self._aspect = self._b / self._a
            # self._circularity



    def __len__(self):
        return len(self._area)

    def __getitem__(self, ind):
        new = Blobs()

        new._area = self._area[ind]
        new._uc = self._uc[ind]
        new._vc = self._vc[ind]
        new._perimeter = self._perimeter[ind]

        new._umin = self._umin[ind]
        new._umax = self._umax[ind]
        new._vmin = self._vmin[ind]
        new._vmax = self._vmax[ind]

        new._a = self._a[ind]
        new._b = self._b[ind]
        new._aspect = self._aspect[ind]
        new._theta = self._theta[ind]
        # new._circularity = self._circularity[ind]

        return new

    # TODO why is self necessary here?
    def _hierarchicalmoments(self, mu):
        # to deliver all the children of i'th contour:
        # first index identifies the row that the next contour at the same
        # hierarchy level starts
        # therefore, to grab all children for given contour, grab all rows
        # up to i-1 of the first row value
        # can only have one parent, so just take the last (4th) column

        # hierarchy order: [Next, Previous, First_Child, Parent]
        # for i in range(len(contours)):
        #    print(i, hierarchy[0,i,:])
        #    0 [ 5 -1  1 -1]
        #    1 [ 4 -1  2  0]
        #    2 [ 3 -1 -1  1]
        #    3 [-1  2 -1  1]
        #    4 [-1  1 -1  0]
        #    5 [ 8  0  6 -1]
        #    6 [ 7 -1 -1  5]
        #    7 [-1  6 -1  5]
        #    8 [-1  5  9 -1]
        #    9 [-1 -1 -1  8]

        mh = mu
        for i in range(len(self._contours)):  # for each contour
            inext = self._hierarchy[i, 0]
            #print('i = ' + str(i))
            #print(inext)
            ichild = self._hierarchy[i, 2]
            if not (ichild == -1):  # then children exist
                ichild = [ichild]  # make first child a list
                # find other children who are less than NEXT in the hierarchy
                # and greater than -1,
                otherkids = [k for k in range(i + 1, len(self._contours)) if
                             ((k < inext) and (inext > 0))]
                if not len(otherkids) == 0:
                    # ichild.append(np.setdiff1d(otherkids, ichild))
                    ichild.extend(list(set(otherkids) - set(ichild)))

                #if inext == (i + 1):
                #    ichildren = np.array([inext])  # what would otherwise be a 0-D array
                #else:
                #    ichildren = np.arange(i+1, inext)

                #import code
                #code.interact(local=dict(globals(), **locals()))

                # print('ichild =', ichild)
                for j in range(ichild[0], ichild[-1]+1):  # for each child
                    # print('j =', j)
                    # all moments that need to be computed
                    # subtract them from the parent moment
                    #mh[i]['m00'] = mh[i]['m00'] - mu[j]['m00']
                    #mh[i]['m01'] = mh[i]['m01'] - mu[j]['m01']
                    #mh[i]['m10'] = mh[i]['m10'] - mu[j]['m10']
                    #mh[i]['m11'] = mh[i]['m11'] - mu[j]['m11']
                    #mh[i]['m20'] = mh[i]['m20'] - mu[j]['m20']
                    #mh[i]['m02'] = mh[i]['m02'] - mu[j]['m02']

                    # do a dictionary comprehension:
                    mh[i] = {key: mh[i][key] - mu[j].get(key, 0) for key in mh[i]}

            # else:
                # no change to mh, because contour i has no children

        return mh

    def drawContours(self,
                     drawing=None,
                     icont=None,
                     colors=None,
                     contourthickness=2,
                     textthickness=2):
        # draw contours of blobs
        # contours - the contour list
        # icont - the index of the contour(s) to plot
        # drawing - the image to draw the contours on
        # colors - the colors for the icont contours to be plotted (3-tuple)
        # return - updated drawing

        if (drawing is None) and (self._image is not None):
            drawing = np.zeros((self._image.shape[0], self._image.shape[1], 3), dtype=np.uint8)

        if icont is None:
            icont = np.arange(0, len(self._contours))

        if colors is None:
            # make colors a list of 3-tuples of random colors
            colors = [None]*len(icont)

            for i in range(len(icont)):
                colors[i] = (rng.randint(0, 256),
                                rng.randint(0, 256),
                                rng.randint(0, 256))
                contourcolors[i] = np.round(colors[i]/2)
            # TODO make a color option, specified through text,
            # as all of a certain color (default white)

        # make contour colours slightly different but similar to the text color
        # (slightly dimmer)?
        # pdb.set_trace()
        cc = [np.uint8(np.array(colors[i])/2) for i in range(len(icont))]
        contourcolors = [(int(cc[i][0]), int(cc[i][1]), int(cc[i][2])) for i in range(len(icont))]

        # pdb.set_trace()

        # TODO check contours, icont, colors, etc are valid
        hierarchy = np.expand_dims(self._hierarchy, axis=0)
        # done because we squeezed hierarchy from a (1,M,4) to an (M,4) earlier

        # plot contours for all icont
        if len(icont) == 1:
            cv.drawContours(drawing, self._contours, icont, contourcolors,
                            thickness=contourthickness, lineType=cv.LINE_8,
                            hierarchy=hierarchy, offset=None)
            cv.putText(drawing, str(icont),
                        (int(self._uc[icont]), int(self._vc[icont])),
                        fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=colors, thickness=textthickness)
        else:
            for i in range(len(icont)):
                ic = icont[i]
                # TODO figure out how to draw alpha/transparencies?
                cv.drawContours(drawing, self._contours, ic, contourcolors[i],
                                thickness=contourthickness, lineType=cv.LINE_8,
                                hierarchy=hierarchy)
                cv.putText(drawing, str(ic),
                            (int(self._uc[ic]), int(self._vc[ic])),
                            fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=colors[i], thickness=textthickness)

        return drawing
    """
    def drawBlobs(self,
                  drawing=None,
                  iblob=None,
                  colors=None)
        # function to plot the blobs (as opposed to contours)
        # TODO function to do contour filling using fillPoly
        cpoly = [cv.approxPolyDP(c, epsilon=3, closed=True)
                     for i, c in enumerate(self._contours)]

        return drawing
    """

    @property
    def area(self):
        return self._area

    @property
    def uc(self):
        return self._uc

    @property
    def vc(self):
        return self._vc

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def theta(self):
        return self._theta

    @property
    def bbox(self):
        return ((self._umin, self._umax), (self._vmin, self._vmax))

    @property
    def umin(self):
        return self._umin

    @property
    def umax(self):
        return self._umax

    @property
    def vmax(self):
        return self._vmax

    @property
    def vmin(self):
        return self._vmin

    @property
    def bboxarea(self):
        return (self._umax - self._umin) * (self._vmax - self._vmin)

    @property
    def centroid(self):
        return (self._uc, self.vc)

    @property
    def perimeter(self):
        return self._perimeter


if __name__ == "__main__":

    # read image
    # im = cv.imread('images/test/longquechen-moon.png', cv.IMREAD_GRAYSCALE)
    #ret = cv.haveImageReader('images/multiblobs.png')
    # print(ret)

    im = cv.imread('images/multiblobs.png', cv.IMREAD_GRAYSCALE)

    # call Blobs class
    b = Blobs(image=im)

    b.area
    b.uc

    # draw detected blobs as red circles
    # DRAW_MATCHES_FLAGS... makes size of circle correspond to size of blob
    # im_kp = cv.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # show keypoints
    #cv.imshow('blob keypoints', im_kp)
    # cv.waitKey(1000)

    b0 = b[0].area
    b02 = b[0:2].uc

    print('Length of b =', len(b))


    # TODO
    # plot image
    # plot centroids of blobs
    # label relevant centroids for the labelled blobs
    import random as rng  # for random colors of blobs
    rng.seed(53467)

    drawing = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
    colors = [None]*len(b)
    icont = [None]*len(b)
    for i in range(len(b)):
        icont[i] = i
        colors[i] = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))

        cv.rectangle(drawing, (b[i].umin, b[i].vmin), (b[i].umax, b[i].vmax),
                     colors[i], thickness=2)
        #cv.putText(drawing, str(i), (int(b[i].uc), int(b[i].vc)),
        #           fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors,
        #           thickness=2)

    drawing = b.drawContours(drawing, icont, colors, contourthickness=cv.FILLED)

    #cv.imshow('blob contours', drawing)
    #cv.waitKey()

    # press Ctrl+D to exit and close the image at the end
    #import code
    #code.interact(local=dict(globals(), **locals()))
    # pdb.set_trace()
