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

import pdb  # for debugging purposes only

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
    _label = []  # TODO label assigned to this region (based on ilabel.m)
    _parent = []  # -1 if no parent, else index points to i'th parent contour
    _children = []  # list of children, -1 if no children
    # _edgepoint = []  # TODO (x,y) of a point on the perimeter
    # _edge = []  # list of edge points # replaced with _contours
    _perimeter = []  # length of edge
    _touch = []  # 0 if bbox doesn't touch the edge, 1 if it does

    _a = []  # major axis length # equivalent ellipse parameters
    _b = []  # minor axis length
    _orientation = []  # angle of major axis wrt the horizontal
    _aspect = []  # b/a < 1.0
    _circularity = []

    _moments = []  # named tuple of m00, m01, m10, m02, m20, m11

    # note that RegionFeature.m has edge, edgepoint - these are the contours
    _contours = []
    _image = []  # keep image saved for each Blobs object
    # probably not necessary in the long run, but for now is useful
    # to retain for debugging purposes. Not practical if blob
    # accepts a large/long sequence of images
    _hierarchy = []

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
            self._touch = None

            self._a = None
            self._b = None
            self._orientation = None
            self._aspect = None
            self._circularity = None
            self._moments = None

            self._contours = None
            self._hierarchy = None
            self._parent = None
            self._children = None

        else:
            # check if image is valid - it should be a binary image, or a
            # thresholded image ()
            # convert to grayscale/mono
            image = mvt.getimage(image)
            image = mvt.mono(image)
            # note: OpenCV doesn't have a binary image type, so it defaults to
            # uint8 0 vs 255
            image = mvt.iint(image)

            # we found cv.simpleblobdetector too simple.
            # Cannot get pixel values/locations of blobs themselves
            # therefore, use cv.findContours approach
            contours, hierarchy = cv.findContours(
                image, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
            self._contours = contours

            # TODO contourpoint, or edgepoint: take first pixel of contours

            # change hierarchy from a (1,M,4) to (M,4)
            self._hierarchy = np.squeeze(hierarchy)
            self._parent = self._hierarchy[:, 2]
            self._children = self._getchildren()

            # get moments as a dictionary for each contour
            mu = [cv.moments(self._contours[i])
                  for i in range(len(self._contours))]

            # recompute moments wrt hierarchy
            mf = self._hierarchicalmoments(mu)
            self._moments = mf

            # get mass centers/centroids:
            mc = np.array(self._computecentroids())
            self._uc = mc[:, 0]
            self._vc = mc[:, 1]

            # get areas:
            self._area = np.array(self._computearea())
            # TODO sort contours wrt area descreasing?

            # get perimeter:
            self._perimeter = np.array(self._computeperimeter())

            # get circularity
            self._circularity = np.array(self._computecircularity())

            # get bounding box:
            bbox = np.array(self._computeboundingbox())

            # bbox in [u0, v0, length, width]
            self._umax = bbox[:, 0] + bbox[:, 2]
            self._umin = bbox[:, 0]
            self._vmax = bbox[:, 1] + bbox[:, 3]
            self._vmin = bbox[:, 1]

            self._touch = self._touchingborder(image.shape)

            # equivalent ellipse from image moments
            a, b, orientation = self._computeequivalentellipse()
            self._a = np.array(a)
            self._b = np.array(b)
            self._orientation = np.array(orientation)
            self._aspect = self._b / self._a

    def _computeboundingbox(self, epsilon=3, closed=True):
        cpoly = [cv.approxPolyDP(c, epsilon=epsilon, closed=closed)
                 for i, c in enumerate(self._contours)]
        bbox = [cv.boundingRect(cpoly[i]) for i in range(len(cpoly))]
        return bbox

    def _computeequivalentellipse(self):
        nc = len(self._contours)
        mf = self._moments
        mc = np.stack((self._uc, self._vc), axis=1)
        w = [None] * nc
        v = [None] * nc
        orientation = [None] * nc
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
            orientation[i] = np.arctan(ev[1] / ev[0])
        return a, b, orientation

    def _computecentroids(self):
        mf = self._moments
        mc = [(mf[i]['m10'] / (mf[i]['m00']), mf[i]['m01'] / (mf[i]['m00']))
              for i in range(len(self._contours))]
        return mc

    def _computearea(self):
        return [self._moments[i]['m00'] for i in range(len(self._contours))]

    def _computecircularity(self):
        # apply Kulpa's correction factor when computing circularity
        # should have max 1 circularity for circle, < 1 for non-circles
        # Peter's reference:
        # Area and perimeter measurement of blobs in discrete binary pictures.
        # Z.Kulpa. Comput. Graph. Image Process., 6:434-451, 1977.
        # Another reference that Dorian found:
        # Methods to Estimate Areas and Perimeters of Blob-like Objects: a
        # Comparison. Proc. IAPR Workshop on Machine Vision Applications.,
        # December 13-15, 1994, Kawasaki, Japan
        # L. Yang, F. Albregtsen, T. Loennestad, P. Groettum
        kulpa = np.pi / 8.0 * (1.0 + np.sqrt(2.0))
        circularity = [((4.0 * np.pi * self._area[i]) /
                        ((self._perimeter[i] * kulpa) ** 2))
                       for i in range(len(self._contours))]
        return circularity

    def _computeperimeter(self):
        nc = len(self._contours)
        edgelist = [None] * nc
        edgediff = [None] * nc
        edgenorm = [None] * nc
        perimeter = [None] * nc
        for i in range(nc):
            edgelist[i] = np.vstack((self._contours[i][0:],
                                     np.expand_dims(self._contours[i][0],
                                                    axis=0)))
            edgediff[i] = np.diff(edgelist[i], axis=0)
            edgenorm[i] = np.linalg.norm(edgediff[i], axis=2)
            perimeter[i] = np.sum(edgenorm[i], axis=0)
        return perimeter

    def _touchingborder(self, imshape):
        t = [False]*len(self._contours)
        # TODO replace with list comprehension?
        for i in range(len(self._contours)):
            if ((self._umin[i] == 0) or (self._umax[i] == imshape[0]) or
                    (self._vmin[i] == 0) or (self._vmax[i] == imshape[1])):
                t[i] = True
        return t

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
        new._orientation = self._orientation[ind]
        new._circularity = self._circularity[ind]
        new._touch = self._touch[ind]

        return new

    # ef label(self, im, connectivity=8, labeltype, cctype):
        # for label.m
        # im = image, binary/boolean in
        # connectivity, 4 or 8-way connectivity
        # labeltype specifies the output label image type - considering the
        # total number of labels, or tot. # of pixels in source image?? (only
        # CV_32S and CV_16U supported), default seems to be CV_32S
        # cctype = labelling algorithm Grana's and Wu's supported

        # output:
        #  labels - a destination labeled image (?)
        #
    #    cv.connectedComponentsWithStats()

    def _hierarchicalmoments(self, mu):
        # for moments in a hierarchy, for any pq moment of a blob ignoring its
        # children you simply subtract the pq moment of each of its children.
        # That gives you the “proper” pq moment for the blob, which you then use
        # to compute area, centroid etc.
        # for each contour
        #   find all children (row i to hierarchy[0,i,0]-1, if same then no
        #   children)
        #   recompute all moments

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
            ichild = self._hierarchy[i, 2]
            if not (ichild == -1):  # then children exist
                ichild = [ichild]  # make first child a list
                # find other children who are less than NEXT in the hierarchy
                # and greater than -1,
                otherkids = [k for k in range(i + 1, len(self._contours)) if
                             ((k < inext) and (inext > 0))]
                if not len(otherkids) == 0:
                    ichild.extend(list(set(otherkids) - set(ichild)))

                for j in range(ichild[0], ichild[-1]+1):  # for each child
                    # all moments that need to be computed
                    # subtract them from the parent moment
                    # mh[i]['m00'] = mh[i]['m00'] - mu[j]['m00'] ...

                    # do a dictionary comprehension:
                    mh[i] = {key: mh[i][key] -
                             mu[j].get(key, 0) for key in mh[i]}
            # else:
                # no change to mh, because contour i has no children

        return mh

    def _getchildren(self):
        # gets list of children for each contour based on hierarchy
        # follows similar for loop logic from _hierarchicalmoments, so
        # TODO finish _getchildren and use the child list to do
        # _hierarchicalmoments

        children = [None]*len(self._contours)
        for i in range(len(self._contours)):
            inext = self._hierarchy[i, 0]
            ichild = self._hierarchy[i, 2]
            if not (ichild == -1):
                # children exist
                ichild = [ichild]
                otherkids = [k for k in range(i + 1, len(self._contours))
                             if ((k < inext) and (inext > 0))]
                if not len(otherkids) == 0:
                    ichild.extend(list(set(otherkids) - set(ichild)))
                children[i] = ichild
            else:
                # else no children
                children[i] = [-1]
        return children

    def drawBlobs(self,
                  image,
                  drawing=None,
                  icont=None,
                  color=None,
                  contourthickness=cv.FILLED,
                  textthickness=2):
        # draw contours of blobs
        # contours - the contour list
        # icont - the index of the contour(s) to plot
        # drawing - the image to draw the contours on
        # colors - the colors for the icont contours to be plotted (3-tuple)
        # return - updated drawing

        # TODO split this up into drawBlobs and drawCentroids methods

        image = mvt.getimage(image)

        if drawing is None:
            drawing = np.zeros(
                (image.shape[0], image.shape[1], 3), dtype=np.uint8)

        if icont is None:
            icont = np.arange(0, len(self._contours))
        else:
            icont = np.array(icont, ndmin=1, copy=True)

        if color is None:
            # make colors a list of 3-tuples of random colors
            color = [None]*len(icont)

            for i in range(len(icont)):
                color[i] = (rng.randint(0, 256),
                            rng.randint(0, 256),
                            rng.randint(0, 256))
                # contourcolors[i] = np.round(colors[i]/2)
            # TODO make a color option, specified through text,
            # as all of a certain color (default white)

        # make contour colours slightly different but similar to the text color
        # (slightly dimmer)?
        cc = [np.uint8(np.array(color[i])/2) for i in range(len(icont))]
        contourcolors = [(int(cc[i][0]), int(cc[i][1]), int(cc[i][2]))
                         for i in range(len(icont))]

        # TODO check contours, icont, colors, etc are valid
        hierarchy = np.expand_dims(self._hierarchy, axis=0)
        # done because we squeezed hierarchy from a (1,M,4) to an (M,4) earlier

        for i in range(len(icont)):
            # TODO figure out how to draw alpha/transparencies?
            cv.drawContours(drawing, self._contours, icont[i], contourcolors[i],
                            thickness=contourthickness, lineType=cv.LINE_8,
                            hierarchy=hierarchy)
        for i in range(len(icont)):
            ic = icont[i]
            cv.putText(drawing, str(ic),
                       (int(self._uc[ic]), int(self._vc[ic])),
                       fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,
                       color=color[i], thickness=textthickness)

        return drawing

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
    def orientation(self):
        return self._orientation

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
        # TODO maybe ind for centroid: b.centroid[0]?

    @property
    def perimeter(self):
        return self._perimeter

    @property
    def touch(self):
        return self._touch

    @property
    def circularity(self):
        return self._circularity

    def printBlobs(self):
        # TODO accept kwargs or args to show/filter relevant parameters

        # convenience function to plot
        for i in range(len(self._contours)):
            print(str.format('({0})  area={1:.1f}, \
                  cent=({2:.1f}, {3:.1f}), \
                  orientation={4:.3f}, \
                  b/a={5:.3f}, \
                  touch={6:d}, \
                  parent={7}, \
                  children={8}',
                             i, self._area[i], self._uc[i], self._vc[i],
                             self._orientation[i], self._aspect[i],
                             self._touch[i], self._parent[i], self._children[i]))


if __name__ == "__main__":

    # read image
    im = cv.imread('images/multiblobs.png', cv.IMREAD_GRAYSCALE)

    # call Blobs class
    b = Blobs(image=im)

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
        colors[i] = (rng.randint(0, 256), rng.randint(
            0, 256), rng.randint(0, 256))

        cv.rectangle(drawing, (b[i].umin, b[i].vmin), (b[i].umax, b[i].vmax),
                     colors[i], thickness=2)
        # cv.putText(drawing, str(i), (int(b[i].uc), int(b[i].vc)),
        #           fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors,
        #           thickness=2)

    drawing = b.drawBlobs(im, drawing, icont, colors,
                          contourthickness=cv.FILLED)
    # mvt.idisp(drawing)

    # import matplotlib.pyplot as plt
    # plt.imshow(d2)
    # plt.show()
    # mvt.idisp(d2)
    im2 = cv.imread('images/multiblobs_edgecase.png', cv.IMREAD_GRAYSCALE)

    # press Ctrl+D to exit and close the image at the end
    import code
    code.interact(local=dict(globals(), **locals()))
