#!/usr/bin/env python
"""
SIFT feature class
@author: Dorian Tsai
@author: Peter Corke
"""

# https://docs.opencv.org/4.4.0/d7/d60/classcv_1_1SIFT.html

import math

import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv
from ansitable import ANSITable, Column
from spatialmath import SE3
from machinevisiontoolbox.ImagePointFeatures import BaseFeature2D


def _fiducial_dict(dict="4x4_1000"):
    tag_dict = {
        "4x4_50": cv.aruco.DICT_4X4_50,
        "4x4_100": cv.aruco.DICT_4X4_100,
        "4x4_250": cv.aruco.DICT_4X4_250,
        "4x4_1000": cv.aruco.DICT_4X4_1000,
        "5x5_50": cv.aruco.DICT_5X5_50,
        "5x5_100": cv.aruco.DICT_5X5_100,
        "5x5_250": cv.aruco.DICT_5X5_250,
        "5x5_1000": cv.aruco.DICT_5X5_1000,
        "6x6_50": cv.aruco.DICT_6X6_50,
        "6x6_100": cv.aruco.DICT_6X6_100,
        "6x6_250": cv.aruco.DICT_6X6_250,
        "6x6_1000": cv.aruco.DICT_6X6_1000,
        "7x7_50": cv.aruco.DICT_7X7_50,
        "7x7_100": cv.aruco.DICT_7X7_100,
        "7x7_250": cv.aruco.DICT_7X7_250,
        "7x7_1000": cv.aruco.DICT_7X7_1000,
        "original": cv.aruco.DICT_ARUCO_ORIGINAL,
        "16h5": cv.aruco.DICT_APRILTAG_16h5,
        "25h9": cv.aruco.DICT_APRILTAG_25h9,
        "36h10": cv.aruco.DICT_APRILTAG_36h10,
        "36h11": cv.aruco.DICT_APRILTAG_36h11,
    }
    if isinstance(dict, str):
        return cv.aruco.getPredefinedDictionary(tag_dict[dict])
    else:
        return dict


class ImageFiducialsMixin:
    def fiducial(self, dict="4x4_1000", K=None, side=None):
        """
        Find fiducial markers in image

        :param dict: marker type, defaults to "4x4_1000"
        :type dict: str, optional
        :param K: camera intrinsics, defaults to None
        :type K: ndarray(3,3), optional
        :param side: side length of the marker, defaults to None
        :type side: float, optional
        :return: markers found in image
        :rtype: list of :class:`Fiducial` instances

        Find ArUco or ApriTag markers in the scene and return a list of
        :class:`Fiducial` objects, one per marker.  If camera intrinsics are
        provided then also compute the marker pose with respect to the camera.

        ``dict`` specifies the marker family or dictionary and describes the
        number of bits in the tag and the number of usable unique tags.

        ============  ========   ===========   =====================
        dict          tag type   marker size   number of unique tags
        ============  ========   ===========   =====================
        ``4x4_50``    Aruco      4x4           50
        ``4x4_100``   Aruco      4x4           100
        ``4x4_250``   Aruco      4x4           250
        ``4x4_1000``  Aruco      4x4           1000
        ``5x5_50``    Aruco      5x5           50
        ``5x5_100``   Aruco      5x5           100
        ``5x5_250``   Aruco      5x5           250
        ``5x5_1000``  Aruco      5x5           1000
        ``6x6_50``    Aruco      6x6           50
        ``6x6_100``   Aruco      6x6           100
        ``6x6_250``   Aruco      6x6           250
        ``6x6_1000``  Aruco      6x6           1000
        ``7x7_50``    Aruco      7x7           50
        ``7x7_100``   Aruco      7x7           100
        ``7x7_250``   Aruco      7x7           250
        ``7x7_1000``  Aruco      7x7           1000
        ``original``  Aruco      ?             ?
        ``16h5``      AprilTag   4x4           30
        ``25h9``      AprilTag   5x5           35
        ``36h10``     AprilTag   6x6           ?
        ``36h11``     AprilTag   6x6           587
        ============  ========   ===========   =====================

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('tags.png')
            >>> fiducials = im.fiducial('5x5_50')
            >>> fiducials
            >>> fiducials[0].corners

        :note: ``side`` is the dimension of the square that contains the
            small white squares inside the black background.

        :references:
            - Robotics, Vision & Control for Python, Section 13.6.1, P. Corke, Springer 2023.

        :seealso: :class:`Fiducial` :class:`ArUcoBoard`
        """

        dictionary = _fiducial_dict(dict)
        cornerss, ids, _ = cv.aruco.detectMarkers(self.mono().A, dictionary)

        # corners is a list of marker corners, one element per tag
        #  each element is 1x4x2 matrix holding corner coordinates

        fiducials = []
        if len(ids) == 0:
            return fiducials  # no markers found
        if K is not None and side is not None:
            rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
                cornerss, side, K, None
            )
            for id, rvec, tvec, corners in zip(ids, rvecs, tvecs, cornerss):
                fiducials.append(Fiducial(id[0], corners[0].T, K, rvec, tvec))
        else:
            for id, corners in zip(ids, cornerss):
                fiducials.append(Fiducial(id[0], corners[0].T))

        return fiducials


# --------------------- supporting classes -------------------------------- #


class Fiducial:
    def __init__(self, id, corners, K=None, rvec=None, tvec=None):
        """
        Properties of a visual fiducial marker

        :param id: identity of the marker
        :type id: int
        :param corners: image plane marker corners
        :type corners: ndarray(2, 4)
        :param K: camera intrinsics
        :type K: ndarray(3,3), optional
        :param rvec: translation of marker with respect to camera, as an Euler vector
        :type rvec: ndarray(3), optional
        :param tvec: translation of marker with respect to camera
        :type tvec: ndarray(3), optional

        :seealso: :meth:`id` :meth:`pose` :meth:`draw`
            :meth:`~machinevisiontoolbox.ImageFeatures.ImageFeaturesMixin.fiducial`
        """
        self._id = id
        self.corners = corners  # strip first dimensions
        self.K = K
        if tvec is not None and rvec is not None:
            self._pose = SE3(tvec) * SE3.EulerVec(rvec.flatten())
        self.rvec = rvec
        self.tvec = tvec

    def __str__(self):
        """
        String representation of fiducial

        :return: Brief readable description of fidicual id and pose
        :rtype: str
        """
        s = f"id={self.id}"
        if self.pose is not None:
            s += ": " + self.pose.strline()
        return s

    def __repr__(self):
        return str(self)

    # def plot(self, ax=None):
    #     ax = _axes_logic(ax, 2)

    @property
    def id(self):
        """
        Fiducial id

        :return: fiducial marker identity
        :rtype: int

        Returns the built in identity code of the April tag or arUco marker.
        """
        return self._id

    @property
    def pose(self):
        """
        Fiducial pose

        :return: marker pose
        :rtype: SE3

        Returns the pose of the tag with respect to the camera.  The x- and
        y-axes are in the marker plane and the z-axis is out of the marker.

        :note: Accurate camera intrinsics and dimension parameters are
            required for this value to be metric.
        """
        return self._pose

    def draw(self, image, length=100, thick=2):
        """
        Draw marker coordinate frame into image

        :param image: image with BGR color order
        :type image: :class:`Image`
        :param length: axis length in pixels, defaults to 100
        :type length: int, optional
        :param thick: axis thickness in pixels, defaults to 2
        :type thick: int, optional
        :raises ValueError: image must have BGR color order

        Draws a coordinate frame into the image representing the pose of the
        marker.  The x-, y- and z-axes are drawn as red, green and blue line
        segments.
        """
        if not image.isbgr:
            raise ValueError("image must have BGR color order")
        cv.drawFrameAxes(
            image.A, self.K, np.array([]), self.rvec, self.tvec, length, thick
        )


class ArUcoBoard:
    # potentially inherit from abstract MarkerBoard class

    def __init__(self, layout, sidelength, separation, dict, name=None, firsttag=0):
        """Create a MarkerBoard object

        :param layout: number of markers in the x- and y-directions
        :type layout: 2-tuple of int
        :param sidelength: Side length of each marker
        :type sidelength: float
        :param separation: White space between markers, must be the same in both directions
        :type separation: float
        :param dict: marker type, eg. '6x6_1000'
        :type dict: str
        :param name: name of the board, defaults to None
        :param firsttag: ID of the first tag, defaults to 0
        :type firsttag: int, optional
        :type name: str, optional
        :raises ValueError: if the ``layout`` is not a 2-tuple of integers

        This object represents a board of markers, such as an ArUco board.  The board comprises
        a regular grid of markers each of which has a known ``sidelength`` and ``separation``.  The grid
        has :math:`n_x \times n_y` markers in the x and y directions respectively, and
        ``layout``=:math:`(n_x, n_y)`.  The type of markers, ArUco or custom, is specified by the
        ``dict`` parameter.

        The markers on the board have ids start at ``firsttag`` and are numbered sequentially.  These ids
        are used to:
            - filter the markers, useful if there are several ArUco boards in the scene
            - create an ArUco board image

        :note: the dimensions must be in the same units as camera focal length and
            pixel size, typically meters.
        """
        self._layout = layout
        if len(layout) != 2:
            raise ValueError("layout must be a tuple of two integers")
        self._sidelength = sidelength
        self._separation = separation
        self._name = name

        self._dict = _fiducial_dict(dict)

        ids = list(range(firsttag, firsttag + layout[0] * layout[1]))

        self._board = cv.aruco.GridBoard(
            layout, sidelength, separation, self._dict, np.array(ids)
        )
        self._ids = set(ids)

    def estimatePose(self, image, camera, return_markers=False):
        """Estimate the pose of the board

        :param image: image containing the board
        :type image: Image
        :param camera: model of the camera, including intrinsics and distortion parameters
        :type camera: :class:`CentralCamera`
        :raises ValueError: the boards pose could not be estimated
        :param return_markers: return the detected markers and their pose, defaults to False
        :type return_markers: bool, optional
        :return: Camera pose with respect to board origin, vector of residuals in units of pixels in marker ID order, corresponding marker IDs
        :rtype: 3-tuple of SE3, numpy.ndarray, numpy.ndarray, optionally list of :class:`Fiducial`

        Residuals are the Euclidean distance between the detected marker corners and the
        reprojected corners in the image plane, in units of pixels.  The mean and maximum
        residuals are useful for assessing the quality of the pose estimate.
        """

        # find the markers in the image
        #  cornnerss is a list of (1,4,2) shaped arrays, each holding the corners of a marker
        #  ids is an (N,1) shaped array of marker ID
        cornerss, ids, rejected = cv.aruco.detectMarkers(image.mono().A, self._dict)

        # filter tags by ID
        cornerss = [corners for corners, id in zip(cornerss, ids) if id[0] in self._ids]

        ids = [id[0] for corners, id in zip(cornerss, ids) if id[0] in self._ids]
        ids = np.reshape(ids, (-1, 1))

        # match the markers to the board
        # print(f"{len(ids)} markers found")
        objPoints, imgPoints = self._board.matchImagePoints(cornerss, ids)

        # solve for camera pose
        retval, rvec, tvec = cv.solvePnP(
            objPoints, imgPoints, camera.K, camera.distortion
        )

        if not retval:
            raise ValueError("solvePnP failed")
        # print(f"rotation: {rvec.T}")
        # print(f"translation: {tvec.T}")
        self._tvec = tvec
        self._rvec = rvec

        # compute the reprojection error
        reprojection, _ = cv.projectPoints(
            objPoints, rvec, tvec, camera.K, camera.distortion
        )
        diff = (imgPoints - reprojection).squeeze()
        residuals = np.linalg.norm(diff, axis=1)

        T = SE3(tvec) * SE3.EulerVec(rvec.flatten())

        if return_markers:
            fiducials = []
            for id, corners in zip(ids, cornerss):
                fiducials.append(Fiducial(id[0], corners[0].T))
            return T, residuals, ids.flatten(), fiducials
        else:
            return T, residuals, ids.flatten()

    def draw(self, image, camera, length=0.1, thick=2):
        """
        Draw board coordinate frame into image

        :param image: image with BGR color order
        :type image: :class:`Image`
        :param length: axis length in metric units, defaults to 0.1
        :type length: float, optional
        :param thick: axis thickness in pixels, defaults to 2
        :type thick: int, optional
        :raises ValueError: image must have BGR color order

        Draws a coordinate frame into the image representing the pose of the
        board.  The x-, y- and z-axes are drawn as red, green and blue line
        segments.

        :note: the ``length`` is specified in the same units as focal length and
            pixel size of the camera, and the marker dimensions, typically meters.
        """
        if not image.isbgr:
            raise ValueError("image must have BGR color order")
        cv.drawFrameAxes(
            image.A, camera.K, camera.distortion, self._rvec, self._tvec, length, thick
        )

    def chart(self, filename=None, dpi=100):
        """Write ArUco chart to a file

        :param filename: name of the file to write chart to, defaults to returning an :class:`Image` instance
        :type filename: str
        :param dpi: dots per inch of printer, defaults to 100
        :type dpi: int, optional
        :return: :class:`Image` if ``filename`` is None
        :rtype: Image or None

        PIL is used to write the file, and can support multiple formats (specified
        by the file extension) such as PNG, PDF, etc.

        The markers have ids that start at the value given in the constructor and are
        numbered sequentially.  The markers are arranged in a grid of size given and
        increase in the x-direction (rows) first.

        If a PDF file is written the chart can be printed at 100% scale factor and will
        have the correct dimensions.  The size is of the chart is invariant to the
        ``dpi`` parameter, that simply affects the resolution of the image and file size.

        :note: This method assumes that the dimensions given in the constructor are in
            meters.
        """
        # dots per m
        dpm = dpi * 1000 / 25.4

        # compute size of chart in metres based on marker size and separation
        width = (
            self._layout[0] * (self._sidelength + self._separation) - self._separation
        )
        height = (
            self._layout[1] * (self._sidelength + self._separation) - self._separation
        )

        # convert to pixels
        width = int(width * dpm)
        height = int(height * dpm)

        # generate the image
        img = self._board.generateImage((width, height))

        if filename is None:
            return Image(img)
        else:
            from PIL import Image

            img = Image.fromarray(img)
            img.save(filename, dpi=(dpi, dpi))
            return None


if __name__ == "__main__":
    from machinevisiontoolbox import Image

    board = ArUcoBoard((5, 7), 28e-3, 3e-3, dict="6x6_1000", firsttag=0)
    board.chart("aruco0.pdf")
    board = ArUcoBoard((5, 7), 28e-3, 3e-3, dict="6x6_1000", firsttag=50)
    board.chart("aruco50.pdf")
