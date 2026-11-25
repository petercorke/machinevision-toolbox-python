"""
Multi-view geometry operations: homography, stereo, and epipolar methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from machinevisiontoolbox._image_typing import ImageMixinBase

from typing import Self

if TYPE_CHECKING:
    from machinevisiontoolbox.ImageCore import Image


class ImageMultiviewMixin(ImageMixinBase):
    # ======================= stereo ================================== #

    def stereo_simple(
        self, right: Image, hw: int, drange: tuple[int, int] | list[int] | int
    ) -> tuple[Self, Self, np.ndarray]:
        """
        Simple stereo matching

        :param right: right image
        :type right: :class:`Image`
        :param hw: window half width
        :type hw: int
        :param drange: disparity range (minimum, maximum), or a single integer
            ``n`` which means ``(0, n)``
        :type drange: array_like(2) or int
        :return: disparity image, similarity image, disparity space image
        :rtype: :class:`Image`, :class:`Image`, ndarray(H,W,D)

        This is a simple stereo matching implementation for pedagogical
        purposes.  It returns:

        - the disparity image, same size as input images, whose elements give
          the integer disparity (in pixels) of the corresponding point in the
          left image.
        - the similarity image, same size as input images, whose elements give
          the strength of the stereo match.  This is a ZNCC measure where 1 is a
          perfect match, greater than 0.8 is a decent match.
        - the disparity space image, a 3D array, whose first two dimensions are
          the same size as the input images and whose third dimension is the
          number of disparities.  Each fibre DSI[v,u,:] is window similarity
          versus disparity.

        .. rubric:: Examples

        .. code-block:: python

            rocks_l = Image.Read("rocks2-l.png", reduce=2)
            rocks_r = Image.Read("rocks2-r.png", reduce=2)
            disparity, similarity, DSI = rocks_l.stereo_simple(rocks_r, hw=3, drange=[40, 90])


        .. note:: The images are assumed to be epipolar aligned.

        .. note:: The disparity and similarity images, and each plane of the DSI,
            have a band of ``hw`` NaN valued pixels along the top, bottom and right
            edges, where the window extends beyond the image.  The band along the
            left edge is ``hw + drange[0]`` wide for the disparity and similarity
            images, and ``hw + drange[0] + d`` wide for the DSI plane of
            disparity ``d`` (counting from 0), since the shifted window in the
            right image must also lie within the image.

        .. rubric:: References

        - |RVC3|, Section 14.4.

        .. warning:: Not fast.

        .. versionchanged:: 2.5.0
            The outputs are the same size as the input images, as documented.
            Previously they were larger by ``2*hw`` in each direction, with
            the result for pixel (u, v) at (u+hw, v+hw).  The disparity at a
            pixel is now the best of its valid disparities, and NaN only if
            none are valid, and an integer image is padded with NaN instead
            of 0, so windows that extend beyond the image are invalid for
            every image type.

        .. seealso:: :meth:`DSI_refine` :meth:`stereo_BM` :meth:`stereo_SGBM`
        """

        def window_stack(image, hw):
            # convert an image to a stack where the planes represent shifted
            # version of image.  For a pixel at (u, v) ...

            stack = []
            w = 2 * hw + 1
            w1 = w - 1

            for upad in range(w):
                for vpad in range(w):
                    stack.append(
                        np.pad(
                            image,
                            ((vpad, w1 - vpad), (upad, w1 - upad)),
                            mode="constant",
                            constant_values=np.nan,
                        )
                    )
            # each padded plane is 2*hw larger than the image in each direction,
            # keep the part whose windows are centred on the image's pixels
            h, w_img = image.shape
            return np.dstack(stack)[hw : hw + h, hw : hw + w_img, :]

        if isinstance(drange, int):
            drange = (0, drange)

        # left = self.mono().image.astype(np.float32)
        # right = right.mono().image.astype(np.float32)
        # work in float32 for any image type, so that the window stack can be
        # padded with NaN (it cannot in an integer array, where it becomes 0)
        left_arr = self.mono()._A.astype(np.float32)
        right_arr = right.mono()._A.astype(np.float32)

        # convert to window stacks
        left_arr = window_stack(left_arr, hw)
        right_arr = window_stack(right_arr, hw)

        # offset the mean value of each template
        left_arr = left_arr - left_arr.mean(axis=2)[..., np.newaxis]
        right_arr = right_arr - right_arr.mean(axis=2)[..., np.newaxis]

        # shift right image to the right
        right_arr = right_arr[:, : right_arr.shape[1] - drange[0], :]
        right_arr = np.pad(
            right_arr,
            ((0, 0), (drange[0], 0), (0, 0)),
            mode="constant",
            constant_values=np.nan,
        )

        similarities = []

        # suppress divide by zero error messages
        # possible ZNCC values include:
        #  - NaN 0 / 0  invalid value encountered in true_divide
        # - inf  x / 0  divide by zero encountered in true_divide
        with np.errstate(divide="ignore", invalid="ignore"):
            for d in np.arange(drange[1] - drange[0]):  # lgtm[py/unused-loop-variable]
                # compute the ZNCC
                sumLL = np.sum(left_arr**2, axis=2)
                sumRR = np.sum(right_arr**2, axis=2)
                sumLR = np.sum(left_arr * right_arr, axis=2)

                denom = np.sqrt(sumLL * sumRR)
                # if (denom == 0).sum() > 0:
                #     print('divide by zero in ZNCC')

                similarity = sumLR / denom

                similarity = np.where(denom == 0, np.nan, similarity)
                similarities.append(similarity)

                # shift right image 1 pixel to the right
                right_arr = right_arr[:, :-1, :]
                right_arr = np.pad(
                    right_arr,
                    ((0, 0), (1, 0), (0, 0)),
                    mode="constant",
                    constant_values=np.nan,
                )

        # stack the similarity images at each disparity into the 3D DSI
        dsi = np.dstack(similarities)

        # disparity is the index of the maxima in the disparity direction.  A
        # NaN similarity means that disparity is not valid for that pixel, for
        # example the window would extend beyond the image, so ignore it.  A
        # pixel with no valid disparity at all is NaN.
        invalid = np.isnan(dsi)
        filled = np.where(invalid, -np.inf, dsi)
        disparity = np.argmax(filled, axis=2).astype(np.float32) + drange[0]

        # maxima is the maximum similarity in the disparity direction
        maxima = np.max(filled, axis=2)

        # wherever there is no valid disparity, the disparity and similarity are NaN
        none_valid = invalid.all(axis=2)
        disparity = np.where(none_valid, np.nan, disparity).astype(np.float32)
        maxima = np.where(none_valid, np.nan, maxima)

        return self.__class__(disparity, dtype=np.float32), self.__class__(maxima), dsi

    @classmethod
    def DSI_refine(cls, DSI: np.ndarray, dmin: int = 0) -> tuple[Self, Self]:
        """
        Refine disparity from disparity space image

        :param DSI: disparity space image
        :type DSI: ndarray(H,W,D)
        :param dmin: disparity of the first plane of the DSI, defaults to 0
        :type dmin: int, optional
        :return: refined disparity image and the curvature at the peak
        :rtype: 2-tuple of :class:`Image`

        Performs subpixel interpolation on the peaks in the DSI to provide disparity
        estimates to a fraction of a pixel.  A parabola is fitted to the peak and
        the two values either side of it, and the position of the parabola's peak
        is the refined disparity.

        Each plane of the DSI is a disparity, and the first plane is the first
        element of the ``drange`` passed to :meth:`stereo_simple`.  The DSI does
        not record this, so by default the returned disparities are relative to
        it: 0 is the first plane, 1 is the second, and so on, and a fraction
        gives the position between planes.  Pass ``dmin`` equal to ``drange[0]``
        to turn them into true disparity values.

        The result is NaN where there is no valid peak: where the best disparity
        is the first or last plane, so one neighbour is missing, where a
        neighbour is NaN, or where every plane is NaN.  Planes that are NaN are
        never chosen as the peak.

        .. rubric:: Examples

        .. code-block:: python

            rocks_l = Image.Read("rocks2-l.png", reduce=2)
            rocks_r = Image.Read("rocks2-r.png", reduce=2)
            disparity, similarity, DSI = rocks_l.stereo_simple(rocks_r, hw=3, drange=[40, 90])
            disparity, curvature = Image.DSI_refine(DSI, dmin=40)

        .. versionchanged:: 2.5.0
            The ``drange`` argument is replaced by ``dmin``.  ``drange`` was
            compared with plane indices, not disparities, which made it unusable
            unless the first disparity was 0.  A DSI plane that is NaN no longer
            gives a NaN result for a pixel that has a valid peak.

        .. rubric:: References

        - |RVC3|, Section 14.4.1.

        .. seealso:: :meth:`stereo_simple`
        """
        invalid = np.isnan(DSI)
        # best plane, never a NaN one (argmax would return the first NaN)
        peak = np.argmax(np.where(invalid, -np.inf, DSI), axis=2)

        # the peak needs a plane either side of it
        nplanes = DSI.shape[2]
        interior = (peak > 0) & (peak < nplanes - 1) & ~invalid.all(axis=2)

        # values at the peak and either side, wherever the peak is not interior
        # use the peak itself, and the result is masked out below
        def plane(offset: int) -> np.ndarray:
            index = np.where(interior, peak + offset, peak)
            return np.take_along_axis(DSI, index[:, :, np.newaxis], axis=2)[:, :, 0]

        YP, Y, YN = plane(-1), plane(0), plane(1)

        A = YP + YN - 2 * Y
        B = YN - YP

        # a flat peak (A == 0) has no parabola
        ok = interior & (A != 0)
        d_subpix = np.full(peak.shape, np.nan)
        d_subpix[ok] = peak[ok] - B[ok] / (2 * A[ok]) + dmin
        A = np.where(interior, A, np.nan)

        return cls(d_subpix), cls(A)

    def stereo_BM(
        self,
        right: Image,
        hw: int,
        drange: tuple[int, int] | list[int] | int,
        speckle: tuple[int, int] | None = None,
    ) -> Self:
        """
        Stereo block matching

        :param right: right image
        :type right: :class:`Image`
        :param hw: window half width
        :type hw: int
        :param drange: disparity range (minimum, maximum), or a single integer
            ``n`` which means ``(0, n)``
        :type drange: array_like(2) or int
        :param speckle: speckle filter parameters, defaults to None
        :type speckle: array_like(2), optional
        :raises ValueError: block size too small
        :return: disparity image
        :rtype: :class:`Image`

        This is an efficient block-matching stereo implementation.  It returns
        the disparity image, same size as input images, whose elements give the
        subpixel-interpolated disparity (in pixels) of the corresponding point
        in the left image.

        Speckle are small regions of anomalous disparity.  A speckle is defined
        as less than A pixels with disparity variation less than V, and the
        filter parameters are (A, V).  The disparity values within a detected
        speckle are set to that of its enclosing region.

        .. rubric:: Examples

        .. code-block:: python

            rocks_l = Image.Read("rocks2-l.png", reduce=2)
            rocks_r = Image.Read("rocks2-r.png", reduce=2)
            disparity = rocks_l.stereo_BM(rocks_r, hw=3, drange=[40, 90], speckle=(200, 2))


        .. note:: The images are assumed to be epipolar aligned.

        .. rubric:: References

        - |RVC3|, Section 14.4.2.7.

        .. seealso:: :meth:`stereo_SGBM` :meth:`stereo_simple` `opencv.StereoBM <https://docs.opencv.org/4.x/d9/dba/classcv_1_1StereoBM.html>`_
        """
        # https://docs.opencv.org/4.x/d9/dba/classcv_1_1StereoBM.html

        if isinstance(drange, int):
            drange = (0, drange)

        if hw < 2:
            raise ValueError("block size too small")

        # number of disparities must be multiple of 16
        ndisparities = drange[1] - drange[0]
        ndisparities = int(np.ceil(ndisparities // 16) * 16)

        # create the stereo matcher
        stereo = cv2.StereoBM_create(numDisparities=ndisparities, blockSize=2 * hw + 1)  # type: ignore[attr-defined]
        stereo.setMinDisparity(drange[0])

        left_arr = self.mono()._A.astype(np.uint8)
        right_arr = right.mono()._A.astype(np.uint8)

        # set speckle filter
        # it seems to make very little difference
        # it's not clear if range is in the int16 units or not
        if speckle is None:
            speckle = (0, 0)

        stereo.setSpeckleWindowSize(speckle[0])
        stereo.setSpeckleRange(int(16 * speckle[1]))

        disparity = stereo.compute(left=left_arr, right=right_arr)

        return self.__class__(disparity / 16.0)

    def stereo_SGBM(
        self,
        right: Image,
        hw: int,
        drange: tuple[int, int] | list[int] | int,
        speckle: tuple[int, int] | None = None,
    ) -> Self:
        """
        Stereo semi-global block matching

        :param right: right image
        :type right: :class:`Image`
        :param hw: window half width
        :type hw: int
        :param drange: disparity range (minimum, maximum), or a single integer
            ``n`` which means ``(0, n)``
        :type drange: array_like(2) or int
        :param speckle: speckle filter parameters, defaults to None
        :type speckle: array_like(2), optional
        :raises ValueError: block size too small
        :return: disparity image
        :rtype: :class:`Image`

        This is an efficient semi-global block-matching stereo implementation.
        It returns the disparity image, same size as input images, whose
        elements give the subpixel-interpolated disparity (in pixels) of the
        corresponding point in the left image.

        Speckle are small regions of anomalous disparity.  A speckle is defined
        as less than A pixels with disparity variation less than V, and the
        filter parameters are (A, V).  The disparity values within a detected
        speckle are set to that of its enclosing region.

        .. rubric:: Examples

        .. code-block:: python

            rocks_l = Image.Read("rocks2-l.png", reduce=2)
            rocks_r = Image.Read("rocks2-r.png", reduce=2)
            disparity = rocks_l.stereo_SGBM(rocks_r, hw=3, drange=[40, 90], speckle=(200, 2))


        .. note:: The images are assumed to be epipolar aligned.

        .. rubric:: References

        - Stereo processing by semiglobal matching and mutual information,
          Heiko Hirschmuller,
          IEEE Transactions on Pattern Analysis and Machine Intelligence,
          30(2):328–341, 2008.
        - |RVC3|, Section 14.4.2.7.

        .. seealso:: :meth:`stereo_SGBM` :meth:`stereo_simple` `opencv.StereoSGBM <https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html>`_
        """
        # https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html#details

        if isinstance(drange, int):
            drange = (0, drange)

        if hw < 2:
            raise ValueError("block size too small")

        # number of disparities must be multiple of 16
        ndisparities = drange[1] - drange[0]
        ndisparities = int(np.ceil(ndisparities // 16) * 16)

        # create the stereo matcher
        stereo = cv2.StereoSGBM_create(  # type: ignore[attr-defined]
            minDisparity=drange[0], numDisparities=ndisparities, blockSize=2 * hw + 1
        )

        left_arr = self.mono()._A.astype(np.uint8)
        right_arr = right.mono()._A.astype(np.uint8)

        # set speckle filter
        # it seems to make very little difference
        # it's not clear if range is in the int16 units or not
        if speckle is not None:
            stereo.setSpeckleWindowSize(speckle[0])
            stereo.setSpeckleRange(speckle[1])

        disparity = stereo.compute(left=left_arr, right=right_arr)

        return self.__class__(disparity / 16.0)

    # def line(self, start, end, color):
    # should be draw_line
    #     return self.__class__(cv2.line(self.image, start, end, color))

    def rectify_homographies(
        self, m: Any, F: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create rectification homographies

        :param m: corresponding points
        :type m: :class:`~machinevisiontoolbox.ImagePointFeatures.FeatureMatch`
        :param F: fundamental matrix
        :type F: ndarray(3,3)
        :return: rectification homographies
        :rtype: ndarray(3,3), ndarray(3,3)

        Given the epipolar geometry between two images, defined by the
        fundamental matrix and corresponding points, compute a pair of
        homographies that can be used to rectify the images so that they are
        epipolar aligned.

        .. rubric:: Examples

        .. code-block:: python

            walls_l = Image.Read('walls-l.png', reduce=2)
            walls_r = Image.Read('walls-r.png', reduce=2)
            sf_l = walls_l.SIFT()
            sf_r = walls_r.SIFT()
            matches = sf_l.match(sf_r);
            F, resid = matches.estimate(CentralCamera.points2F, method="ransac", confidence=0.95);
            H_l, H_r = walls_l.rectify_homographies(matches, F)
            walls_l_rect = walls_l.warp_perspective(H_l)
            walls_r_rect = walls_r.warp_perspective(H_r)


        .. rubric:: References

        - |RVC3|, Section 14.4.3.

        .. seealso:: :meth:`warp_perspective` :class:`Match` `opencv.stereoRectifyUncalibrated <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaadc5b14471ddc004939471339294f052>`_
        """
        retval, H1, H2 = cv2.stereoRectifyUncalibrated(
            points1=m.inliers.p1.T, points2=m.inliers.p2.T, F=F, imgSize=self.size
        )
        return H1, H2
