#!/usr/bin/env python
import numpy as np
from spatialmath.base import argcheck, getvector, e2h, h2e, transl2
import cv2 as cv

class ImageMultiviewMixin:
    # ======================= stereo ================================== #

    def stereo_simple(self, right, hw, drange):
        """
        Simple stereo matching

        :param right: right image
        :type right: :class:`Image`
        :param hw: window half width
        :type hw: int
        :param drange: disparity range
        :type drange: array_like(2)
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
          the same size as the input imagesm and whose third dimension is the
          number of disparities.  Each fibre DSI[v,u,:] is window similarity
          versus disparity.

        Example::

            >>> rocks_l = Image.Read("rocks2-l.png", reduce=2)
            >>> rocks_r = Image.Read("rocks2-r.png", reduce=2)
            >>> disparity, similarity, DSI = rocks_l.stereo_simple(rocks_r, hw=3, drange=[40, 90])

        .. note:: The images are assumed to be epipolar aligned.

        :references:
            - Robotics, Vision & Control for Python, Section 14.4, P. Corke, Springer 2023.

        .. warning:: Not fast.

        :seealso: :meth:`DSI_refine` :meth:`stereo_BM` :meth:`stereo_SGBM`
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
                        np.pad(image, ((vpad, w1 - vpad), (upad, w1- upad)),
                        mode='constant', constant_values=np.nan)
                    )
            return np.dstack(stack)

        if isinstance(drange, int):
            drange = (0, drange)

        # left = self.mono().image.astype(np.float32)
        # right = right.mono().image.astype(np.float32)
        left = self.mono().image
        right = right.mono().image

        # convert to window stacks
        left = window_stack(left, hw)
        right = window_stack(right, hw)

        # offset the mean value of each template
        left = left - left.mean(axis=2)[..., np.newaxis]
        right = right - right.mean(axis=2)[..., np.newaxis]

        # idisp(np.sum(left ** 2, axis=2))
        # idisp(np.sum(right ** 2, axis=2))

        # shift right image to the right
        right = right[:, :-drange[0], :]
        right = np.pad(right, ((0, 0), (drange[0], 0), (0, 0)),
            mode='constant', constant_values=np.nan)

        similarities = []

        # suppress divide by zero error messages
        # possible ZNCC values include:
        #  - NaN 0 / 0  invalid value encountered in true_divide
        # - inf  x / 0  divide by zero encountered in true_divide
        with np.errstate(divide='ignore', invalid='ignore'):

            for d in np.arange(drange[1] - drange[0]):  # lgtm[py/unused-loop-variable]

                # compute the ZNCC
                sumLL = np.sum(left ** 2, axis=2)
                sumRR = np.sum(right ** 2, axis=2)
                sumLR = np.sum(left * right, axis=2)

                denom = np.sqrt(sumLL * sumRR)
                # if (denom == 0).sum() > 0:
                #     print('divide by zero in ZNCC')

                similarity = sumLR / denom

                similarity = np.where(denom==0, np.nan, similarity)
                similarities.append(similarity)

                # shift right image 1 pixel to the right
                right = right[:, :-1, :]
                right = np.pad(right, ((0, 0), (1, 0), (0, 0)),
                    mode='constant', constant_values=np.nan)

        # stack the similarity images at each disparity into the 3D DSI
        dsi = np.dstack(similarities)
        
        # disparity is the index of the maxima in the disparity direction
        disparity = np.argmax(dsi, axis=2).astype(np.float32) + drange[0]

        # maxima is the maximum similarity in the disparity direction
        maxima = np.max(dsi, axis=2)

        # whereever maxima is nan set disparity to nan, similarity will be 
        # done for border regions
        disparity = np.where(np.isnan(maxima), np.nan, disparity)

        disparity[:, :drange[0]] = np.nan

        return self.__class__(disparity, dtype=np.float32), \
               self.__class__(maxima), \
               dsi

    @classmethod
    def DSI_refine(cls, DSI, drange=None):
        """
        Refine disparity from disparity space image

        :param DSI: disparity space image
        :type DSI: ndarray(H,W,D)
        :param drange: disparity range, defaults to span of DSI values
        :type drange: array_like(2), optional
        :return: refined disparity image
        :rtype: :class:`Image`

        Performs subpixel interpolation on the peaks in the DSI to provide disparity
        estimates to a fraction of a pixel.

        Example::

            >>> rocks_l = Image.Read("rocks2-l.png", reduce=2)
            >>> rocks_r = Image.Read("rocks2-r.png", reduce=2)
            >>> disparity, similarity, DSI = rocks_l.stereo_simple(rocks_r, hw=3, drange=[40, 90])
            >>> disparity = Image.DSI_refine(DSI)

        :references:
            - Robotics, Vision & Control for Python, Section 14.4.1, P. Corke, Springer 2023.

        :seealso: :meth:`stereo_simple`
        """
        DSI_flat = DSI.reshape((-1,DSI.shape[2]))

        YP = []
        Y = []
        YN = []
        
        if drange is None:
            disparity = np.argmax(DSI, axis=2)
            drange = [disparity.min(), disparity.max()]
        for i, d in enumerate(np.argmax(DSI, axis=2).ravel()):
            if drange[0] < d < drange[1]:
                YP.append(DSI_flat[i, d-1])
                Y.append(DSI_flat[i, d])
                YN.append(DSI_flat[i, d+1])
            else:
                YP.append(np.nan)
                Y.append(np.nan)
                YN.append(np.nan)
        
        YP= np.array(YP).reshape(DSI.shape[:2])
        Y = np.array(Y).reshape(DSI.shape[:2])
        YN = np.array(YN).reshape(DSI.shape[:2])

        A = YP + YN - 2 * Y
        B = YN - YP

        d_subpix = disparity - B / (2 * A)

        return cls(d_subpix), cls(A)

    def stereo_BM(self, right, hw, drange, speckle=None):
        """
        Stereo block matching

        :param right: right image
        :type right: :class:`Image`
        :param hw: window half width
        :type hw: int
        :param drange: disparity range
        :type drange: array_like(2)
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

        Example::

            >>> rocks_l = Image.Read("rocks2-l.png", reduce=2)
            >>> rocks_r = Image.Read("rocks2-r.png", reduce=2)
            >>> disparity = rocks_l.stereo_BM(rocks_r, hw=3, drange=[40, 90], speckle=(200, 2))

        .. note:: The images are assumed to be epipolar aligned.

        :references:
            - Robotics, Vision & Control for Python, Section 14.4.2.7, P. Corke, Springer 2023.

        :seealso: :meth:`stereo_SGBM` :meth:`stereo_simple` `opencv.StereoBM <https://docs.opencv.org/3.4/d9/dba/classcv_1_1StereoBM.html>`_
        """
        # https://docs.opencv.org/master/d9/dba/classcv_1_1StereoBM.html
        
        if isinstance(drange, int):
            drange = (0, drange)
        
        if hw < 2:
            raise ValueError('block size too small')

        # number of disparities must be multiple of 16
        ndisparities = drange[1] - drange[0]
        ndisparities = int(np.ceil(ndisparities // 16) * 16)

        # create the stereo matcher
        stereo = cv.StereoBM_create(
            numDisparities=ndisparities,
            blockSize=2*hw+1)
        stereo.setMinDisparity(drange[0])

        left = self.mono().image.astype(np.uint8)
        right = right.mono().image.astype(np.uint8)

        # set speckle filter
        # it seems to make very little difference
        # it's not clear if range is in the int16 units or not
        if speckle is None:
            speckle = (0, 0)

        stereo.setSpeckleWindowSize(speckle[0])
        stereo.setSpeckleRange(int(16 * speckle[1]))

        disparity = stereo.compute(
            left=left,
            right=right)

        return self.__class__(disparity / 16.0)

    def stereo_SGBM(self, right, hw, drange, speckle=None):
        """
        Stereo semi-global block matching

        :param right: right image
        :type right: :class:`Image`
        :param hw: window half width
        :type hw: int
        :param drange: disparity range
        :type drange: array_like(2)
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

        Example::

            >>> rocks_l = Image.Read("rocks2-l.png", reduce=2)
            >>> rocks_r = Image.Read("rocks2-r.png", reduce=2)
            >>> disparity = rocks_l.stereo_SGBM(rocks_r, hw=3, drange=[40, 90], speckle=(200, 2))

        .. note:: The images are assumed to be epipolar aligned.

        :references:
            - Stereo processing by semiglobal matching and mutual information,
              Heiko Hirschmuller,
              IEEE Transactions on Pattern Analysis and Machine Intelligence, 
              30(2):328–341, 2008.
            - Robotics, Vision & Control for Python, Section 14.4.2.7, P. Corke, Springer 2023.

        :seealso: :meth:`stereo_SGBM` :meth:`stereo_simple` `opencv.StereoSGBM <https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html>`_
        """
        # https://docs.opencv.org/master/d2/d85/classcv_1_1StereoSGBM.html#details
        
        if isinstance(drange, int):
            drange = (0, drange)
        
        if hw < 2:
            raise ValueError('block size too small')

        # number of disparities must be multiple of 16
        ndisparities = drange[1] - drange[0]
        ndisparities = int(np.ceil(ndisparities // 16) * 16)

        # create the stereo matcher
        stereo = cv.StereoSGBM_create(
            minDisparity=drange[0],
            numDisparities=ndisparities,
            blockSize=2*hw+1)


        left = self.mono().image.astype(np.uint8)
        right = right.mono().image.astype(np.uint8)

        # set speckle filter
        # it seems to make very little difference
        # it's not clear if range is in the int16 units or not
        if speckle is not None:
            stereo.setSpeckleWindowSize(speckle[0])
            stereo.setSpeckleRange(speckle[1])

        disparity = stereo.compute(
            left=left,
            right=right)

        return self.__class__(disparity / 16.0)

    # def line(self, start, end, color):
    # should be draw_line
    #     return self.__class__(cv.line(self.image, start, end, color))



    def rectify_homographies(self, m, F):
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

        Examples::

            >>> walls_l = Image.Read('walls-l.png', reduce=2)
            >>> walls_r = Image.Read('walls-r.png', reduce=2)
            >>> sf_l = walls_l.SIFT()
            >>> sf_r = walls_r.SIFT()
            >>> matches = sf_l.match(sf_r);
            >>> F, resid = matches.estimate(CentralCamera.points2F, method="ransac", confidence=0.95);
            >>> H_l, H_r = walls_l.rectify_homographies(matches, F)
            >>> walls_l_rect = walls_l.warp_perspective(H_l)
            >>> walls_r_rect = walls_r.warp_perspective(H_r)

        :references:
            - Robotics, Vision & Control for Python, Section 14.4.3, P. Corke, Springer 2023.

        :seealso: :meth:`warp_perspective` :class:`Match` `opencv.stereoRectifyUncalibrated <https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gaadc5b14471ddc004939471339294f052>`_
        """
        retval, H1, H2 = cv.stereoRectifyUncalibrated(m.inliers.p1, m.inliers.p2, F, self.size)
        return H1, H2