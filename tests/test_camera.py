#!/usr/bin/env python

import unittest

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt
from spatialmath import SE3

from machinevisiontoolbox import CentralCamera, Image


class TestCamera(unittest.TestCase):

    def test_parameters(self):

        c = CentralCamera(f=0.2)
        nt.assert_array_almost_equal(c.f, (0.2, 0.2))
        self.assertEqual(c.camtype, "perspective")
        self.assertEqual(c.fu, 0.2)
        self.assertEqual(c.fv, 0.2)

        c.f = 0.3
        nt.assert_array_almost_equal(c.f, (0.3, 0.3))

        c = CentralCamera(imagesize=4000)
        nt.assert_array_almost_equal(c.imagesize, (4000, 4000))
        self.assertEqual(c.width, 4000)
        self.assertEqual(c.height, 4000)
        self.assertEqual(c.nu, 4000)
        self.assertEqual(c.nv, 4000)
        nt.assert_array_almost_equal(c.pp, (2000, 2000))

        c = CentralCamera(imagesize=(4000, 3000))
        nt.assert_array_almost_equal(c.imagesize, (4000, 3000))
        self.assertEqual(c.width, 4000)
        self.assertEqual(c.height, 3000)
        self.assertEqual(c.nu, 4000)
        self.assertEqual(c.nv, 3000)
        nt.assert_array_almost_equal(c.pp, (2000, 1500))

        c.imagesize = 5000
        self.assertEqual(c.width, 5000)
        self.assertEqual(c.height, 5000)
        self.assertEqual(c.nu, 5000)
        self.assertEqual(c.nv, 5000)

        c.imagesize = (5000, 4000)
        self.assertEqual(c.width, 5000)
        self.assertEqual(c.height, 4000)
        self.assertEqual(c.nu, 5000)
        self.assertEqual(c.nv, 4000)

        c = CentralCamera(rho=0.1)
        self.assertEqual(c.rhou, 0.1)
        self.assertEqual(c.rhov, 0.1)

        c = CentralCamera(rho=(0.1, 0.2))
        nt.assert_array_almost_equal(c.rho, (0.1, 0.2))
        self.assertEqual(c.rhou, 0.1)
        self.assertEqual(c.rhov, 0.2)

        c = CentralCamera(f=2, rho=(0.1, 0.2))

        c = CentralCamera(pp=(200, 300))
        self.assertEqual(c.u0, 200)
        self.assertEqual(c.v0, 300)
        nt.assert_array_almost_equal(c.pp, (200, 300))

        c = CentralCamera(pp=(200, 300), imagesize=3000)
        self.assertEqual(c.u0, 200)
        self.assertEqual(c.v0, 300)
        nt.assert_array_almost_equal(c.pp, (200, 300))

        # Setters must update both components, not silently fail.
        c = CentralCamera(f=0.015)
        c.f = 0.02
        nt.assert_allclose(c.f, [0.02, 0.02])
        c.pp = (400, 300)
        nt.assert_array_equal(c.pp, [400, 300])

    def test_pose(self):

        c = CentralCamera()
        self.assertTrue(c.pose == SE3())

        x = SE3(1, 2, 3)
        y = SE3(4, 5, 6)

        # Test setting pose
        c.pose = x
        self.assertTrue(c.pose == x)

        # Test different pose
        c.pose = y
        self.assertTrue(c.pose == y)

    def test_project_point(self):
        """Check pinhole projection with points stored in columns."""
        c = CentralCamera(f=0.015, rho=10e-6, imagesize=[1280, 1024])
        p = c.project_point([0.1, 0.2, 3.0])
        self.assertEqual(p.shape, (2, 1))
        nt.assert_allclose(p, [[690], [612]])

        P = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [3.0, 3.0, 3.0]])
        p = c.project_point(P)
        self.assertEqual(p.shape, (2, 3))
        nt.assert_allclose(p, [[690, 740, 790], [612, 662, 712]])

    def test_K_matrix(self):
        """Test camera calibration matrix"""
        c = CentralCamera(f=0.015, rho=10e-6, imagesize=[1280, 1024], pp=(640, 512))
        K = c.K

        # Check shape
        self.assertEqual(K.shape, (3, 3))

        # Check that K is upper triangular (mostly)
        self.assertNotEqual(K[0, 0], 0)  # fu
        self.assertNotEqual(K[1, 1], 0)  # fv
        self.assertNotEqual(K[0, 2], 0)  # u0
        self.assertNotEqual(K[1, 2], 0)  # v0
        self.assertEqual(K[2, 2], 1.0)  # Bottom right should be 1

    def test_C_matrix(self):
        """Test camera matrix"""
        c = CentralCamera(f=0.015, rho=10e-6)
        C = c.C()

        # Check shape
        self.assertEqual(C.shape, (3, 4))

    def test_different_camera_types(self):
        """Test different camera models"""
        # Perspective camera
        c_persp = CentralCamera(f=0.015)
        self.assertEqual(c_persp.camtype, "perspective")

        c_dist = CentralCamera(f=0.015, distortion=[0.1, 0.01])
        nt.assert_array_equal(c_dist.distortion, [0.1, 0.01])

    def test_ray_direction(self):
        """Rays contain the camera centre and follow the pixel direction."""
        c = CentralCamera(f=0.015, rho=10e-6, imagesize=[1280, 1024], pose=SE3(1, 2, 3))
        for pixel, direction in [([640, 512], [0, 0, 1]), ([790, 512], [0.1, 0, 1])]:
            with self.subTest(pixel=pixel):
                ray = c.ray(pixel)
                self.assertEqual(len(ray), 1)
                nt.assert_allclose(ray.w / ray.w[2], direction, atol=1e-12)
                self.assertTrue(ray.contains(c.centre))
                self.assertTrue(ray.contains(c.centre + np.array(direction)))

    def test_move_camera(self):
        """Move returns a new camera and leaves the original pose unchanged."""
        original_pose = SE3(1, 2, 3)
        c = CentralCamera(f=0.015, pose=original_pose)
        T = SE3(0.5, 0.25, 0)
        moved = c.move(T)
        self.assertIsNot(c, moved)
        nt.assert_allclose(moved.pose.A, T.A)
        nt.assert_allclose(c.pose.A, original_pose.A)
        nt.assert_allclose(moved.f, c.f)
        relative = c.move(T, relative=True)
        nt.assert_allclose(relative.pose.A, (original_pose * T).A)
        nt.assert_allclose(c.pose.A, original_pose.A)

    def test_str(self):
        c = CentralCamera(f=0.123, imagesize=(2000, 3000), name="fred")
        s = str(c)
        self.assertIsInstance(s, str)
        self.assertTrue("fred" in s)
        self.assertTrue("CentralCamera" in s)
        self.assertTrue("2000" in s)
        self.assertTrue("3000" in s)
        self.assertTrue("0.123" in s)

    def test_project(self):

        c = CentralCamera(f=1, rho=1, pp=(0, 0))
        p = c.project_point([0, 0, 5])
        self.assertTrue(p.shape == (2, 1))
        nt.assert_array_almost_equal(p.flatten(), [0, 0])

        p = c.project_point([1, 0, 5])
        nt.assert_array_almost_equal(p.flatten(), [0.2, 0])

        p = c.project_point([-1, 0, 5])
        nt.assert_array_almost_equal(p.flatten(), [-0.2, 0])

        p = c.project_point([0, 1, 5])
        nt.assert_array_almost_equal(p.flatten(), [0, 0.2])

        p = c.project_point([0, -1, 5])
        nt.assert_array_almost_equal(p.flatten(), [0, -0.2])

        p = c.project_point([1, -1, 5])
        nt.assert_array_almost_equal(p.flatten(), [0.2, -0.2])

        p = c.project_point([1, -1, 2.5])
        nt.assert_array_almost_equal(p.flatten(), [0.4, -0.4])

        p = c.project_point([1, -1, 5], pose=SE3(1, -1, 0))
        nt.assert_array_almost_equal(p.flatten(), [0, 0])

        p = c.project_point([1, -1, 5], pose=SE3(0, 0, 2.5))
        nt.assert_array_almost_equal(p.flatten(), [0.4, -0.4])

        p = c.project_point([1, -1, 5], objpose=SE3(-1, 1, 0))
        nt.assert_array_almost_equal(p.flatten(), [0, 0])

        c = CentralCamera(f=1, rho=1, pp=(1, 1))

        p = c.project_point([0, -1, 5])
        nt.assert_array_almost_equal(p.flatten(), [1, 0.8])

        c = CentralCamera(f=2, rho=1, pp=(0, 0))
        p = c.project_point([1, -1, 5])
        nt.assert_array_almost_equal(p.flatten(), [0.4, -0.4])

        c = CentralCamera(f=2, rho=2, pp=(0, 0))
        p = c.project_point([1, -1, 5])
        nt.assert_array_almost_equal(p.flatten(), [0.2, -0.2])

    def test_graphics(self):
        c = CentralCamera.Default()

        c.plot_point([1, -1, 5])
        c.clf()
        p = c.project_point([1, -1, 5])
        c.plot_point(p)

        plt.close()

    # test visibility
    # test multiple points

    def test_epiline(self):
        c1 = CentralCamera.Default()
        c2 = c1.move(SE3(0.1, 0, 0))
        P = [1, -1, 5]
        p1 = c1.project_point(P)
        F = c1.F(c2)

        c1.plot_epiline(F, p1)

        plt.close()

    def test_plucker(self):

        c = CentralCamera.Default()
        P = [1, -1, 5]
        p = c.project_point(P)

        line = c.ray(p)
        self.assertTrue(line.contains(P))

    def test_points2F_too_few_points_raises(self):
        """Fewer than the minimum required correspondences must raise a
        clear ValueError before ever calling cv2.findFundamentalMat --
        regression test for a bare AttributeError:
        'NoneType' object has no attribute 'ravel' when cv2 silently
        returned (None, None)"""
        p1 = np.array([[10, 20, 30], [10, 20, 30]], dtype=float)
        p2 = np.array([[11, 21, 31], [11, 21, 31]], dtype=float)

        with self.assertRaises(ValueError) as cm:
            CentralCamera.points2F(p1, p2, method="ransac")
        self.assertIn("at least 8", str(cm.exception))

    def test_points2F_degenerate_points_raises(self):
        """Enough points (>= 8), but collinear -- cv2.findFundamentalMat
        itself fails (returns None), which must raise a clear ValueError
        distinct from the too-few-points case above"""
        t = np.linspace(0, 1, 10)
        p1 = np.vstack([10 + 5 * t, 10 + 5 * t])
        p2 = np.vstack([11 + 5 * t, 11 + 5 * t])

        with self.assertRaises(ValueError) as cm:
            CentralCamera.points2F(p1, p2, method="ransac")
        self.assertIn("degenerate", str(cm.exception))

    def test_camera_attributes_access(self):
        """Test accessing various camera attributes"""
        c = CentralCamera(f=0.015, rho=10e-6, imagesize=[1280, 1024], pp=(640, 512))

        # Test accessing imaging plane size
        self.assertEqual(c.width, 1280)
        self.assertEqual(c.height, 1024)

        # Test pixel size
        self.assertEqual(c.rhou, 10e-6)
        self.assertEqual(c.rhov, 10e-6)

    def test_camera_pose_transformations(self):
        """Test camera pose and transformation"""
        c = CentralCamera()

        # Initial pose
        initial_pose = c.pose
        self.assertTrue(isinstance(initial_pose, SE3))

        # Set new pose
        new_pose = SE3(0.5, 0.5, 0.5)
        c.pose = new_pose
        self.assertTrue(c.pose == new_pose)

    def test_camera_matrix_access(self):
        """Check calibration and extrinsics against an analytic matrix."""
        c = CentralCamera(f=0.015, rho=10e-6, pp=(640, 512), pose=SE3(1, 2, 3))
        K = np.array([[1500, 0, 640], [0, 1500, 512], [0, 0, 1]])
        nt.assert_allclose(c.K, K)
        C = np.array([[1500, 0, 640, -3420], [0, 1500, 512, -4536], [0, 0, 1, -3]])
        nt.assert_allclose(c.C(), C)
        # C is the projection matrix; there is no separate P() API.
        P = np.array([1.1, 2.2, 6, 1])
        projected = C @ P
        nt.assert_allclose(c.project_point(P[:3]).ravel(), projected[:2] / projected[2])

    def test_camera_clone_methods(self):
        """A separately constructed camera can reuse a pose without sharing it."""
        c = CentralCamera(f=0.015, pose=SE3(1, 2, 3))
        c_copy = CentralCamera(f=0.015, pose=c.pose)
        self.assertIsNot(c, c_copy)
        nt.assert_allclose(c_copy.pose.A, c.pose.A)
        c_copy.pose = SE3(4, 5, 6)
        nt.assert_allclose(c.pose.t, [1, 2, 3])

    def test_images2C(self):
        """Checkerboard calibration end-to-end (regression: OpenCV 5's
        findChessboardCorners/cornerSubPix return (N,2) instead of OpenCV
        4's (N,1,2) -- verified empirically that cv2.calibrateCamera and
        cv2.drawChessboardCorners both accept either shape, so no source
        change was needed here, but this pipeline had zero prior test
        coverage at all)"""
        sq = 20
        board = np.zeros((8 * sq, 8 * sq), dtype=np.uint8)
        for r in range(8):
            for c in range(8):
                if (r + c) % 2 == 0:
                    board[r * sq : (r + 1) * sq, c * sq : (c + 1) * sq] = 255

        canvas = np.full((300, 300), 128, dtype=np.uint8)
        images = []
        for dx, dy, scale in [(50, 50, 1.0), (60, 40, 1.05), (40, 60, 0.95)]:
            view = canvas.copy()
            bsz = int(board.shape[0] * scale)
            resized = cv2.resize(board, (bsz, bsz))
            view[dy : dy + bsz, dx : dx + bsz] = resized
            images.append(Image(view))

        result = CentralCamera.images2C(images, gridshape=(7, 7), squaresize=0.025)

        self.assertIsNotNone(result)
        C, distortion, frames = result
        self.assertEqual(C.shape, (3, 3))
        self.assertEqual(distortion.shape, (5,))
        self.assertEqual(len(frames), len(images))
        for frame in frames:
            self.assertIsInstance(frame.image, Image)
            self.assertIsInstance(frame.pose, SE3)


class TestCameraPlot(unittest.TestCase):
    """Characterization tests for CameraBase.plot(). This method had zero
    test coverage before an extract-method refactor -- these establish a
    concrete before/after baseline (artist counts on the returned axes)
    rather than deeply validating rendering correctness."""

    def _camera(self):
        return CentralCamera(
            f=0.015, rho=10e-6, imagesize=[1280, 1024], pp=[640, 512], name="cam1"
        )

    def setUp(self):
        # plot()'s ax=None path reuses whatever matplotlib considers the
        # "current" 3D axes (spatialmath.base.graphics.axes_logic) -- close
        # everything first so a figure left open by an unrelated test
        # elsewhere in the suite can't be silently reused here, which would
        # make the artist counts below meaningless.
        plt.close("all")

    def tearDown(self):
        plt.close("all")

    def test_plot_frustum(self):
        ax = self._camera().plot(shape="frustum")
        self.assertIsNotNone(ax)
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(len(ax.lines), 0)

    def test_plot_camera(self):
        ax = self._camera().plot(shape="camera")
        self.assertIsNotNone(ax)
        self.assertEqual(len(ax.collections), 3)
        self.assertEqual(len(ax.lines), 0)

    def test_plot_frame(self):
        # shape defaults to "camera", so this also draws the camera icon
        # in addition to the pose-frame overlay
        ax = self._camera().plot(frame=True)
        self.assertIsNotNone(ax)
        self.assertEqual(len(ax.collections), 4)
        self.assertEqual(len(ax.lines), 3)

    def test_plot_unbound_with_pose(self):
        # eg. CentralCamera.plot(pose=frame.pose, ...) as used to plot a
        # set of calibration frames with no single camera instance
        ax = CentralCamera.plot(pose=SE3.Tz(1), shape="camera")
        self.assertIsNotNone(ax)
        self.assertEqual(len(ax.collections), 3)
        self.assertEqual(len(ax.lines), 0)

    def test_plot_unbound_without_pose_raises(self):
        with self.assertRaises(ValueError):
            CentralCamera.plot(shape="camera")


# ----------------------------------------------------------------------- #
if __name__ == "__main__":

    unittest.main()
