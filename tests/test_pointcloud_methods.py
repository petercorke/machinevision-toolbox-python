#!/usr/bin/env python
"""Tests of the PointCloud methods that wrap Open3D: copy, transforms, downsampling,
normals, selection, painting, ICP and VoxelGrid.write.  These use real Open3D
objects, not mocks."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as nt
from spatialmath import SE3

from machinevisiontoolbox import PointCloud

try:
    import open3d  # noqa: F401

    open3d_available = True
except ImportError:
    open3d_available = False


def _cloud(n: int = 200, seed: int = 0) -> PointCloud:
    return PointCloud(np.random.default_rng(seed).random((3, n)))


@unittest.skipUnless(open3d_available, "open3d not installed")
class TestPointCloudBasics(unittest.TestCase):
    def test_constructor_rejects_other_types(self):
        with self.assertRaisesRegex(ValueError, "arg must be"):
            PointCloud("not a cloud")

    def test_wrapping_an_open3d_cloud(self):
        pc = _cloud()
        again = PointCloud(pc.pcd)
        self.assertIs(again.pcd, pc.pcd)  # wraps, does not copy
        self.assertEqual(len(again), 200)

    def test_copy_is_independent(self):
        pc = _cloud()
        c = pc.copy()
        self.assertEqual(len(c), len(pc))
        nt.assert_allclose(c.points, pc.points)
        c.paint("red")
        self.assertFalse(pc.iscolor)
        self.assertTrue(c.iscolor)

    def test_str_and_repr(self):
        pc = _cloud(50)
        self.assertIn("50", str(pc))
        self.assertEqual(repr(pc), "PointCloud(npoints=50, colored=False)")
        self.assertEqual(
            repr(pc.copy().paint("red")), "PointCloud(npoints=50, colored=True)"
        )

    def test_open3d_method_is_wrapped_and_returns_a_pointcloud(self):
        pc = _cloud()
        # an Open3D method that returns an Open3D point cloud
        out = pc.crop(
            __import__("open3d").geometry.AxisAlignedBoundingBox(
                [0, 0, 0], [0.5, 0.5, 0.5]
            )
        )
        self.assertIsInstance(out, PointCloud)
        self.assertLess(len(out), len(pc))
        # an Open3D method that returns something else is passed through
        self.assertEqual(len(pc.get_center()), 3)

    def test_unknown_attribute_suggests_a_name(self):
        with self.assertRaisesRegex(AttributeError, "Did you mean"):
            _cloud().get_axis_alligned_bounding_box


@unittest.skipUnless(open3d_available, "open3d not installed")
class TestPointCloudTransform(unittest.TestCase):
    def setUp(self):
        self.pc = _cloud()
        self.T = SE3.Tz(2) * SE3.Rx(0.3)

    def expected(self):
        return self.T.A[:3, :3] @ self.pc.points + self.T.A[:3, 3:4]

    def test_transform_in_place(self):
        want = self.expected()
        out = self.pc.transform(self.T)
        self.assertIs(out, self.pc)
        nt.assert_allclose(self.pc.points, want, atol=1e-6)

    def test_transform_to_a_copy(self):
        before = self.pc.points.copy()
        want = self.expected()
        out = self.pc.transform(self.T, inplace=False)
        self.assertIsNot(out, self.pc)
        nt.assert_allclose(self.pc.points, before)
        nt.assert_allclose(out.points, want, atol=1e-6)

    def test_premultiply_gives_a_new_cloud(self):
        before = self.pc.points.copy()
        out = self.T * self.pc
        nt.assert_allclose(self.pc.points, before)
        nt.assert_allclose(out.points, self.expected(), atol=1e-6)

    def test_inplace_multiply(self):
        want = self.expected()
        pc = self.pc
        pc *= self.T
        self.assertIs(pc, self.pc)
        nt.assert_allclose(pc.points, want, atol=1e-6)

    def test_multiply_by_something_else_is_refused(self):
        with self.assertRaises(TypeError):
            self.pc *= 3
        with self.assertRaises(TypeError):
            3 * self.pc


@unittest.skipUnless(open3d_available, "open3d not installed")
class TestPointCloudProcessing(unittest.TestCase):
    def test_sum_of_clouds(self):
        a, b = _cloud(10, 1), _cloud(20, 2)
        self.assertEqual(len(sum([a, b])), 30)
        self.assertIs(0 + a, a)  # what sum() starts from
        with self.assertRaises(TypeError):
            "x" + a

    def test_downsample_voxel_reduces_the_cloud(self):
        pc = _cloud(2000)
        out = pc.downsample_voxel(0.2)
        self.assertIsInstance(out, PointCloud)
        self.assertLess(len(out), len(pc))
        self.assertGreater(len(out), 0)

    def test_downsample_random_fraction(self):
        pc = _cloud(1000)
        self.assertEqual(len(pc.downsample_random(0.25, seed=1)), 250)
        # without a seed Open3D chooses, the size is still the fraction
        out = pc.downsample_random(0.25)
        self.assertEqual(len(out), 250)

    def test_downsample_random_is_repeatable_with_a_seed(self):
        pc = _cloud(1000)
        a = pc.downsample_random(0.1, seed=7)
        b = pc.downsample_random(0.1, seed=7)
        nt.assert_array_equal(a.points, b.points)

    def test_estimate_normals(self):
        pc = _cloud(500)
        self.assertEqual(pc.normals.shape, (3, 0))
        pc.estimate_normals(radius=0.3, max_nn=20)
        self.assertEqual(pc.normals.shape, (3, 500))
        nt.assert_allclose(np.linalg.norm(pc.normals, axis=0), 1, atol=1e-5)

    def test_remove_outlier_is_deprecated(self):
        pc = _cloud(500)
        with self.assertWarns(DeprecationWarning):
            out = pc.remove_outlier(nb_points=5, radius=0.2)
        self.assertIsInstance(out, PointCloud)
        self.assertLessEqual(len(out), len(pc))

    def test_select_by_index_and_function(self):
        pts = np.array([[0.0, 1, 2, 3], [0, 0, 0, 0], [0, 1, 2, 3]])
        pc = PointCloud(pts)
        sel = pc.select([1, 3])
        nt.assert_allclose(sel.points, pts[:, [1, 3]])
        inv = pc.select([1, 3], invert=True)
        nt.assert_allclose(inv.points, pts[:, [0, 2]])
        near = pc.select(lambda p: p[2] < 1.5)
        nt.assert_allclose(near.points, pts[:, [0, 1]])

    def test_paint_with_a_name_or_a_value(self):
        pc = _cloud(10)
        nt.assert_allclose(pc.paint("red").colors[:, 0], [1, 0, 0])
        nt.assert_allclose(pc.paint([0, 0, 1]).colors[:, 3], [0, 0, 1])
        self.assertIs(pc.paint("green"), pc)  # returns self

    def test_paint_unknown_name(self):
        with self.assertRaisesRegex(ValueError, "unknown color name"):
            _cloud(10).paint("not-a-color")

    def test_segment_plane(self):
        rng = np.random.default_rng(3)
        flat = np.vstack([rng.random((2, 300)), np.zeros(300)])  # z = 0
        noise = rng.random((3, 30)) + np.array([[0], [0], [2]])
        pc = PointCloud(np.hstack([flat, noise]))
        model, inliers, outliers = pc.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=200
        )
        self.assertEqual(len(inliers) + len(outliers), len(pc))
        self.assertGreaterEqual(len(inliers), 290)
        self.assertAlmostEqual(abs(model[2]), 1, places=2)  # normal is z


@unittest.skipUnless(open3d_available, "open3d not installed")
class TestPointCloudICP(unittest.TestCase):
    def test_icp_recovers_a_small_motion(self):
        target = _cloud(800, 5)
        T_true = SE3.Trans(0.02, -0.01, 0.015) * SE3.Rz(0.02)
        moved = target.transform(T_true, inplace=False)
        T, status = moved.ICP(target, max_correspondence_distance=0.2)
        self.assertIsInstance(T, SE3)
        self.assertGreater(status.fitness, 0.9)
        # moving the cloud by T lands it on the target, so T is the inverse motion
        nt.assert_allclose(T.A, T_true.inv().A, atol=5e-3)

    def test_icp_starts_from_an_initial_guess(self):
        target = _cloud(800, 6)
        moved = target.transform(SE3.Trans(0.05, 0, 0), inplace=False)
        T, status = moved.ICP(
            target, T0=SE3.Trans(-0.05, 0, 0), max_correspondence_distance=0.1
        )
        nt.assert_allclose(T.t, [-0.05, 0, 0], atol=5e-3)


@unittest.skipUnless(open3d_available, "open3d not installed")
class TestPointCloudFiles(unittest.TestCase):
    def test_write_and_read_back(self):
        pc = _cloud(40).paint("blue")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cloud.ply"
            self.assertTrue(pc.write(path))
            back = PointCloud.Read(str(path))
        self.assertEqual(len(back), 40)
        nt.assert_allclose(back.points, pc.points, atol=1e-5)
        self.assertTrue(back.iscolor)

    def test_voxelgrid_write(self):
        vg = _cloud(500).voxel_grid(0.25)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "grid.ply"
            vg.write(path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


@unittest.skipUnless(open3d_available, "open3d not installed")
class TestDepthImageErrors(unittest.TestCase):
    def test_camera_is_required(self):
        from machinevisiontoolbox import Image

        with self.assertRaisesRegex(ValueError, "camera"):
            PointCloud.DepthImage(Image(np.ones((4, 4), dtype=np.float32)), None)

    def test_depth_must_be_an_image(self):
        from machinevisiontoolbox import CentralCamera

        camera = CentralCamera(f=0.008, rho=10e-6, imagesize=(640, 480))
        with self.assertRaisesRegex(ValueError, "Image"):
            PointCloud.DepthImage(np.ones((4, 4)), camera)


if __name__ == "__main__":
    unittest.main()
