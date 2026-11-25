#!/usr/bin/env python

import unittest

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt
from spatialmath import SE3

from machinevisiontoolbox import CentralCamera, Image, PointCloud

try:
    import open3d  # noqa: F401

    open3d_available = True
except ImportError:
    open3d_available = False


@unittest.skipUnless(open3d_available, "open3d not installed")
class TestPointCloud(unittest.TestCase):
    def test_constructor(self):
        pts = np.random.rand(3, 100)
        pc = PointCloud(pts)
        self.assertIsInstance(pc, PointCloud)
        self.assertEqual(len(pc), 100)

        from open3d.data import SampleTUMRGBDImage

        data = SampleTUMRGBDImage()

        rgb = Image.Read(data.color_path)
        d = Image.Read(data.depth_path)
        camera = CentralCamera(f=0.008, rho=10e-6, imagesize=(640, 480))

        pc = PointCloud.DepthImage(d, camera, depth_scale=0.001)

        pc = PointCloud.DepthImage(d, camera, rgb=rgb)

        rgbd = Image.Pstack((d, rgb.astype("uint16")), colororder="DRGB")
        pc = PointCloud.DepthImage(rgbd, camera)

    def test_read(self):
        # bare filename -- the documented convention
        pc = PointCloud.Read("bunny.ply")
        self.assertIsInstance(pc, PointCloud)
        self.assertEqual(len(pc), 35947)

        # a caller-supplied redundant "data/" prefix (the older convention
        # some book examples still use) must resolve to the same file,
        # not double-join to data/data/bunny.ply
        pc2 = PointCloud.Read("data/bunny.ply")
        self.assertEqual(len(pc2), len(pc))


try:
    import plotly  # noqa: F401

    plotly_available = True
except ImportError:
    plotly_available = False


@unittest.skipUnless(open3d_available, "open3d not installed")
class TestPointCloudNotebookDisplay(unittest.TestCase):
    def setUp(self):
        self.pc = PointCloud(np.random.rand(3, 500))

    @unittest.skipUnless(plotly_available, "plotly not installed")
    def test_shown_once_as_a_figure_with_points(self):
        from unittest.mock import patch

        with patch("plotly.graph_objects.Figure.show") as show:
            result = self.pc.disp(notebook=True, title="My cloud")
        self.assertEqual(show.call_count, 1)
        # nothing is returned, or a cell ending in the call would show it again
        self.assertIsNone(result)

    @unittest.skipUnless(plotly_available, "plotly not installed")
    def test_axes_have_the_same_scale(self):
        from unittest.mock import patch

        # a long thin cloud, for which Plotly's default would draw a cube
        pc = PointCloud(np.random.rand(3, 500) * np.array([[10.0], [0.5], [0.5]]))
        shown = []
        with patch(
            "plotly.graph_objects.Figure.show",
            autospec=True,
            side_effect=lambda fig, *a, **k: shown.append(fig),
        ):
            pc.disp(notebook=True)
        self.assertEqual(shown[0].layout.scene.aspectmode, "data")

    @unittest.skipUnless(plotly_available, "plotly not installed")
    def test_figure_has_the_points_and_title(self):
        from unittest.mock import patch

        shown = []
        with patch(
            "plotly.graph_objects.Figure.show",
            autospec=True,
            side_effect=lambda fig, *a, **k: shown.append(fig),
        ):
            self.pc.disp(notebook=True, title="My cloud", width=300, height=200)
        fig = shown[0]
        self.assertEqual(len(fig.data[0].x), 500)
        self.assertEqual(fig.layout.title.text, "My cloud")
        self.assertEqual((fig.layout.width, fig.layout.height), (300, 200))

    @unittest.skipUnless(plotly_available, "plotly not installed")
    def test_point_sample_factor_thins_the_cloud(self):
        from unittest.mock import patch

        shown = []
        with patch(
            "plotly.graph_objects.Figure.show",
            autospec=True,
            side_effect=lambda fig, *a, **k: shown.append(fig),
        ):
            self.pc.disp(notebook=True, point_sample_factor=0.2)
        self.assertLess(len(shown[0].data[0].x), 500)

    def test_plotly_missing_says_how_to_install_it(self):
        from unittest.mock import patch

        # a None entry in sys.modules makes the import raise ImportError
        with patch.dict("sys.modules", {"plotly": None}):
            with self.assertRaisesRegex(ImportError, "pip install plotly"):
                self.pc.disp(notebook=True)

    def test_file_cannot_be_used(self):
        with self.assertRaisesRegex(ValueError, "file"):
            self.pc.disp(notebook=True, file="x.png")


@unittest.skipUnless(open3d_available, "open3d not installed")
class TestVoxelGridNotebookDisplay(unittest.TestCase):
    def setUp(self):
        self.vg = PointCloud(np.random.rand(3, 2000)).voxel_grid(0.25)

    def test_one_cube_per_voxel_with_the_bounds_of_the_grid(self):
        n = len(self.vg._voxels.get_voxels())
        mesh = self.vg._cubes()
        self.assertEqual(len(mesh.triangles), 12 * n)
        self.assertEqual(len(mesh.vertices), 8 * n)
        nt.assert_allclose(mesh.get_min_bound(), self.vg._voxels.get_min_bound())
        nt.assert_allclose(mesh.get_max_bound(), self.vg._voxels.get_max_bound())

    def test_cubes_are_gray_when_the_voxels_have_no_color(self):
        colors = np.asarray(self.vg._cubes().vertex_colors)
        nt.assert_allclose(colors, 0.7)

    def test_cubes_keep_the_colors_of_the_voxels(self):
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.random.rand(500, 3))
        pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (500, 1)))
        colors = np.asarray(PointCloud(pcd).voxel_grid(0.25)._cubes().vertex_colors)
        nt.assert_allclose(colors, np.tile([1.0, 0.0, 0.0], (len(colors), 1)))

    @unittest.skipUnless(plotly_available, "plotly not installed")
    def test_shown_once_with_a_mesh_of_the_voxels(self):
        from unittest.mock import patch

        shown = []
        with patch(
            "plotly.graph_objects.Figure.show",
            autospec=True,
            side_effect=lambda fig, *a, **k: shown.append(fig),
        ):
            result = self.vg.disp(notebook=True, title="Voxels")
        self.assertIsNone(result)
        self.assertEqual(len(shown), 1)
        self.assertEqual(shown[0].layout.title.text, "Voxels")
        self.assertEqual(len(shown[0].data), 1)
        self.assertEqual(len(shown[0].data[0].x), len(self.vg._cubes().vertices))

    @unittest.skipUnless(plotly_available, "plotly not installed")
    def test_axes_have_the_same_scale(self):
        from unittest.mock import patch

        shown = []
        with patch(
            "plotly.graph_objects.Figure.show",
            autospec=True,
            side_effect=lambda fig, *a, **k: shown.append(fig),
        ):
            self.vg.disp(notebook=True)
        self.assertEqual(shown[0].layout.scene.aspectmode, "data")

    def test_plotly_missing_says_how_to_install_it(self):
        from unittest.mock import patch

        with patch.dict("sys.modules", {"plotly": None}):
            with self.assertRaisesRegex(ImportError, "pip install plotly"):
                self.vg.disp(notebook=True)

    def test_file_cannot_be_used(self):
        with self.assertRaisesRegex(ValueError, "file"):
            self.vg.disp(notebook=True, file="x.png")


@unittest.skipUnless(open3d_available, "open3d not installed")
class TestConcatenateAndUnpaint(unittest.TestCase):
    def setUp(self):
        self.red = PointCloud(np.random.rand(3, 4)).paint("red")
        self.plain = PointCloud(np.random.rand(3, 3))
        self.empty = PointCloud(np.empty((3, 0)))

    def test_both_colored(self):
        c = self.red.concatenate(PointCloud(np.random.rand(3, 2)).paint("blue"))
        self.assertEqual(c.colors.shape, (3, 6))
        nt.assert_allclose(c.colors[:, 0], [1, 0, 0])
        nt.assert_allclose(c.colors[:, 5], [0, 0, 1])

    def test_neither_colored_gives_no_colors(self):
        c = self.plain.concatenate(self.plain)
        self.assertEqual(len(c), 6)
        self.assertFalse(c.iscolor)

    def test_uncolored_points_get_the_default_gray(self):
        for c in (self.red + self.plain, self.plain + self.red):
            self.assertEqual(len(c), 7)
            self.assertEqual(c.colors.shape, (3, 7))
        c = self.red + self.plain
        nt.assert_allclose(c.colors[:, 3], [1, 0, 0])
        nt.assert_allclose(c.colors[:, 4], 0.7)

    def test_uncolored_points_get_the_given_color(self):
        c = self.red.concatenate(self.plain, color="blue")
        nt.assert_allclose(c.colors[:, 4:], np.tile([[0], [0], [1]], (1, 3)))
        c = self.red.concatenate(self.plain, color=[0, 1, 0])
        nt.assert_allclose(c.colors[:, 4:], np.tile([[0], [1], [0]], (1, 3)))

    def test_color_none_drops_all_colors(self):
        c = self.red.concatenate(self.plain, color=None)
        self.assertEqual(len(c), 7)
        self.assertFalse(c.iscolor)

    def test_unknown_color_name(self):
        with self.assertRaisesRegex(ValueError, "unknown color name"):
            self.red.concatenate(self.plain, color="not-a-color")

    def test_several_clouds(self):
        c = self.red.concatenate(self.plain, self.red, color="blue")
        self.assertEqual(len(c), 11)
        self.assertEqual(c.colors.shape, (3, 11))
        nt.assert_allclose(c.colors[:, 5], [0, 0, 1])
        nt.assert_allclose(c.colors[:, 8], [1, 0, 0])

    def test_inputs_are_not_changed(self):
        self.red.concatenate(self.plain)
        self.assertEqual((len(self.red), len(self.plain)), (4, 3))
        self.assertFalse(self.plain.iscolor)

    def test_empty_clouds(self):
        for c in (self.red + self.empty, self.empty + self.red):
            self.assertEqual(c.colors.shape, (3, 4))
        for c in (self.plain + self.empty, self.empty + self.plain):
            self.assertEqual(len(c), 3)
            self.assertFalse(c.iscolor)
        c = self.empty + self.empty
        self.assertEqual(len(c), 0)
        self.assertFalse(c.iscolor)
        # an empty cloud does not make the others colored
        c = sum([self.empty, self.plain, self.empty])
        self.assertFalse(c.iscolor)
        c = sum([self.red, self.empty, self.plain])
        self.assertEqual(c.colors.shape, (3, 7))

    def test_unpaint_removes_the_colors(self):
        result = self.red.unpaint()
        self.assertIs(result, self.red)
        self.assertFalse(self.red.iscolor)
        self.assertEqual(len(self.red), 4)
        # and it can be painted again
        self.assertTrue(self.red.paint("green").iscolor)

    def test_unpaint_of_an_uncolored_cloud(self):
        self.assertFalse(self.plain.unpaint().iscolor)


@unittest.skipUnless(open3d_available, "open3d not installed")
class TestSegmentPlane(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        ground = np.vstack((rng.random((2, 800)) * 10, rng.normal(0, 0.01, 800)))
        objects = rng.random((3, 200)) * [[1], [1], [1]] + np.array([[4], [4], [0.5]])
        self.pc = PointCloud(np.hstack((ground, objects)))

    def test_splits_the_points_into_plane_and_rest(self):
        plane, on_plane, rest = self.pc.segment_plane(distance_threshold=0.05)
        self.assertEqual(len(on_plane) + len(rest), len(self.pc))
        self.assertEqual(np.shape(plane), (4,))
        # the ground is z = 0
        nt.assert_allclose(np.abs(plane[2]), 1, atol=0.01)
        self.assertGreater(len(on_plane), 780)
        self.assertEqual(len(self.pc), 1000)  # not changed

    def test_default_is_1000_iterations_of_3_points(self):
        import open3d as o3d

        o3d.utility.random.seed(1)
        a = self.pc.segment_plane(distance_threshold=0.05)
        o3d.utility.random.seed(1)
        b = self.pc.segment_plane(
            distance_threshold=0.05, ransac_n=3, num_iterations=1000
        )
        nt.assert_allclose(a[0], b[0])
        self.assertEqual(len(a[1]), len(b[1]))

    def test_distance_threshold_is_required(self):
        with self.assertRaises(TypeError):
            self.pc.segment_plane()


# ----------------------------------------------------------------------- #
if __name__ == "__main__":
    unittest.main()
