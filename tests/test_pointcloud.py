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
class TestNormals(unittest.TestCase):
    """``normals`` is a property, and also still the old method of that name"""

    def setUp(self):
        self.pc = PointCloud.Read("bunny.ply")

    def test_property_is_an_array_of_columns(self):
        self.assertEqual(self.pc.normals.shape, (3, 0))  # none yet
        self.pc.estimate_normals(radius=0.1, max_nn=30)
        n = self.pc.normals
        self.assertEqual(n.shape, (3, len(self.pc)))
        nt.assert_allclose(np.linalg.norm(n, axis=0), 1)
        self.assertIsInstance(n, np.ndarray)

    def test_old_call_estimates_the_normals_and_warns(self):
        # the printed book has pcd.normals(radius=0.1, max_nn=30)
        with self.assertWarnsRegex(DeprecationWarning, "estimate_normals") as w:
            result = self.pc.normals(radius=0.1, max_nn=30)
        self.assertIsNone(result)
        # the warning is reported at the caller's line, not inside the Toolbox
        self.assertEqual(w.filename, __file__)

        other = PointCloud.Read("bunny.ply")
        other.estimate_normals(radius=0.1, max_nn=30)
        nt.assert_allclose(self.pc.normals, other.normals)

    def test_array_behaves_as_an_array(self):
        self.pc.estimate_normals(radius=0.1, max_nn=30)
        n = self.pc.normals
        self.assertEqual(n.T.shape, (len(self.pc), 3))
        self.assertEqual(n[:, 0].shape, (3,))
        nt.assert_allclose(np.sum(n * n, axis=0), 1)
        self.assertEqual(type(np.asarray(n)), np.ndarray)


# ----------------------------------------------------------------------- #
if __name__ == "__main__":
    unittest.main()
