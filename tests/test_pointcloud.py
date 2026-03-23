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


# ----------------------------------------------------------------------- #
if __name__ == "__main__":
    unittest.main()
