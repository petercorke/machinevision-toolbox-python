"""
Tests for ``from machinevisiontoolbox import *`` behaviour.

Verifies that:
- the star-import executes without error
- all intended public names land in the namespace
- ``mvb`` is the base module (base functions accessible via ``mvb.iread``, etc.)
- internal / third-party names are NOT polluted into the namespace
"""

import unittest
import types


class TestStarImport(unittest.TestCase):

    def setUp(self):
        """Execute the star-import in an isolated namespace."""
        self.ns = {}
        exec("from machinevisiontoolbox import *", self.ns)

    # ------------------------------------------------------------------
    # Core classes
    # ------------------------------------------------------------------
    def test_image_present(self):
        self.assertIn("Image", self.ns)

    def test_pointcloud_present(self):
        self.assertIn("PointCloud", self.ns)

    # ------------------------------------------------------------------
    # Source classes
    # ------------------------------------------------------------------
    def test_videofile_present(self):
        self.assertIn("VideoFile", self.ns)

    def test_videocamera_present(self):
        self.assertIn("VideoCamera", self.ns)

    def test_imagecollection_present(self):
        self.assertIn("ImageCollection", self.ns)

    # ------------------------------------------------------------------
    # Processing helpers
    # ------------------------------------------------------------------
    def test_kernel_present(self):
        self.assertIn("Kernel", self.ns)

    def test_histogram_present(self):
        self.assertIn("Histogram", self.ns)

    # ------------------------------------------------------------------
    # Camera classes
    # ------------------------------------------------------------------
    def test_centralcamera_present(self):
        self.assertIn("CentralCamera", self.ns)

    def test_fisheye_present(self):
        self.assertIn("FishEyeCamera", self.ns)

    def test_sphericalcamera_present(self):
        self.assertIn("SphericalCamera", self.ns)

    # ------------------------------------------------------------------
    # mvb alias
    # ------------------------------------------------------------------
    def test_mvb_is_module(self):
        self.assertIn("mvb", self.ns)
        self.assertIsInstance(self.ns["mvb"], types.ModuleType)

    def test_mvb_has_iread(self):
        self.assertTrue(hasattr(self.ns["mvb"], "iread"))

    def test_mvb_has_idisp(self):
        self.assertTrue(hasattr(self.ns["mvb"], "idisp"))

    def test_mvb_has_mkgrid(self):
        self.assertTrue(hasattr(self.ns["mvb"], "mkgrid"))

    # ------------------------------------------------------------------
    # No namespace pollution
    # ------------------------------------------------------------------
    def test_se3_not_exported(self):
        """SE3 comes from spatialmath and must not leak into the MVTB namespace."""
        self.assertNotIn("SE3", self.ns)

    def test_numpy_not_exported(self):
        self.assertNotIn("np", self.ns)
        self.assertNotIn("numpy", self.ns)

    def test_cv2_not_exported(self):
        self.assertNotIn("cv2", self.ns)

    def test_os_not_exported(self):
        self.assertNotIn("os", self.ns)

    def test_rectangleselector_not_exported(self):
        self.assertNotIn("RectangleSelector", self.ns)

    # ------------------------------------------------------------------
    # Base functions not exported directly via star
    # ------------------------------------------------------------------
    def test_iread_not_in_star_namespace(self):
        """iread is accessible via mvb.iread but not as a bare name after star-import."""
        self.assertNotIn("iread", self.ns)

    def test_idisp_not_in_star_namespace(self):
        self.assertNotIn("idisp", self.ns)

    def test_mkgrid_not_in_star_namespace(self):
        self.assertNotIn("mkgrid", self.ns)

    def test_blackbody_not_in_star_namespace(self):
        self.assertNotIn("blackbody", self.ns)


if __name__ == "__main__":
    unittest.main()
