#!/usr/bin/env python

import unittest

import numpy as np
import numpy.testing as nt

from machinevisiontoolbox import Image


class TestImageSpatial(unittest.TestCase):
    # new test
    def test_gauss_kernel(self):
        """Test Gaussian kernel class method"""
        kernel = Image.Gauss(sigma=1.0)
        self.assertIsNotNone(kernel)
        # Check kernel is normalized
        # self.assertAlmostEqual(np.sum(kernel.K), 1.0, places=5)

    # new test
    def test_sobel_kernel(self):
        """Test Sobel kernel class method"""
        kernel_x = Image.Sobel(direction="x")
        kernel_y = Image.Sobel(direction="y")
        self.assertIsNotNone(kernel_x)
        self.assertIsNotNone(kernel_y)

    # new test
    def test_laplace_kernel(self):
        """Test Laplacian kernel class method"""
        kernel = Image.Laplace()
        self.assertIsNotNone(kernel)

    # new test
    def test_dog_kernel(self):
        """Test Difference of Gaussians kernel"""
        kernel = Image.DoG(sigma1=1.0, sigma2=2.0)
        self.assertIsNotNone(kernel)

    # new test
    def test_log_kernel(self):
        """Test Laplacian of Gaussian kernel"""
        kernel = Image.LoG(sigma=1.0)
        self.assertIsNotNone(kernel)

    # new test
    def test_dgauss_kernel(self):
        """Test derivative of Gaussian kernel"""
        kernel = Image.DGauss(sigma=1.0)
        self.assertIsNotNone(kernel)

    # new test
    def test_convolve(self):
        """Test image convolution"""
        im = Image.Random(size=(50, 50), dtype="float32")
        kernel = np.ones((3, 3)) / 9.0  # Simple averaging kernel
        result = im.convolve(kernel)
        self.assertEqual(result.size, im.size)

    # new test
    def test_smooth(self):
        """Test image smoothing"""
        im = Image.Random(size=(50, 50), dtype="uint8")
        smoothed = im.smooth(sigma=1.0)
        self.assertEqual(smoothed.size, im.size)

    def test_medianfilter(self):
        """Test median filter"""
        im = Image.Random(size=(20, 20), dtype="uint8")
        filtered = im.medianfilter()
        self.assertEqual(filtered.size, im.size)

    # new test
    def test_canny(self):
        """Test Canny edge detection"""
        im = Image.Circles(3, size=128, fg=255, bg=0, dtype="uint8")
        edges = im.canny(sigma=1.0, th0=0.1, th1=0.2)
        self.assertEqual(edges.size, im.size)
        # Should detect some edges
        # self.assertGreater(np.sum(edges.A > 0), 0)

    # new test
    def test_harris_corner_strength(self):
        """Test Harris corner strength computation"""
        im = Image.Squares(3, size=128, fg=255, bg=0, dtype="uint8")
        corners = im.Harris_corner_strength()
        self.assertEqual(corners.size, im.size)

    def test_smooth_non_scalar_sigma_raises(self):
        im = Image.Random(size=(20, 20))
        with self.assertRaises(ValueError):
            im.smooth([1.0, 2.0])

    def test_convolve_border_constant(self):
        im = Image.Random(size=(10, 10))
        K = Image.Gauss(sigma=1.0)
        out = im.convolve(K, border="constant")
        self.assertEqual(out.size, im.size)

    def test_convolve_mode_valid(self):
        im = Image.Random(size=(20, 20))
        K = Image.Gauss(sigma=1.0)
        out = im.convolve(K, mode="valid")
        # valid mode crops edges
        self.assertLessEqual(out.width, im.width)

    def test_sobel_direction_y(self):
        kernel = Image.Sobel(direction="y")
        self.assertIsNotNone(kernel)

    def test_sobel_invalid_direction_raises(self):
        with self.assertRaises(ValueError):
            Image.Sobel(direction="z")


class TestKernel(unittest.TestCase):
    """Tests for the Kernel class"""

    def setUp(self):
        from machinevisiontoolbox import Kernel

        self.Kernel = Kernel

    def test_kernel_str(self):
        K = self.Kernel.Gauss(sigma=2)
        s = str(K)
        self.assertIn("Kernel:", s)
        self.assertIn("min=", s)
        self.assertIn("max=", s)

    def test_kernel_repr(self):
        K = self.Kernel.Gauss(sigma=2)
        r = repr(K)
        self.assertIn("Kernel:", r)

    def test_kernel_symmetric_str(self):
        K = self.Kernel.Gauss(sigma=2)
        s = str(K)
        self.assertIn("SYMMETRIC", s)

    def test_kernel_name_in_str(self):
        K = self.Kernel.Gauss(sigma=2)
        s = str(K)
        self.assertIn("Gauss", s)

    def test_kernel_print(self):
        K = self.Kernel.Gauss(sigma=1)
        import io
        import sys

        captured = io.StringIO()
        sys.stdout = captured
        try:
            K.print()
        finally:
            sys.stdout = sys.__stdout__
        output = captured.getvalue()
        self.assertGreater(len(output), 0)

    def test_kernel_print_with_fmt(self):
        K = self.Kernel.Box(2)
        import io
        import sys

        captured = io.StringIO()
        sys.stdout = captured
        try:
            K.print(fmt=" {:5.3f}")
        finally:
            sys.stdout = sys.__stdout__
        self.assertGreater(len(captured.getvalue()), 0)

    def test_kernel_shape_property(self):
        K = self.Kernel.Gauss(sigma=2, h=3)
        self.assertEqual(K.shape, (7, 7))

    def test_kernel_box(self):
        K = self.Kernel.Box(2)
        self.assertEqual(K.shape, (5, 5))
        # normalized: sum should be 1
        self.assertAlmostEqual(np.sum(K.K), 1.0, places=5)

    def test_kernel_box_no_normalize(self):
        K = self.Kernel.Box(2, normalize=False)
        self.assertEqual(K.shape, (5, 5))
        nt.assert_array_equal(K.K, np.ones((5, 5)))

    def test_kernel_circle_disc(self):
        K = self.Kernel.Circle(radius=3)
        # kernel should be square
        self.assertEqual(K.shape[0], K.shape[1])
        # should contain ones inside circle
        self.assertEqual(K.K[int(K.shape[0] / 2), int(K.shape[1] / 2)], 1)

    def test_kernel_circle_annulus(self):
        # radius as [rmin, rmax] → annulus
        K = self.Kernel.Circle(radius=[2, 4])
        self.assertEqual(K.shape[0], K.shape[1])
        # centre of annulus should be 0 (inside inner radius)
        c = K.shape[0] // 2
        self.assertEqual(K.K[c, c], 0)

    def test_kernel_dog_auto_sigma2(self):
        # sigma2=None: auto-set to 1.6*sigma1
        K = self.Kernel.DoG(sigma1=2.0)
        self.assertIsNotNone(K)
        self.assertIsInstance(K.K, np.ndarray)

    def test_kernel_dog_explicit_sigma2(self):
        # When sigma2 > sigma1, they are swapped internally
        K1 = self.Kernel.DoG(sigma1=2.0, sigma2=3.0)
        K2 = self.Kernel.DoG(sigma1=3.0, sigma2=2.0)
        nt.assert_array_almost_equal(K1.K, K2.K)

    def test_bordertype_sp_exclude_raises(self):
        im = Image.Random(size=(10, 10))
        from machinevisiontoolbox.ImageSpatial import ImageSpatialMixin

        with self.assertRaises(ValueError):
            ImageSpatialMixin._bordertype_sp("constant", exclude=["constant"])

    def test_bordertype_sp_invalid_raises(self):
        from machinevisiontoolbox.ImageSpatial import ImageSpatialMixin

        with self.assertRaises(ValueError):
            ImageSpatialMixin._bordertype_sp("not_a_valid_border")


# ----------------------------------------------------------------------- #
if __name__ == "__main__":
    unittest.main()
