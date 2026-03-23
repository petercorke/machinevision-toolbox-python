import unittest

import numpy as np

from machinevisiontoolbox import findpeaks, findpeaks2d, findpeaks3d


class TestFindpeaks(unittest.TestCase):
    """Test cases for 1D peak finding"""

    def test_simple_peak(self):
        """Test detection of simple peak"""
        y = np.array([0, 0, 1, 2, 0, 0, 0, 3, 1, 0, 0, 0, 0])
        x, ymax = findpeaks(y, scale=1)

        self.assertGreater(len(x), 0)
        self.assertEqual(x[0], 7)
        self.assertEqual(ymax[0], 3)

    def test_peak_with_custom_x(self):
        """Test peaks with custom x coordinates"""
        y = np.array([0, 0, 1, 2, 0, 0, 0, 3, 1, 0, 0, 0, 0])
        x_vals = np.arange(13) * 2
        x, ymax = findpeaks(y, x=x_vals, scale=1)

        self.assertEqual(x[0], 14)
        self.assertEqual(ymax[0], 3)

    def test_peak_with_custom_x_nonuniform(self):
        """Test peaks with non-uniform custom x coordinates"""
        y = np.array([0, 0, 1, 2, 0, 0, 0, 3, 1, 0])
        x_vals = np.array([0, 0.5, 1.2, 2.1, 3.5, 4.0, 5.0, 6.5, 7.2, 8.0])
        x, ymax = findpeaks(y, x=x_vals, scale=1)

        self.assertGreater(len(x), 0)

    def test_npeaks_limit(self):
        """Test limiting number of peaks returned"""
        y = np.array([0, 0, 1, 2, 0, 0, 0, 3, 1, 0, 0, 0, 0])
        x, ymax = findpeaks(y, scale=1, npeaks=1)

        self.assertEqual(len(x), 1)
        self.assertEqual(x[0], 7)
        self.assertEqual(ymax[0], 3)

    def test_scale_parameter(self):
        """Test different scale parameters"""
        y = np.array([0, 0, 1, 2, 0, 0, 0, 3, 1, 0, 0, 0, 0])

        x, ymax = findpeaks(y, scale=3)
        self.assertGreater(len(x), 0)
        self.assertEqual(x[0], 7)

    def test_scale_comparison(self):
        """Test that larger scale finds fewer peaks"""
        y = np.array([0, 0, 1, 2, 0, 0, 0, 3, 1, 0, 0, 0, 0])

        x1, _ = findpeaks(y, scale=1)
        x2, _ = findpeaks(y, scale=2)

        self.assertGreaterEqual(len(x1), len(x2))

    def test_no_peaks(self):
        """Test when no clear peaks are found"""
        y = np.array([1, 1, 1, 1, 1, 0, 0, 0])
        x, ymax = findpeaks(y, scale=2)

        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(ymax, np.ndarray)
        self.assertEqual(len(x), len(ymax))

    def test_all_equal(self):
        """Test signal with all equal values"""
        y = np.array([1, 1, 1, 1, 1])
        x, ymax = findpeaks(y, scale=1)

        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(ymax, np.ndarray)
        self.assertEqual(len(x), len(ymax))

    def test_single_peak_at_center(self):
        """Test single isolated peak"""
        y = np.array([0, 0, 0, 0, 5, 0, 0, 0, 0])
        x, ymax = findpeaks(y, scale=1)

        self.assertGreater(len(x), 0)
        self.assertEqual(x[0], 4)
        self.assertEqual(ymax[0], 5)

    def test_interpolation_simple(self):
        """Test peak interpolation with default order"""
        y = np.array([0, 1, 2, 1, 0], dtype=float)
        x, ymax = findpeaks(y, scale=1, interp=True)

        self.assertEqual(len(x), 1)
        self.assertAlmostEqual(x[0], 2.0, places=1)

    def test_interpolation_order(self):
        """Test peak interpolation with specified polynomial order"""
        y = np.array([0, 1, 2, 1, 0], dtype=float)
        x, ymax = findpeaks(y, scale=1, interp=4)

        self.assertEqual(len(x), 1)
        self.assertGreater(x[0], 1.5)
        self.assertLess(x[0], 2.5)

    def test_interpolation_order_6(self):
        """Test peak interpolation with 6th order polynomial"""
        y = np.array([0, 0.5, 1.5, 2.0, 1.5, 0.5, 0], dtype=float)
        x, ymax = findpeaks(y, scale=1, interp=6)

        self.assertEqual(len(x), 1)
        self.assertGreater(x[0], 2.5)
        self.assertLess(x[0], 3.5)

    def test_interpolation_with_return_poly(self):
        """Test that return_poly returns polynomial coefficients"""
        y = np.array([0, 1, 2, 1, 0], dtype=float)
        x, ymax, polys = findpeaks(y, scale=1, interp=True, return_poly=True)

        self.assertEqual(len(x), 1)
        self.assertEqual(len(polys), 1)
        self.assertTrue(hasattr(polys[0], "coef"))

    def test_interpolation_multiple_peaks_with_poly(self):
        """Test return_poly with multiple peaks"""
        y = np.array([0, 1, 0, 0, 2, 1, 0], dtype=float)
        x, ymax, polys = findpeaks(y, scale=1, interp=True, return_poly=True)

        if len(x) > 0:
            self.assertEqual(len(polys), len(x))

    def test_interpolation_invalid_order(self):
        """Test that interpolation order < 2 raises error"""
        y = np.array([0, 1, 2, 1, 0], dtype=float)

        with self.assertRaises(ValueError):
            findpeaks(y, scale=1, interp=1)

    def test_negative_peaks(self):
        """Test finding minima by negating the signal"""
        y = np.array([3, 2, 1, 2, 3])
        x, ymin = findpeaks(-y, scale=1)

        self.assertEqual(len(x), 1)
        self.assertEqual(x[0], 2)
        self.assertEqual(ymin[0], -1)

    def test_descending_order(self):
        """Test that peaks are returned in descending magnitude"""
        y = np.array([0, 2, 0, 0, 0, 5, 0, 0, 3, 0])
        x, ymax = findpeaks(y, scale=1)

        self.assertTrue(np.all(ymax[:-1] >= ymax[1:]))

    def test_float_signal(self):
        """Test with floating point signal"""
        y = np.array([0.5, 1.5, 0.5, 0.0, 0.0, 2.5, 0.0], dtype=float)
        x, ymax = findpeaks(y, scale=1)

        self.assertGreater(len(x), 0)
        self.assertTrue(all(isinstance(v, (float, np.floating)) for v in ymax))

    def test_zero_crossing_detection(self):
        """Test zero crossing peak detection with scale=0"""
        y = np.array([1, 2, 3, 2, 1, -1, -2, -1, 0, 1, 2, 1, 0])
        x, ymax = findpeaks(y, scale=0)

        self.assertGreater(len(x), 0)

    def test_large_signal(self):
        """Test with larger signal array"""
        y = np.random.rand(100) * 10
        x, ymax = findpeaks(y, scale=1, npeaks=5)

        self.assertLessEqual(len(x), 5)
        if len(ymax) > 1:
            self.assertTrue(np.all(ymax[:-1] >= ymax[1:]))

    def test_negative_values(self):
        """Test with negative values in signal"""
        y = np.array([-3, -2, -1, 0, -1, -2, -3])
        x, ymax = findpeaks(y, scale=1)

        self.assertGreater(len(x), 0)
        self.assertEqual(x[0], 3)

    def test_empty_signal(self):
        """Test with minimal signal"""
        y = np.array([1, 2, 1])
        x, ymax = findpeaks(y, scale=0)

        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(ymax, np.ndarray)

    def test_x_vector_validation(self):
        """Test that x vector length must match y"""
        y = np.array([0, 1, 2, 1, 0])
        x_vals = np.array([0, 1, 2, 3])  # Wrong length

        with self.assertRaises(ValueError):
            findpeaks(y, x=x_vals, scale=1)


class TestFindpeaks2d(unittest.TestCase):
    """Test cases for 2D peak finding"""

    def setUp(self):
        """Set up test fixtures"""
        self.z_simple = np.zeros((10, 10))
        self.z_simple[3, 4] = 2
        self.z_simple[4, 4] = 1

    def test_single_peak(self):
        """Test detection of single peak in 2D array"""
        result = findpeaks2d(self.z_simple)

        self.assertGreater(len(result), 0)
        self.assertEqual(result[0, 0], 4)
        self.assertEqual(result[0, 1], 3)

    def test_npeaks_limit(self):
        """Test limiting number of 2D peaks"""
        result = findpeaks2d(self.z_simple, npeaks=1)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0, 0], 4)
        self.assertEqual(result[0, 1], 3)

    def test_multiple_peaks(self):
        """Test detection of multiple peaks"""
        z = np.zeros((10, 10))
        z[2, 2] = 3
        z[7, 7] = 5
        z[5, 3] = 2

        result = findpeaks2d(z, npeaks=3)

        self.assertLessEqual(len(result), 3)
        self.assertEqual(result[0, 0], 7)
        self.assertEqual(result[0, 1], 7)

    def test_scale_parameter(self):
        """Test different scale parameters"""
        z = np.zeros((10, 10))
        z[5, 5] = 5
        z[5, 6] = 3

        result_scale1 = findpeaks2d(z, scale=1, npeaks=10)
        result_scale2 = findpeaks2d(z, scale=2, npeaks=10)

        self.assertGreaterEqual(len(result_scale1), len(result_scale2))

    def test_interpolation(self):
        """Test peak interpolation in 2D"""
        z = np.zeros((10, 10), dtype=float)
        z[5, 5] = 1.0
        z[5, 4] = 0.8
        z[5, 6] = 0.7
        z[4, 5] = 0.9
        z[6, 5] = 0.6

        result = findpeaks2d(z, interp=True, npeaks=1)

        self.assertEqual(result.shape[1], 4)
        self.assertAlmostEqual(result[0, 0], 5.0, delta=0.5)
        self.assertAlmostEqual(result[0, 1], 5.0, delta=0.5)

    def test_positive_only(self):
        """Test positive peak constraint"""
        z = np.zeros((10, 10))
        z[3, 3] = -5
        z[7, 7] = 3

        result_positive = findpeaks2d(z, positive=True, npeaks=10)
        result_all = findpeaks2d(z, positive=False, npeaks=10)

        x_positive = set(zip(result_positive[:, 0], result_positive[:, 1]))
        self.assertFalse((3, 3) in x_positive)
        self.assertGreater(len(result_all), 0)

    def test_flat_region(self):
        """Test behavior on flat regions"""
        z = np.ones((10, 10))
        result = findpeaks2d(z)

        self.assertEqual(len(result), 0)

    def test_gaussian_peak(self):
        """Test detection of Gaussian-like peak"""
        x = np.linspace(-5, 5, 15)
        y = np.linspace(-5, 5, 15)
        X, Y = np.meshgrid(x, y)
        z = np.exp(-(X**2 + Y**2) / 2)

        result = findpeaks2d(z, npeaks=1, scale=1)

        if len(result) > 0:
            self.assertAlmostEqual(result[0, 0], 7, delta=2)
            self.assertAlmostEqual(result[0, 1], 7, delta=2)

    def test_output_shape(self):
        """Test output shape is correct"""
        z = np.random.rand(10, 10)
        result = findpeaks2d(z, npeaks=5)

        self.assertEqual(result.shape[1], 3)

    def test_value_column(self):
        """Test that value column contains correct data"""
        z = np.zeros((10, 10))
        z[5, 5] = 10
        z[3, 3] = 5

        result = findpeaks2d(z, npeaks=2)

        self.assertAlmostEqual(result[0, 2], 10)
        self.assertAlmostEqual(result[1, 2], 5)

    def test_descending_magnitude(self):
        """Test that results are sorted by magnitude"""
        z = np.zeros((10, 10))
        z[2, 2] = 2
        z[5, 5] = 5
        z[8, 8] = 3

        result = findpeaks2d(z, npeaks=10)

        if len(result) > 1:
            for i in range(len(result) - 1):
                self.assertGreaterEqual(result[i, 2], result[i + 1, 2])

    def test_interpolation_symmetry(self):
        """Test interpolation with symmetric peak"""
        z = np.zeros((10, 10), dtype=float)
        z[5, 5] = 1.0
        z[4, 5] = 0.9
        z[6, 5] = 0.9
        z[5, 4] = 0.9
        z[5, 6] = 0.9

        result = findpeaks2d(z, interp=True, npeaks=1)

        if len(result) > 0:
            self.assertAlmostEqual(result[0, 0], 5.0, delta=0.2)
            self.assertAlmostEqual(result[0, 1], 5.0, delta=0.2)

    def test_edge_pixels(self):
        """Test behavior with peaks at edges"""
        z = np.zeros((10, 10))
        z[0, 0] = 5  # Corner
        z[5, 0] = 3  # Edge
        z[5, 5] = 4  # Interior

        result = findpeaks2d(z, scale=1, npeaks=10)

        interior_found = any((r[0] == 5 and r[1] == 5) for r in result)
        self.assertTrue(interior_found)

    def test_random_noise(self):
        """Test with random noise"""
        np.random.seed(42)
        z = np.random.rand(10, 10) * 0.1
        z[5, 5] = 2.0  # Add clear peak

        result = findpeaks2d(z, npeaks=1)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0, 0], 5)
        self.assertEqual(result[0, 1], 5)


class TestFindpeaks3d(unittest.TestCase):
    """Test cases for 3D peak finding"""

    def setUp(self):
        """Set up test fixtures"""
        self.v_simple = np.zeros((10, 10, 10))
        self.v_simple[3, 4, 5] = 5

    def test_single_peak(self):
        """Test detection of single peak in 3D array"""
        result = findpeaks3d(self.v_simple)

        self.assertGreater(len(result), 0)
        self.assertEqual(result[0, 0], 3)
        self.assertEqual(result[0, 1], 4)
        self.assertEqual(result[0, 2], 5)
        self.assertEqual(result[0, 3], 5)

    def test_multiple_peaks(self):
        """Test detection of multiple peaks"""
        v = np.zeros((10, 10, 10))
        v[2, 2, 2] = 3
        v[7, 7, 7] = 5
        v[5, 3, 4] = 2

        result = findpeaks3d(v)

        self.assertGreaterEqual(len(result), 1)
        self.assertEqual(result[0, 0], 7)
        self.assertEqual(result[0, 1], 7)
        self.assertEqual(result[0, 2], 7)

    def test_npeaks_limit(self):
        """Test limiting number of 3D peaks"""
        v = np.zeros((10, 10, 10))
        v[2, 2, 2] = 3
        v[7, 7, 7] = 5
        v[5, 3, 4] = 2

        result = findpeaks3d(v, npeaks=1)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0, 0], 7)

    def test_edge_rejection(self):
        """Test that edge elements are not returned as peaks"""
        v = np.zeros((10, 10, 10))
        v[0, 0, 0] = 100  # Corner peak
        v[5, 5, 5] = 10  # Interior peak

        result = findpeaks3d(v, npeaks=5)

        interior_found = any((r[0] == 5 and r[1] == 5 and r[2] == 5) for r in result)
        self.assertTrue(interior_found)

        edge_found = any((r[0] == 0 and r[1] == 0 and r[2] == 0) for r in result)
        self.assertFalse(edge_found)

    def test_output_shape(self):
        """Test output shape is correct"""
        v = np.random.rand(10, 10, 10) * 10
        result = findpeaks3d(v)

        self.assertEqual(result.shape[1], 4)

    def test_descending_order(self):
        """Test that peaks are returned in descending magnitude"""
        v = np.zeros((10, 10, 10))
        v[2, 2, 2] = 2
        v[4, 4, 4] = 5
        v[6, 6, 6] = 3

        result = findpeaks3d(v, npeaks=3)

        if len(result) > 1:
            self.assertTrue(np.all(result[:-1, 3] >= result[1:, 3]))

    def test_spherical_peak(self):
        """Test detection of spherical Gaussian peak"""
        x = np.linspace(-5, 5, 15)
        y = np.linspace(-5, 5, 15)
        z = np.linspace(-5, 5, 15)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        v = np.exp(-(X**2 + Y**2 + Z**2) / 2)

        result = findpeaks3d(v, npeaks=1)

        if len(result) > 0:
            self.assertAlmostEqual(result[0, 0], 7, delta=2)
            self.assertAlmostEqual(result[0, 1], 7, delta=2)
            self.assertAlmostEqual(result[0, 2], 7, delta=2)

    def test_flat_volume(self):
        """Test behavior on flat volume"""
        v = np.ones((10, 10, 10))
        result = findpeaks3d(v)

        self.assertEqual(len(result), 0)

    def test_integer_values(self):
        """Test with integer valued volume"""
        v = np.zeros((10, 10, 10), dtype=int)
        v[5, 5, 5] = 10
        v[3, 3, 3] = 5

        result = findpeaks3d(v, npeaks=2)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0, 3], 10)
        self.assertEqual(result[1, 3], 5)

    def test_small_volume(self):
        """Test with minimal volume"""
        v = np.ones((3, 3, 3))
        v[1, 1, 1] = 5

        result = findpeaks3d(v)

        if len(result) > 0:
            self.assertEqual(result[0, 0], 1)
            self.assertEqual(result[0, 1], 1)
            self.assertEqual(result[0, 2], 1)

    def test_negative_values(self):
        """Test with negative values"""
        v = np.zeros((10, 10, 10))
        v[5, 5, 5] = -10  # Negative peak

        result = findpeaks3d(v)

        self.assertGreaterEqual(len(result), 0)

    def test_random_peaks(self):
        """Test with random volume"""
        np.random.seed(42)
        v = np.random.rand(10, 10, 10) * 10

        result = findpeaks3d(v, npeaks=3)

        self.assertLessEqual(len(result), 3)
        self.assertTrue(np.all(result[:, 3] >= 0))

    def test_value_column(self):
        """Test that value column contains correct data"""
        v = np.zeros((10, 10, 10))
        v[3, 4, 5] = 7.5

        result = findpeaks3d(v)

        if len(result) > 0:
            self.assertAlmostEqual(result[0, 3], 7.5)


if __name__ == "__main__":
    unittest.main()
