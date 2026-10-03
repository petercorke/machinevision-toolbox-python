#!/usr/bin/env python

import os
import tempfile
import unittest

import numpy as np
import numpy.testing as nt

from machinevisiontoolbox import Image


class TestImageCoreOperations(unittest.TestCase):

    def test_copy(self):
        """Test image copying"""
        img = Image.Read("monalisa.png")
        img_copy = img.copy()
        self.assertEqual(img_copy.shape, img.shape)
        nt.assert_array_equal(img_copy.array, img.array)

    def test_write_read_roundtrip(self):
        """Test writing and reading back an image (PNG is lossless)"""
        img = Image.Read("monalisa.png", dtype="uint8")

        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "roundtrip.png")
            img.write(fname)
            img_read = Image.Read(fname)

        self.assertEqual(img_read.shape, img.shape)
        nt.assert_array_equal(img_read.array, img.array)

    def test_colorspace_conversion_roundtrip(self):
        """Test colorspace conversions"""
        img = Image.Read("flowers1.png")

        # RGB to HSV
        hsv = img.colorspace("hsv", src="rgb")
        self.assertEqual(hsv.nplanes, 3)
        self.assertEqual(hsv.shape[:2], img.shape[:2])

        # RGB to lab
        lab = img.colorspace("lab")
        self.assertEqual(lab.nplanes, 3)
        self.assertEqual(lab.shape[:2], img.shape[:2])

    def test_cast_operations(self):
        """Test image type casting"""
        # float to uint8
        img_float = Image(np.random.rand(10, 10))
        img_uint8 = img_float.array_as("uint8")
        self.assertEqual(img_uint8.dtype, np.uint8)

        # uint8 to float
        img_uint8 = Image(np.random.rand(10, 10) * 255, dtype="uint8")
        img_float2 = img_uint8.array_as("float32")
        self.assertEqual(img_float2.dtype, np.float32)

    def test_matrix_conversion(self):
        """Test matrix/array conversion"""
        img = Image(np.random.rand(10, 10, 3))

        # To array
        arr = img.array
        self.assertEqual(arr.shape, img.shape)

    def test_hstack(self):
        """Test horizontal concatenation"""
        img1 = Image(np.random.rand(10, 10))
        img2 = Image(np.random.rand(10, 10))

        stacked = Image.Hstack([img1, img2], sep=0)
        self.assertEqual(stacked.shape, (10, 20))
        nt.assert_array_equal(stacked.array[:, :10], img1.array)
        nt.assert_array_equal(stacked.array[:, 10:], img2.array)

        # a separator adds columns between the images
        stacked = Image.Hstack([img1, img2], sep=2)
        self.assertEqual(stacked.shape, (10, 22))

    def test_vstack(self):
        """Test vertical concatenation"""
        img1 = Image(np.random.rand(10, 10))
        img2 = Image(np.random.rand(10, 10))

        stacked = Image.Vstack([img1, img2], sep=0)
        self.assertEqual(stacked.shape, (20, 10))
        nt.assert_array_equal(stacked.array[:10, :], img1.array)
        nt.assert_array_equal(stacked.array[10:, :], img2.array)

        # a separator adds rows between the images
        stacked = Image.Vstack([img1, img2], sep=2)
        self.assertEqual(stacked.shape, (22, 10))

    def test_interp2d(self):
        """Test 2D interpolation: sampling at integer pixel coordinates must
        reproduce the image values"""
        img = Image.Read("monalisa.png", mono=True, dtype="float32")

        # U, V are (Ho, Wo) coordinate arrays for the output image
        U, V = np.meshgrid(np.arange(100, 110), np.arange(150, 160))
        interp = img.interp2d(U, V)

        self.assertEqual(interp.shape, (10, 10))
        nt.assert_allclose(interp.array, img.array[150:160, 100:110], rtol=1e-5)

    def test_get_pixel(self):
        """Test getting pixel values; the arguments are (u, v), i.e. (column,
        row), the opposite order to NumPy indexing"""
        img = Image(np.random.rand(10, 12))
        self.assertEqual(img.pixel(5, 3), img.array[3, 5])

        # color image: result is a vector over planes
        img3 = Image(np.random.rand(10, 12, 3))
        nt.assert_array_equal(img3.pixel(5, 3), img3.array[3, 5, :])


class TestImagePointFeatures(unittest.TestCase):

    def test_sift(self):
        """Test SIFT feature detection"""
        img = Image.Read("monalisa.png", mono=True)

        sift = img.SIFT()
        self.assertGreater(len(sift), 0)

    @unittest.skip(
        "Image.SURF() is not implemented, and SURF is non-free in pip OpenCV "
        "builds; see https://github.com/petercorke/machinevision-toolbox-python/issues/113"
    )
    def test_surf(self):
        """Test SURF feature detection"""
        img = Image.Read("flowers1.png", mono=True)

        surf = img.SURF()
        self.assertGreater(len(surf), 0)

    def test_orb(self):
        """Test ORB feature detection"""
        img = Image.Read("monalisa.png", mono=True)

        orb = img.ORB()
        self.assertGreater(len(orb), 0)

    def test_match_orb_auto_metric(self):
        """match() must auto-select hamming distance for binary (ORB)
        descriptors -- regression test for the L2 default giving
        near-meaningless distances for binary descriptors, which silently
        yields far fewer matches than the correct metric rather than
        raising an error"""
        img1 = Image.Read("eiffel-1.png", mono=True)
        img2 = Image.Read("eiffel-2.png", mono=True)
        orb1 = img1.ORB(nfeatures=200)
        orb2 = img2.ORB(nfeatures=200)

        m_auto = orb1.match(orb2)
        m_hamming = orb1.match(orb2, metric="hamming")
        m_l2 = orb1.match(orb2, metric="L2")

        # auto-detected metric must agree with explicit hamming, not L2
        self.assertEqual(len(m_auto), len(m_hamming))
        self.assertGreater(len(m_auto), len(m_l2))

        # float descriptors (SIFT) must still default to L2
        sift1 = img1.SIFT()
        sift2 = img2.SIFT()
        m_sift_auto = sift1.match(sift2)
        m_sift_l2 = sift1.match(sift2, metric="L2")
        self.assertEqual(len(m_sift_auto), len(m_sift_l2))

    def test_harris(self):
        """Test Harris corner detection"""
        img = Image.Read("monalisa.png", mono=True)

        corners = img.Harris()
        self.assertGreater(len(corners), 0)

    def test_draw2(self):
        """draw2() with a named color and a colorized (colororder-bearing)
        image must not raise -- regression test for name2color() leaking a
        plain list instead of an ndarray when colororder is given"""
        img = Image.Read("monalisa.png", mono=True)
        orb = img.ORB(nfeatures=20)
        self.assertGreater(len(orb), 0)

        color_img = img.colorize()
        result = orb.draw2(color_img, color="y")
        self.assertIsInstance(result, Image)

    def test_features_list_operations(self):
        """Test feature list operations"""
        img = Image.Read("monalisa.png", mono=True)
        sift = img.SIFT()
        self.assertGreater(len(sift), 5)

        # Test slicing
        slice_sift = sift[:5]
        self.assertEqual(len(slice_sift), 5)

        # Test indexing
        first_feature = sift[0]
        self.assertEqual(len(first_feature), 1)

    def test_gridify_scalar_nbins(self):
        """gridify() with a scalar nbins must not raise (regression: numpy
        float // int stayed float64, bins[iy, ix] then rejected as an index)"""
        img = Image.Read("monalisa.png", mono=True)
        sift = img.SIFT()
        self.assertGreater(len(sift), 0)

        gridded = sift.gridify(nbins=4, nfeat=2)
        self.assertLessEqual(len(gridded), len(sift))

    def test_gridify_tuple_nbins(self):
        """gridify() with a (nw, nh) tuple must not raise, same root cause
        as the scalar case above."""
        img = Image.Read("monalisa.png", mono=True)
        sift = img.SIFT()
        self.assertGreater(len(sift), 0)

        gridded = sift.gridify(nbins=(4, 3), nfeat=2)
        self.assertLessEqual(len(gridded), len(sift))

    def test_brisk(self):
        """BRISK feature detection (regression: OpenCV 5 moved BRISK_create
        from cv2 to cv2.xfeatures2d -- no bare except here, a broken
        detector must fail this test, not pass silently)"""
        img = Image.Read("monalisa.png", mono=True)
        brisk = img.BRISK()
        self.assertGreater(len(brisk), 0)

    def test_akaze(self):
        """AKAZE feature detection (regression: OpenCV 5 moved AKAZE_create
        from cv2 to cv2.xfeatures2d)"""
        img = Image.Read("monalisa.png", mono=True)
        akaze = img.AKAZE()
        self.assertGreater(len(akaze), 0)

    def test_feature_properties(self):
        """Test feature properties"""
        img = Image.Read("monalisa.png", mono=True)
        sift = img.SIFT()
        n = len(sift)
        self.assertGreater(n, 0)

        # coordinates: u, v are per-feature lists, p is a 2xN array
        self.assertEqual(len(sift.u), n)
        self.assertEqual(len(sift.v), n)
        self.assertEqual(sift.p.shape, (2, n))
        nt.assert_array_equal(sift.p[0, :], sift.u)
        nt.assert_array_equal(sift.p[1, :], sift.v)

        # strength
        self.assertEqual(len(sift.strength), n)


if __name__ == "__main__":
    unittest.main()
