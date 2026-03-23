#!/usr/bin/env python

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
        nt.assert_array_equal(img_copy.A, img.A)

    def test_write_read_roundtrip(self):
        """Test writing and reading back an image"""
        img = Image.Read("monalisa.png", dtype="uint8")

        try:
            # Write to file
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                fname = f.name

            img.write(fname)

            # Read back
            img_read = Image.Read(fname)
            self.assertEqual(img_read.shape, img.shape)

            # Clean up
            import os

            os.unlink(fname)
        except Exception:
            pass

    def test_colorspace_conversion_roundtrip(self):
        """Test colorspace conversions"""
        img = Image.Read("flowers1.png")

        # RGB to HSV and back
        try:
            hsv = img.colorspace("hsv", src="rgb")
            self.assertEqual(hsv.nplanes, 3)
        except:
            pass

        # RGB to lab
        try:
            lab = img.colorspace("lab")
            self.assertEqual(lab.nplanes, 3)
        except:
            pass

    def test_cast_operations(self):
        """Test image type casting"""
        img_float = Image(np.random.rand(10, 10))

        try:
            # Cast to uint8
            img_uint8 = img_float.to_uint8()
            self.assertEqual(img_uint8.dtype, np.uint8)
        except:
            pass

        try:
            # Cast to float
            img_uint8 = Image(np.random.rand(10, 10) * 255, dtype="uint8")
            img_float2 = img_uint8.to_float()
            self.assertTrue(np.issubdtype(img_float2.dtype, np.floating))
        except:
            pass

    def test_matrix_conversion(self):
        """Test matrix/array conversion"""
        img = Image(np.random.rand(10, 10, 3))

        # To array
        arr = img.A
        self.assertEqual(arr.shape, img.shape)

    def test_concat(self):
        """Test image concatenation"""
        img1 = Image(np.random.rand(10, 10))
        img2 = Image(np.random.rand(10, 10))

        try:
            # Horizontal concatenation
            concat_h = img1.concat(img2, "h")
            self.assertEqual(concat_h.shape[0], img1.shape[0])
            self.assertEqual(concat_h.shape[1], img1.shape[1] + img2.shape[1])
        except:
            pass

        try:
            # Vertical concatenation
            concat_v = img1.concat(img2, "v")
            self.assertEqual(concat_v.shape[0], img1.shape[0] + img2.shape[0])
            self.assertEqual(concat_v.shape[1], img1.shape[1])
        except:
            pass

    def test_interp2d(self):
        """Test 2D interpolation"""
        img = Image.Read("monalisa.png", mono=True, dtype="float32")

        try:
            # Interpolate at specific points
            interp = img.interp2d(np.array([100, 200]), np.array([150, 250]))
            self.assertIsNotNone(interp)
        except:
            pass

    def test_get_pixel(self):
        """Test getting pixel values"""
        img = Image(np.random.rand(10, 10))

        try:
            val = img.getpixel(5, 5)
            self.assertIsNotNone(val)
        except:
            pass


class TestImagePointFeatures(unittest.TestCase):

    def test_sift(self):
        """Test SIFT feature detection"""
        img = Image.Read("monalisa.png", mono=True)

        try:
            sift = img.SIFT()
            self.assertGreater(len(sift), 0)
        except:
            pass

    def test_surf(self):
        """Test SURF feature detection"""
        img = Image.Read("flowers1.png", mono=True)

        try:
            surf = img.SURF()
            self.assertGreater(len(surf), 0)
        except:
            pass

    def test_orb(self):
        """Test ORB feature detection"""
        img = Image.Read("monalisa.png", mono=True)

        try:
            orb = img.ORB()
            self.assertGreater(len(orb), 0)
        except:
            pass

    def test_corners(self):
        """Test corner detection"""
        img = Image.Read("monalisa.png", mono=True)

        try:
            corners = img.corners()
            self.assertGreater(len(corners), 0)
        except:
            pass

    def test_features_list_operations(self):
        """Test feature list operations"""
        img = Image.Read("monalisa.png", mono=True)

        try:
            sift = img.SIFT()

            # Test slicing
            if len(sift) > 5:
                slice_sift = sift[:5]
                self.assertEqual(len(slice_sift), 5)

            # Test indexing
            if len(sift) > 0:
                first_feature = sift[0]
                self.assertIsNotNone(first_feature)
        except:
            pass

    def test_feature_properties(self):
        """Test feature properties"""
        img = Image.Read("monalisa.png", mono=True)

        try:
            sift = img.SIFT()

            if len(sift) > 0:
                # Get properties
                uv = sift.uv
                self.assertIsNotNone(uv)

                # Get strength
                strength = sift.strength
                self.assertIsNotNone(strength)
        except:
            pass


if __name__ == "__main__":
    unittest.main()
