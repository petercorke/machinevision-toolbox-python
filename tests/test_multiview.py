#!/usr/bin/env python

import unittest

import numpy as np
import numpy.testing as nt

from machinevisiontoolbox import Image


class TestImageMultiview(unittest.TestCase):
    # new test
    def test_DSI_refine(self):
        """Test DSI refinement class method"""
        # TODO: Create minimal DSI for testing
        pass

    # new test
    def test_rectify_homographies(self):
        """Test stereo rectification using homographies"""
        # TODO: Create stereo pair and test rectification
        pass

    # new test
    def test_stereo_BM(self):
        """Test Block Matching stereo algorithm"""
        im_left = Image.Random(size=(100, 100), dtype="uint8")
        im_right = Image.Random(size=(100, 100), dtype="uint8")
        # TODO: Test with actual stereo pair
        pass

    # new test
    def test_stereo_SGBM(self):
        """Test Semi-Global Block Matching stereo algorithm"""
        im_left = Image.Random(size=(100, 100), dtype="uint8")
        im_right = Image.Random(size=(100, 100), dtype="uint8")
        # TODO: Test with actual stereo pair
        pass

    # new test
    def test_stereo_simple(self):
        """Test simple stereo matching"""
        im_left = Image.Random(size=(100, 100), dtype="uint8")
        im_right = Image.Random(size=(100, 100), dtype="uint8")
        # TODO: Test with actual stereo pair
        pass


# ----------------------------------------------------------------------- #
if __name__ == "__main__":
    unittest.main()
