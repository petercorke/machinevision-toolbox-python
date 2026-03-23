#!/usr/bin/env python

import unittest

import numpy as np
import numpy.testing as nt

from machinevisiontoolbox import Image


class TestImageRegionFeatures(unittest.TestCase):

    def test_mser_basic(self):
        """Test MSER feature detection"""
        im = Image.Read("castle.png")
        msers = im.MSER()

        # Should detect some features
        self.assertGreater(len(msers), 0)

        # Check that we can access bbox attribute
        bbox = msers.bbox
        self.assertIsNotNone(bbox)

    def test_mser_indexing(self):
        """Test MSER list-like behavior"""
        im = Image.Read("shark2.png")
        msers = im.MSER()

        # Test length
        n = len(msers)
        self.assertGreater(n, 0)

        # Test indexing
        if n > 0:
            mser_first = msers[0]
            self.assertIsNotNone(mser_first)

        # Test slicing
        if n > 5:
            mser_slice = msers[:5]
            self.assertIsNotNone(mser_slice)

    def test_mser_parameters(self):
        """Test MSER with custom parameters"""
        im = Image.Read("flowers1.png")

        # Test with different delta values
        msers1 = im.MSER(delta=3)
        msers2 = im.MSER(delta=10)

        # Different parameters should potentially detect different number of regions
        self.assertIsInstance(msers1, type(msers2))


if __name__ == "__main__":
    unittest.main()
