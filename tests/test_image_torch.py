#!/usr/bin/env python

import unittest

import numpy as np

from machinevisiontoolbox import Image

try:
    import torch

    _torch_available = True
except ImportError:
    _torch_available = False


@unittest.skipUnless(_torch_available, "PyTorch not installed")
class TestImageTorch(unittest.TestCase):

    def test_to_tensor_shape_mono(self):
        """Mono image produces tensor of shape (1, H, W)"""
        img = Image.Zeros(64, 64, dtype="uint8")
        t = img.to_tensor()
        self.assertEqual(t.shape, (1, 64, 64))

    def test_to_tensor_shape_color(self):
        """Color image produces tensor of shape (C, H, W)"""
        img = Image.Zeros(64, 64, colororder="RGB", dtype="uint8")
        t = img.to_tensor()
        self.assertEqual(t.shape, (3, 64, 64))

    def test_from_tensor_roundtrip(self):
        """Round-trip: Image → tensor → Image preserves pixel values"""
        img = Image.Read("monalisa.png")
        t = img.to_tensor()
        img2 = Image.from_tensor(t, colororder=img.colororder_str)
        np.testing.assert_array_equal(img2.A, img.A)


if __name__ == "__main__":
    unittest.main()
