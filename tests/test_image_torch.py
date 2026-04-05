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
        t = img.tensor()
        self.assertEqual(t.shape, (1, 64, 64))

    def test_to_tensor_shape_color(self):
        """Color image produces tensor of shape (C, H, W)"""
        img = Image.Zeros(64, 64, colororder="RGB", dtype="uint8")
        t = img.tensor()
        self.assertEqual(t.shape, (1, 3, 64, 64))

    def test_from_tensor_roundtrip(self):
        """Round-trip: Image → tensor → Image preserves pixel values"""
        img = Image.Read("monalisa.png")
        t = img.tensor()
        img2 = Image.Tensor(t, colororder=img.colororder_str)
        np.testing.assert_array_equal(img2.A, img.A)

    def test_to_tensor_dtype_option(self):
        """tensor(dtype=...) converts to requested tensor dtype."""
        img = Image.Zeros(32, 32, dtype="uint8")
        t = img.tensor(dtype=torch.float32)
        self.assertEqual(t.dtype, torch.float32)

    def test_to_tensor_dtype_with_normalize(self):
        """dtype option is applied when normalisation is requested."""
        img = Image.Zeros(32, 32, colororder="RGB", dtype="uint8")
        t = img.tensor(normalize="imagenet", dtype=torch.float64)
        self.assertEqual(t.dtype, torch.float64)

    def test_from_tensor_logits_option(self):
        """Tensor(logits=True) applies argmax over class axis."""
        x = torch.tensor(
            [[[1.0, 2.0], [3.0, 1.0]], [[0.0, 5.0], [1.0, 2.0]]],
            dtype=torch.float32,
        )
        img = Image.Tensor(x, logits=True)
        np.testing.assert_array_equal(img.A, np.array([[0, 1], [0, 1]], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
