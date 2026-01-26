#!/usr/bin/env python

# test for Image input/output

import numpy as np
import os
import numpy.testing as nt
import unittest
import contextlib
import io

# import machinevisiontoolbox as mvt
from machinevisiontoolbox import Image, ImageCollection
from machinevisiontoolbox.base import iread
from pathlib import Path
from collections.abc import Iterable


class TestBaseImageIO(unittest.TestCase):

    def test_iread(self):
        # greyscale image
        im = iread("wally.png")
        self.assertIsInstance(im[0], np.ndarray)
        self.assertIsInstance(im[1], str)
        self.assertEqual(im[0].shape, (25, 21))

        # color image
        im = iread("monalisa.png")
        self.assertIsInstance(im[0], np.ndarray)
        self.assertIsInstance(im[1], str)
        self.assertEqual(im[0].shape, (700, 677, 3))

        # greyscale image sequence
        im = iread("seq/im*.png")
        self.assertIsInstance(im[0], list)
        self.assertEqual(len(im[0]), 9)
        self.assertIsInstance(im[0][0], np.ndarray)
        self.assertEqual(im[0][0].shape, (512, 512), self.assertIsInstance(im[1], list))
        self.assertEqual(len(im[1]), 9)
        self.assertIsInstance(im[1][0], str)

        # color image sequence
        im = iread("campus/holdout/*.png")
        self.assertIsInstance(im[0], list)
        self.assertEqual(len(im[0]), 5)
        self.assertIsInstance(im[0][0], np.ndarray)
        self.assertEqual(im[0][0].shape, (426, 640, 3))
        self.assertIsInstance(im[1], list)
        self.assertEqual(len(im[1]), 5)
        self.assertIsInstance(im[1][0], str)

        # URL image
        im = iread("https://petercorke.com/files/images/monalisa.png")
        self.assertIsInstance(im[0], np.ndarray)
        self.assertIsInstance(im[1], str)
        self.assertEqual(im[0].shape, (700, 677, 3))

    # TODO unit tests:
    # test the various options to iread()
    # test writing
    # test_isimage - make sure Image rejects/fails with invalid input
    # test_imtypes - test Image works on different Image types?
    # test_getimage - make sure Image returns the same array but with valid
    # typing?
    # test_imwrite - test write/successfully save file?

    def tearDown(self):
        # Cleanup code if needed
        pass


# ------------------------------------------------------------------------ #
if __name__ == "__main__":

    unittest.main()
