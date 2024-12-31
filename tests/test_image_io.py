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


class TestImage(unittest.TestCase):

    def test_iread(self):
        # see ioTest.m
        # test image:
        im = iread("wally.png")
        self.assertIsInstance(im[0], np.ndarray)
        self.assertIsInstance(im[1], str)
        self.assertEqual(im[0].shape, (25, 21))

        im = iread("monalisa.png")
        self.assertIsInstance(im[0], np.ndarray)
        self.assertIsInstance(im[1], str)
        self.assertEqual(im[0].shape, (700, 677, 3))

    def test_isimage(self):

        # create mini image (Bayer pattern)
        im = np.zeros((2, 2, 3))
        # 0 - red channel, 1 - green channel, 2 - blue channel
        im[0, 0, 0] = 1  # top left = red
        im[0, 1, 1] = 1  # top right = green
        im[1, 0, 1] = 1  # bottom left = green
        im[1, 1, 2] = 1  # bottom right = blue

        # a single grayscale image
        img = Image(im)
        self.assertIsInstance(img, Image)
        self.assertEqual(img.shape, im.shape)
        self.assertEqual(img.dtype, np.float64)

        # set type as float, then make sure isimage is true
        img = Image(im.astype(np.float32))
        self.assertIsInstance(img, Image)
        self.assertEqual(img.shape, im.shape)
        self.assertEqual(img.dtype, np.float32)

        img = Image(im.astype(np.uint8))
        self.assertIsInstance(img, Image)
        self.assertEqual(img.shape, im.shape)
        self.assertEqual(img.dtype, np.uint8)

    def test_str(self):
        # single color image as str
        # print('test_str')
        imname = "monalisa.png"

        im = Image.Read(imname)
        # check attributes
        nt.assert_array_equal(im.shape, (700, 677, 3))
        self.assertEqual(os.path.split(im.name)[1], imname)
        self.assertEqual(im.iscolor, True)
        self.assertEqual(im.dtype, "uint8")
        self.assertEqual(im.width, 677)
        self.assertEqual(im.height, 700)
        self.assertEqual(im.ndim, 3)
        self.assertEqual(im.colororder_str, "R:G:B")
        self.assertEqual(im.nplanes, 3)

    def test_filecollection(self):
        # single str with wild card for folder of images
        # print('test_wildcardstr')
        images = ImageCollection("campus/*.png")

        self.assertEqual(len(images), 20)
        self.assertIsInstance(images, Iterable)
        self.assertEqual(images[0], (426, 640, 3))
        self.assertEqual(images[0].dtype, "uint8")
        self.assertEqual(images[0].colororder_str, "R:G:B")
        self.assertEqual(images[0].nplanes, 3)

    def test_image(self):
        # Image object
        # print('test_image')
        imname = "shark1.png"
        im0 = Image.Read(imname)

        im1 = Image(im0)
        # TODO consider __eq__ to compare Image objects directly im0 == im1
        nt.assert_array_almost_equal(im1.A, im0.A)
        self.assertEqual(im1.shape, im0.shape)
        self.assertEqual(im1.iscolor, im0.iscolor)
        # ... for the rest of the attributes

    def test_array(self):
        # test single numpy array
        # print('test_numpyarray')
        imarray = iread("walls-l.png")

        im = Image(imarray[0])
        self.assertEqual(im.shape, (2448, 3264, 3))
        self.assertEqual(im.iscolor, True)

    def test_options(self):

        imname = "monalisa.png"
        im = Image.Read(imname)

        # check predicatives
        self.assertFalse(im.isfloat)
        self.assertTrue(im.isint)
        self.assertIsInstance(im, Image)
        self.assertEqual(im.bgr.shape, im.shape)
        self.assertEqual(im.rgb.shape, im.shape)
        self.assertEqual(im.size, (677, 700))

        # check one element for rgb vs bgr ordering
        v = round(im.shape[0] / 2)  # rows
        u = round(im.shape[1] / 2)  # cols
        bgr = im.bgr[v, u, :]
        nt.assert_array_equal(im.rgb[v, u, :], bgr[::-1])

        self.assertTrue(im.isrgb)
        self.assertFalse(im.isbgr)

        self.assertTrue(im.iscolor)

    def test_im_from_string(self):
        img = Image.String("01234|56789|87654")

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            img.print()
        self.assertEqual(f.getvalue(), " 0 1 2 3 4\n 5 6 7 8 9\n 8 7 6 5 4\n")

    def test_im_to_string(self):
        img = Image.String("01234|56789|87654")

        s = Image.strhcat(img)
        self.assertEqual(
            s,
            "\n      0 1 2 3 4  \n      - - - - -  \n  0:  0 1 2 3 4\n  1:  5 6 7 8 9\n  2:  8 7 6 5 4\n",
        )

        s = Image.strhcat(img, widths=3)
        self.assertEqual(
            s,
            "\n        0   1   2   3   4  \n        -   -   -   -   -  \n  0:    0   1   2   3   4\n  1:    5   6   7   8   9\n  2:    8   7   6   5   4\n",
        )

        s = Image.strhcat(img, img)
        self.assertEqual(
            s,
            "\n      0 1 2 3 4   0 1 2 3 4  \n      - - - - -   - - - - -  \n  0:  0 1 2 3 4 0 1 2 3 4\n  1:  5 6 7 8 9 5 6 7 8 9\n  2:  8 7 6 5 4 8 7 6 5 4\n",
        )

        s = Image.strhcat(img, img, labels=["a", "b"])
        self.assertEqual(
            s,
            "     a           b\n      0 1 2 3 4   0 1 2 3 4  \n      - - - - -   - - - - -  \n  0:  0 1 2 3 4 0 1 2 3 4\n  1:  5 6 7 8 9 5 6 7 8 9\n  2:  8 7 6 5 4 8 7 6 5 4\n",
        )

    # TODO unit tests:
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
