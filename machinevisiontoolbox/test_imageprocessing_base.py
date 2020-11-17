#!/usr/bin/env python

import numpy as np
import numpy.testing as nt
import unittest

from machinevisiontoolbox.Image import Image
from pathlib import Path


class TestImageProcessingBase(unittest.TestCase):

    def test_int(self):

        # test for uint8
        im = np.zeros((2, 2), np.float)
        im = Image(im)
        nt.assert_array_almost_equal(im.int().image,
                                     np.zeros((2, 2), np.uint8))

        im = np.ones((2, 2), np.float)
        im = Image(im)
        nt.assert_array_almost_equal(im.int().image,
                                     255 * np.ones((2, 2)).astype(np.uint8))

        # tests for shape
        im = np.random.randint(1, 255, (3, 5), int)
        im = Image(im)
        imi = im.int()
        nt.assert_array_almost_equal(imi.shape, im.shape)

        im = np.random.randint(1, 255, (3, 5, 3), int)
        im = Image(im)
        imi = im.int()
        nt.assert_array_almost_equal(imi.shape, im.shape)

        im = np.random.randint(1, 255, (3, 5, 3, 10), int)
        im = Image(im)
        imi = im.int()
        nt.assert_array_almost_equal(imi.shape, im.shape)

    def test_float(self):
        # test for uint8
        im = np.zeros((2, 2), np.uint8)
        im = Image(im)
        nt.assert_array_almost_equal(im.float().image,
                                     np.zeros((2, 2), np.float32))

        im = 128 * np.ones((2, 2), np.uint8)
        im = Image(im)
        nt.assert_array_almost_equal(im.float().image,
                                     (128.0/255.0 * np.ones((2, 2))))

        im = 255 * np.ones((2, 2), np.uint8)
        im = Image(im)
        nt.assert_array_almost_equal(im.float().image,
                                     (np.ones((2, 2))))

        # test for uint16
        im = np.zeros((2, 2), np.uint16)
        im = Image(im)
        nt.assert_array_almost_equal(im.float().image,
                                     np.zeros((2, 2), np.float32))

        im = 128 * np.ones((2, 2), np.uint16)
        im = Image(im)
        nt.assert_array_almost_equal(im.float().image,
                                     (128.0/65535.0 * np.ones((2, 2))))

        im = 65535 * np.ones((2, 2), np.uint16)
        im = Image(im)
        nt.assert_array_almost_equal(im.float().image,
                                     (np.ones((2, 2))))

        # test for sequence of images
        im = np.random.randint(low=1,
                               high=255,
                               size=(5, 8, 3, 4),
                               dtype=np.uint8)
        im = Image(im)
        imf = im.float()
        nt.assert_array_almost_equal(imf.shape, im.shape)
        nt.assert_array_almost_equal(imf.image,
                                     im.image.astype(np.float32) / 255.0)

        im = np.random.randint(low=1,
                               high=65535,
                               size=(3, 10, 3, 7),
                               dtype=np.uint16)
        im = Image(im)
        imf = im.float()
        nt.assert_array_almost_equal(imf.shape, im.shape)
        nt.assert_array_almost_equal(imf.image,
                                     im.image.astype(np.float32) / 65535.0)

    def test_mono(self):
        # input an image that is not mono
        imname = 'monalisa.png'
        im = Image(imname)
        imm = im.mono()
        self.assertEqual(imm.iscolor, False)
        self.assertEqual(imm.shape, im.size)

    # TODO
    # test_stretch
    # test_thresh
    # test_otsu
    # test_imeshgrid
    # test_hist
    # test_plothist
    # test_normhist
    # test_replicate
    # test_decimate
    # test_testpattern
    # test_scale
    # test_rotate
    # test_samesize
    # test_paste
    # test_peak2
    # test_roi
    # test_pixelswitch


# ----------------------------------------------------------------------- #
if __name__ == '__main__':

    unittest.main()
