#!/usr/bin/env python

import numpy as np
import numpy.testing as nt
import unittest

from machinevisiontoolbox.Image import Image
# from pathlib import Path


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

    def test_testpattern(self):
        im = Image()
        tp = im.testpattern('rampx', 10, 2)
        self.assertEqual(tp.shape, (10, 10))

        tp = im.testpattern('rampx', (20, 10), 2)
        self.assertEqual(tp.shape, (10, 20))
        r = np.linspace(0, 1, 10, endpoint=True)
        out = np.hstack((r, r))
        nt.assert_array_almost_equal(tp.image[5, :], out)

        tp = im.testpattern('rampy', (10, 20), 2)
        self.assertEqual(tp.shape, (20, 10))
        nt.assert_array_almost_equal(tp.image[:, 5], out.T)

        tp = im.testpattern('sinx', 12, 1)
        self.assertEqual(tp.shape, (12, 12))
        nt.assert_almost_equal(np.sum(tp.image), 0, decimal=6)
        nt.assert_almost_equal(np.diff(tp.image[:, 2]),
                               np.zeros((11)),
                               decimal=6)

        tp = im.testpattern('siny', 12, 1)
        self.assertEqual(tp.shape, (12, 12))
        nt.assert_almost_equal(np.sum(tp.image), 0, decimal=6)
        nt.assert_almost_equal(np.diff(tp.image[2, :]),
                               np.zeros((11)),
                               decimal=6)

        tp = im.testpattern('dots', 100, 20, 10)
        self.assertEqual(tp.shape, (100, 100))
        # TODO [l,ml,p,c] = ilabel(im);
        # tc.verifyEqual(sum(c), 25);

        tp = im.testpattern('squares', 100, 20, 10)
        self.assertEqual(tp.shape, (100, 100))
        # TODO [l,ml,p,c] = ilabel(im);
        # tc.verifyEqual(sum(c), 25);

        # TODO not yet converted to python:
        tp = im.testpattern('line', 20, np.pi / 6, 10)
        self.assertEqual(tp.shape, (20, 20))
        self.assertEqual(tp.image[10, 0], 1)
        self.assertEqual(tp.image[11, 1], 1)
        self.assertEqual(tp.image[16, 11], 1)
        self.assertEqual(np.sum(tp.image), 17)

    def test_paste(self):

        im = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
        im = Image(im)
        canvas = np.zeros((5, 5))
        canvas = Image(canvas)

        out = np.array([[0, 0, 0, 0, 0],
                        [0, 0, 1, 2, 3],
                        [0, 0, 4, 5, 6],
                        [0, 0, 7, 8, 9],
                        [0, 0, 0, 0, 0]])
        cp = canvas.paste(im, (2, 1))
        nt.assert_array_almost_equal(cp.image, out)

        canvas = np.zeros((5, 5))
        canvas = Image(canvas)
        cp = canvas.paste(im, (3, 2), centre=True)
        nt.assert_array_almost_equal(cp.image, out)

        canvas = np.zeros((5, 5))
        canvas = Image(canvas)
        cp = canvas.paste(im, (2, 1), opt='set')
        nt.assert_array_almost_equal(cp.image, out)

        canvas = np.zeros((5, 5))
        canvas = Image(canvas)
        cp = canvas.paste(im, (2, 1), opt='mean')
        nt.assert_array_almost_equal(cp.image, out / 2)

        canvas = np.zeros((5, 5))
        canvas = Image(canvas)
        cp = canvas.paste(im, (2, 1), opt='add')
        cp2 = cp.paste(im, (2, 1), opt='add')
        nt.assert_array_almost_equal(cp2.image, out * 2)

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
    # test_testpattern (half done)
    # test_scale
    # test_rotate
    # test_samesize
    # test_peak2
    # test_roi
    # test_pixelswitch


# ----------------------------------------------------------------------- #
if __name__ == '__main__':

    unittest.main()

    # import code
    # code.interact(local=dict(globals(), **locals()))
