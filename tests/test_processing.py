#!/usr/bin/env python

import numpy as np
import numpy.testing as nt
import unittest

from machinevisiontoolbox import Image
# from pathlib import Path

class TestImageProcessingBase(unittest.TestCase):

    def test_read(self):

        im = Image.Read('penguins.png')
        self.assertEqual(im.size, (1047, 730))
        self.assertFalse(im.iscolor)
        self.assertEqual(im.dtype, np.uint8)

        im = Image.Read('penguins.png', dtype='float32')
        self.assertEqual(im.size, (1047, 730))
        self.assertFalse(im.iscolor)
        self.assertEqual(im.dtype, np.float32)

        im = Image.Read('monalisa.png')
        self.assertEqual(im.size, (677, 700))
        self.assertTrue(im.iscolor)
        self.assertEqual(im.dtype, np.uint8)
        self.assertEqual(im.colororder_str, "R:G:B")

        im = Image.Read('monalisa.png', mono=True)
        self.assertEqual(im.size, (677, 700))
        self.assertFalse(im.iscolor)
        self.assertEqual(im.dtype, np.uint8)


        im = Image.Read('monalisa.png', dtype='float32')
        self.assertEqual(im.size, (677, 700))
        self.assertTrue(im.iscolor)
        self.assertEqual(im.dtype, np.float32)
        self.assertEqual(im.colororder_str, "R:G:B")

        im = Image.Read('monalisa.png', dtype='float32', mono=True)
        self.assertEqual(im.size, (677, 700))
        self.assertFalse(im.iscolor)
        self.assertEqual(im.dtype, np.float32)

        im = Image.Read('monalisa.png', dtype='float32', mono=True)
        self.assertEqual(im.size, (677, 700))
        self.assertFalse(im.iscolor)
        self.assertEqual(im.dtype, np.float32)

        im = Image.Read('monalisa.png', reduce=2)
        self.assertEqual(im.size, (339, 350))
        self.assertTrue(im.iscolor)
        self.assertEqual(im.dtype, np.uint8)
        self.assertEqual(im.colororder_str, "R:G:B")

    def test_int(self):

        # test for uint8
        im = np.zeros((2, 2), np.float32)
        im = Image(im)
        nt.assert_array_almost_equal(im.to_int(),
                                     np.zeros((2, 2), np.uint8))

        im = np.ones((2, 2), np.float32)
        im = Image(im)
        nt.assert_array_almost_equal(im.to_int(),
                                     255 * np.ones((2, 2)).astype(np.uint8))

        # tests for shape
        im = np.random.randint(1, 255, (3, 5), int)
        im = Image(im)
        imi = im.to_int()
        nt.assert_array_almost_equal(imi.shape, im.shape)

        im = np.random.randint(1, 255, (3, 5, 3), int)
        im = Image(im)
        imi = im.to_int()
        nt.assert_array_almost_equal(imi.shape, im.shape)

        im = np.random.randint(1, 255, (3, 5, 3, 10), int)
        im = Image(im)
        imi = im.to_int()
        nt.assert_array_almost_equal(imi.shape, im.shape)

    def test_float(self):
        # test for uint8
        im = np.zeros((2, 2), np.uint8)
        im = Image(im)
        nt.assert_array_almost_equal(im.to_float(),
                                     np.zeros((2, 2), np.float32))

        im = 128 * np.ones((2, 2), np.uint8)
        im = Image(im)
        nt.assert_array_almost_equal(im.to_float(),
                                     (128.0/255.0 * np.ones((2, 2))))

        im = 255 * np.ones((2, 2), np.uint8)
        im = Image(im)
        nt.assert_array_almost_equal(im.to_float(),
                                     (np.ones((2, 2))))

        # test for uint16
        im = np.zeros((2, 2), np.uint16)
        im = Image(im)
        nt.assert_array_almost_equal(im.to_float(),
                                     np.zeros((2, 2), np.float32))

        im = 128 * np.ones((2, 2), np.uint16)
        im = Image(im)
        nt.assert_array_almost_equal(im.to_float(),
                                     (128.0/65535.0 * np.ones((2, 2))))

        im = 65535 * np.ones((2, 2), np.uint16)
        im = Image(im)
        nt.assert_array_almost_equal(im.to_float(),
                                     (np.ones((2, 2))))

        im = Image(im)
        imf = im.to_float()
        nt.assert_array_almost_equal(imf.shape, im.shape)
        nt.assert_array_almost_equal(imf,
                                     im.image.astype(np.float32) / 65535.0)

    def test_testpattern(self):
        tp = Image.Ramp(dir='x', size=20, cycles=2)
        self.assertEqual(tp.shape, (20, 20))

        r = np.linspace(0, 1, 10, endpoint=True)
        out = np.hstack((r, r))
        nt.assert_array_almost_equal(tp.A[5, :], out)

        tp = Image.Ramp(dir='y', size=20, cycles=2)
        self.assertEqual(tp.shape, (20, 20))
        nt.assert_array_almost_equal(tp.A[:, 5], out.T)

        tp = Image.Sin(dir='x', size=12, cycles=1)
        self.assertEqual(tp.shape, (12, 12))
        x = tp.A[2,:] # take a line
        self.assertTrue(x[5] > x[0])
        self.assertEqual(x[0], x[6])
        self.assertEqual(x[1] - x[0], x[6] - x[7])

        tp = Image.Sin(dir='y', size=12, cycles=1)
        self.assertEqual(tp.shape, (12, 12))
        x = tp.A[:,2] # take a line
        self.assertTrue(x[5] > x[0])
        self.assertEqual(x[0], x[6])
        self.assertEqual(x[1] - x[0], x[6] - x[7])

        tp = Image.Circles(size=100, number=10)
        self.assertEqual(tp.shape, (100, 100))
        # TODO [l,ml,p,c] = ilabel(im);
        _, n = tp.labels_binary()
        self.assertEqual(n, 101)

        tp = Image.Squares(size=100, number=10)
        self.assertEqual(tp.shape, (100, 100))
        _, n = tp.labels_binary()
        self.assertEqual(n, 101)

        # TODO not yet converted to python:
        # tp = im.testpattern('line', 20, np.pi / 6, 10)
        # self.assertEqual(tp.shape, (20, 20))
        # self.assertEqual(tp.image[10, 0], 1)
        # self.assertEqual(tp.image[11, 1], 1)
        # self.assertEqual(tp.image[16, 11], 1)
        # self.assertEqual(np.sum(tp.image), 17)

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
        cp = canvas.paste(im, (3, 2), position='centre')
        nt.assert_array_almost_equal(cp.image, out)

        canvas = np.zeros((5, 5))
        canvas = Image(canvas)
        cp = canvas.paste(im, (2, 1), method='set')
        nt.assert_array_almost_equal(cp.image, out)

        canvas = np.zeros((5, 5))
        canvas = Image(canvas)
        cp = canvas.paste(im, (2, 1), method='mean')
        nt.assert_array_almost_equal(cp.image, out / 2)

        canvas = np.zeros((5, 5))
        canvas = Image(canvas)
        cp = canvas.paste(im, (2, 1), method='add')
        cp2 = cp.paste(im, (2, 1), method='add')
        nt.assert_array_almost_equal(cp2.image, out * 2)

    def test_choose(self):

        # test monochrome image
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])

        a = Image(a)
        b = Image(b)

        nt.assert_array_almost_equal(a.choose(b, np.zeros((2, 2))).A,
                                     a.A)
        nt.assert_array_almost_equal(a.choose(b, np.ones((2, 2))).A,
                                     b.A)
        mask = np.array([[0, 1], [1, 0]])
        nt.assert_array_almost_equal(a.choose(b, mask).A,
                                     np.array([[1, 6], [7, 4]]))

        mask = np.array([[0, 1], [1, 0]])
        # nt.assert_array_almost_equal(a.switch(mask, 77).A,
        #                              np.array([[1, 77], [77, 4]]))

        # test color image
        a = np.random.randint(0, 255, (2, 2, 3), dtype='uint8')
        b = np.random.randint(0, 255, (2, 2, 3), dtype='uint8')
        a = Image(a)
        b = Image(b)
        mask = np.array([[0, 1], [0, 0]])
        out = a.choose(b, mask)
        nt.assert_array_almost_equal(out.A[0, 0, :], a.A[0, 0, :])
        nt.assert_array_almost_equal(out.A[0, 1, :], b.A[0, 1, :])
        nt.assert_array_almost_equal(out.A[1, 0, :], a.A[1, 0, :])
        nt.assert_array_almost_equal(out.A[1, 1, :], a.A[1, 1, :])

        out = a.choose((10,11,12), mask)
        nt.assert_array_almost_equal(out.A[0,1,:], (10,11,12))
        nt.assert_array_almost_equal(out.A[0,0,:], a.A[0,0,:])
        nt.assert_array_almost_equal(out.A[1,0,:], a.A[1,0,:])
        nt.assert_array_almost_equal(out.A[1,1,:], a.A[1,1,:])

        out = a.choose('red', mask)
        nt.assert_array_almost_equal(out.A[0,1,:], (255,0,0))
        nt.assert_array_almost_equal(out.A[0,0,:], a.A[0,0,:])
        nt.assert_array_almost_equal(out.A[1,0,:], a.A[1,0,:])
        nt.assert_array_almost_equal(out.A[1,1,:], a.A[1,1,:])

    def test_labels_binary(self):

        a = np.zeros((20,20))
        a[8:13, 8:13] = 1

        L, n = Image(a, dtype='float32').labels_binary()
        self.assertEqual(n, 2)
        self.assertEqual(L.A[10,10], 1)
        self.assertEqual(L.A[0,0], 0)
        self.assertEqual(np.sum(L.A), 25)

        a[8:13, 8:13] = 100
        L, n = Image(a, dtype='uint8').labels_binary()
        self.assertEqual(n, 2)
        self.assertEqual(L.A[10,10], 1)
        self.assertEqual(L.A[0,0], 0)
        self.assertEqual(np.sum(L.A), 25)

    # @skip
    # def test_labels_MSER(self):

    #     a = np.zeros((20,20))
    #     a[8:13, 8:13] = 0.5

    #     L, n = Image(a).labels_MSER()
    #     print(L)
    #     self.assertEqual(n, 3)
    #     self.assertEqual(L.A[10,10], 1)
    #     self.assertEqual(L.A[0,0], 0)
    #     self.assertEqual(np.sum(L.A), 25)

    def test_LUT(self):

        im = np.random.randint(1, 255, (4, 5), np.uint8)
        i = np.arange(256)
        lut = (2 * i).astype('uint8')

        # single channel LUT, single channel image
        x = Image(im).LUT(lut)
        self.assertEqual(x.shape, (4,5))
        nt.assert_almost_equal(x.A, 2 * im)

        # triple channel LUT, single channel image
        lut = np.column_stack(((2 * i), (3 * i), (256 - i))).astype('uint8')
        x = Image(im).LUT(lut, colororder="RGB")
        self.assertEqual(x.shape, (4,5,3))
        nt.assert_almost_equal(x.A[:,:,0], 2 * im)
        nt.assert_almost_equal(x.A[:,:,1], 3 * im)
        nt.assert_almost_equal(x.A[:,:,2], 256 - im)

        # single channel LUT, triple channel image
        im = np.random.randint(1, 255, (4, 5, 3), np.uint8)
        lut = (2 * i).astype('uint8')
        x = Image(im, colororder="RGB").LUT(lut)
        self.assertEqual(x.shape, (4,5,3))
        nt.assert_almost_equal(x.A[:,:,0], 2 * im[:,:,0])
        nt.assert_almost_equal(x.A[:,:,1], 2 * im[:,:,1])
        nt.assert_almost_equal(x.A[:,:,2], 2 * im[:,:,2])

        # triple channel LUT, triple channel image
        lut = np.column_stack(((2 * i), (3 * i), (256 - i))).astype('uint8')
        x = Image(im, colororder="RGB").LUT(lut)
        self.assertEqual(x.shape, (4,5,3))
        nt.assert_almost_equal(x.A[:,:,0], 2 * im[:,:,0])
        nt.assert_almost_equal(x.A[:,:,1], 3 * im[:,:,1])
        nt.assert_almost_equal(x.A[:,:,2], 256 - im[:,:,2])

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


# ----------------------------------------------------------------------- #
if __name__ == '__main__':

    unittest.main()
