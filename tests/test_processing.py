#!/usr/bin/env python

import unittest

import numpy as np
import numpy.testing as nt

from machinevisiontoolbox import Image

# from pathlib import Path

# TODO
#  - test rank/median filter with a known image and footprint, check output is correct
#  - test rank/median filter for color images, check output is correct


class TestImageProcessingBase(unittest.TestCase):

    def test_LUT_basic(self):
        """Test lookup table application"""
        # Create simple image
        img = Image([[100, 150], [200, 250]], dtype="uint8")

        # Invert LUT
        lut_invert = np.arange(255, -1, -1, dtype="uint8")
        result = img.LUT(lut_invert)
        self.assertEqual(result.A[0, 0], 155)  # 255 - 100
        self.assertEqual(result.A[0, 1], 105)  # 255 - 150

        # Identity LUT
        lut_identity = np.arange(256, dtype="uint8")
        result_id = img.LUT(lut_identity)
        nt.assert_array_equal(result_id.A, img.A)

        # Multi-plane LUT for grayscale
        lut_multi = np.stack([np.arange(256), np.arange(256)[::-1]], axis=1).astype(
            "uint8"
        )
        result_multi = img.LUT(lut_multi)
        self.assertEqual(result_multi.nplanes, 2)

    def test_apply_function(self):
        """Test applying arbitrary functions"""
        img = Image(np.ones((5, 5)) * 0.5)

        # Test with simple function
        result = img.apply(lambda x: x * 2)
        nt.assert_array_almost_equal(result.A, np.ones((5, 5)))

        # Test with vectorized function
        result_vec = img.apply(lambda x: x * 3, vectorize=True)
        nt.assert_array_almost_equal(result_vec.A, np.ones((5, 5)) * 1.5)

    def test_threshold(self):
        """Test image thresholding"""
        img = Image(np.array([[50, 100], [150, 200]], dtype="uint8"))

        # Binary threshold
        result = img.threshold(125)
        self.assertEqual(result.A[0, 0], 0)
        self.assertEqual(result.A[1, 1], 255)

    def test_otsu_threshold(self):
        """Test threshold() with Otsu correctly partitions a bimodal image"""
        img_data = np.zeros((20, 10), dtype=np.uint8)
        img_data[:10] = 20
        img_data[10:] = 220
        img = Image(img_data)
        result, t = img.threshold(t="otsu")
        # threshold is applied as pixels > t, so 20-valued pixels → 0, 220-valued → 255
        self.assertGreaterEqual(t, 20)
        self.assertLess(t, 220)
        nt.assert_array_equal(result.A[:10], 0)  # low-valued half should be zero
        nt.assert_array_equal(result.A[10:], 255)  # high-valued half should be maxval

    def test_otsu(self):
        """Test otsu() returns a scalar threshold that separates the two pixel classes"""
        img_data = np.zeros((20, 10), dtype=np.uint8)
        img_data[:10] = 20
        img_data[10:] = 220
        img = Image(img_data)
        t = img.otsu()
        self.assertGreaterEqual(t, 20)
        self.assertLess(t, 220)

    def test_read(self):

        im = Image.Read("penguins.png")
        self.assertEqual(im.size, (1047, 730))
        self.assertFalse(im.iscolor)
        self.assertEqual(im.dtype, np.uint8)

        im = Image.Read("penguins.png", dtype="float32")
        self.assertEqual(im.size, (1047, 730))
        self.assertFalse(im.iscolor)
        self.assertEqual(im.dtype, np.float32)

        im = Image.Read("monalisa.png")
        self.assertEqual(im.size, (677, 700))
        self.assertTrue(im.iscolor)
        self.assertEqual(im.dtype, np.uint8)
        self.assertEqual(im.colororder_str, "R:G:B")

        im = Image.Read("monalisa.png", mono=True)
        self.assertEqual(im.size, (677, 700))
        self.assertFalse(im.iscolor)
        self.assertEqual(im.dtype, np.uint8)

        im = Image.Read("monalisa.png", dtype="float32")
        self.assertEqual(im.size, (677, 700))
        self.assertTrue(im.iscolor)
        self.assertEqual(im.dtype, np.float32)
        self.assertEqual(im.colororder_str, "R:G:B")

        im = Image.Read("monalisa.png", dtype="float32", mono=True)
        self.assertEqual(im.size, (677, 700))
        self.assertFalse(im.iscolor)
        self.assertEqual(im.dtype, np.float32)

        im = Image.Read("monalisa.png", dtype="float32", mono=True)
        self.assertEqual(im.size, (677, 700))
        self.assertFalse(im.iscolor)
        self.assertEqual(im.dtype, np.float32)

        im = Image.Read("monalisa.png", reduce=2)
        self.assertEqual(im.size, (339, 350))
        self.assertTrue(im.iscolor)
        self.assertEqual(im.dtype, np.uint8)
        self.assertEqual(im.colororder_str, "R:G:B")

    def test_int(self):

        # test for uint8
        im = np.zeros((2, 2), np.float32)
        im = Image(im)
        nt.assert_array_almost_equal(im.array_as("uint8"), np.zeros((2, 2), np.uint8))

        im = np.ones((2, 2), np.float32)
        im = Image(im)
        nt.assert_array_almost_equal(
            im.array_as("uint8"), 255 * np.ones((2, 2)).astype(np.uint8)
        )

        # tests for shape
        im = np.random.randint(1, 255, (3, 5), int)
        im = Image(im)
        imi = im.array_as("uint8")
        nt.assert_array_almost_equal(imi.shape, im.shape)

        im = np.random.randint(1, 255, (3, 5, 3), int)
        im = Image(im)
        imi = im.array_as("uint8")
        nt.assert_array_almost_equal(imi.shape, im.shape)

    def test_float(self):
        # test for uint8
        im = np.zeros((2, 2), np.uint8)
        im = Image(im)
        nt.assert_array_almost_equal(
            im.array_as("float32"), np.zeros((2, 2), np.float32)
        )

        im = 128 * np.ones((2, 2), np.uint8)
        im = Image(im)
        nt.assert_array_almost_equal(
            im.array_as("float32"), (128.0 / 255.0 * np.ones((2, 2)))
        )

        im = 255 * np.ones((2, 2), np.uint8)
        im = Image(im)
        nt.assert_array_almost_equal(im.array_as("float32"), (np.ones((2, 2))))

        # test for uint16
        im = np.zeros((2, 2), np.uint16)
        im = Image(im)
        nt.assert_array_almost_equal(
            im.array_as("float32"), np.zeros((2, 2), np.float32)
        )

        im = 128 * np.ones((2, 2), np.uint16)
        im = Image(im)
        nt.assert_array_almost_equal(
            im.array_as("float32"), (128.0 / 65535.0 * np.ones((2, 2)))
        )

        im = 65535 * np.ones((2, 2), np.uint16)
        im = Image(im)
        nt.assert_array_almost_equal(im.array_as("float32"), (np.ones((2, 2))))

        im = Image(im)
        imf = im.array_as("float32")
        nt.assert_array_almost_equal(imf.shape, im.shape)
        nt.assert_array_almost_equal(imf, im.array.astype(np.float32) / 65535.0)

    def test_testpattern(self):
        tp = Image.Ramp(dir="x", size=20, cycles=2)
        self.assertEqual(tp.shape, (20, 20))

        r = np.linspace(0, 1, 10, endpoint=True)
        out = np.hstack((r, r))
        nt.assert_array_almost_equal(tp.A[5, :], out)

        tp = Image.Ramp(dir="y", size=20, cycles=2)
        self.assertEqual(tp.shape, (20, 20))
        nt.assert_array_almost_equal(tp.A[:, 5], out.T)

        tp = Image.Sin(dir="x", size=12, cycles=1)
        self.assertEqual(tp.shape, (12, 12))
        x = tp.A[2, :]  # take a line
        self.assertTrue(x[5] > x[0])
        self.assertEqual(x[0], x[6])
        self.assertEqual(x[1] - x[0], x[6] - x[7])

        tp = Image.Sin(dir="y", size=12, cycles=1)
        self.assertEqual(tp.shape, (12, 12))
        x = tp.A[:, 2]  # take a line
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
        # self.assertEqual(tp.array[10, 0], 1)
        # self.assertEqual(tp.array[11, 1], 1)
        # self.assertEqual(tp.array[16, 11], 1)
        # self.assertEqual(np.sum(tp.array), 17)

    def test_paste(self):

        im = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        im = Image(im)
        canvas = np.zeros((5, 5))
        canvas = Image(canvas)

        out = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 2, 3],
                [0, 0, 4, 5, 6],
                [0, 0, 7, 8, 9],
                [0, 0, 0, 0, 0],
            ]
        )
        cp = canvas.paste(im, (2, 1))
        nt.assert_array_almost_equal(cp.array, out)

        canvas = np.zeros((5, 5))
        canvas = Image(canvas)
        cp = canvas.paste(im, (3, 2), position="centre")
        nt.assert_array_almost_equal(cp.array, out)

        canvas = np.zeros((5, 5))
        canvas = Image(canvas)
        cp = canvas.paste(im, (2, 1), method="set")
        nt.assert_array_almost_equal(cp.array, out)

        canvas = np.zeros((5, 5))
        canvas = Image(canvas)
        cp = canvas.paste(im, (2, 1), method="mean")
        nt.assert_array_almost_equal(cp.array, out / 2)

        canvas = np.zeros((5, 5))
        canvas = Image(canvas)
        cp = canvas.paste(im, (2, 1), method="add")
        cp2 = cp.paste(im, (2, 1), method="add")
        nt.assert_array_almost_equal(cp2.array, out * 2)

    def test_choose(self):

        # test monochrome image
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])

        a = Image(a)
        b = Image(b)

        nt.assert_array_almost_equal(a.choose(b, np.zeros((2, 2))).A, a.A)
        nt.assert_array_almost_equal(a.choose(b, np.ones((2, 2))).A, b.A)
        mask = np.array([[0, 1], [1, 0]])
        nt.assert_array_almost_equal(a.choose(b, mask).A, np.array([[1, 6], [7, 4]]))

        mask = np.array([[0, 1], [1, 0]])
        # nt.assert_array_almost_equal(a.switch(mask, 77).A,
        #                              np.array([[1, 77], [77, 4]]))

        # test color image
        a = np.random.randint(0, 255, (2, 2, 3), dtype="uint8")
        b = np.random.randint(0, 255, (2, 2, 3), dtype="uint8")
        a = Image(a)
        b = Image(b)
        mask = np.array([[0, 1], [0, 0]])
        out = a.choose(b, mask)
        nt.assert_array_almost_equal(out.A[0, 0, :], a.A[0, 0, :])
        nt.assert_array_almost_equal(out.A[0, 1, :], b.A[0, 1, :])
        nt.assert_array_almost_equal(out.A[1, 0, :], a.A[1, 0, :])
        nt.assert_array_almost_equal(out.A[1, 1, :], a.A[1, 1, :])

        out = a.choose((10, 11, 12), mask)
        nt.assert_array_almost_equal(out.A[0, 1, :], (10, 11, 12))
        nt.assert_array_almost_equal(out.A[0, 0, :], a.A[0, 0, :])
        nt.assert_array_almost_equal(out.A[1, 0, :], a.A[1, 0, :])
        nt.assert_array_almost_equal(out.A[1, 1, :], a.A[1, 1, :])

        out = a.choose("red", mask)
        nt.assert_array_almost_equal(out.A[0, 1, :], (255, 0, 0))
        nt.assert_array_almost_equal(out.A[0, 0, :], a.A[0, 0, :])
        nt.assert_array_almost_equal(out.A[1, 0, :], a.A[1, 0, :])
        nt.assert_array_almost_equal(out.A[1, 1, :], a.A[1, 1, :])

    def test_labels_binary(self):

        a = np.zeros((20, 20))
        a[8:13, 8:13] = 1

        L, n = Image(a, dtype="float32").labels_binary()
        self.assertEqual(n, 2)
        self.assertEqual(L.A[10, 10], 1)
        self.assertEqual(L.A[0, 0], 0)
        self.assertEqual(np.sum(L.A), 25)

        a[8:13, 8:13] = 100
        L, n = Image(a, dtype="uint8").labels_binary()
        self.assertEqual(n, 2)
        self.assertEqual(L.A[10, 10], 1)
        self.assertEqual(L.A[0, 0], 0)
        self.assertEqual(np.sum(L.A), 25)

    def test_histogram(self):
        """Test histogram computation"""
        img = Image.Read("monalisa.png", mono=True)
        hist = img.hist()
        self.assertIsNotNone(hist)

    def test_stats(self):
        """Test image statistics"""
        img = Image(np.random.rand(10, 10))
        stats = img.stats()
        self.assertIsNotNone(stats)

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
        lut = (2 * i).astype("uint8")

        # single channel LUT, single channel image
        x = Image(im).LUT(lut)
        self.assertEqual(x.shape, (4, 5))
        nt.assert_almost_equal(x.A, 2 * im)

        # triple channel LUT, single channel image
        lut = np.column_stack(((2 * i), (3 * i), (255 - i))).astype("uint8")
        x = Image(im).LUT(lut, colororder="RGB")
        self.assertEqual(x.shape, (4, 5, 3))
        nt.assert_almost_equal(x.A[:, :, 0], 2 * im)
        nt.assert_almost_equal(x.A[:, :, 1], 3 * im)
        nt.assert_almost_equal(x.A[:, :, 2], 255 - im)

        # single channel LUT, triple channel image
        im = np.random.randint(1, 255, (4, 5, 3), np.uint8)
        lut = (2 * i).astype("uint8")
        x = Image(im, colororder="RGB").LUT(lut)
        self.assertEqual(x.shape, (4, 5, 3))
        nt.assert_almost_equal(x.A[:, :, 0], 2 * im[:, :, 0])
        nt.assert_almost_equal(x.A[:, :, 1], 2 * im[:, :, 1])
        nt.assert_almost_equal(x.A[:, :, 2], 2 * im[:, :, 2])

        # triple channel LUT, triple channel image
        lut = np.column_stack(((2 * i), (3 * i), (255 - i))).astype("uint8")
        x = Image(im, colororder="RGB").LUT(lut)
        self.assertEqual(x.shape, (4, 5, 3))
        nt.assert_almost_equal(x.A[:, :, 0], 2 * im[:, :, 0])
        nt.assert_almost_equal(x.A[:, :, 1], 3 * im[:, :, 1])
        nt.assert_almost_equal(x.A[:, :, 2], 255 - im[:, :, 2])


class TestImageProcessingOperations(unittest.TestCase):

    def test_clip(self):
        img = Image([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        out = img.clip(3, 7)
        self.assertEqual(out.A[0, 0], 3)  # 1 clipped to 3
        self.assertEqual(out.A[1, 1], 5)  # 5 unchanged
        self.assertEqual(out.A[2, 2], 7)  # 9 clipped to 7

    def test_roll_dx_dy(self):
        img = Image([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # dx is alias for ru (column roll)
        r1 = img.roll(ru=1)
        r2 = img.roll(dx=1)
        nt.assert_array_equal(r1.A, r2.A)
        # dy is alias for rv (row roll)
        r3 = img.roll(rv=1)
        r4 = img.roll(dy=1)
        nt.assert_array_equal(r3.A, r4.A)

    def test_normhist(self):
        img = Image(np.array([[10, 20, 30], [40, 41, 42], [70, 80, 90]], dtype="uint8"))
        out = img.normhist()
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, img.dtype)

    def test_stretch_with_range(self):
        img = Image([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        out = img.stretch(range=[2, 8])
        # pixel at 2 → 0.0, pixel at 8 → 1.0
        self.assertAlmostEqual(out.A[0, 1], 0.0, places=5)  # value 2
        self.assertAlmostEqual(out.A[2, 1], 1.0, places=5)  # value 8

    def test_stretch_clip_false(self):
        img = Image([[0, 5, 10]])
        out = img.stretch(range=[2, 8], clip=False)
        # pixel at 0 < range[0] → negative value (not clipped)
        self.assertLess(out.A[0, 0], 0.0)

    def test_thresh_deprecated(self):
        img = Image(np.array([[50, 150], [200, 250]], dtype="uint8"))
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = img.thresh(t=100)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(out.shape, img.shape)

    def test_ithresh_with_threshold(self):
        img = Image(np.array([[50, 150], [200, 250]], dtype="uint8"))
        # With threshold supplied, acts as regular threshold (no interaction)
        out = img.ithresh(threshold=100)
        self.assertEqual(out.shape, img.shape)

    def test_threshold_adaptive(self):
        img = Image.Read("monalisa.png", mono=True)
        out = img.threshold_adaptive(h=15)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_threshold_adaptive_blocksize(self):
        img = Image.Read("monalisa.png", mono=True)
        out = img.threshold_adaptive(blocksize=31)
        self.assertEqual(out.shape, img.shape)

    def test_blend(self):
        img1 = Image(np.full((3, 3), 4, dtype="uint8"))
        img2 = Image(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="uint8"))
        out = img1.blend(img2, alpha=0.5)
        self.assertEqual(out.shape, img1.shape)

    def test_blend_size_mismatch_raises(self):
        img1 = Image(np.ones((3, 3), dtype="uint8"))
        img2 = Image(np.ones((4, 4), dtype="uint8"))
        with self.assertRaises(ValueError):
            img1.blend(img2, alpha=0.5)

    def test_apply2_size_mismatch_raises(self):
        img1 = Image(np.ones((3, 3)))
        img2 = Image(np.ones((4, 4)))
        with self.assertRaises(ValueError):
            img1.apply2(img2, lambda a, b: a + b)

    def test_invert_int(self):
        img = Image(np.array([[0, 128, 255]], dtype="uint8"))
        out = img.invert()
        self.assertEqual(out.A[0, 0], 255)
        self.assertEqual(out.A[0, 1], 127)
        self.assertEqual(out.A[0, 2], 0)

    def test_invert_float(self):
        img = Image(np.array([[0.0, 0.5, 1.0]]))
        out = img.invert()
        nt.assert_array_almost_equal(out.A, [[1.0, 0.5, 0.0]])

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


class TestImageProcessingKernel(unittest.TestCase):

    def test_similarity(self):

        a = np.array([[0.9280, 0.3879, 0.8679],
                      [0.1695, 0.3826, 0.7415],
                      [0.8837, 0.2715, 0.4479]])
        a = Image(a, dtype='float64')

        eps = 1e-6
        self.assertEqual(abs(a.sad(a)) < eps, True)
        self.assertEqual(abs(a.sad(Image(a.A + 0.1))) < eps, False)

        self.assertEqual(abs(a.zsad(a)) < eps, True)
        self.assertEqual(abs(a.zsad(a + 0.1)) < eps, True)

        self.assertEqual(abs(a.ssd(a)) < eps, True)
        self.assertEqual(abs(a.ssd(a + 0.1)) < eps, False)

        self.assertEqual(abs(a.zssd(a)) < eps, True)
        self.assertEqual(abs(a.zssd(a + 0.1)) < eps, True)

        self.assertEqual(abs(1 - a.ncc(a)) < eps, True)
        self.assertEqual(abs(1 - a.ncc(a * 2)) < eps, True)

        self.assertEqual(abs(1 - a.zncc(a)) < eps, True)
        self.assertEqual(abs(1 - a.zncc(a + 0.1)) < eps, True)
        self.assertEqual(abs(1 - a.zncc(a * 2)) < eps, True)

    def test_window(self):
        im = np.array([[3,     5,     8,    10,     9],
                       [7,    10,     3,     6,     3],
                       [7,     4,     6,     2,     9],
                       [2,     6,     7,     2,     3],
                       [2,     3,     9,     3,    10]])
        img = Image(im)
        se = np.ones((1, 1))

        nt.assert_array_almost_equal(img.window(np.sum, se=se).A, im)

        se = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        out = np.array([[43,    47,    57,    56,    59],
                        [46,    43,    51,    50,    57],
                        [45,   48,    40,    39,    31],
                        [33,    40,    35,    49,    48],
                        [22,    40,    36,    53,    44]])
        nt.assert_array_almost_equal(img.window(np.sum, se=se).A, out)


# ----------------------------------------------------------------------- #
if __name__ == "__main__":

    unittest.main()
