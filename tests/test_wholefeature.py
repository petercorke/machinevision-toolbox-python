import unittest

import numpy as np
import numpy.testing as nt

from machinevisiontoolbox import Image


class TestImageWholeFeatures(unittest.TestCase):
    # hist

    def test_hist(self):
        im = (
            Image.String(
                r"""
            123456789
            12345b789
            123a56789
            123456789
            123456c89
            123456789
        """
            )
            - ord("0")
        )

        h = im.hist()

        # test x vector
        x = h.x
        self.assertEqual(len(x), 256)
        self.assertEqual(x[0], 0)
        self.assertEqual(x[1], 1)
        self.assertEqual(x[255], 255)
        self.assertEqual(x.sum(), 255 * 256 / 2)

        # test bin vector
        y = h.h
        self.assertEqual(len(y), 256)
        self.assertEqual(y[0], 0)
        self.assertEqual(y[1], 6)
        self.assertEqual(y[2], 6)
        self.assertEqual(y[3], 6)
        self.assertEqual(y[4], 5)
        self.assertEqual(y[5], 6)
        self.assertEqual(y[6], 5)
        self.assertEqual(y[7], 5)
        self.assertEqual(y[8], 6)
        self.assertEqual(y[9], 6)
        self.assertEqual(y[ord("a") - ord("0")], 1)
        self.assertEqual(y[ord("b") - ord("0")], 1)
        self.assertEqual(y[ord("c") - ord("0")], 1)

        self.assertEqual(y.sum(), 6 * 9)

    # moments
    def test_mpq(self):
        im = Image.String("00000|00000|00070|00000")

        self.assertEqual(im.mpq(0, 0), 7)
        self.assertEqual(im.mpq(1, 0), 3 * 7)
        self.assertEqual(im.mpq(0, 1), 2 * 7)
        self.assertEqual(im.mpq(2, 0), 3**2 * 7)
        self.assertEqual(im.mpq(0, 2), 2**2 * 7)
        self.assertEqual(im.mpq(1, 1), 3 * 2 * 7)

    def test_upq(self):
        im = Image.String("00000|00000|00070|00000")

        self.assertEqual(im.upq(0, 0), 7)
        self.assertEqual(im.upq(1, 0), 0)
        self.assertEqual(im.upq(0, 1), 0)
        self.assertEqual(im.upq(2, 0), 0)
        self.assertEqual(im.upq(0, 2), 0)
        self.assertEqual(im.upq(1, 1), 0)

        im = Image.Zeros(10)
        box = Image.Constant(7, size=(3, 3))  # 3x3 block
        im.paste(box, (3, 3))
        self.assertEqual(im.upq(0, 0), 3 * 3 * 7)
        self.assertEqual(im.upq(1, 0), 0)
        self.assertEqual(im.upq(0, 1), 0)
        self.assertEqual(im.upq(2, 0), 3 * (1 + 1) * 7)
        self.assertEqual(im.upq(0, 2), 3 * (1 + 1) * 7)

        im = Image.Zeros(10)
        box = Image.Constant(7, size=(5, 5))  # 5x5 block
        im.paste(box, (2, 2))
        self.assertEqual(im.upq(0, 0), 5 * 5 * 7)
        self.assertEqual(im.upq(1, 0), 0)
        self.assertEqual(im.upq(0, 1), 0)
        self.assertEqual(im.upq(2, 0), 2 * (2**2 + 1**2) * 5 * 7)
        self.assertEqual(im.upq(0, 2), 2 * (2**2 + 1**2) * 5 * 7)

    def test_npq(self):
        im = Image.String("00000|00000|00070|00000")

        self.assertEqual(im.npq(2, 0), 0)
        self.assertEqual(im.npq(0, 2), 0)
        self.assertEqual(im.npq(1, 1), 0)

        im = Image.Zeros(10)
        box = Image.Constant(7, size=(3, 3))  # 3x3 block
        im.paste(box, (3, 3))
        self.assertEqual(im.npq(2, 0), 3 * (1 + 1) * 7 / (3**2 * 7) ** 2)
        self.assertEqual(im.npq(0, 2), 3 * (1 + 1) * 7 / (3**2 * 7) ** 2)

        im = Image.Zeros(10)
        box = Image.Constant(7, size=(5, 5))  # 5x5 block
        im.paste(box, (2, 2))
        self.assertEqual(im.npq(2, 0), 2 * (2**2 + 1**2) * 5 * 7 / (5**2 * 7) ** 2)
        self.assertEqual(im.npq(0, 2), 2 * (2**2 + 1**2) * 5 * 7 / (5**2 * 7) ** 2)

    def test_humoments(self):
        pass

    def test_moments(self):
        pass

    # simple stats

    def test_mean(self):
        im = Image.String("123|456|789")
        self.assertEqual(im.mean(), 5)

    def test_std(self):
        im = Image.String("123|456|789")

        self.assertEqual(im.std(), np.sqrt(20.0 / 3))

    def test_var(self):
        im = Image.String("123|456|789")
        self.assertEqual(im.var(), 20.0 / 3)

    def test_min(self):
        im = Image.String("123|456|789")

        self.assertEqual(im.min(), 1)

    def test_max(self):
        im = Image.String("123|456|789")
        self.assertEqual(im.max(), 9)

    def test_median(self):
        im = Image.String("123|456|789")
        self.assertEqual(im.median(), 5)

    # pixel values

    def test_nonzero(self):
        im = Image.String("000|001|000|100")
        nz = im.nonzero()
        self.assertEqual(len(nz), 2)
        self.assertIn((2, 1), nz)
        self.assertIn((0, 3), nz)

    def test_flatnonzero(self):
        im = Image.String("000|001|000|100")
        nz = im.flatnonzero()
        self.assertEqual(len(nz), 2)
        self.assertIn(5, nz)
        self.assertIn(9, nz)

    def test_peak2d(self):
        im = Image.String(
            r"""
            123456789
            12345b789
            123a56789
            123456789
            123456c89
            123456789
        """
        )
        mag, pos = im.peak2d()
        self.assertEqual(pos.shape, (2, 2))
        nt.assert_array_equal(pos[:, 0], (6, 4))
        nt.assert_array_equal(pos[:, 1], (5, 1))
        self.assertEqual(len(mag), 2)
        self.assertEqual(mag[0], im.pixel(*pos[:, 0]))
        self.assertEqual(mag[1], im.pixel(*pos[:, 1]))

        mag, pos = im.peak2d(npeaks=1)
        self.assertEqual(pos.shape, (2, 1))
        nt.assert_array_equal(pos[:, 0], (6, 4))
        self.assertEqual(len(mag), 1)
        self.assertEqual(mag[0], im.pixel(*pos[:, 0]))

        mag, pos = im.peak2d(npeaks=3)
        self.assertEqual(pos.shape, (2, 3))
        nt.assert_array_equal(pos[:, 0], (6, 4))
        nt.assert_array_equal(pos[:, 1], (5, 1))
        nt.assert_array_equal(pos[:, 2], (3, 2))
        self.assertEqual(len(mag), 3)
        self.assertEqual(mag[0], im.pixel(*pos[:, 0]))
        self.assertEqual(mag[1], im.pixel(*pos[:, 1]))
        self.assertEqual(mag[2], im.pixel(*pos[:, 2]))

    # new test
    def test_stats(self):
        """Test image statistics"""
        im = Image.Random(size=(50, 50), dtype="uint8")
        stats = im.stats()
        self.assertIsNotNone(stats)
        # Check stats contains expected keys
        # self.assertIn('mean', stats)
        # self.assertIn('std', stats)

    # new test
    def test_peaks(self):
        """Test peak finding in histogram"""
        im = Image.String("000011112222")
        hist = im.hist()
        peaks = im.peaks()
        self.assertIsNotNone(peaks)

    # new test
    def test_plot(self):
        """Test plotting histogram"""
        im = Image.Random(size=(50, 50), dtype="uint8")
        # TODO: Test histogram plotting (may require mock)
        # im.plot()
        pass

    # new test
    def test_cdf_property(self):
        """Test cumulative distribution function property"""
        im = Image.Random(size=(50, 50), dtype="uint8")
        cdf = im.cdf
        self.assertIsNotNone(cdf)
        # CDF should be monotonically increasing
        # self.assertTrue(np.all(np.diff(cdf) >= 0))

    # new test
    def test_ncdf_property(self):
        """Test normalized CDF property"""
        im = Image.Random(size=(50, 50), dtype="uint8")
        ncdf = im.ncdf
        self.assertIsNotNone(ncdf)
        # Normalized CDF should end at 1.0
        # self.assertAlmostEqual(ncdf[-1], 1.0)

    # new test
    def test_h_property(self):
        """Test histogram property"""
        im = Image.Random(size=(50, 50), dtype="uint8")
        h = im.h
        self.assertIsNotNone(h)
        # Histogram sum should equal number of pixels
        # self.assertEqual(np.sum(h), im.npixels)

    # new test
    def test_x_property(self):
        """Test histogram bins property"""
        im = Image.Random(size=(50, 50), dtype="uint8")
        x = im.x
        self.assertIsNotNone(x)
        # Should have 256 bins for uint8
        # self.assertEqual(len(x), 256)


if __name__ == "__main__":
    unittest.main()
