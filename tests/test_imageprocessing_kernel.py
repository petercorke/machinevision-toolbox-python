#!/usr/bin/env python

import numpy as np
import numpy.testing as nt
import unittest

from machinevisiontoolbox import Image
# from pathlib import Path


class TestImageProcessingKernel(unittest.TestCase):

    def test_similarity(self):

        a = np.array([[0.9280, 0.3879, 0.8679],
                      [0.1695, 0.3826, 0.7415],
                      [0.8837, 0.2715, 0.4479]])
        a = Image(a, dtype='float64')

        eps = 1e-6
        # self.assertAlmostEqual(a.sad(a), eps)
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

        # TODO check imatch.m, as test_similarity calls imatch, which has not
        # yet been implemented in mvt

    def test_window(self):
        im = np.array([[3,     5,     8,    10,     9],
                       [7,    10,     3,     6,     3],
                       [7,     4,     6,     2,     9],
                       [2,     6,     7,     2,     3],
                       [2,     3,     9,     3,    10]])
        img = Image(im)
        se = np.ones((1, 1))
        # se must be same number of dimensions as input image for scipy
        # TODO maybe detect this in im.window and perform this line?

        # test with different input formats
        nt.assert_array_almost_equal(img.window(np.sum, se=se).A, im)

        se = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        out = np.array([[43,    47,    57,    56,    59],
                        [46,    43,    51,    50,    57],
                        [45,   48,    40,    39,    31],
                        [33,    40,    35,    49,    48],
                        [22,    40,    36,    53,    44]])
        nt.assert_array_almost_equal(img.window(np.sum, se=se).A, out)

    # TODO
    # kgauss
    # klaplace
    # ksobel
    # kdog
    # klog
    # kdgauss
    # kcircle
    # smooth
    # similarity
    # pyramid
    # convolve
    # canny


# ----------------------------------------------------------------------- #
if __name__ == '__main__':

    unittest.main()

