#!/usr/bin/env python

import numpy as np
import numpy.testing as nt
import unittest

from machinevisiontoolbox.Image import Image
# from pathlib import Path


class TestImageProcessingKernel(unittest.TestCase):

    def test_similarity(self):

        a = np.array([[0.9280, 0.3879, 0.8679],
                      [0.1695, 0.3826, 0.7415],
                      [0.8837, 0.2715, 0.4479]])
        a = Image(a)

        eps = 2.224e-16 * 100
        # self.assertAlmostEqual(a.sad(a), eps)
        self.assertEqual(abs(a.sad(a)) < eps, True)
        self.assertEqual(abs(a.sad(Image(a.image + 0.1))) < eps, False)

        self.assertEqual(abs(a.zsad(a)) < eps, True)
        self.assertEqual(abs(a.zsad(Image(a.image + 0.1))) < eps, True)

        self.assertEqual(abs(a.ssd(a)) < eps, True)
        self.assertEqual(abs(a.ssd(Image(a.image + 0.1))) < eps, False)

        self.assertEqual(abs(a.zssd(a)) < eps, True)
        self.assertEqual(abs(a.zssd(Image(a.image + 0.1))) < eps, True)

        self.assertEqual(abs(1 - a.ncc(a)) < eps, True)
        self.assertEqual(abs(1 - a.ncc(Image(a.image * 2))) < eps, True)

        self.assertEqual(abs(1 - a.zncc(a)) < eps, True)
        self.assertEqual(abs(1 - a.zncc(Image(a.image + 0.1))) < eps, True)
        self.assertEqual(abs(1 - a.zncc(Image(a.image * 2))) < eps, True)

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
    # window
    # rank
    # convolve
    # canny


# ----------------------------------------------------------------------- #
if __name__ == '__main__':

    unittest.main()
