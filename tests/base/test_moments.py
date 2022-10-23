import math
import numpy as np
import numpy.testing as nt
import unittest
from machinevisiontoolbox.base.moments import *

class TestMoments(unittest.TestCase):

    def test_mpq(self):

        im = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ])

        self.assertEqual(mpq(im, 0, 0), 6)
        self.assertEqual(mpq(im, 1, 0), 9)
        self.assertEqual(mpq(im, 0, 1), 18)
        self.assertEqual(mpq(im, 1, 1), 27)
        self.assertEqual(mpq(im, 2, 0), 15)
        self.assertEqual(mpq(im, 0, 2), 58)

    def test_npq(self):

        im = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ])

        self.assertEqual(upq(im, 0, 0), 6)
        self.assertEqual(upq(im, 1, 0), 0)
        self.assertEqual(upq(im, 0, 1), 0)
        self.assertEqual(upq(im, 1, 1), 0)
        self.assertEqual(upq(im, 2, 0), 1.5)
        self.assertEqual(upq(im, 0, 2), 4)

    def test_npq(self):

        im = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ])

        self.assertEqual(npq(im, 2, 0), 1/24)
        self.assertEqual(npq(im, 0, 2), 1/9)


# ------------------------------------------------------------------------ #
if __name__ == '__main__':

    unittest.main()