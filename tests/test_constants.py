#!/usr/bin/env python

import numpy as np
import numpy.testing as nt
import unittest

from machinevisiontoolbox import Image
from machinevisiontoolbox.base.color import *


class TestImageConstants(unittest.TestCase):

    def test_zeros(self):
        im = Image.Zeros(5, 9)
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(np.all(im.A == 0))

        im = Image.Zeros(5)
        self.assertEqual(im.size, (5, 5))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(np.all(im.A == 0))

        im = Image.Zeros((5, 9))
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(np.all(im.A == 0))

        im = Image.Zeros((5, 9), colororder="ABCD")
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertTrue(np.all(im.A == 0))
        self.assertTrue(im.iscolor)
        self.assertEqual(im.nplanes, 4)
        self.assertEqual(im.colororder_str, "A:B:C:D")

        im = Image.Zeros((5, 9), dtype="float32")
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.float32)
        self.assertTrue(np.all(im.A == 0.0))

        im = Image.Zeros((5, 9), dtype="float32", colororder="ABCD")
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.float32)
        self.assertTrue(im.iscolor)
        self.assertEqual(im.nplanes, 4)
        self.assertEqual(im.colororder_str, "A:B:C:D")

    def test_constant(self):
        im = Image.Constant(5, 9, value=42)
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(np.all(im.A == 42))

        im = Image.Constant(5, value=42)
        self.assertEqual(im.size, (5, 5))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(np.all(im.A == 42))

        im = Image.Constant((5, 9), value=42)
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(np.all(im.A == 42))

        im = Image.Constant((5, 9), value="red")
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertTrue(im.iscolor)
        self.assertEqual(im.nplanes, 3)
        self.assertEqual(im.colororder_str, "R:G:B")
        self.assertTrue(np.all(im.A[:, :, 0] == 255))
        self.assertTrue(np.all(im.A[:, :, 1] == 0))
        self.assertTrue(np.all(im.A[:, :, 2] == 0))

        im = Image.Constant((5, 9), value="red", colororder="XYZ")
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertTrue(im.iscolor)
        self.assertEqual(im.nplanes, 3)
        self.assertEqual(im.colororder_str, "X:Y:Z")
        self.assertTrue(np.all(im.A[:, :, 0] == 255))
        self.assertTrue(np.all(im.A[:, :, 1] == 0))
        self.assertTrue(np.all(im.A[:, :, 2] == 0))

        im = Image.Constant((5, 9), value=(40, 41, 42, 43), colororder="ABCD")
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertTrue(im.iscolor)
        self.assertEqual(im.nplanes, 4)
        self.assertEqual(im.colororder_str, "A:B:C:D")
        self.assertTrue(np.all(im.A[:, :, 0] == 40))
        self.assertTrue(np.all(im.A[:, :, 1] == 41))
        self.assertTrue(np.all(im.A[:, :, 2] == 42))
        self.assertTrue(np.all(im.A[:, :, 3] == 43))

        im = Image.Constant(
            (5, 9), dtype="float32", value=(0.40, 0.41, 0.42, 0.43), colororder="ABCD"
        )
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.float32)
        self.assertTrue(im.iscolor)
        self.assertEqual(im.nplanes, 4)
        self.assertEqual(im.colororder_str, "A:B:C:D")
        self.assertTrue(np.all(im.A[:, :, 0] == 0.40))
        self.assertTrue(np.all(im.A[:, :, 1] == 0.41))
        self.assertTrue(np.all(im.A[:, :, 2] == 0.42))
        self.assertTrue(np.all(im.A[:, :, 3] == 0.43))

        im = Image.Constant((5, 9), value=0.74, dtype="float32")
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.float32)
        self.assertTrue(np.all(im.A == 0.74))

        with self.assertRaises(ValueError) as context:
            im = Image.Constant(5, 9, value=(40, 41, 42), colororder="ABCD")
        with self.assertRaises(ValueError) as context:
            im = Image.Constant((5, 9), value=(40, 41, 42), colororder="ABCD")

    def test_string(self):
        im = Image.String(
            r"""
                    ..##..
                    ...##.
                """,
            binary=True,
        )

        self.assertEqual(im.size, (6, 2))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(
            np.all(
                im.A
                == np.array(
                    [
                        [False, False, True, True, False, False],
                        [False, False, False, True, True, False],
                    ]
                )
            )
        )

        im = (
            Image.String(
                r"""
                    001200
                    000345
                """
            )
            - ord("0")
        )

        self.assertEqual(im.size, (6, 2))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(
            np.all(
                im.A
                == np.array(
                    [
                        [0, 0, 1, 2, 0, 0],
                        [0, 0, 0, 3, 4, 5],
                    ]
                )
            )
        )
        im = Image.String(
            r"""


                    ..##..

                    ...##.


                """,
            binary=True,
        )

        self.assertEqual(im.size, (6, 2))

    def test_string2(self):

        img = Image.String("01234|56789|01234")
        self.assertEqual(img.size, (5, 3))
        self.assertEqual(img.dtype, np.uint8)
        self.assertFalse(img.iscolor)
        self.assertTrue(
            np.all(
                img.A
                == np.array(
                    [
                        [0, 1, 2, 3, 4],
                        [5, 6, 7, 8, 9],
                        [0, 1, 2, 3, 4],
                    ]
                )
            )
        )

    def test_random(self):
        im = Image.Random(5, 9)
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im.A.var() > 10)

        im = Image.Random(5)
        self.assertEqual(im.size, (5, 5))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im.A.var() > 10)

        im = Image.Random((5, 9))
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im.A.var() > 10)

        im = Image.Random((5, 9), maxval=10)
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im.A.var() > 3)
        self.assertTrue(im.A.max() == 9)

        im = Image.Random((5, 9), colororder="ABCD")
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertTrue(im.iscolor)
        self.assertEqual(im.nplanes, 4)
        self.assertEqual(im.colororder_str, "A:B:C:D")
        self.assertTrue(np.all(im.A.var(axis=(0, 1)) > 10))

        im = Image.Random((5, 9), dtype="float32")
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.float32)
        self.assertTrue(im.A.var() > 0.05)
        self.assertTrue(np.all(im.A.max() < 1.0))
        self.assertTrue(np.any(im.A.max() > 0.5))

        im = Image.Random((5, 9), dtype="float32", maxval=10)
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.float32)
        self.assertTrue(im.A.var() > 0.5)
        self.assertTrue(np.all(im.A.max() < 10.0))
        self.assertTrue(np.any(im.A.max() > 5))

    def test_squares(self):
        im = Image.Squares(1, size=100)
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im.A.min() == 0)
        self.assertTrue(im.A.max() == 1)

        im = Image.Squares(1, size=100, fg=30, bg=20)
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im.A.min() == 20)
        self.assertTrue(im.A.max() == 30)

        im = Image.Squares(1, size=100, fg=3, bg=2, dtype="float32")
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.float32)
        self.assertFalse(im.iscolor)
        self.assertTrue(im.A.min() == 2.0)
        self.assertTrue(im.A.max() == 3.0)

        im = Image.Squares(1, size=5, fg=3, bg=2)
        self.assertEqual(im.size, (5, 5))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(
            np.all(
                im.A
                == np.array(
                    [
                        [2, 2, 2, 2, 2],
                        [2, 3, 3, 3, 2],
                        [2, 3, 3, 3, 2],
                        [2, 3, 3, 3, 2],
                        [2, 2, 2, 2, 2],
                    ]
                ),
            )
        )

    def test_circles(self):
        im = Image.Circles(1, size=100)
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im.A.min() == 0)
        self.assertTrue(im.A.max() == 1)

        im = Image.Circles(1, size=100, fg=30, bg=20)
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im.A.min() == 20)
        self.assertTrue(im.A.max() == 30)

        im = Image.Circles(1, size=100, fg=3, bg=2, dtype="float32")
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.float32)
        self.assertFalse(im.iscolor)
        self.assertTrue(im.A.min() == 2.0)
        self.assertTrue(im.A.max() == 3.0)

        im = Image.Circles(1, size=5, fg=3, bg=2)
        self.assertEqual(im.size, (5, 5))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(
            np.all(
                im.A
                == np.array(
                    [
                        [2, 2, 2, 2, 2],
                        [2, 2, 3, 2, 2],
                        [2, 3, 3, 3, 2],
                        [2, 2, 3, 2, 2],
                        [2, 2, 2, 2, 2],
                    ]
                ),
            )
        )

    def test_ramp(self):
        im = Image.Ramp(size=100, dtype="uint8")
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im.A.min() == 0)
        self.assertTrue(im.A.max() == 255)
        self.assertTrue(im.A[0, 0] < im.A[0, 1])
        self.assertTrue(im.A[0, 0] == im.A[1, 0])

        im = Image.Ramp(size=100, dir="y", dtype="uint8")
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im.A.min() == 0)
        self.assertTrue(im.A.max() == 255)
        self.assertTrue(im.A[0, 0] < im.A[1, 0])
        self.assertTrue(im.A[0, 0] == im.A[0, 1])

        im = Image.Ramp(size=100)
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.float32)
        self.assertFalse(im.iscolor)
        self.assertTrue(im.A.min() == 0)
        self.assertTrue(im.A.max() == 1.0)
        self.assertTrue(im.A[0, 0] < im.A[0, 1])
        self.assertTrue(im.A[0, 0] == im.A[1, 0])

        im = Image.Ramp(size=20)
        nt.assert_almost_equal(
            im.A[0, :],
            np.array(
                [
                    0.0,
                    0.11111111,
                    0.22222222,
                    0.33333334,
                    0.44444445,
                    0.5555556,
                    0.6666667,
                    0.7777778,
                    0.8888889,
                    1.0,
                    0.0,
                    0.11111111,
                    0.22222222,
                    0.33333334,
                    0.44444445,
                    0.5555556,
                    0.6666667,
                    0.7777778,
                    0.8888889,
                    1.0,
                ]
            ),
        )

    def test_sine(self):
        im = Image.Sin(size=100)
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.float32)
        self.assertFalse(im.iscolor)
        self.assertTrue(im.A.min() < 0.05)
        self.assertTrue(im.A.max() > 0.95)
        self.assertTrue(im.A[0, 0] < im.A[0, 1])
        self.assertTrue(im.A[0, 0] == im.A[1, 0])

        im = Image.Sin(size=100, dir="y")
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.float32)
        self.assertFalse(im.iscolor)
        self.assertTrue(im.A.min() < 0.05)
        self.assertTrue(im.A.max() > 0.95)
        self.assertTrue(im.A[0, 0] < im.A[1, 0])
        self.assertTrue(im.A[0, 0] == im.A[0, 1])

        im = Image.Sin(size=100, dtype="uint8")
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im.A.min() < 5)
        self.assertTrue(im.A.max() > 250)
        self.assertTrue(im.A[0, 0] < im.A[0, 1])
        self.assertTrue(im.A[0, 0] == im.A[1, 0])

        im = Image.Sin(size=20)
        nt.assert_almost_equal(
            im.A[0, :],
            np.array(
                [
                    0.5,
                    0.7938926,
                    0.97552824,
                    0.97552824,
                    0.7938926,
                    0.5,
                    0.20610738,
                    0.02447174,
                    0.02447174,
                    0.20610738,
                    0.5,
                    0.7938926,
                    0.97552824,
                    0.97552824,
                    0.7938926,
                    0.5,
                    0.20610738,
                    0.02447174,
                    0.02447174,
                    0.20610738,
                ],
            ),
        )


# ----------------------------------------------------------------------------#
if __name__ == "__main__":

    unittest.main()

    # import code
    # code.interact(local=dict(globals(), **locals()))
