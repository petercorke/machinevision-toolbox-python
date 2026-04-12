#!/usr/bin/env python

import unittest

import numpy as np
import numpy.testing as nt
from spatialmath import Polygon2

from machinevisiontoolbox import Image
from machinevisiontoolbox.base.color import *


class TestImageConstants(unittest.TestCase):
    def test_zeros(self):
        im = Image.Zeros(5, 9)
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(np.all(im._A == 0))

        im = Image.Zeros(5)
        self.assertEqual(im.size, (5, 5))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(np.all(im._A == 0))

        im = Image.Zeros((5, 9))
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(np.all(im._A == 0))

        im = Image.Zeros((5, 9), colororder="ABCD")
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertTrue(np.all(im._A == 0))
        self.assertTrue(im.iscolor)
        self.assertEqual(im.nplanes, 4)
        self.assertEqual(im.colororder_str, "A:B:C:D")

        im = Image.Zeros((5, 9), dtype="float32")
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.float32)
        self.assertTrue(np.all(im._A == 0.0))

        im = Image.Zeros((5, 9), dtype="float32", colororder="ABCD")
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.float32)
        self.assertTrue(im.iscolor)
        self.assertEqual(im.nplanes, 4)
        self.assertEqual(im.colororder_str, "A:B:C:D")

    def test_constant(self):
        im = Image.Constant(42, size=(5, 9))
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(np.all(im._A == 42))

        im = Image.Constant(42, size=5)
        self.assertEqual(im.size, (5, 5))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(np.all(im._A == 42))

        im = Image.Constant(42, size=(5, 9))
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(np.all(im._A == 42))

        im = Image.Constant("red", size=(5, 9))
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertTrue(im.iscolor)
        self.assertEqual(im.nplanes, 3)
        self.assertEqual(im.colororder_str, "R:G:B")
        self.assertTrue(np.all(im._A[:, :, 0] == 255))
        self.assertTrue(np.all(im._A[:, :, 1] == 0))
        self.assertTrue(np.all(im._A[:, :, 2] == 0))

        im = Image.Constant("red", size=(5, 9), colororder="XYZ")
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertTrue(im.iscolor)
        self.assertEqual(im.nplanes, 3)
        self.assertEqual(im.colororder_str, "X:Y:Z")
        self.assertTrue(np.all(im._A[:, :, 0] == 255))
        self.assertTrue(np.all(im._A[:, :, 1] == 0))
        self.assertTrue(np.all(im._A[:, :, 2] == 0))

        im = Image.Constant((40, 41, 42, 43), size=(5, 9), colororder="ABCD")
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertTrue(im.iscolor)
        self.assertEqual(im.nplanes, 4)
        self.assertEqual(im.colororder_str, "A:B:C:D")
        self.assertTrue(np.all(im._A[:, :, 0] == 40))
        self.assertTrue(np.all(im._A[:, :, 1] == 41))
        self.assertTrue(np.all(im._A[:, :, 2] == 42))
        self.assertTrue(np.all(im._A[:, :, 3] == 43))

        im = Image.Constant(
            (0.40, 0.41, 0.42, 0.43), size=(5, 9), dtype="float32", colororder="ABCD"
        )
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.float32)
        self.assertTrue(im.iscolor)
        self.assertEqual(im.nplanes, 4)
        self.assertEqual(im.colororder_str, "A:B:C:D")
        self.assertTrue(np.all(im._A[:, :, 0] == 0.40))
        self.assertTrue(np.all(im._A[:, :, 1] == 0.41))
        self.assertTrue(np.all(im._A[:, :, 2] == 0.42))
        self.assertTrue(np.all(im._A[:, :, 3] == 0.43))

        im = Image.Constant(0.74, size=(5, 9), dtype="float32")
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.float32)
        self.assertTrue(np.all(im._A == 0.74))

        with self.assertRaises(ValueError) as context:
            im = Image.Constant((40, 41, 42), size=(5, 9), colororder="ABCD")

    def test_constant_new_style(self):
        im = Image.Constant(42, size=(5, 9))
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(np.all(im._A == 42))

    def test_constant_legacy_warning_includes_preferred_call(self):
        with self.assertWarnsRegex(
            DeprecationWarning,
            r"Image\.Constant\(42, size=\(5, 9\)\)",
        ):
            im = Image.Constant(5, 9, 42)

        self.assertEqual(im.size, (5, 9))
        self.assertTrue(np.all(im._A == 42))

    def test_constant_legacy_warning_includes_kwargs(self):
        with self.assertWarnsRegex(
            DeprecationWarning,
            r"Image\.Constant\('red', size=\(5, 9\), colororder='XYZ', dtype='uint8'\)",
        ):
            im = Image.Constant(5, 9, value="red", colororder="XYZ", dtype="uint8")

        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.colororder_str, "X:Y:Z")

    def test_constant_legacy_tuple_size_and_value(self):
        with self.assertWarnsRegex(
            DeprecationWarning,
            r"Image\.Constant\(3, size=\(4, 4\)\)",
        ):
            im = Image.Constant((4, 4), 3)

        self.assertEqual(im.size, (4, 4))
        self.assertTrue(np.all(im._A == 3))

    def test_string_binary_and_numeric(self):
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
                im._A
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
                im._A
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

    def test_string_pipe_separator(self):
        img = Image.String("01234|56789|01234")
        self.assertEqual(img.size, (5, 3))
        self.assertEqual(img.dtype, np.uint8)
        self.assertFalse(img.iscolor)
        self.assertTrue(
            np.all(
                img._A
                == np.array(
                    [
                        [0, 1, 2, 3, 4],
                        [5, 6, 7, 8, 9],
                        [0, 1, 2, 3, 4],
                    ]
                )
            )
        )

    def test_string_multiplane(self):
        img = Image.String("12|34", "56|78", "9A|BC")
        self.assertEqual(img.size, (2, 2))
        self.assertEqual(img.nplanes, 3)
        self.assertEqual(img.colororder_str, "R:G:B")
        self.assertEqual(img[0], Image.String("12|34"))
        self.assertEqual(img[1], Image.String("56|78"))
        self.assertEqual(img[2], Image.String("9A|BC"))

    def test_random(self):
        im = Image.Random(size=(5, 9))
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im._A.var() > 10)

        im = Image.Random(size=(5, 9, 3))
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.nplanes, 3)
        self.assertEqual(im.dtype, np.uint8)
        self.assertTrue(im.iscolor)
        self.assertTrue(im._A.var() > 10)

        im = Image.Random(size=(5, 9), colororder="ABCD")
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.nplanes, 4)
        self.assertEqual(im.dtype, np.uint8)
        self.assertTrue(im.iscolor)
        self.assertTrue(im._A.var() > 10)

        im = Image.Random(5)
        self.assertEqual(im.size, (5, 5))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im._A.var() > 10)

        im = Image.Random((5, 9))
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im._A.var() > 10)

        im = Image.Random((5, 9), maxval=10)
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im._A.var() > 3)
        self.assertTrue(im._A.max() == 9)

        im = Image.Random((5, 9), colororder="ABCD")
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.uint8)
        self.assertTrue(im.iscolor)
        self.assertEqual(im.nplanes, 4)
        self.assertEqual(im.colororder_str, "A:B:C:D")
        self.assertTrue(np.all(im._A.var(axis=(0, 1)) > 10))

        im = Image.Random((5, 9), dtype="float32")
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.float32)
        self.assertTrue(im._A.var() > 0.05)
        self.assertTrue(np.all(im._A.max() < 1.0))
        self.assertTrue(np.any(im._A.max() > 0.5))

        im = Image.Random((5, 9), dtype="float32", maxval=10)
        self.assertEqual(im.size, (5, 9))
        self.assertEqual(im.dtype, np.float32)
        self.assertTrue(im._A.var() > 0.5)
        self.assertTrue(np.all(im._A.max() < 10.0))
        self.assertTrue(np.any(im._A.max() > 5))

    def test_squares(self):
        im = Image.Squares(1, size=100)
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im._A.min() == 0)
        self.assertTrue(im._A.max() == 1)

        im = Image.Squares(1, size=100, fg=30, bg=20)
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im._A.min() == 20)
        self.assertTrue(im._A.max() == 30)

        im = Image.Squares(1, size=100, fg=3, bg=2, dtype="float32")
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.float32)
        self.assertFalse(im.iscolor)
        self.assertTrue(im._A.min() == 2.0)
        self.assertTrue(im._A.max() == 3.0)

        im = Image.Squares(1, size=5, fg=3, bg=2)
        self.assertEqual(im.size, (5, 5))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(
            np.all(
                im._A
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

        like = Image.Zeros((7, 9), dtype="float32", colororder="ABCD")
        im = Image.Squares(1, like=like)
        self.assertEqual(im.size, (7, 9))
        self.assertEqual(im.dtype, np.float32)
        self.assertTrue(im.iscolor)
        self.assertEqual(im.colororder_str, "A:B:C:D")

    def test_circles(self):
        im = Image.Circles(1, size=100)
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im._A.min() == 0)
        self.assertTrue(im._A.max() == 1)

        im = Image.Circles(1, size=100, fg=30, bg=20)
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im._A.min() == 20)
        self.assertTrue(im._A.max() == 30)

        im = Image.Circles(1, size=100, fg=3, bg=2, dtype="float32")
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.float32)
        self.assertFalse(im.iscolor)
        self.assertTrue(im._A.min() == 2.0)
        self.assertTrue(im._A.max() == 3.0)

        im = Image.Circles(1, size=5, fg=3, bg=2)
        self.assertEqual(im.size, (5, 5))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(
            np.all(
                im._A
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
        self.assertTrue(im._A.min() == 0)
        self.assertTrue(im._A.max() == 255)
        self.assertTrue(im._A[0, 0] < im._A[0, 1])
        self.assertTrue(im._A[0, 0] == im._A[1, 0])

        im = Image.Ramp(size=100, dir="y", dtype="uint8")
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im._A.min() == 0)
        self.assertTrue(im._A.max() == 255)
        self.assertTrue(im._A[0, 0] < im._A[1, 0])
        self.assertTrue(im._A[0, 0] == im._A[0, 1])

        im = Image.Ramp(size=100)
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.float32)
        self.assertFalse(im.iscolor)
        self.assertTrue(im._A.min() == 0)
        self.assertTrue(im._A.max() == 1.0)
        self.assertTrue(im._A[0, 0] < im._A[0, 1])
        self.assertTrue(im._A[0, 0] == im._A[1, 0])

        im = Image.Ramp(size=20)
        nt.assert_almost_equal(
            im._A[0, :],
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
        self.assertTrue(im._A.min() < 0.05)
        self.assertTrue(im._A.max() > 0.95)
        self.assertTrue(im._A[0, 0] < im._A[0, 1])
        self.assertTrue(im._A[0, 0] == im._A[1, 0])

        im = Image.Sin(size=100, dir="y")
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.float32)
        self.assertFalse(im.iscolor)
        self.assertTrue(im._A.min() < 0.05)
        self.assertTrue(im._A.max() > 0.95)
        self.assertTrue(im._A[0, 0] < im._A[1, 0])
        self.assertTrue(im._A[0, 0] == im._A[0, 1])

        im = Image.Sin(size=100, dtype="uint8")
        self.assertEqual(im.size, (100, 100))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im._A.min() < 5)
        self.assertTrue(im._A.max() > 250)
        self.assertTrue(im._A[0, 0] < im._A[0, 1])
        self.assertTrue(im._A[0, 0] == im._A[1, 0])

        im = Image.Sin(size=(100, 200), dtype="uint8")
        self.assertEqual(im.size, (100, 200))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertTrue(im._A.min() < 5)
        self.assertTrue(im._A.max() > 250)
        self.assertTrue(im._A[0, 0] < im._A[0, 1])
        self.assertTrue(im._A[0, 0] == im._A[1, 0])

        im = Image.Sin(size=20)
        nt.assert_almost_equal(
            im._A[0, :],
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

    def test_chequerboard(self):
        im = Image.Chequerboard(size=64)
        self.assertEqual(im.size, (64, 64))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertEqual(im._A.min(), 0)
        self.assertEqual(im._A.max(), 255)
        self.assertTrue(im._A.sum(), 64 * 64 * 255 / 2)

        im = Image.Chequerboard(size=(64, 128))
        self.assertEqual(im.size, (64, 128))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertEqual(im._A.min(), 0)
        self.assertEqual(im._A.max(), 255)
        self.assertTrue(im._A.sum(), 64 * 128 * 255 / 2)

        im = Image.Chequerboard(size=(64, 128), dtype="float32")
        self.assertEqual(im.size, (64, 128))
        self.assertEqual(im.dtype, np.float32)
        self.assertFalse(im.iscolor)
        self.assertEqual(im._A.min(), 0.0)
        self.assertEqual(im._A.max(), 1.0)
        self.assertTrue(im._A.sum(), 64 * 128 / 2)

    def test_polygons(self):
        from spatialmath import Polygon2

        p1 = Polygon2([(10, 10), (12, 10), (12, 12), (10, 12)])  # 2x2
        p2 = Polygon2([(30, 30), (40, 30), (40, 40), (30, 40)])  # 10x10

        im = Image.Polygons(p1, size=(50, 60))
        self.assertEqual(im.size, (50, 60))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertEqual(im.sum(), 9)

        im = Image.Polygons([p1, p2], size=(50, 60), color=1)
        self.assertEqual(im.size, (50, 60))
        self.assertEqual(im.dtype, np.uint8)
        self.assertFalse(im.iscolor)
        self.assertEqual(im.sum(), 9 + 11 * 11)

        im = Image.Polygons([p1, p2], size=(50, 60), color=[5, 1])
        self.assertEqual(im.sum(), 5 * 9 + 11 * 11)


# ----------------------------------------------------------------------------#
if __name__ == "__main__":
    unittest.main()

    # import code
    # code.interact(local=dict(globals(), **locals()))
