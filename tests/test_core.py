from unittest.case import skip
import numpy as np
import numpy.testing as nt
import unittest
from machinevisiontoolbox import Image

from pathlib import Path


class TestImage(unittest.TestCase):

    def test_pixel(self):
        im = Image(np.arange(80).reshape((10, 8)), dtype="int64")  # 8x10 image
        pix = im.pixel(5, 6)
        self.assertIsInstance(pix, np.int64)
        self.assertEqual(pix, im.A[6, 5])

        im = Image(np.arange(240).reshape((10, 8, 3)), dtype="int64")  # 8x10 image
        pix = im.pixel(5, 6)
        self.assertIsInstance(pix, np.ndarray)
        self.assertEqual(pix.shape, (3,))
        nt.assert_array_equal(pix, im.A[6, 5, :])

    def test_getitem_grey(self):

        im = Image(np.arange(80).reshape((10, 8)), dtype="int64")  # 8x10 image

        sim = im[0:4, 0:6]
        self.assertEqual(sim.size, (4, 6))
        nt.assert_array_equal(sim.A, im.A[0:6, 0:4])

        # single column
        sim = im[5, 0:6]
        self.assertEqual(sim.size, (1, 6))
        nt.assert_array_equal(sim.A.squeeze(), im.A[0:6, 5])

        sim = im[7:, 0:6]
        self.assertEqual(sim.size, (1, 6))
        nt.assert_array_equal(sim.A.squeeze(), im.A[0:6, 7])

        sim = im[5, :]
        self.assertEqual(sim.size, (1, 10))
        nt.assert_array_equal(sim.A.squeeze(), im.A[:, 5])

        # single row
        sim = im[0:5, 6]
        self.assertEqual(sim.size, (5, 1))
        nt.assert_array_equal(sim.A.squeeze(), im.A[6, 0:5])

        sim = im[0:5, 9:]
        self.assertEqual(sim.size, (5, 1))
        nt.assert_array_equal(sim.A.squeeze(), im.A[9, 0:5])

        sim = im[:, 5]
        self.assertEqual(sim.size, (8, 1))
        nt.assert_array_equal(sim.A.squeeze(), im.A[5, :])

        # single pixel
        pix = im[5, 6]
        self.assertIsInstance(pix, np.int64)
        self.assertEqual(pix, im.A[6, 5])

        sim = im[5:5, 6:6]
        self.assertEqual(sim.size, (0, 0))

    def test_getitem_color(self):

        im = Image(np.arange(240).reshape((10, 8, 3)), dtype="int64")  # 8x10 image

        sim = im[0:4, 0:6]
        self.assertEqual(sim.size, (4, 6))
        self.assertTrue(sim.iscolor)
        self.assertEqual(sim.nplanes, 3)
        nt.assert_array_equal(sim.A, im.A[0:6, 0:4, :])

        sim = im[0:4, 0:6, 0:2]
        self.assertEqual(sim.size, (4, 6))
        self.assertTrue(sim.iscolor)
        self.assertEqual(sim.nplanes, 2)
        nt.assert_array_equal(sim.A, im.A[0:6, 0:4, 0:2])

        sim = im[0:4, 0:6, 1]
        self.assertEqual(sim.size, (4, 6))
        self.assertFalse(sim.iscolor)
        self.assertEqual(sim.nplanes, 1)
        nt.assert_array_equal(sim.A, im.A[0:6, 0:4, 1])

        sim = im[0:4, 0:6, 2:]
        self.assertEqual(sim.size, (4, 6))
        self.assertFalse(sim.iscolor)
        self.assertEqual(sim.nplanes, 1)
        nt.assert_array_equal(sim.A, im.A[0:6, 0:4, 2])

        # single column
        sim = im[5, 0:6]
        self.assertEqual(sim.size, (1, 6))
        self.assertTrue(sim.iscolor)
        self.assertEqual(sim.nplanes, 3)
        nt.assert_array_equal(sim.A.squeeze(), im.A[0:6, 5, :])

        sim = im[7:, 0:6]
        self.assertEqual(sim.size, (1, 6))
        self.assertTrue(sim.iscolor)
        self.assertEqual(sim.nplanes, 3)
        nt.assert_array_equal(sim.A.squeeze(), im.A[0:6, 7, :])

        sim = im[5, 0:6, 2]
        self.assertEqual(sim.size, (1, 6))
        self.assertFalse(sim.iscolor)
        nt.assert_array_equal(sim.A.squeeze(), im.A[0:6, 5, 2])

        sim = im[5, :]
        self.assertEqual(sim.size, (1, 10))
        nt.assert_array_equal(sim.A.squeeze(), im.A[:, 5, :])

        sim = im[7:, 0:6, 2:]
        self.assertEqual(sim.size, (1, 6))
        self.assertFalse(sim.iscolor)
        self.assertEqual(sim.nplanes, 1)
        nt.assert_array_equal(sim.A.squeeze(), im.A[0:6, 7, 2])

        sim = im[7:, 0:6, 0:2]
        self.assertEqual(sim.size, (1, 6))
        self.assertTrue(sim.iscolor)
        self.assertEqual(sim.nplanes, 2)
        nt.assert_array_equal(sim.A.squeeze(), im.A[0:6, 7, 0:2])

        # single row
        sim = im[0:5, 6]
        self.assertEqual(sim.size, (5, 1))
        self.assertTrue(sim.iscolor)
        self.assertEqual(sim.nplanes, 3)
        nt.assert_array_equal(sim.A.squeeze(), im.A[6, 0:5])

        sim = im[0:5, 9:]
        self.assertEqual(sim.size, (5, 1))
        self.assertTrue(sim.iscolor)
        self.assertEqual(sim.nplanes, 3)
        nt.assert_array_equal(sim.A.squeeze(), im.A[9, 0:5])

        sim = im[0:5, 6, 2]
        self.assertEqual(sim.size, (5, 1))
        self.assertFalse(sim.iscolor)
        nt.assert_array_equal(sim.A.squeeze(), im.A[6, 0:5, 2])

        sim = im[:, 5]
        self.assertEqual(sim.size, (8, 1))
        self.assertTrue(sim.iscolor)
        nt.assert_array_equal(sim.A.squeeze(), im.A[5, :, :])

        sim = im[0:5, 9:, 2:]
        self.assertEqual(sim.size, (5, 1))
        self.assertFalse(sim.iscolor)
        nt.assert_array_equal(sim.A.squeeze(), im.A[9, 0:5, 2])

        sim = im[0:5, 9:, 0:2]
        self.assertEqual(sim.size, (5, 1))
        self.assertTrue(sim.iscolor)
        self.assertEqual(sim.nplanes, 2)
        nt.assert_array_equal(sim.A.squeeze(), im.A[9, 0:5, 0:2])

        # single pixel
        pix = im[5, 6]
        self.assertIsInstance(pix, np.ndarray)
        self.assertEqual(pix.shape, (3,))
        nt.assert_array_equal(pix, im.A[6, 5, :])

        pix = im[5, 6, 2]
        self.assertIsInstance(pix, np.int64)
        nt.assert_array_equal(pix, im.A[6, 5, 2])

        sim = im[5, 6, 2:]
        self.assertEqual(sim.size, (1, 1))
        self.assertFalse(sim.iscolor)
        nt.assert_array_equal(sim.A.squeeze(), im.A[6, 5, 2:])

        sim = im[5, 6, 0:2]
        self.assertEqual(sim.size, (1, 1))
        self.assertTrue(sim.iscolor)
        self.assertEqual(sim.nplanes, 2)
        nt.assert_array_equal(sim.A.squeeze(), im.A[6, 5, 0:2])

    def test_ndarray_integer(self):
        im = Image([[1, 2], [3, 4], [5, 6]])  # 2x3 image

        self.assertEqual(im.height, 3)
        self.assertEqual(im.width, 2)
        self.assertEqual(im.shape, (3, 2))
        self.assertEqual(im.size, (2, 3))
        self.assertEqual(im.npixels, 6)
        self.assertEqual(im.ndim, 2)
        self.assertEqual(im.nplanes, 1)
        self.assertFalse(im.iscolor)
        self.assertFalse(im.isfloat)
        self.assertEqual(im.dtype, np.dtype(np.uint8))

        im = Image([[1, 2], [3, 257], [5, 6]])

        self.assertEqual(im.height, 3)
        self.assertEqual(im.width, 2)
        self.assertEqual(im.shape, (3, 2))
        self.assertEqual(im.size, (2, 3))
        self.assertEqual(im.npixels, 6)
        self.assertEqual(im.ndim, 2)
        self.assertEqual(im.nplanes, 1)
        self.assertFalse(im.iscolor)
        self.assertFalse(im.isfloat)
        self.assertEqual(im.dtype, np.dtype(np.uint16))

        im = Image([[1, 2], [3, 70000], [5, 6]])

        self.assertEqual(im.height, 3)
        self.assertEqual(im.width, 2)
        self.assertEqual(im.shape, (3, 2))
        self.assertEqual(im.size, (2, 3))
        self.assertEqual(im.npixels, 6)
        self.assertEqual(im.ndim, 2)
        self.assertEqual(im.nplanes, 1)
        self.assertFalse(im.iscolor)
        self.assertFalse(im.isfloat)
        self.assertEqual(im.dtype, np.dtype(np.uint32))

        im = Image([[1, 2], [3, 4.0], [5, 6]])

        self.assertEqual(im.height, 3)
        self.assertEqual(im.width, 2)
        self.assertEqual(im.shape, (3, 2))
        self.assertEqual(im.size, (2, 3))
        self.assertEqual(im.npixels, 6)
        self.assertEqual(im.ndim, 2)
        self.assertEqual(im.nplanes, 1)
        self.assertFalse(im.iscolor)
        self.assertTrue(im.isfloat)
        self.assertEqual(im.dtype, np.dtype(np.float32))

    def test_span(self):
        x = np.zeros((10, 12, 3), dtype="uint8")
        img = Image(x)

        self.assertEqual(img.umax, 11)
        self.assertEqual(img.vmax, 9)
        nt.assert_array_equal(img.uspan(), np.arange(0, 12))
        nt.assert_array_equal(img.vspan(), np.arange(0, 10))
        nt.assert_array_equal(img.uspan(step=2), np.arange(0, 12, 2))
        nt.assert_array_equal(img.vspan(step=2), np.arange(0, 10, 2))

    def test_ndarray_float(self):

        x = np.zeros((3, 4), dtype="float32")
        im = Image(x)

        self.assertEqual(im.height, 3)
        self.assertEqual(im.width, 4)
        self.assertEqual(im.shape, (3, 4))
        self.assertEqual(im.npixels, 12)
        self.assertEqual(im.ndim, 2)
        self.assertEqual(im.nplanes, 1)

        self.assertIs(im.dtype, np.dtype(np.float32))
        self.assertTrue(im.isfloat)
        self.assertFalse(im.isint)

        self.assertIs(im.A, x)

    def test_ndarray_float32(self):

        x = np.zeros((3, 4))
        im = Image(x, dtype="float32")

        self.assertEqual(im.height, 3)
        self.assertEqual(im.width, 4)
        self.assertEqual(im.shape, (3, 4))
        self.assertEqual(im.npixels, 12)
        self.assertEqual(im.ndim, 2)
        self.assertEqual(im.nplanes, 1)

        self.assertIs(im.dtype, np.dtype(np.float32))
        self.assertTrue(im.isfloat)
        self.assertFalse(im.isint)

        self.assertIsNot(im.A, x)

    def test_ndarray_float_copy(self):

        x = np.zeros((3, 4))
        im = Image(x, copy=True)

        self.assertEqual(im.height, 3)
        self.assertEqual(im.width, 4)
        self.assertEqual(im.shape, (3, 4))
        self.assertEqual(im.npixels, 12)
        self.assertEqual(im.ndim, 2)
        self.assertEqual(im.nplanes, 1)

        self.assertTrue(im.isfloat)
        self.assertFalse(im.isint)

        self.assertIsNot(im.A, x)
        nt.assert_almost_equal(im.A, x)

    def test_ndarray_float_shape(self):

        x = np.arange(12.0)
        im = Image(x, shape=(3, 4))

        self.assertEqual(im.height, 3)
        self.assertEqual(im.width, 4)
        self.assertEqual(im.shape, (3, 4))
        self.assertEqual(im.npixels, 12)
        self.assertEqual(im.ndim, 2)
        self.assertEqual(im.nplanes, 1)

        self.assertTrue(im.isfloat)
        self.assertFalse(im.isint)

        self.assertIsNot(im.A, x)

    def test_ndarray_Image(self):

        im1 = Image(np.zeros((3, 4)))

        im = Image(im1)
        self.assertEqual(im.height, 3)
        self.assertEqual(im.width, 4)
        self.assertEqual(im.shape, (3, 4))
        self.assertEqual(im.npixels, 12)
        self.assertEqual(im.ndim, 2)
        self.assertEqual(im.nplanes, 1)

        self.assertTrue(im.isfloat)
        self.assertFalse(im.isint)

    def test_color(self):

        x = np.arange(24).reshape((2, 4, 3))

        im = Image(x)

        self.assertEqual(im.height, 2)
        self.assertEqual(im.width, 4)
        self.assertEqual(im.shape, (2, 4, 3))
        self.assertEqual(im.npixels, 8)
        self.assertEqual(im.ndim, 3)
        self.assertEqual(im.nplanes, 3)

        self.assertFalse(im.isfloat)
        self.assertTrue(im.isint)

        nt.assert_array_almost_equal(im.A, x)

        self.assertIsInstance(im.colororder, dict)
        self.assertEqual(len(im.colororder), 3)
        self.assertEqual(im.colororder["R"], 0)
        self.assertEqual(im.colororder["G"], 1)
        self.assertEqual(im.colororder["B"], 2)
        self.assertEqual(im.colororder_str, "R:G:B")

    def test_bool(self):
        im = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        img = Image(im.astype(bool))
        self.assertEqual(img.dtype, np.bool_)
        nt.assert_array_almost_equal(img.A * 1, im)

        x = img.to_int()
        self.assertEqual(x.dtype, np.uint8)
        nt.assert_array_almost_equal(x, im * 255)

        x = img.to_int("uint16")
        self.assertEqual(x.dtype, np.uint16)
        nt.assert_array_almost_equal(x, im * 65535)

        x = img.to_float()
        self.assertEqual(x.dtype, np.float32)
        nt.assert_array_almost_equal(x, im)

    def test_colororder(self):

        x = np.arange(24).reshape((2, 4, 3))

        im = Image(x, dtype="uint8")
        for i in range(3):
            P = im.plane(i)
            self.assertEqual(P.shape, (2, 4))
            self.assertEqual(P.nplanes, 1)
            nt.assert_array_almost_equal(P.A, x[:, :, i])
            self.assertIs(P.colororder, None)

        for i, color in enumerate("RGB"):
            P = im.plane(color)
            self.assertEqual(P.shape, (2, 4))
            self.assertEqual(P.nplanes, 1)
            nt.assert_array_almost_equal(P.A, x[:, :, i])
            self.assertIs(P.colororder, None)

        P = im.red()
        self.assertEqual(P.shape, (2, 4))
        self.assertEqual(P.nplanes, 1)
        nt.assert_array_almost_equal(P.A, x[:, :, 0])
        self.assertIs(P.colororder, None)

        P = im.green()
        self.assertEqual(P.shape, (2, 4))
        self.assertEqual(P.nplanes, 1)
        nt.assert_array_almost_equal(P.A, x[:, :, 1])
        self.assertIs(P.colororder, None)

        P = im.blue()
        self.assertEqual(P.shape, (2, 4))
        self.assertEqual(P.nplanes, 1)
        nt.assert_array_almost_equal(P.A, x[:, :, 2])
        self.assertIs(P.colororder, None)

        P = im.plane(1)
        self.assertEqual(P.shape, (2, 4))
        self.assertEqual(P.nplanes, 1)
        nt.assert_array_almost_equal(P.A, x[:, :, 1])

        P = im.plane([1, 2])
        self.assertEqual(P.shape, (2, 4, 2))
        self.assertEqual(P.nplanes, 2)
        nt.assert_array_almost_equal(P.A, x[:, :, [1, 2]])
        self.assertEqual(P.colororder_str, "G:B")

        P = im.plane("GB")
        self.assertEqual(P.shape, (2, 4, 2))
        self.assertEqual(P.nplanes, 2)
        nt.assert_array_almost_equal(P.A, x[:, :, [1, 2]])
        self.assertEqual(P.colororder_str, "G:B")

        P = im.plane("G:B")
        self.assertEqual(P.shape, (2, 4, 2))
        self.assertEqual(P.nplanes, 2)
        nt.assert_array_almost_equal(P.A, x[:, :, [1, 2]])
        self.assertEqual(P.colororder_str, "G:B")

    def test_arith_float(self):

        x = np.arange(6).reshape((2, 3))
        imx = Image(x, dtype="float32")
        y = np.arange(6, 12).reshape((2, 3))
        imy = Image(y, dtype="float32")

        nt.assert_array_almost_equal((imx + imy).A, x + y)
        nt.assert_array_almost_equal((imx + 2).A, x + 2)
        nt.assert_array_almost_equal((2 + imy).A, 2 + y)

        nt.assert_array_almost_equal((imx - imy).A, x - y)
        nt.assert_array_almost_equal((imx - 2).A, x - 2)
        nt.assert_array_almost_equal((2 - imy).A, 2 - y)

        nt.assert_array_almost_equal((imx * imy).A, x * y)
        nt.assert_array_almost_equal((imx * 2).A, x * 2)
        nt.assert_array_almost_equal((2 * imy).A, 2 * y)

        nt.assert_array_almost_equal((imx / imy).A, x / y)
        nt.assert_array_almost_equal((imx / 2).A, x / 2)
        nt.assert_array_almost_equal((2 / imy).A, 2 / y)

        nt.assert_array_almost_equal((imx**2).A, x**2)

    def test_logical_float(self):

        x = np.arange(6).reshape((2, 3))
        imx = Image(x, dtype="float32")
        y = np.arange(6, 12).reshape((2, 3))
        imy = Image(y, dtype="float32")
        nt.assert_array_almost_equal(
            (imx == imx).A, np.ones(imx.shape, dtype="float32")
        )
        nt.assert_array_almost_equal(
            (imx != imy).A, np.ones(imx.shape, dtype="float32")
        )
        nt.assert_array_almost_equal((imx < imy).A, np.ones(imx.shape, dtype="float32"))
        nt.assert_array_almost_equal((imy > imx).A, np.ones(imx.shape, dtype="float32"))

        imx = Image(np.array([[1, 2], [3, 4]]), dtype="float32")
        imy = Image(np.array([[1, 3], [2, 5]]), dtype="float32")
        T = 1
        F = 0
        nt.assert_array_almost_equal((imx == imy).A, np.array([[T, F], [F, F]]))
        nt.assert_array_almost_equal((imx == 2).A, np.array([[F, T], [F, F]]))
        nt.assert_array_almost_equal((2 == imy).A, np.array([[F, F], [T, F]]))

        nt.assert_array_almost_equal((imx != imy).A, np.array([[F, T], [T, T]]))
        nt.assert_array_almost_equal((imx != 2).A, np.array([[T, F], [T, T]]))
        nt.assert_array_almost_equal((2 != imy).A, np.array([[T, T], [F, T]]))

        nt.assert_array_almost_equal((imx > imy).A, np.array([[F, F], [T, F]]))
        nt.assert_array_almost_equal((imx > 2).A, np.array([[F, F], [T, T]]))
        nt.assert_array_almost_equal((2 > imy).A, np.array([[T, F], [F, F]]))

        nt.assert_array_almost_equal((imx >= imy).A, np.array([[T, F], [T, F]]))
        nt.assert_array_almost_equal((imx >= 2).A, np.array([[F, T], [T, T]]))
        nt.assert_array_almost_equal((2 >= imy).A, np.array([[T, F], [T, F]]))

        nt.assert_array_almost_equal((imx < imy).A, np.array([[F, T], [F, T]]))
        nt.assert_array_almost_equal((imx < 2).A, np.array([[T, F], [F, F]]))
        nt.assert_array_almost_equal((2 < imy).A, np.array([[F, T], [F, T]]))

        nt.assert_array_almost_equal((imx <= imy).A, np.array([[T, T], [F, T]]))
        nt.assert_array_almost_equal((imx <= 2).A, np.array([[T, T], [F, F]]))
        nt.assert_array_almost_equal((2 <= imy).A, np.array([[F, T], [T, T]]))

        x = imx <= imy
        nt.assert_array_almost_equal((~x).A, (imx > imy).A)

    def test_arith_int(self):

        x = np.arange(6, dtype="uint8").reshape((2, 3))
        imx = Image(x, dtype="uint8")
        y = np.arange(6, 12, dtype="uint8").reshape((2, 3))
        imy = Image(y, dtype="uint8")

        nt.assert_array_almost_equal((imx + imy).A, x + y)
        nt.assert_array_almost_equal((imx + 2).A, x + 2)
        nt.assert_array_almost_equal((2 + imy).A, 2 + y)

        nt.assert_array_almost_equal((imx - imy).A, x - y)
        nt.assert_array_almost_equal((imx - 2).A, x - 2)
        nt.assert_array_almost_equal((2 - imy).A, 2 - y)

        nt.assert_array_almost_equal((imx * imy).A, x * y)
        nt.assert_array_almost_equal((imx * 2).A, x * 2)
        nt.assert_array_almost_equal((2 * imy).A, 2 * y)

        nt.assert_array_almost_equal((imx / imy).A, x / y)
        nt.assert_array_almost_equal((imx / 2).A, x / 2)
        nt.assert_array_almost_equal((2 / imy).A, 2 / y)

        nt.assert_array_almost_equal((imx**2).A, x**2)

    def test_logical_int(self):

        x = np.arange(6).reshape((2, 3))
        imx = Image(x, dtype="uint8")
        y = np.arange(6, 12).reshape((2, 3))
        imy = Image(y, dtype="uint8")
        nt.assert_array_almost_equal((imx == imx).A, np.ones(imx.shape, dtype="bool"))
        nt.assert_array_almost_equal((imx != imy).A, np.ones(imx.shape, dtype="bool"))
        nt.assert_array_almost_equal((imx < imy).A, np.ones(imx.shape, dtype="bool"))
        nt.assert_array_almost_equal((imy > imx).A, np.ones(imx.shape, dtype="bool"))

        imx = Image([[1, 2], [3, 4]], dtype="uint8")
        imy = Image([[1, 3], [2, 5]], dtype="uint8")
        T = True
        F = False
        nt.assert_array_almost_equal((imx == imy).A, np.array([[T, F], [F, F]]))
        nt.assert_array_almost_equal((imx == 2).A, np.array([[F, T], [F, F]]))
        nt.assert_array_almost_equal((2 == imy).A, np.array([[F, F], [T, F]]))

        nt.assert_array_almost_equal((imx != imy).A, np.array([[F, T], [T, T]]))
        nt.assert_array_almost_equal((imx != 2).A, np.array([[T, F], [T, T]]))
        nt.assert_array_almost_equal((2 != imy).A, np.array([[T, T], [F, T]]))

        nt.assert_array_almost_equal((imx > imy).A, np.array([[F, F], [T, F]]))
        nt.assert_array_almost_equal((imx > 2).A, np.array([[F, F], [T, T]]))
        nt.assert_array_almost_equal((2 > imy).A, np.array([[T, F], [F, F]]))

        nt.assert_array_almost_equal((imx >= imy).A, np.array([[T, F], [T, F]]))
        nt.assert_array_almost_equal((imx >= 2).A, np.array([[F, T], [T, T]]))
        nt.assert_array_almost_equal((2 >= imy).A, np.array([[T, F], [T, F]]))

        nt.assert_array_almost_equal((imx < imy).A, np.array([[F, T], [F, T]]))
        nt.assert_array_almost_equal((imx < 2).A, np.array([[T, F], [F, F]]))
        nt.assert_array_almost_equal((2 < imy).A, np.array([[F, T], [F, T]]))

        nt.assert_array_almost_equal((imx <= imy).A, np.array([[T, T], [F, T]]))
        nt.assert_array_almost_equal((imx <= 2).A, np.array([[T, T], [F, F]]))
        nt.assert_array_almost_equal((2 <= imy).A, np.array([[F, T], [T, T]]))

        x = imx <= imy
        nt.assert_array_almost_equal((~x).A, (imx > imy).A)

    def test_types(self):
        imx = Image(np.ones((2, 3), dtype="float32"))
        self.assertEqual(imx.dtype, np.dtype("float32"))
        self.assertFalse(imx.isint)
        self.assertTrue(imx.isfloat)
        nt.assert_array_almost_equal(imx.A, np.ones((2, 3)))

        imx = Image(np.ones((2, 3), dtype="float64"))
        self.assertEqual(imx.dtype, np.dtype("float64"))
        self.assertFalse(imx.isint)
        self.assertTrue(imx.isfloat)
        nt.assert_array_almost_equal(imx.A, np.ones((2, 3)))

        imx = Image(np.ones((2, 3)), dtype="float64")
        self.assertEqual(imx.dtype, np.dtype("float64"))
        self.assertFalse(imx.isint)
        self.assertTrue(imx.isfloat)
        nt.assert_array_almost_equal(imx.A, np.ones((2, 3)))

        imx = Image(np.ones((2, 3), dtype="int16"))
        self.assertEqual(imx.dtype, np.dtype("int16"))
        self.assertTrue(imx.isint)
        self.assertFalse(imx.isfloat)
        nt.assert_array_almost_equal(imx.A, np.ones((2, 3)))

        imx = Image(np.ones((2, 3), dtype="uint8"))
        self.assertEqual(imx.dtype, np.dtype("uint8"))
        self.assertTrue(imx.isint)
        self.assertFalse(imx.isfloat)
        nt.assert_array_almost_equal(imx.A, np.ones((2, 3)))

        #
        imx = Image(np.ones((2, 3), dtype="int64"), dtype="uint8")
        self.assertEqual(imx.dtype, np.dtype("uint8"))
        self.assertTrue(imx.isint)
        self.assertFalse(imx.isfloat)
        nt.assert_array_almost_equal(imx.A, np.ones((2, 3)))

        imx = Image(np.ones((2, 3), dtype="float64"), dtype="uint8")
        self.assertEqual(imx.dtype, np.dtype("uint8"))
        self.assertTrue(imx.isint)
        self.assertFalse(imx.isfloat)
        nt.assert_array_almost_equal(imx.A, np.ones((2, 3)))

        imx = Image(np.ones((2, 3), dtype="uint8"), dtype="uint8")
        self.assertEqual(imx.dtype, np.dtype("uint8"))
        self.assertTrue(imx.isint)
        self.assertFalse(imx.isfloat)
        nt.assert_array_almost_equal(imx.A, np.ones((2, 3)))

        imx = Image(np.ones((2, 3), dtype="uint8"), dtype="float32")
        self.assertEqual(imx.dtype, np.dtype("float32"))
        self.assertFalse(imx.isint)
        self.assertTrue(imx.isfloat)
        nt.assert_array_almost_equal(imx.A, np.ones((2, 3)))

    def test_minmax(self):
        im = Image(np.ones((2, 3)) * 100, dtype="uint8")
        self.assertEqual(im.minval, 0)
        self.assertEqual(im.maxval, 255)

    def test_truefalse(self):
        im = Image(np.ones((2, 3)) * 100, dtype="uint8")
        self.assertEqual(im.true, 255)
        self.assertEqual(im.false, 0)

        im = Image(np.ones((2, 3)) * 100, dtype="float32")
        self.assertEqual(im.true, 1.0)
        self.assertEqual(im.false, 0.0)

    def test_cast(self):
        im = Image(np.ones((2, 3)) * 100, dtype="uint8")
        self.assertIsInstance(im.cast(2.3), np.uint8)
        self.assertEqual(im.cast(2.3), 2)

        im = Image(np.ones((2, 3)) * 100, dtype="float32")
        self.assertIsInstance(im.cast(2), np.float32)
        nt.assert_array_almost_equal(im.cast(2.3), 2.3)

    def test_like(self):
        imi = Image(np.ones((2, 3)) * 100, dtype="uint8")
        imf = Image(np.ones((2, 3)) * 0.5, dtype="float32")

        x = imi.like(np.uint8(100))
        self.assertEqual(x.dtype, np.uint8)
        self.assertEqual(x, 100)

        x = imi.like(0.5)
        self.assertEqual(x.dtype, np.uint8)
        self.assertEqual(x, 127)

        x = imf.like(np.uint8(100))
        self.assertEqual(x.dtype, np.float32)
        nt.assert_array_almost_equal(x, 100 / 255.0)

        x = imf.like(0.5)
        self.assertEqual(x.dtype, np.float32)
        nt.assert_array_almost_equal(x, 0.5)

    def test_to(self):

        img = Image([[1, 2], [3, 4]])
        self.assertEqual(img.dtype, np.uint8)
        z = img.to("uint16")
        self.assertEqual(z.dtype, np.uint16)
        self.assertEqual(z.A[0, 0], 257)

        z = img.to("float32")
        self.assertEqual(z.dtype, np.float32)
        self.assertAlmostEqual(z.A[0, 0], 1 / 255.0)

        im = np.array([[0.1, 0.3], [0.3, 0.4]])
        img = Image(im, dtype="float64")
        z = img.to("uint8")
        self.assertEqual(z.dtype, np.uint8)
        self.assertEqual(z.A[0, 0], round(0.1 * 255.0))

        z = img.to("uint16")
        self.assertEqual(z.dtype, np.uint16)
        self.assertEqual(z.A[0, 0], round(0.1 * 65535.0))

        z = img.to("float32")
        self.assertEqual(z.dtype, np.float32)
        self.assertAlmostEqual(z.A[0, 0], 0.1)

    def test_arith_color(self):
        pass


# ------------------------------------------------------------------------ #
if __name__ == "__main__":

    unittest.main()
