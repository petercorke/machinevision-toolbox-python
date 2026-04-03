import unittest
from pathlib import Path
from unittest.case import skip

import numpy as np
import numpy.testing as nt
from spatialmath import Polygon2

from machinevisiontoolbox import Image


class TestImage(unittest.TestCase):
    def test_pixel(self):
        im = Image(np.arange(80).reshape((10, 8)), dtype="int64")  # 8x10 image
        pix = im.pixel(5, 6)
        self.assertIsInstance(pix, np.int64)
        self.assertEqual(pix, im.array[6, 5])

        im = Image(np.arange(240).reshape((10, 8, 3)), dtype="int64")  # 8x10 image
        pix = im.pixel(5, 6)
        self.assertIsInstance(pix, np.ndarray)
        self.assertEqual(pix.shape, (3,))
        nt.assert_array_equal(pix, im.array[6, 5, :])

    def test_getitem_grey(self):
        im = Image(np.arange(80).reshape((10, 8)), dtype="int64")  # 8x10 image

        sim = im[0:4, 0:6]
        self.assertEqual(sim.size, (4, 6))
        nt.assert_array_equal(sim.array, im.array[0:6, 0:4])

        # single column
        sim = im[5, 0:6]
        self.assertEqual(sim.size, (1, 6))
        nt.assert_array_equal(sim.array.squeeze(), im.array[0:6, 5])

        sim = im[7:, 0:6]
        self.assertEqual(sim.size, (1, 6))
        nt.assert_array_equal(sim.array.squeeze(), im.array[0:6, 7])

        sim = im[5, :]
        self.assertEqual(sim.size, (1, 10))
        nt.assert_array_equal(sim.array.squeeze(), im.array[:, 5])

        # single row
        sim = im[0:5, 6]
        self.assertEqual(sim.size, (5, 1))
        nt.assert_array_equal(sim.array.squeeze(), im.array[6, 0:5])

        sim = im[0:5, 9:]
        self.assertEqual(sim.size, (5, 1))
        nt.assert_array_equal(sim.array.squeeze(), im.array[9, 0:5])

        sim = im[:, 5]
        self.assertEqual(sim.size, (8, 1))
        nt.assert_array_equal(sim.array.squeeze(), im.array[5, :])

        # single pixel
        pix = im[5, 6]
        self.assertIsInstance(pix, np.int64)
        self.assertEqual(pix, im.array[6, 5])

        # single pixel as an Image
        sim = im[5:6, 6:7]
        self.assertEqual(sim.size, (1, 1))

    def test_colordict(self):
        cdict = Image.colororder2dict("RGBA")
        self.assertIsInstance(cdict, dict)
        self.assertEqual(len(cdict), 4)
        self.assertEqual(cdict["R"], 0)
        self.assertEqual(cdict["G"], 1)
        self.assertEqual(cdict["B"], 2)
        self.assertEqual(cdict["A"], 3)

        cdict = Image.colororder2dict("red:green:blue:Z")
        self.assertIsInstance(cdict, dict)
        self.assertEqual(len(cdict), 4)
        self.assertEqual(cdict["red"], 0)
        self.assertEqual(cdict["green"], 1)
        self.assertEqual(cdict["blue"], 2)
        self.assertEqual(cdict["Z"], 3)

        cdict = Image.colororder2dict("red:green:blue", start=5)
        self.assertIsInstance(cdict, dict)
        self.assertEqual(len(cdict), 3)
        self.assertEqual(cdict["red"], 5)
        self.assertEqual(cdict["green"], 6)
        self.assertEqual(cdict["blue"], 7)

        cdict = Image.colororder2dict("red:green:blue:Z")
        clist = Image.colordict2list(cdict)
        self.assertIsInstance(clist, list)
        self.assertEqual(len(clist), 4)
        self.assertEqual(clist[0], "red")
        self.assertEqual(clist[1], "green")
        self.assertEqual(clist[2], "blue")
        self.assertEqual(clist[3], "Z")

        self.assertEqual(Image.colordict2str(cdict), "red:green:blue:Z")

    def test_pstack(self):
        im1 = Image.Random(size=(100, 120))
        im2 = Image.Pstack((im1, im1, im1), colororder="RGB")
        self.assertEqual(im2.nplanes, 3)
        self.assertEqual(im2.size, (100, 120))
        nt.assert_array_equal(im2[0, 0], np.array([im1[0, 0], im1[0, 0], im1[0, 0]]))
        self.assertEqual(im2.colororder_str, "R:G:B")

        r = Image.Random(size=(100, 120), colororder="R")
        g = Image.Random(size=(100, 120), colororder="G")
        b = Image.Random(size=(100, 120), colororder="B")

        im2 = Image.Pstack((g, r, b))
        self.assertEqual(im2.nplanes, 3)
        self.assertEqual(im2.size, (100, 120))
        self.assertEqual(im2.colororder_str, "G:R:B")
        nt.assert_array_equal(im2[0, 0], np.r_[g[0, 0], r[0, 0], b[0, 0]])

        r = Image.Random(size=(100, 120), colororder="R")
        g = Image.Random(size=(100, 120), colororder="G")
        b = Image.Random(size=(100, 120), colororder="B")
        gr = Image.Pstack((g, r))
        self.assertEqual(gr.nplanes, 2)
        self.assertEqual(gr.size, (100, 120))
        self.assertEqual(gr.colororder_str, "G:R")
        nt.assert_array_equal(gr[0, 0], np.r_[g[0, 0], r[0, 0]])

        grb = Image.Pstack((gr, b))
        self.assertEqual(grb.nplanes, 3)
        self.assertEqual(grb.size, (100, 120))
        self.assertEqual(grb.colororder_str, "G:R:B")
        nt.assert_array_equal(grb[0, 0], np.r_[g[0, 0], r[0, 0], b[0, 0]])

        im1 = Image.Random(size=(100, 120))
        im2 = Image.Pstack((im1, im1))
        self.assertEqual(im2.colororder_str, None)

    def pstack_mod(self):
        x = np.arange(12).reshape((3, 4))
        y = np.arange(100, 112).reshape((3, 4))

        im = Image(x) % Image(y)
        self.assertEqual(im.shape, (3, 4))
        self.assertEqual(im.nplanes, 2)
        nt.assert_array_equal(im[0], x)
        nt.assert_array_equal(im[1], y)

        im = Image(x) % 7 % Image(y)
        self.assertEqual(im.shape, (3, 4))
        self.assertEqual(im.nplanes, 3)
        nt.assert_array_equal(im[0], x)
        nt.assert_array_equal(im[1], 7)
        nt.assert_array_equal(im[2], y)

    def test_getitem_color(self):
        im = Image(np.arange(240).reshape((10, 8, 3)), dtype="int64")  # 8x10 image

        sim = im[0:4, 0:6]
        self.assertEqual(sim.size, (4, 6))
        self.assertTrue(sim.iscolor)
        self.assertEqual(sim.nplanes, 3)
        nt.assert_array_equal(sim.array, im.array[0:6, 0:4, :])

        sim = im[0:4, 0:6, 0:2]
        self.assertEqual(sim.size, (4, 6))
        self.assertTrue(sim.iscolor)
        self.assertEqual(sim.nplanes, 2)
        nt.assert_array_equal(sim.array, im.array[0:6, 0:4, 0:2])

        sim = im[0:4, 0:6, 1]
        self.assertEqual(sim.size, (4, 6))
        self.assertFalse(sim.iscolor)
        self.assertEqual(sim.nplanes, 1)
        nt.assert_array_equal(sim.array, im.array[0:6, 0:4, 1])

        sim = im[0:4, 0:6, 2:]
        self.assertEqual(sim.size, (4, 6))
        self.assertFalse(sim.iscolor)
        self.assertEqual(sim.nplanes, 1)
        nt.assert_array_equal(sim.array, im.array[0:6, 0:4, 2])

        # single column
        sim = im[5, 0:6]
        self.assertEqual(sim.size, (1, 6))
        self.assertTrue(sim.iscolor)
        self.assertEqual(sim.nplanes, 3)
        nt.assert_array_equal(sim.array.squeeze(), im.array[0:6, 5, :])

        sim = im[7:, 0:6]
        self.assertEqual(sim.size, (1, 6))
        self.assertTrue(sim.iscolor)
        self.assertEqual(sim.nplanes, 3)
        nt.assert_array_equal(sim.array.squeeze(), im.array[0:6, 7, :])

        sim = im[5, 0:6, 2]
        self.assertEqual(sim.size, (1, 6))
        self.assertFalse(sim.iscolor)
        nt.assert_array_equal(sim.array.squeeze(), im.array[0:6, 5, 2])

        sim = im[5, :]
        self.assertEqual(sim.size, (1, 10))
        nt.assert_array_equal(sim.array.squeeze(), im.array[:, 5, :])

        sim = im[7:, 0:6, 2:]
        self.assertEqual(sim.size, (1, 6))
        self.assertFalse(sim.iscolor)
        self.assertEqual(sim.nplanes, 1)
        nt.assert_array_equal(sim.array.squeeze(), im.array[0:6, 7, 2])

        sim = im[7:, 0:6, 0:2]
        self.assertEqual(sim.size, (1, 6))
        self.assertTrue(sim.iscolor)
        self.assertEqual(sim.nplanes, 2)
        nt.assert_array_equal(sim.array.squeeze(), im.array[0:6, 7, 0:2])

        # single row
        sim = im[0:5, 6]
        self.assertEqual(sim.size, (5, 1))
        self.assertTrue(sim.iscolor)
        self.assertEqual(sim.nplanes, 3)
        nt.assert_array_equal(sim.array.squeeze(), im.array[6, 0:5])

        sim = im[0:5, 9:]
        self.assertEqual(sim.size, (5, 1))
        self.assertTrue(sim.iscolor)
        self.assertEqual(sim.nplanes, 3)
        nt.assert_array_equal(sim.array.squeeze(), im.array[9, 0:5])

        sim = im[0:5, 6, 2]
        self.assertEqual(sim.size, (5, 1))
        self.assertFalse(sim.iscolor)
        nt.assert_array_equal(sim.array.squeeze(), im.array[6, 0:5, 2])

        sim = im[:, 5]
        self.assertEqual(sim.size, (8, 1))
        self.assertTrue(sim.iscolor)
        nt.assert_array_equal(sim.array.squeeze(), im.array[5, :, :])

        sim = im[0:5, 9:, 2:]
        self.assertEqual(sim.size, (5, 1))
        self.assertFalse(sim.iscolor)
        nt.assert_array_equal(sim.array.squeeze(), im.array[9, 0:5, 2])

        sim = im[0:5, 9:, 0:2]
        self.assertEqual(sim.size, (5, 1))
        self.assertTrue(sim.iscolor)
        self.assertEqual(sim.nplanes, 2)
        nt.assert_array_equal(sim.array.squeeze(), im.array[9, 0:5, 0:2])

        # single pixel
        pix = im[5, 6]
        self.assertIsInstance(pix, np.ndarray)
        self.assertEqual(pix.shape, (3,))
        nt.assert_array_equal(pix, im.array[6, 5, :])

        pix = im[5, 6, 2]
        self.assertIsInstance(pix, np.int64)
        nt.assert_array_equal(pix, im.array[6, 5, 2])

        sim = im[5, 6, 2:]
        self.assertEqual(sim.size, (1, 1))
        self.assertFalse(sim.iscolor)
        nt.assert_array_equal(sim.array.squeeze(), im.array[6, 5, 2:])

        sim = im[5, 6, 0:2]
        self.assertEqual(sim.size, (1, 1))
        self.assertTrue(sim.iscolor)
        self.assertEqual(sim.nplanes, 2)
        nt.assert_array_equal(sim.array.squeeze(), im.array[6, 5, 0:2])

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

        self.assertIs(im.array, x)

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

        self.assertIsNot(im.array, x)

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

        self.assertIsNot(im.array, x)
        nt.assert_almost_equal(im.array, x)

    def test_ndarray_float_size(self):
        x = np.arange(12.0)
        im = Image(x, size=(4, 3))  # rows x cols
        self.assertEqual(im.width, 4)
        self.assertEqual(im.height, 3)
        self.assertEqual(im.shape, (3, 4))
        self.assertEqual(im.npixels, 12)
        self.assertEqual(im.ndim, 2)
        self.assertEqual(im.nplanes, 1)
        self.assertTrue(im.isfloat)
        self.assertFalse(im.isint)
        self.assertIsNot(im.array, x)
        self.assertEqual(im[0, 0], 0.0)
        self.assertEqual(im[1, 0], 1.0)
        self.assertEqual(im[0, 1], 4.0)
        self.assertEqual(im[3, 2], 11.0)

        x = np.arange(12.0)
        im = Image(x.reshape(-1, 1), size=(4, 3))  # rows x cols
        self.assertEqual(im.width, 4)
        self.assertEqual(im.height, 3)
        self.assertEqual(im.shape, (3, 4))
        self.assertEqual(im.npixels, 12)
        self.assertEqual(im.ndim, 2)
        self.assertEqual(im.nplanes, 1)
        self.assertTrue(im.isfloat)
        self.assertFalse(im.isint)
        self.assertIsNot(im.array, x)
        self.assertEqual(im[0, 0], 0.0)
        self.assertEqual(im[1, 0], 1.0)
        self.assertEqual(im[0, 1], 4.0)
        self.assertEqual(im[3, 2], 11.0)

        x = np.arange(12.0)
        im = Image(x.reshape(1, -1), size=(4, 3))  # rows x cols
        self.assertEqual(im.width, 4)
        self.assertEqual(im.height, 3)
        self.assertEqual(im.shape, (3, 4))
        self.assertEqual(im.npixels, 12)
        self.assertEqual(im.ndim, 2)
        self.assertEqual(im.nplanes, 1)
        self.assertTrue(im.isfloat)
        self.assertFalse(im.isint)
        self.assertIsNot(im.array, x)
        self.assertEqual(im[0, 0], 0.0)
        self.assertEqual(im[1, 0], 1.0)
        self.assertEqual(im[0, 1], 4.0)
        self.assertEqual(im[3, 2], 11.0)

        im = Image(x, size=(3, 4))  # width x height

        self.assertEqual(im.width, 3)
        self.assertEqual(im.height, 4)
        self.assertEqual(im.shape, (4, 3))
        self.assertEqual(im.npixels, 12)
        self.assertEqual(im.ndim, 2)
        self.assertEqual(im.nplanes, 1)
        self.assertTrue(im.isfloat)
        self.assertFalse(im.isint)
        self.assertIsNot(im.array, x)
        self.assertEqual(im[0, 0], 0.0)
        self.assertEqual(im[1, 0], 1.0)
        self.assertEqual(im[0, 1], 3.0)
        self.assertEqual(im[2, 3], 11.0)

        im = Image.Random(size=(5, 6, 3))
        self.assertEqual(im.size, (5, 6))
        self.assertEqual(im.nplanes, 3)

        im = Image.Random(size=(5, 6), colororder="RGB")
        self.assertEqual(im.size, (5, 6))
        self.assertEqual(im.nplanes, 3)

        c = im.view1d()
        self.assertEqual(c.shape, (30, 3))

        im2 = Image(c, size=(5, 6, 3))
        self.assertEqual(im2.size, (5, 6))
        self.assertEqual(im2.nplanes, 3)

        im2 = Image(c, size=(5, 6))
        self.assertEqual(im2.size, (5, 6))
        self.assertEqual(im2.nplanes, 3)

        im2 = Image(c, size=im)
        self.assertEqual(im2.size, (5, 6))
        self.assertEqual(im2.nplanes, 3)

        r = np.arange(10, 16)
        g = np.arange(20, 26)
        b = np.arange(30, 36)

        x = np.stack((r, g, b), axis=0)
        im = Image(x, size=(2, 3), colororder="RGB")
        self.assertEqual(im[0][0, 0], 10)
        self.assertEqual(im[0][1, 2], 15)
        self.assertEqual(im[1][0, 0], 20)
        self.assertEqual(im[1][1, 2], 25)
        self.assertEqual(im[2][0, 0], 30)
        self.assertEqual(im[2][1, 2], 35)

        im = Image(x.T, size=(2, 3), colororder="RGB")
        self.assertEqual(im[0][0, 0], 10)
        self.assertEqual(im[0][1, 2], 15)
        self.assertEqual(im[1][0, 0], 20)
        self.assertEqual(im[1][1, 2], 25)
        self.assertEqual(im[2][0, 0], 30)
        self.assertEqual(im[2][1, 2], 35)

        x = x.T.reshape(-1)
        im = Image(x.T, size=(2, 3, 3), colororder="RGB")
        self.assertEqual(im[0][0, 0], 10)
        self.assertEqual(im[0][1, 2], 15)
        self.assertEqual(im[1][0, 0], 20)
        self.assertEqual(im[1][1, 2], 25)
        self.assertEqual(im[2][0, 0], 30)
        self.assertEqual(im[2][1, 2], 35)

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

        nt.assert_array_almost_equal(im.array, x)

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
        nt.assert_array_almost_equal(img.array * 1, im)

        x = img.array_as("uint8")
        self.assertEqual(x.dtype, np.uint8)
        nt.assert_array_almost_equal(x, im * 255)

        x = img.array_as("uint16")
        self.assertEqual(x.dtype, np.uint16)
        nt.assert_array_almost_equal(x, im * 65535)

        x = img.array_as("float32")
        self.assertEqual(x.dtype, np.float32)
        nt.assert_array_almost_equal(x, im)

    def test_colororder(self):
        x = np.arange(24).reshape((2, 4, 3))

        im = Image(x, dtype="uint8")
        for i in range(3):
            P = im.plane(i)
            self.assertEqual(P.shape, (2, 4))
            self.assertEqual(P.nplanes, 1)
            nt.assert_array_almost_equal(P.array, x[:, :, i])
            self.assertIs(P.colororder, None)

        for i, color in enumerate("RGB"):
            P = im.plane(color)
            self.assertEqual(P.shape, (2, 4))
            self.assertEqual(P.nplanes, 1)
            nt.assert_array_almost_equal(P.array, x[:, :, i])
            self.assertIs(P.colororder, None)

        P = im.red()
        self.assertEqual(P.shape, (2, 4))
        self.assertEqual(P.nplanes, 1)
        nt.assert_array_almost_equal(P.array, x[:, :, 0])
        self.assertIs(P.colororder, None)

        P = im.green()
        self.assertEqual(P.shape, (2, 4))
        self.assertEqual(P.nplanes, 1)
        nt.assert_array_almost_equal(P.array, x[:, :, 1])
        self.assertIs(P.colororder, None)

        P = im.blue()
        self.assertEqual(P.shape, (2, 4))
        self.assertEqual(P.nplanes, 1)
        nt.assert_array_almost_equal(P.array, x[:, :, 2])
        self.assertIs(P.colororder, None)

        P = im.plane(1)
        self.assertEqual(P.shape, (2, 4))
        self.assertEqual(P.nplanes, 1)
        nt.assert_array_almost_equal(P.array, x[:, :, 1])

        P = im.plane([1, 2])
        self.assertEqual(P.shape, (2, 4, 2))
        self.assertEqual(P.nplanes, 2)
        nt.assert_array_almost_equal(P.array, x[:, :, [1, 2]])
        self.assertEqual(P.colororder_str, "G:B")

        P = im.plane("GB")
        self.assertEqual(P.shape, (2, 4, 2))
        self.assertEqual(P.nplanes, 2)
        nt.assert_array_almost_equal(P.array, x[:, :, [1, 2]])
        self.assertEqual(P.colororder_str, "G:B")

        P = im.plane("G:B")
        self.assertEqual(P.shape, (2, 4, 2))
        self.assertEqual(P.nplanes, 2)
        nt.assert_array_almost_equal(P.array, x[:, :, [1, 2]])
        self.assertEqual(P.colororder_str, "G:B")

    def test_arith_float(self):
        x = np.arange(6).reshape((2, 3))
        imx = Image(x, dtype="float32")
        y = np.arange(6, 12).reshape((2, 3))
        imy = Image(y, dtype="float32")

        nt.assert_array_almost_equal((imx + imy).array, x + y)
        nt.assert_array_almost_equal((imx + 2).array, x + 2)
        nt.assert_array_almost_equal((2 + imy).array, 2 + y)

        nt.assert_array_almost_equal((imx - imy).array, x - y)
        nt.assert_array_almost_equal((imx - 2).array, x - 2)
        nt.assert_array_almost_equal((2 - imy).array, 2 - y)

        nt.assert_array_almost_equal((imx * imy).array, x * y)
        nt.assert_array_almost_equal((imx * 2).array, x * 2)
        nt.assert_array_almost_equal((2 * imy).array, 2 * y)

        nt.assert_array_almost_equal((imx / imy).array, x / y)
        nt.assert_array_almost_equal((imx / 2).array, x / 2)
        nt.assert_array_almost_equal((2 / imy).array, 2 / y)

        nt.assert_array_almost_equal((imx**2).array, x**2)

    def test_logical_float(self):
        x = np.arange(6).reshape((2, 3))
        imx = Image(x, dtype="float32")
        y = np.arange(6, 12).reshape((2, 3))
        imy = Image(y, dtype="float32")
        nt.assert_array_almost_equal(
            (imx == imx).array, np.ones(imx.shape, dtype="float32")
        )
        nt.assert_array_almost_equal(
            (imx != imy).array, np.ones(imx.shape, dtype="float32")
        )
        nt.assert_array_almost_equal(
            (imx < imy).array, np.ones(imx.shape, dtype="float32")
        )
        nt.assert_array_almost_equal(
            (imy > imx).array, np.ones(imx.shape, dtype="float32")
        )

        imx = Image(np.array([[1, 2], [3, 4]]), dtype="float32")
        imy = Image(np.array([[1, 3], [2, 5]]), dtype="float32")
        T = 1
        F = 0
        nt.assert_array_almost_equal((imx == imy).array, np.array([[T, F], [F, F]]))
        nt.assert_array_almost_equal((imx == 2).array, np.array([[F, T], [F, F]]))
        nt.assert_array_almost_equal((2 == imy).array, np.array([[F, F], [T, F]]))

        nt.assert_array_almost_equal((imx != imy).array, np.array([[F, T], [T, T]]))
        nt.assert_array_almost_equal((imx != 2).array, np.array([[T, F], [T, T]]))
        nt.assert_array_almost_equal((2 != imy).array, np.array([[T, T], [F, T]]))

        nt.assert_array_almost_equal((imx > imy).array, np.array([[F, F], [T, F]]))
        nt.assert_array_almost_equal((imx > 2).array, np.array([[F, F], [T, T]]))
        nt.assert_array_almost_equal((2 > imy).array, np.array([[T, F], [F, F]]))

        nt.assert_array_almost_equal((imx >= imy).array, np.array([[T, F], [T, F]]))
        nt.assert_array_almost_equal((imx >= 2).array, np.array([[F, T], [T, T]]))
        nt.assert_array_almost_equal((2 >= imy).array, np.array([[T, F], [T, F]]))

        nt.assert_array_almost_equal((imx < imy).array, np.array([[F, T], [F, T]]))
        nt.assert_array_almost_equal((imx < 2).array, np.array([[T, F], [F, F]]))
        nt.assert_array_almost_equal((2 < imy).array, np.array([[F, T], [F, T]]))

        nt.assert_array_almost_equal((imx <= imy).array, np.array([[T, T], [F, T]]))
        nt.assert_array_almost_equal((imx <= 2).array, np.array([[T, T], [F, F]]))
        nt.assert_array_almost_equal((2 <= imy).array, np.array([[F, T], [T, T]]))

        x = imx <= imy
        nt.assert_array_almost_equal((~x).array, (imx > imy).array)

    def test_arith_int(self):
        x = np.arange(6, dtype="uint8").reshape((2, 3))
        imx = Image(x, dtype="uint8")
        y = np.arange(6, 12, dtype="uint8").reshape((2, 3))
        imy = Image(y, dtype="uint8")

        nt.assert_array_almost_equal((imx + imy).array, x + y)
        nt.assert_array_almost_equal((imx + 2).array, x + 2)
        nt.assert_array_almost_equal((2 + imy).array, 2 + y)

        nt.assert_array_almost_equal((imx - imy).array, x - y)
        nt.assert_array_almost_equal((imx - 2).array, x - 2)
        nt.assert_array_almost_equal((2 - imy).array, 2 - y)

        nt.assert_array_almost_equal((imx * imy).array, x * y)
        nt.assert_array_almost_equal((imx * 2).array, x * 2)
        nt.assert_array_almost_equal((2 * imy).array, 2 * y)

        nt.assert_array_almost_equal((imx / imy).array, x / y)
        nt.assert_array_almost_equal((imx / 2).array, x / 2)
        nt.assert_array_almost_equal((2 / imy).array, 2 / y)

        nt.assert_array_almost_equal((imx**2).array, x**2)

    def test_logical_int(self):
        x = np.arange(6).reshape((2, 3))
        imx = Image(x, dtype="uint8")
        y = np.arange(6, 12).reshape((2, 3))
        imy = Image(y, dtype="uint8")
        nt.assert_array_almost_equal(
            (imx == imx).array, np.ones(imx.shape, dtype="bool")
        )
        nt.assert_array_almost_equal(
            (imx != imy).array, np.ones(imx.shape, dtype="bool")
        )
        nt.assert_array_almost_equal(
            (imx < imy).array, np.ones(imx.shape, dtype="bool")
        )
        nt.assert_array_almost_equal(
            (imy > imx).array, np.ones(imx.shape, dtype="bool")
        )

        imx = Image([[1, 2], [3, 4]], dtype="uint8")
        imy = Image([[1, 3], [2, 5]], dtype="uint8")
        T = True
        F = False
        nt.assert_array_almost_equal((imx == imy).array, np.array([[T, F], [F, F]]))
        nt.assert_array_almost_equal((imx == 2).array, np.array([[F, T], [F, F]]))
        nt.assert_array_almost_equal((2 == imy).array, np.array([[F, F], [T, F]]))

        nt.assert_array_almost_equal((imx != imy).array, np.array([[F, T], [T, T]]))
        nt.assert_array_almost_equal((imx != 2).array, np.array([[T, F], [T, T]]))
        nt.assert_array_almost_equal((2 != imy).array, np.array([[T, T], [F, T]]))

        nt.assert_array_almost_equal((imx > imy).array, np.array([[F, F], [T, F]]))
        nt.assert_array_almost_equal((imx > 2).array, np.array([[F, F], [T, T]]))
        nt.assert_array_almost_equal((2 > imy).array, np.array([[T, F], [F, F]]))

        nt.assert_array_almost_equal((imx >= imy).array, np.array([[T, F], [T, F]]))
        nt.assert_array_almost_equal((imx >= 2).array, np.array([[F, T], [T, T]]))
        nt.assert_array_almost_equal((2 >= imy).array, np.array([[T, F], [T, F]]))

        nt.assert_array_almost_equal((imx < imy).array, np.array([[F, T], [F, T]]))
        nt.assert_array_almost_equal((imx < 2).array, np.array([[T, F], [F, F]]))
        nt.assert_array_almost_equal((2 < imy).array, np.array([[F, T], [F, T]]))

        nt.assert_array_almost_equal((imx <= imy).array, np.array([[T, T], [F, T]]))
        nt.assert_array_almost_equal((imx <= 2).array, np.array([[T, T], [F, F]]))
        nt.assert_array_almost_equal((2 <= imy).array, np.array([[F, T], [T, T]]))

        x = imx <= imy
        nt.assert_array_almost_equal((~x).array, (imx > imy).array)

    def test_types(self):
        imx = Image(np.ones((2, 3), dtype="float32"))
        self.assertEqual(imx.dtype, np.dtype("float32"))
        self.assertFalse(imx.isint)
        self.assertTrue(imx.isfloat)
        nt.assert_array_almost_equal(imx.array, np.ones((2, 3)))

        imx = Image(np.ones((2, 3), dtype="float64"))
        self.assertEqual(imx.dtype, np.dtype("float32"))
        self.assertFalse(imx.isint)
        self.assertTrue(imx.isfloat)
        nt.assert_array_almost_equal(imx.array, np.ones((2, 3)))

        imx = Image(np.ones((2, 3), dtype="float64"), dtype=True)
        self.assertEqual(imx.dtype, np.dtype("float64"))
        self.assertFalse(imx.isint)
        self.assertTrue(imx.isfloat)
        nt.assert_array_almost_equal(imx.array, np.ones((2, 3)))

        imx = Image(np.ones((2, 3)), dtype="float64")
        self.assertEqual(imx.dtype, np.dtype("float64"))
        self.assertFalse(imx.isint)
        self.assertTrue(imx.isfloat)
        nt.assert_array_almost_equal(imx.array, np.ones((2, 3)))

        imx = Image(np.ones((2, 3)), dtype="float64")
        self.assertEqual(imx.dtype, np.dtype("float64"))
        self.assertFalse(imx.isint)
        self.assertTrue(imx.isfloat)
        nt.assert_array_almost_equal(imx.array, np.ones((2, 3)))

        imx = Image(np.ones((2, 3), dtype="int16"))
        self.assertEqual(imx.dtype, np.dtype("uint8"))
        self.assertTrue(imx.isint)
        self.assertFalse(imx.isfloat)
        nt.assert_array_almost_equal(imx.array, np.ones((2, 3)))

        imx = Image(np.ones((2, 3), dtype="int16"), dtype=True)
        self.assertEqual(imx.dtype, np.dtype("int16"))
        self.assertTrue(imx.isint)
        self.assertFalse(imx.isfloat)
        nt.assert_array_almost_equal(imx.array, np.ones((2, 3)))

        imx = Image(np.ones((2, 3)), dtype="int16")
        self.assertEqual(imx.dtype, np.dtype("int16"))
        self.assertTrue(imx.isint)
        self.assertFalse(imx.isfloat)
        nt.assert_array_almost_equal(imx.array, np.ones((2, 3)))

        imx = Image(np.ones((2, 3), dtype="uint8"))
        self.assertEqual(imx.dtype, np.dtype("uint8"))
        self.assertTrue(imx.isint)
        self.assertFalse(imx.isfloat)
        nt.assert_array_almost_equal(imx.array, np.ones((2, 3)))

        #
        imx = Image(np.ones((2, 3), dtype="int64"), dtype="uint8")
        self.assertEqual(imx.dtype, np.dtype("uint8"))
        self.assertTrue(imx.isint)
        self.assertFalse(imx.isfloat)
        nt.assert_array_almost_equal(imx.array, np.ones((2, 3)))

        imx = Image(np.ones((2, 3), dtype="float64"), dtype="uint8")
        self.assertEqual(imx.dtype, np.dtype("uint8"))
        self.assertTrue(imx.isint)
        self.assertFalse(imx.isfloat)
        nt.assert_array_almost_equal(imx.array, np.ones((2, 3)))

        imx = Image(np.ones((2, 3), dtype="uint8"), dtype="uint8")
        self.assertEqual(imx.dtype, np.dtype("uint8"))
        self.assertTrue(imx.isint)
        self.assertFalse(imx.isfloat)
        nt.assert_array_almost_equal(imx.array, np.ones((2, 3)))

        imx = Image(np.ones((2, 3), dtype="uint8"), dtype="float32")
        self.assertEqual(imx.dtype, np.dtype("float32"))
        self.assertFalse(imx.isint)
        self.assertTrue(imx.isfloat)
        nt.assert_array_almost_equal(imx.array, np.ones((2, 3)))

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
        self.assertEqual(z.array[0, 0], 257)

        z = img.to("float32")
        self.assertEqual(z.dtype, np.float32)
        self.assertAlmostEqual(z.array[0, 0], 1 / 255.0)

        im = np.array([[0.1, 0.3], [0.3, 0.4]])
        img = Image(im, dtype="float64")
        z = img.to("uint8")
        self.assertEqual(z.dtype, np.uint8)
        self.assertEqual(z.array[0, 0], round(0.1 * 255.0))

        z = img.to("uint16")
        self.assertEqual(z.dtype, np.uint16)
        self.assertEqual(z.array[0, 0], round(0.1 * 65535.0))

        z = img.to("float32")
        self.assertEqual(z.dtype, np.float32)
        self.assertAlmostEqual(z.array[0, 0], 0.1)

    def test_arith_color(self):
        pass

    def test_inplace_arith(self):
        x = np.array([[1, 2], [3, 4]], dtype="float32")
        imx = Image(x.copy())

        imx += 1
        nt.assert_array_almost_equal(imx.array, x + 1)

        imx = Image(x.copy())
        imx -= 1
        nt.assert_array_almost_equal(imx.array, x - 1)

        imx = Image(x.copy())
        imx *= 2
        nt.assert_array_almost_equal(imx.array, x * 2)

        imx = Image(x.copy())
        imx /= 2
        nt.assert_array_almost_equal(imx.array, x / 2)

        imx = Image(x.copy())
        imx //= 2
        nt.assert_array_almost_equal(imx.array, x // 2)

    def test_floordiv(self):
        x = np.array([[5, 10], [15, 20]], dtype="float32")
        imx = Image(x)

        nt.assert_array_almost_equal((imx // 3).array, x // 3)
        nt.assert_array_almost_equal((12 // imx).array, 12 // x)

        xi = np.array([[5, 10], [15, 20]], dtype="uint8")
        imxi = Image(xi)
        nt.assert_array_almost_equal((imxi // 3).array, xi // 3)

    def test_unary_neg(self):
        x = np.array([[1, -2], [-3, 4]], dtype="int8")
        imx = Image(x)
        nt.assert_array_almost_equal((-imx).array, -x)

    def test_mod_plane_stack(self):
        a = Image([[1, 2], [3, 4]])
        b = Image([[5, 6], [7, 8]])
        z = a % b
        self.assertEqual(z.nplanes, 2)
        nt.assert_array_equal(z.plane(0).array, a.array)
        nt.assert_array_equal(z.plane(1).array, b.array)

        # scalar appended as a plane
        z = a % 0
        self.assertEqual(z.nplanes, 2)
        nt.assert_array_equal(z.plane(0).array, a.array)
        nt.assert_array_equal(z.plane(1).array, np.zeros_like(a.array))

    def test_imod_plane_stack(self):
        a = Image([[1, 2], [3, 4]])
        b = Image([[5, 6], [7, 8]])
        a_orig = a.array.copy()
        a %= b
        self.assertEqual(a.nplanes, 2)
        nt.assert_array_equal(a.plane(0).array, a_orig)
        nt.assert_array_equal(a.plane(1).array, b.array)

        # scalar appended as a plane
        a = Image([[1, 2], [3, 4]])
        a_orig = a.array.copy()
        a %= 0
        self.assertEqual(a.nplanes, 2)
        nt.assert_array_equal(a.plane(0).array, a_orig)
        nt.assert_array_equal(a.plane(1).array, np.zeros_like(a_orig))

    def test_bitwise(self):
        x = np.array([[0b0011, 0b0101], [0b1010, 0b1100]], dtype="uint8")
        y = np.array([[0b0110, 0b0011], [0b1001, 0b0101]], dtype="uint8")
        imx = Image(x)
        imy = Image(y)

        nt.assert_array_equal((imx & imy).array, x & y)
        nt.assert_array_equal((imx & 0b0101).array, x & 0b0101)

        nt.assert_array_equal((imx | imy).array, x | y)
        nt.assert_array_equal((imx | 0b0101).array, x | 0b0101)

        nt.assert_array_equal((imx ^ imy).array, x ^ y)
        nt.assert_array_equal((imx ^ 0b0101).array, x ^ 0b0101)

        nt.assert_array_equal((imx << 1).array, x << 1)
        nt.assert_array_equal((imx >> 1).array, x >> 1)

    def test_bitwise_inplace(self):
        x = np.array([[0b0011, 0b0101], [0b1010, 0b1100]], dtype="uint8")
        y = np.array([[0b0110, 0b0011], [0b1001, 0b0101]], dtype="uint8")

        imx = Image(x.copy())
        imx &= Image(y)
        nt.assert_array_equal(imx.array, x & y)

        imx = Image(x.copy())
        imx |= Image(y)
        nt.assert_array_equal(imx.array, x | y)

        imx = Image(x.copy())
        imx ^= Image(y)
        nt.assert_array_equal(imx.array, x ^ y)

        imx = Image(x.copy())
        imx <<= 1
        nt.assert_array_equal(imx.array, x << 1)

        imx = Image(x.copy())
        imx >>= 1
        nt.assert_array_equal(imx.array, x >> 1)

    def test_bad_values(self):
        img = Image([[1, 2, np.nan], [4, 5, 6], [-np.inf, 8, np.inf]])
        self.assertEqual(img.numnan, 1)
        self.assertEqual(img.numinf, 2)

        im2 = img.fixbad(nan=0, posinf=99, neginf=-99)
        self.assertEqual(im2[2, 0], 0)
        self.assertEqual(im2[0, 2], -99)
        self.assertEqual(im2[2, 2], 99)

    def test_pixels_mask(self):
        x = np.arange(24)
        im = Image(x, size=(4, 6))  # width x height
        self.assertEqual(im.size, (4, 6))

        mask = Image.String(
            r"""
            ....
            ....
            .##.
            .##.
            .##.
            ...."""
        )
        self.assertEqual(mask.size, (4, 6))
        pix = im.pixels_mask(mask)
        self.assertEqual(pix.shape, (6,))
        nt.assert_array_equal(pix, [9, 10, 13, 14, 17, 18])

        x = np.arange(24 * 3)
        im = Image(x, size=(4, 6, 3))  # width x height
        self.assertEqual(im.size, (4, 6))
        pix = im.pixels_mask(mask)
        self.assertEqual(pix.shape, (3, 6))
        expected = np.array(
            [
                [27, 30, 39, 42, 51, 54],
                [28, 31, 40, 43, 52, 55],
                [29, 32, 41, 44, 53, 56],
            ]
        )
        nt.assert_array_equal(pix, expected)

        poly = Polygon2([(1, 2), (2, 2), (2, 4), (1, 4)])
        x = np.arange(24)
        im = Image(x, size=(4, 6))  # width x height
        pix = im.pixels_mask(mask)
        self.assertEqual(pix.shape, (6,))
        nt.assert_array_equal(pix, [9, 10, 13, 14, 17, 18])

    # new test
    def test_copy(self):
        im = Image.Random(size=(10, 10))
        im_copy = im.copy()
        self.assertIsNot(im.array, im_copy.array)
        nt.assert_array_equal(im.array, im_copy.array)
        # Modify copy and ensure original unchanged
        im_copy.array[0, 0] = 255
        self.assertNotEqual(im[0, 0], im_copy[0, 0])

    # new test
    def test_contains(self):
        im = Image.Zeros(size=(10, 10))
        # Test point inside
        self.assertTrue(im.contains((5, 5)))
        # Test point on boundary
        self.assertTrue(im.contains((0, 0)))
        self.assertTrue(im.contains((9, 9)))
        # Test point outside
        self.assertFalse(im.contains((10, 10)))
        self.assertFalse(im.contains((-1, 5)))

    # new test
    def test_shape_property(self):
        im = Image.Random(size=(10, 15))
        self.assertEqual(im.shape, (15, 10))
        im_color = Image.Random(size=(10, 15), colororder="RGB")
        self.assertEqual(im_color.shape, (15, 10, 3))

    # new test
    def test_size_property(self):
        im = Image.Random(size=(10, 15))
        self.assertEqual(im.size, (10, 15))

    # new test
    def test_center_property(self):
        im = Image.Random(size=(10, 15))
        self.assertEqual(im.center, (5, 7.5))

    # new test
    def test_width_height_properties(self):
        im = Image.Random(size=(10, 15))
        self.assertEqual(im.width, 10)
        self.assertEqual(im.height, 15)

    # new test
    def test_npixels_property(self):
        im = Image.Random(size=(10, 15))
        self.assertEqual(im.npixels, 150)

    # new test
    def test_nplanes_property(self):
        im_grey = Image.Random(size=(10, 10))
        self.assertEqual(im_grey.nplanes, 1)
        im_rgb = Image.Random(size=(10, 10), colororder="RGB")
        self.assertEqual(im_rgb.nplanes, 3)

    def test_planes_iterator(self):
        x = np.arange(24).reshape((2, 4, 3))
        im = Image(x, size=(4, 2), colororder="RGB")

        planes = list(im.planes())
        self.assertEqual(len(planes), 3)
        for i, plane in enumerate(im.planes()):
            self.assertIsInstance(plane, Image)
            self.assertEqual(plane.shape, (2, 4))
            self.assertEqual(plane.nplanes, 1)
            nt.assert_array_equal(planes[i].array, x[:, :, i])

    def test_init_input_check(self):
        # 1D array without size
        x = np.zeros((10,))
        with self.assertRaises(ValueError):
            y = Image(x)

        # 4D array without size
        x = np.zeros((10, 10, 3, 2))
        with self.assertRaises(ValueError):
            y = Image(x)

        # incorrect argument type
        with self.assertRaises(ValueError):
            y = Image("image")

    def test_init_singleton_removal(self):
        x = np.zeros((5, 10, 1))
        im = Image(x)
        self.assertEqual(im.shape, (5, 10))
        self.assertEqual(im.nplanes, 1)


# ------------------------------------------------------------------------ #
if __name__ == "__main__":
    unittest.main()
