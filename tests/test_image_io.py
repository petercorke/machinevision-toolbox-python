#!/usr/bin/env python

# test for Image input/output

import contextlib
import io
import os
import unittest
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.testing as nt

# import machinevisiontoolbox as mvt
from machinevisiontoolbox import Image
from machinevisiontoolbox.base import iread


class TestImage(unittest.TestCase):
    def test_isimage(self):
        # create mini image (Bayer pattern)
        im = np.zeros((2, 2, 3))
        # 0 - red channel, 1 - green channel, 2 - blue channel
        im[0, 0, 0] = 1  # top left = red
        im[0, 1, 1] = 1  # top right = green
        im[1, 0, 1] = 1  # bottom left = green
        im[1, 1, 2] = 1  # bottom right = blue

        # a single grayscale image
        img = Image(im)
        self.assertIsInstance(img, Image)
        self.assertEqual(img.shape, im.shape)
        self.assertEqual(img.dtype, np.float32)

        # set type as float, then make sure isimage is true
        img = Image(im.astype(np.float32))
        self.assertIsInstance(img, Image)
        self.assertEqual(img.shape, im.shape)
        self.assertEqual(img.dtype, np.float32)

        img = Image(im.astype(np.uint8))
        self.assertIsInstance(img, Image)
        self.assertEqual(img.shape, im.shape)
        self.assertEqual(img.dtype, np.uint8)

    def test_str(self):
        # single color image as str
        # print('test_str')
        imname = "monalisa.png"

        im = Image.Read(imname)
        # check attributes
        nt.assert_array_equal(im.shape, (700, 677, 3))
        self.assertEqual(os.path.split(im.name)[1], imname)
        self.assertEqual(im.iscolor, True)
        self.assertEqual(im.dtype, "uint8")
        self.assertEqual(im.width, 677)
        self.assertEqual(im.height, 700)
        self.assertEqual(im.ndim, 3)
        self.assertEqual(im.colororder_str, "R:G:B")
        self.assertEqual(im.nplanes, 3)

    def test_image(self):
        # Image object
        # print('test_image')
        imname = "shark1.png"
        im0 = Image.Read(imname)

        im1 = Image(im0)
        # TODO consider __eq__ to compare Image objects directly im0 == im1
        nt.assert_array_almost_equal(im1.A, im0.A)
        self.assertEqual(im1.shape, im0.shape)
        self.assertEqual(im1.iscolor, im0.iscolor)
        # ... for the rest of the attributes

    def test_array(self):
        # test single numpy array
        # print('test_numpyarray')
        imarray = iread("walls-l.png")

        im = Image(imarray[0])
        self.assertEqual(im.shape, (2448, 3264, 3))
        self.assertEqual(im.iscolor, True)

    def test_options(self):
        imname = "monalisa.png"
        im = Image.Read(imname)

        # check predicatives
        self.assertFalse(im.isfloat)
        self.assertTrue(im.isint)
        self.assertIsInstance(im, Image)
        self.assertEqual(im.bgr.shape, im.shape)
        self.assertEqual(im.rgb.shape, im.shape)
        self.assertEqual(im.size, (677, 700))

        # check one element for rgb vs bgr ordering
        v = round(im.shape[0] / 2)  # rows
        u = round(im.shape[1] / 2)  # cols
        bgr = im.bgr[v, u, :]
        nt.assert_array_equal(im.rgb[v, u, :], bgr[::-1])

        self.assertTrue(im.isrgb)
        self.assertFalse(im.isbgr)

        self.assertTrue(im.iscolor)

    def test_im_from_string(self):
        img = Image.String("01234|56789|87654")
        nt.assert_array_equal(
            img.A, np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [8, 7, 6, 5, 4]])
        )

    def test_print(self):

        img = Image.String("01234|56789|87654")

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            img.print(header=False)
        self.assertEqual(f.getvalue(), "   0 1 2 3 4\n   5 6 7 8 9\n   8 7 6 5 4\n")

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            img.print(header=True)
        self.assertEqual(
            f.getvalue(),
            "Image: 5 x 3 (uint8), 1 anonymous plane\n  span=[0, 9]; mean=5, 𝜎=2.58199; median=5\n   0 1 2 3 4\n   5 6 7 8 9\n   8 7 6 5 4\n",
        )

    def test_im_to_string(self):
        img = Image.String("01234|56789|87654")

        s = Image.strhcat(img)
        self.assertEqual(
            s,
            "\n      0 1 2 3 4  \n      - - - - -  \n  0:  0 1 2 3 4\n  1:  5 6 7 8 9\n  2:  8 7 6 5 4\n",
        )

        s = Image.strhcat(img, widths=3)
        self.assertEqual(
            s,
            "\n        0   1   2   3   4  \n        -   -   -   -   -  \n  0:    0   1   2   3   4\n  1:    5   6   7   8   9\n  2:    8   7   6   5   4\n",
        )

        s = Image.strhcat(img, img)
        self.assertEqual(
            s,
            "\n      0 1 2 3 4   0 1 2 3 4  \n      - - - - -   - - - - -  \n  0:  0 1 2 3 4 0 1 2 3 4\n  1:  5 6 7 8 9 5 6 7 8 9\n  2:  8 7 6 5 4 8 7 6 5 4\n",
        )

        s = Image.strhcat(img, img, labels=["a", "b"])
        self.assertEqual(
            s,
            "     a           b\n      0 1 2 3 4   0 1 2 3 4  \n      - - - - -   - - - - -  \n  0:  0 1 2 3 4 0 1 2 3 4\n  1:  5 6 7 8 9 5 6 7 8 9\n  2:  8 7 6 5 4 8 7 6 5 4\n",
        )

    # TODO unit tests:
    # test_isimage - make sure Image rejects/fails with invalid input
    # test_imtypes - test Image works on different Image types?
    # test_getimage - make sure Image returns the same array but with valid
    # typing?
    # test_imwrite - test write/successfully save file?

    # new test
    def test_read(self):
        """Test reading image from file"""
        # TODO: Test with actual image file
        # im = Image.Read('flowers1.png')
        # self.assertIsNotNone(im)
        # self.assertGreater(im.width, 0)
        # self.assertGreater(im.height, 0)
        pass

    # new test
    def test_anaglyph(self):
        """Test anaglyph stereo image creation"""
        im_left = Image.Random(size=(100, 100), colororder="RGB")
        im_right = Image.Random(size=(100, 100), colororder="RGB")
        anaglyph = im_left.anaglyph(im_right)
        self.assertEqual(anaglyph.size, im_left.size)
        self.assertEqual(anaglyph.nplanes, 3)

    # new test
    def test_disp(self):
        """Test image display"""
        im = Image.Random(size=(50, 50), dtype="uint8")
        # TODO: Test display (may require mock or headless mode)
        # im.disp(block=False)
        pass

    # new test
    def test_metadata(self):
        """Test reading image metadata"""
        # TODO: Test with image file that has metadata
        # im = Image.Read('test_image.jpg')
        # metadata = im.metadata()
        # self.assertIsNotNone(metadata)
        pass

    # new test
    def test_showpixels(self):
        """Test showing pixel values overlay"""
        im = Image.String("123|456|789")
        # TODO: Test pixel value display
        # im.showpixels()
        pass

    # new test
    def test_stdisp(self):
        """Test stereo image pair display"""
        im_left = Image.Random(size=(50, 50), dtype="uint8")
        im_right = Image.Random(size=(50, 50), dtype="uint8")
        # TODO: Test stereo display
        # im_left.stdisp(im_right)
        pass

    # new test
    def test_write(self):
        """Test writing image to file"""
        im = Image.Random(size=(50, 50), dtype="uint8")
        # TODO: Test writing to temp file and reading back
        # import tempfile
        # with tempfile.NamedTemporaryFile(suffix='.png') as f:
        #     im.write(f.name)
        #     im2 = Image.Read(f.name)
        #     nt.assert_array_equal(im.A, im2.A)
        pass

    def tearDown(self):
        # Cleanup code if needed
        pass


# ------------------------------------------------------------------------ #

# Stripe fixtures are in tests/data/ (generated by examples/stripe.py).
# See tests/base/test_io.py for the fixture layout description.

_STRIPE_DATA = Path(__file__).parent / "data"
_STRIPES_RGB_FILE = _STRIPE_DATA / "stripes.png"
_STRIPES_RGBA_FILE = _STRIPE_DATA / "stripes_a.png"

_STRIPE_CENTRES = [
    (100, [0, 0, 0]),  # black
    (300, [255, 0, 0]),  # red
    (500, [0, 255, 0]),  # green
    (700, [0, 0, 255]),  # blue
    (900, [255, 255, 255]),  # white
]
_V = 200  # sample row


class TestStripeImageIO(unittest.TestCase):
    """Tests for Image.Read / Image.disp using pre-committed stripe fixtures."""

    @classmethod
    def setUpClass(cls):
        import matplotlib

        matplotlib.use("Agg")

    def tearDown(self):
        import matplotlib.pyplot as plt

        plt.close("all")

    # ------------------------------------------------------------------
    # helper
    # ------------------------------------------------------------------

    def _disp_pixel(self, h, v, u):
        """Return displayed pixel at (v, u) from an AxesImage handle."""
        return np.asarray(h.get_array())[v, u]

    # ------------------------------------------------------------------
    # Image.Read – array values, colororder_str, colororder dict
    # ------------------------------------------------------------------

    def test_read_rgb(self):
        """Image.Read default: RGB array, correct colororder."""
        im = Image.Read(str(_STRIPES_RGB_FILE))
        self.assertEqual(im.shape, (400, 1000, 3))
        self.assertEqual(im.dtype, np.uint8)
        self.assertEqual(im.colororder_str, "R:G:B")
        self.assertEqual(im.colororder, {"R": 0, "G": 1, "B": 2})
        for u, rgb in _STRIPE_CENTRES:
            nt.assert_array_equal(
                im.array[_V, u], rgb, err_msg=f"RGB mismatch at u={u}"
            )
            self.assertEqual(im.array[_V, u, im.colororder["R"]], rgb[0])

    def test_read_bgr(self):
        """Image.Read rgb=False: BGR array, colororder B:G:R."""
        im = Image.Read(str(_STRIPES_RGB_FILE), rgb=False)
        self.assertEqual(im.shape, (400, 1000, 3))
        self.assertEqual(im.colororder_str, "B:G:R")
        self.assertEqual(im.colororder, {"B": 0, "G": 1, "R": 2})
        for u, rgb in _STRIPE_CENTRES:
            bgr = [rgb[2], rgb[1], rgb[0]]
            nt.assert_array_equal(
                im.array[_V, u], bgr, err_msg=f"BGR mismatch at u={u}"
            )
            # Accessing the R plane via the dict always gives the R value
            self.assertEqual(im.array[_V, u, im.colororder["R"]], rgb[0])

    def test_read_rgba_file_default(self):
        """Image.Read RGBA file with default alpha=False: alpha stripped, R:G:B."""
        im = Image.Read(str(_STRIPES_RGBA_FILE))
        self.assertEqual(im.shape, (400, 1000, 3))
        self.assertEqual(im.colororder_str, "R:G:B")
        self.assertEqual(im.colororder, {"R": 0, "G": 1, "B": 2})
        for u, rgb in _STRIPE_CENTRES:
            nt.assert_array_equal(
                im.array[_V, u], rgb, err_msg=f"RGBA→RGB mismatch at u={u}"
            )

    def test_read_rgba(self):
        """Image.Read RGBA file with alpha=True: RGBA array, R:G:B:A."""
        im = Image.Read(str(_STRIPES_RGBA_FILE), alpha=True)
        self.assertEqual(im.shape, (400, 1000, 4))
        self.assertEqual(im.colororder_str, "R:G:B:A")
        self.assertEqual(im.colororder, {"R": 0, "G": 1, "B": 2, "A": 3})
        for u, rgb in _STRIPE_CENTRES:
            nt.assert_array_equal(
                im.array[_V, u], rgb + [255], err_msg=f"RGBA mismatch at u={u}"
            )

    def test_read_bgra(self):
        """Image.Read RGBA file with rgb=False, alpha=True: BGRA, B:G:R:A.
        The underlying array stays in OpenCV native BGRA order; the colororder
        dict maps each plane name to the correct index."""
        im = Image.Read(str(_STRIPES_RGBA_FILE), rgb=False, alpha=True)
        self.assertEqual(im.shape, (400, 1000, 4))
        self.assertEqual(im.colororder_str, "B:G:R:A")
        self.assertEqual(im.colororder, {"B": 0, "G": 1, "R": 2, "A": 3})
        for u, rgb in _STRIPE_CENTRES:
            bgra = [rgb[2], rgb[1], rgb[0], 255]
            nt.assert_array_equal(
                im.array[_V, u], bgra, err_msg=f"BGRA mismatch at u={u}"
            )
            self.assertEqual(im.array[_V, u, im.colororder["R"]], rgb[0])

    def test_read_mono(self):
        """Image.Read mono=True: 2-D greyscale array, no colororder."""
        im = Image.Read(str(_STRIPES_RGB_FILE), mono=True)
        self.assertEqual(im.array.ndim, 2)
        self.assertIsNone(im.colororder_str)
        self.assertIsNone(im.colororder)
        self.assertEqual(int(im.array[_V, 100]), 0)  # black → 0
        self.assertEqual(int(im.array[_V, 900]), 255)  # white → 255

    # ------------------------------------------------------------------
    # Image.disp – all formats must display with correct RGB colour
    # ------------------------------------------------------------------

    def test_disp_rgb(self):
        """Image.disp on RGB image: displayed pixels in RGB order."""
        im = Image.Read(str(_STRIPES_RGB_FILE))
        h = im.disp(block=None)
        for u, rgb in _STRIPE_CENTRES:
            nt.assert_array_equal(
                self._disp_pixel(h, _V, u), rgb, err_msg=f"disp RGB mismatch at u={u}"
            )

    def test_disp_bgr(self):
        """Image.disp on BGR image: idisp reorders to RGB for display."""
        im = Image.Read(str(_STRIPES_RGB_FILE), rgb=False)
        h = im.disp(block=None)
        for u, rgb in _STRIPE_CENTRES:
            nt.assert_array_equal(
                self._disp_pixel(h, _V, u),
                rgb,
                err_msg=f"disp BGR→RGB mismatch at u={u}",
            )

    def test_disp_rgba(self):
        """Image.disp on RGBA image: displayed pixels include alpha."""
        im = Image.Read(str(_STRIPES_RGBA_FILE), alpha=True)
        h = im.disp(block=None)
        for u, rgb in _STRIPE_CENTRES:
            nt.assert_array_equal(
                self._disp_pixel(h, _V, u),
                rgb + [255],
                err_msg=f"disp RGBA mismatch at u={u}",
            )

    def test_disp_bgra(self):
        """Image.disp on BGRA image: idisp reorders to RGBA for display."""
        im = Image.Read(str(_STRIPES_RGBA_FILE), rgb=False, alpha=True)
        h = im.disp(block=None)
        for u, rgb in _STRIPE_CENTRES:
            nt.assert_array_equal(
                self._disp_pixel(h, _V, u),
                rgb + [255],
                err_msg=f"disp BGRA→RGBA mismatch at u={u}",
            )

    def test_disp_mono(self):
        """Image.disp on greyscale image: black=0 and white=255 are preserved."""
        im = Image.Read(str(_STRIPES_RGB_FILE), mono=True)
        h = im.disp(block=None)
        self.assertEqual(int(self._disp_pixel(h, _V, 100)), 0)  # black
        self.assertEqual(int(self._disp_pixel(h, _V, 900)), 255)  # white


# ------------------------------------------------------------------------ #
if __name__ == "__main__":
    unittest.main()
