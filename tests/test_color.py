#!/usr/bin/env python

import unittest

import numpy as np
import numpy.testing as nt

from machinevisiontoolbox import Image
from machinevisiontoolbox.base.color import *


class TestImageProcessingColor(unittest.TestCase):

    def test_color2name(self):
        name = color2name([1, 0, 0])
        self.assertIsInstance(name, str)
        self.assertIn("red", name.lower())

    def test_color2name_xy(self):
        name = color2name([0.33, 0.33], colorspace="xy")
        self.assertIsInstance(name, str)

    def test_name2color(self):
        nt.assert_array_almost_equal(name2color("r"), (1, 0, 0))
        nt.assert_array_almost_equal(
            name2color("g", "xyz"), (0.17879, 0.35758, 0.059597)
        )
        nt.assert_array_almost_equal(name2color("g", "xy"), (0.3, 0.6))

    @unittest.skip("broken")
    def test_showcolorspace(self):

        # test it runs and is the correct shape
        # may also be able to test values of specific coordinates?
        imcs = Image().showcolorspace("xy")
        self.assertEqual(imcs.shape, (451, 401, 3))

        imcs = Image().showcolorspace("ab")
        self.assertEqual(imcs.shape, (501, 501, 3))

    @unittest.skip("gamma has changed")
    def test_gamma(self):

        a = Image(np.array([[0.4]]))
        g = a.gamma_encode(0.5)
        nt.assert_array_almost_equal(g.array * g.array, a.array)

        a = Image(np.array([[64.0]]))
        g = a.gamma_encode(0.5)
        nt.assert_array_almost_equal(g.array * g.array, a.array)

        # test for shape
        g = a.gamma("srgb")
        self.assertEqual(g.shape, a.shape)

        a = Image(np.random.rand(5, 5))
        g = a.gamma(0.5)
        nt.assert_array_almost_equal(g.shape, a.shape)
        nt.assert_array_almost_equal(g.gamma(2).array, a.array)

        a = Image(np.random.rand(5, 5, 3))
        g = a.gamma(0.5)
        nt.assert_array_almost_equal(g.shape, a.shape)

    def test_colorize(self):

        im = np.array([[1, 2, 3], [1, 2, 3], [1, 3, 3]]) / 10
        im = Image(im)
        out = im.colorize(color=[0, 0, 1])

        # quick element teste
        self.assertAlmostEqual(out.A[0, 0, 0], 0)
        self.assertAlmostEqual(out.A[0, 0, 1], 0)
        self.assertAlmostEqual(out.A[0, 0, 2], 0.1)
        # TODO mask functionality not yet implemented

    @unittest.skip("Code has a bug: mono.array should be mono")
    def test_mono(self):
        # input an image that is not mono
        im = Image.Read("monalisa.png")
        imm = im.mono()
        self.assertEqual(imm.iscolor, False)
        self.assertEqual(imm.shape, im.shape[:2])

        # Test different modes
        imm_r709 = im.mono(opt="r709")
        self.assertEqual(imm_r709.iscolor, False)

        imm_value = im.mono(opt="value")
        self.assertEqual(imm_value.iscolor, False)

        # Test mono on already mono image (should return self)
        imm2 = imm.mono()
        self.assertTrue(imm2 is imm)

    def test_chromaticity(self):
        # Create RGB image from known data
        im = np.ones((5, 5, 3))
        im[:, :, 1] = 0.5  # Reduce green channel
        im_obj = Image(im, colororder="RGB")
        chrom = im_obj.chromaticity()
        self.assertEqual(chrom.shape, (5, 5, 2))
        self.assertEqual(chrom.colororder_str, "r:g")

    @unittest.skip("Code has issues with colorize")
    def test_colorize_variations(self):
        # Test with string color
        im = Image(np.ones((5, 5)) * 0.5)
        out = im.colorize("red")
        self.assertEqual(out.nplanes, 3)
        self.assertAlmostEqual(out.A[0, 0, 0], 0.5)
        self.assertAlmostEqual(out.A[0, 0, 1], 0.0, places=5)
        self.assertAlmostEqual(out.A[0, 0, 2], 0.0, places=5)

        # Test with alpha channel
        out_alpha = im.colorize([1, 0, 0], alpha=True)
        self.assertEqual(out_alpha.nplanes, 4)
        self.assertAlmostEqual(out_alpha.A[0, 0, 3], 1.0)

        # Test with scalar alpha
        out_alpha_scalar = im.colorize([1, 0, 0], alpha=0.5)
        self.assertEqual(out_alpha_scalar.nplanes, 4)
        self.assertAlmostEqual(out_alpha_scalar.A[0, 0, 3], 0.5)

    def test_kmeans_color(self):
        # Create simple color image
        im = Image(np.random.rand(20, 20, 3))

        # Test training mode
        labels, centroids, residual = im.kmeans_color(k=3, seed=42)
        self.assertEqual(labels.shape, (20, 20))
        self.assertEqual(centroids.shape[1], 3)
        self.assertTrue(isinstance(residual, (int, float)))

        # Test classification mode with existing centroids
        labels2 = im.kmeans_color(centroids=centroids)
        self.assertEqual(labels2.shape, (20, 20))

    def test_colorspace_conversions(self):
        # Create RGB image
        im = Image.Read("flowers1.png")

        # Test various colorspace conversions
        hsv = im.colorspace("hsv")
        self.assertEqual(hsv.nplanes, 3)

        xyz = im.colorspace("xyz")
        self.assertEqual(xyz.nplanes, 3)

        lab = im.colorspace("lab")
        self.assertEqual(lab.nplanes, 3)

        gray = im.colorspace("gray")
        self.assertEqual(gray.nplanes, 1)

    def test_plane_extraction(self):
        # Test color plane extraction
        im = Image.Read("flowers1.png")
        red = im.plane("R")
        self.assertEqual(red.shape, im.shape[:2])

        green = im.plane("G")
        self.assertEqual(green.shape, im.shape[:2])

        blue = im.plane("B")
        self.assertEqual(blue.shape, im.shape[:2])

    def test_mono_r709(self):
        im = Image.Read("flowers1.png")
        mono = im.mono("r709")
        self.assertFalse(mono.iscolor)
        self.assertEqual(mono.shape, im.shape[:2])

    def test_mono_value(self):
        im = Image.Read("flowers1.png")
        mono = im.mono("value")
        self.assertFalse(mono.iscolor)
        self.assertEqual(mono.shape, im.shape[:2])

    def test_mono_cv(self):
        im = Image.Read("flowers1.png")
        mono = im.mono("cv")
        self.assertFalse(mono.iscolor)
        self.assertEqual(mono.shape, im.shape[:2])

    def test_mono_already_greyscale(self):
        im = Image.Read("flowers1.png", mono=True)
        result = im.mono()
        self.assertIs(result, im)

    def test_mono_unknown_opt_raises(self):
        im = Image.Read("flowers1.png")
        with self.assertRaises(TypeError):
            im.mono("invalid")

    def test_colorize_string_name(self):
        im = Image(np.ones((5, 5)) * 0.5)
        out = im.colorize("r")
        self.assertEqual(out.nplanes, 3)
        self.assertAlmostEqual(out.A[0, 0, 0], 0.5, places=3)
        self.assertAlmostEqual(out.A[0, 0, 1], 0.0, places=3)

    def test_colorize_color_image_raises(self):
        im = Image.Read("flowers1.png")
        with self.assertRaises(ValueError):
            im.colorize([1, 0, 0])

    def test_gamma_encode_sRGB(self):
        im = Image(np.array([[0.5]]))
        out = im.gamma_encode("sRGB")
        self.assertTrue(out.isfloat)
        # sRGB encodes lighter (raises value for mid-tones)
        self.assertGreater(out.A[0, 0], 0.5)

    def test_gamma_decode_sRGB(self):
        im = Image(np.array([[0.7]]))
        out = im.gamma_decode("sRGB")
        self.assertTrue(out.isfloat)
        self.assertLess(out.A[0, 0], 0.7)

    def test_gamma_encode_float_param(self):
        im = Image(np.array([[0.5]]))
        out = im.gamma_encode(2.2)
        nt.assert_almost_equal(out.A[0, 0], 0.5**2.2, decimal=5)

    def test_gamma_decode_float_param(self):
        im = Image(np.array([[0.5]]))
        out = im.gamma_decode(2.2)
        nt.assert_almost_equal(out.A[0, 0], 0.5**2.2, decimal=5)

    def test_overlay(self):
        im1 = Image.Read("eiffel-1.png", mono=True)
        im2 = Image.Read("eiffel-2.png", mono=True)
        result = Image.Overlay(im1, im2)
        self.assertTrue(result.iscolor)
        self.assertEqual(result.height, max(im1.height, im2.height))
        self.assertEqual(result.width, max(im1.width, im2.width))

    def test_overlay_custom_colors(self):
        im1 = Image.Read("eiffel-1.png", mono=True)
        im2 = Image.Read("eiffel-2.png", mono=True)
        result = Image.Overlay(im1, im2, colors="rg")
        self.assertTrue(result.iscolor)

    def test_overlay_raises_on_color(self):
        im_grey = Image.Read("flowers1.png", mono=True)
        im_color = Image.Read("flowers1.png")
        with self.assertRaises(ValueError):
            Image.Overlay(im_grey, im_color)

    def test_chromaticity_greyscale_raises(self):
        im = Image(np.ones((5, 5)))
        with self.assertRaises(ValueError):
            im.chromaticity()

    def test_chromaticity_alt_channels(self):
        im = np.ones((5, 5, 3))
        im_obj = Image(im, colororder="RGB")
        chrom = im_obj.chromaticity("RB")
        self.assertEqual(chrom.shape, (5, 5, 2))
        self.assertEqual(chrom.colororder_str, "r:b")

    def test_kmeans_no_args_raises(self):
        im = Image(np.random.rand(20, 20, 3))
        with self.assertRaises(ValueError):
            im.kmeans_color()


class TestBaseColorFunctions(unittest.TestCase):

    def test_luminos_scalar(self):
        y = luminos(555e-9)
        # peak photopic luminosity (555 nm) ≈ 683 lm/W
        self.assertGreater(y, 600)

    def test_luminos_vector(self):
        wl = np.array([450e-9, 555e-9, 650e-9])
        y = luminos(wl)
        self.assertEqual(len(y), 3)
        # peak is near 555 nm
        self.assertGreater(y[1], y[0])
        self.assertGreater(y[1], y[2])

    def test_ccxyz(self):
        wl = np.linspace(400e-9, 700e-9, 31)
        cc = ccxyz(wl)
        self.assertEqual(cc.shape[0], 31)
        self.assertEqual(cc.shape[1], 3)
        nt.assert_array_almost_equal(np.sum(cc, axis=1), np.ones(31), decimal=4)

    def test_lambda2xy(self):
        wl = np.linspace(400, 700, 20)
        xy = lambda2xy(wl)
        self.assertEqual(xy.shape[0], 20)
        self.assertEqual(xy.shape[1], 2)

    def test_XYZ2RGBxform_D65(self):
        M = XYZ2RGBxform(white="D65")
        self.assertEqual(M.shape, (3, 3))

    def test_XYZ2RGBxform_E(self):
        M = XYZ2RGBxform(white="E")
        self.assertEqual(M.shape, (3, 3))

    def test_gamma_encode_float_direct(self):
        img = np.array([[0.5]])
        out = gamma_encode(img, 2.0)
        nt.assert_almost_equal(out[0, 0], 0.5**2.0, decimal=5)

    def test_gamma_encode_sRGB_direct_grey(self):
        img = np.array([[0.5]])
        out = gamma_encode(img, "sRGB")
        self.assertEqual(out.shape, (1, 1))
        self.assertGreater(out[0, 0], 0.5)  # sRGB encodes lighter

    def test_gamma_encode_sRGB_direct_color(self):
        img = np.ones((4, 4, 3)) * 0.5
        out = gamma_encode(img, "sRGB")
        self.assertEqual(out.shape, (4, 4, 3))

    def test_gamma_encode_int_image(self):
        img = np.full((2, 2), 128, dtype=np.uint8)
        out = gamma_encode(img, 2.2)
        # gamma_encode returns float32 for integer input with numeric gamma
        self.assertEqual(out.dtype, np.float32)

    def test_gamma_decode_float_direct(self):
        img = np.array([[0.5]])
        out = gamma_decode(img, 2.0)
        nt.assert_almost_equal(out[0, 0], 0.5**2.0, decimal=5)

    def test_gamma_decode_sRGB_direct_grey(self):
        img = np.array([[0.7]])
        out = gamma_decode(img, "sRGB")
        self.assertEqual(out.shape, (1, 1))
        self.assertLess(out[0, 0], 0.7)

    def test_gamma_decode_sRGB_direct_color(self):
        img = np.ones((4, 4, 3)) * 0.7
        out = gamma_decode(img, "sRGB")
        self.assertEqual(out.shape, (4, 4, 3))

    def test_colorspace_convert_rgb_to_ycrcb(self):
        img = np.ones((4, 4, 3), dtype=np.float32) * 0.5
        out = colorspace_convert(img, "rgb", "ycrcb")
        self.assertEqual(out.shape, (4, 4, 3))

    def test_colorspace_convert_rgb_to_hls(self):
        img = np.ones((4, 4, 3), dtype=np.float32) * 0.5
        out = colorspace_convert(img, "rgb", "hls")
        self.assertEqual(out.shape, (4, 4, 3))

    def test_colorspace_convert_rgb_to_luv(self):
        img = np.ones((4, 4, 3), dtype=np.float32) * 0.5
        out = colorspace_convert(img, "rgb", "luv")
        self.assertEqual(out.shape, (4, 4, 3))


# ----------------------------------------------------------------------------#
if __name__ == "__main__":

    unittest.main()

    # import code
    # code.interact(local=dict(globals(), **locals()))
