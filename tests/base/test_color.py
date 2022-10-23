#!/usr/bin/env python

import numpy as np
import numpy.testing as nt
import unittest

from machinevisiontoolbox.base import *
import matplotlib.pyplot as plt
class TestImageProcessingColor(unittest.TestCase):

    def test_loaddata(self):

        l = np.linspace(380, 700, 10) * 1e-9  # visible spectrum
        sun = loadspectrum(l, "solar")
        self.assertIsInstance(sun, np.ndarray)

        with self.assertRaises(ValueError):
            sun = loadspectrum(l, "blah")

    def test_blackbody(self):

        nt.assert_array_almost_equal(blackbody(500e-9, 6500)/1e13, 4.60974312)
        nt.assert_array_almost_equal(blackbody([500e-9, 550e-9], 6500)/1e13, 
            np.array([4.60974312, 4.30633287]))
        nt.assert_array_almost_equal(blackbody(np.array([500e-9, 550e-9]), 6500)/1e13,
            np.array([4.60974312, 4.30633287]))
        nt.assert_array_almost_equal(blackbody(500e-9, np.array([6000, 6500]))/1e13,
            np.array([3.17570908, 4.60974312]))

    def test_lambda2rg(self):
        nt.assert_array_almost_equal(
            lambda2rg(550e-9),
            np.array([0.09739732, 0.90508141])
        )
        
        nt.assert_array_almost_equal(
            lambda2rg([550e-9, 600e-9]),        
            np.array([[0.09739732, 0.90508141],
                    [0.84746222, 0.15374391]])
        )
        nt.assert_array_almost_equal(
            lambda2rg(np.array([550e-9, 600e-9])),        
            np.array([[0.09739732, 0.90508141],
                    [0.84746222, 0.15374391]])
        )

        l = np.array([400, 500, 550, 600, 700]) * 1e-9
        e = np.array([0, 0, 1, 0, 0])
        nt.assert_array_almost_equal(
            lambda2rg(l, e),
            np.array([0.09739732, 0.90508141])
        )

        l = np.linspace(380, 700, 10) * 1e-9  # visible spectrum
        e = loadspectrum(l, "solar")
        nt.assert_array_almost_equal(
            lambda2rg(l, e),
            np.array([0.33080601, 0.35469871])
        )

    def test_cmfrgb(self):
        nt.assert_array_almost_equal(
            cmfrgb(550e-9),
            np.array([0.02279,  0.21178, -0.00058])
        )
        
        nt.assert_array_almost_equal(
            cmfrgb([550e-9, 600e-9]),        
            np.array([[ 0.02279,  0.21178, -0.00058],
                [ 0.34429,  0.06246, -0.00049]])
        )
        nt.assert_array_almost_equal(
            cmfrgb(np.array([550e-9, 600e-9])),        
            np.array([[ 0.02279,  0.21178, -0.00058],
                [ 0.34429,  0.06246, -0.00049]])        )

        l = np.linspace(380, 700, 10) * 1e-9  # visible spectrum
        e = loadspectrum(l, "solar")
        nt.assert_array_almost_equal(
            cmfrgb(l, e),
            np.array([2.426528, 2.601785, 2.306885])
        )

    def test_tristim2cc(self):
        nt.assert_array_almost_equal(tristim2cc([2, 3, 5]), [0.2, 0.3])
        nt.assert_array_almost_equal(tristim2cc(np.array([2, 3, 5])), [0.2, 0.3])

        nt.assert_array_almost_equal(
            tristim2cc(np.array([
                [2, 3, 5],
                [2, 3, 15]])),
                np.array([
                    [0.2, 0.3],
                    [0.1, 0.15]])
            )

        im = np.dstack([
            np.array([[2,2]]),
            np.array([[3,3]]),
            np.array([[5,15]])
        ])
        out = np.dstack([
            np.array([[0.2,0.1]]),
            np.array([[0.3,0.15]]),
        ])


    def test_lambda2xy(self):
        nt.assert_array_almost_equal(
            lambda2xy(550e-9),
            np.array([0.301604, 0.692308])
        )
        
        nt.assert_array_almost_equal(
            lambda2xy([550e-9, 600e-9]),        
            np.array([[0.301604, 0.692308],
                [0.627037, 0.372491]])
        )
        nt.assert_array_almost_equal(
            lambda2xy(np.array([550e-9, 600e-9])),        
            np.array([[0.301604, 0.692308],
                [0.627037, 0.372491]])
        )

        l = np.array([400, 500, 550, 600, 700]) * 1e-9
        e = np.array([0, 0, 1, 0, 0])
        nt.assert_array_almost_equal(
            lambda2xy(l, e),
            np.array([0.301604, 0.692308])
        )

        l = np.linspace(380, 700, 10) * 1e-9  # visible spectrum
        e = loadspectrum(l, "solar")
        nt.assert_array_almost_equal(
            lambda2xy(l, e),
            np.array([0.33498, 0.350072])
        )

    def test_cmfxyz(self):
        nt.assert_array_almost_equal(
            cmfxyz(550e-9),
            np.array([0.43345, 0.99495, 0.00875])
        )
        
        nt.assert_array_almost_equal(
            cmfxyz([550e-9, 600e-9]),        
            np.array([[4.334499e-01, 9.949501e-01, 8.749999e-03],
                [1.062200e+00, 6.310000e-01, 8.000000e-04]])
        )
        nt.assert_array_almost_equal(
            cmfxyz(np.array([550e-9, 600e-9])),        
            np.array([[4.334499e-01, 9.949501e-01, 8.749999e-03],
                [1.062200e+00, 6.310000e-01, 8.000000e-04]])        )

        l = np.linspace(380, 700, 10) * 1e-9  # visible spectrum
        e = loadspectrum(l, "solar")
        nt.assert_array_almost_equal(
            cmfxyz(l, e),
            np.array([138.830936, 145.08582 , 130.528957])
        )
        
    def test_luminos(self):
        nt.assert_array_almost_equal(
            luminos(550e-9),
            679.585
        )

        nt.assert_array_almost_equal(
            luminos([550e-9, 600e-9]),
            np.array([679.585, 430.973])
        )
        nt.assert_array_almost_equal(
            luminos(np.array([550e-9, 600e-9])),
            np.array([679.585, 430.973])
        )

    def test_rluminos(self):
        nt.assert_array_almost_equal(
            rluminos(550e-9),
            0.99495
        )

        nt.assert_array_almost_equal(
            rluminos([550e-9, 600e-9]),
            np.array([0.99495, 0.631 ])
        )
        nt.assert_array_almost_equal(
            rluminos(np.array([550e-9, 600e-9])),
            np.array([0.99495, 0.631 ])
        )


    def test_ccxyz(self):
        nt.assert_array_almost_equal(
            ccxyz(550e-9),
            np.array([0.301604, 0.692308, 0.006088])
        )
        
        nt.assert_array_almost_equal(
            ccxyz([550e-9, 600e-9]),        
            np.array([[3.016038e-01, 6.923078e-01, 6.088438e-03],
       [6.270366e-01, 3.724911e-01, 4.722550e-04]])
        )
        nt.assert_array_almost_equal(
            ccxyz(np.array([550e-9, 600e-9])),        
            np.array([[3.016038e-01, 6.923078e-01, 6.088438e-03],
       [6.270366e-01, 3.724911e-01, 4.722550e-04]])        )

        l = np.array([400, 500, 550, 600, 700]) * 1e-9
        e = np.array([0, 0, 1, 0, 0])
        nt.assert_array_almost_equal(
            ccxyz(l, e),
            np.array([0.301604, 0.692308, 0.006088])
        )

        l = np.linspace(380, 700, 10) * 1e-9  # visible spectrum
        e = loadspectrum(l, "solar")
        nt.assert_array_almost_equal(
            ccxyz(l, e),
            np.array([0.33498 , 0.350072, 0.314948])
        )

    def test_color2name(self):
        self.assertEqual(color2name([1, 0, 0]), 'red')
        self.assertEqual(color2name([0, 1, 0]), 'lime')
        self.assertEqual(color2name([0, 0, 1]), 'blue')
        self.assertEqual(color2name((0.2, 0.3), 'xy'), 'deepskyblue')

    def test_name2color(self):
        nt.assert_array_almost_equal(name2color('r'), (1,0,0))
        nt.assert_array_almost_equal(name2color('g', 'xyz'), (0.17879 , 0.35758 , 0.059597))
        nt.assert_array_almost_equal(name2color('g', 'xy'), (0.3, 0.6))

    def test_XYZ2RGBxform(self):
        T = XYZ2RGBxform(primaries='CIE')
        XYZ = np.linalg.inv(T) @ [1, 0, 0]
        self.assertEqual(color2name(tristim2cc(XYZ), 'xy'), 'darkred')

        XYZ = np.linalg.inv(T) @ [0, 1, 0]
        self.assertEqual(color2name(tristim2cc(XYZ), 'xy'), 'green')

        XYZ = np.linalg.inv(T) @ [0, 0, 1]
        self.assertEqual(color2name(tristim2cc(XYZ), 'xy'), 'darkblue')

        XYZ = np.linalg.inv(T) @ [1, 1, 1]
        self.assertEqual(color2name(tristim2cc(XYZ), 'xy'), 'xkcd:dusty red')

        T = XYZ2RGBxform(white='E', primaries='CIE')
        XYZ = np.linalg.inv(T) @ [1, 1, 1]
        self.assertEqual(color2name(tristim2cc(XYZ), 'xy'), 'xkcd:reddish grey')

    def test_showcolorspace(self):

        # test it runs and is the correct shape
        # may also be able to test values of specific coordinates?
        imcs = plot_chromaticity_diagram(N=300)
        self.assertIsInstance(imcs, np.ndarray)
        self.assertEqual(imcs.shape, (300, 300, 3))

        imcs = plot_chromaticity_diagram(colorspace='ab')
        plt.close('all')

    def test_plot_spectral_locus(test):

        plot_spectral_locus('xy', labels=True)
        plot_spectral_locus('xy', labels=True, lambda_ticks=np.r_[500, 550, 600]*1e-9)
        plot_spectral_locus('rg', labels=True)
        plt.close('all')

    def test_cie_primaries(self):
        p = cie_primaries()
        self.assertIsInstance(p, np.ndarray)
        self.assertEqual(p.shape, (3,))

    def test_colorspace_convert(self):
        nt.assert_array_almost_equal(
            colorspace_convert([1, 0, 0], 'rgb', 'hls'),
            [0, 0.5, 1]
        )
        nt.assert_array_almost_equal(
            colorspace_convert([0, 1, 0], 'rgb', 'hls'),
            [120, 0.5, 1]
        )
        nt.assert_array_almost_equal(
            colorspace_convert([0, 0, 1], 'rgb', 'hls'),
            [240, 0.5, 1]
        )

        nt.assert_array_almost_equal(
            colorspace_convert([1, 0, 0], 'rgb', 'xyz'),
            [0.412453, 0.212671, 0.019334]
        )
        nt.assert_array_almost_equal(
            colorspace_convert([0, 1, 0], 'rgb', 'xyz'),
            [0.35758 , 0.71516 , 0.119193]
        )
        nt.assert_array_almost_equal(
            colorspace_convert([0, 0, 1], 'rgb', 'xyz'),
            [0.180423, 0.072169, 0.950227]
        )

        v1 = [0.2, 0.3, 0.4]
        v2 = colorspace_convert(v1, 'xyz', 'bgr')
        v3 = colorspace_convert(v2, 'bgr', 'xyz')
        nt.assert_array_almost_equal(v1, v3)

    def test_gamma(self):

        # test the round trip
        im = np.array([[0.0005, 0.1, 0.5, 0.7, 1]])
        nt.assert_almost_equal(gamma_decode(gamma_encode(im)), im)
        nt.assert_almost_equal(gamma_encode(gamma_decode(im)), im)

        im = np.array([[0, 1, 50, 100, 200, 255]], dtype='uint8')
        self.assertTrue(np.all(abs(gamma_decode(gamma_encode(im)) - im) <= 1))
        # the reverse version has significant errors for uint8 images, upto 10

        # float image
        im = np.array([[0.0]])
        nt.assert_array_almost_equal(gamma_encode(im, gamma=0.5), 0)
        im = np.array([[1.0]])
        nt.assert_array_almost_equal(gamma_encode(im, gamma=0.5), 1.0)
        im = np.array([[0.25]])
        nt.assert_array_almost_equal(gamma_encode(im, gamma=0.5), 0.5)
        im = np.dstack([
            np.array([[0.0]]),
            np.array([[0.25]]),
            np.array([[1.0]])
        ])
        nt.assert_array_almost_equal(gamma_encode(im, gamma=0.5).squeeze(), [0.0, 0.5, 1.0])

        im = np.array([[0.0]])
        nt.assert_array_almost_equal(gamma_decode(im, gamma=2), 0)
        im = np.array([[1.0]])
        nt.assert_array_almost_equal(gamma_decode(im, gamma=2), 1.0)
        im = np.array([[0.5]])
        nt.assert_array_almost_equal(gamma_encode(im, gamma=2), 0.25)
        im = np.dstack([
            np.array([[0.0]]),
            np.array([[0.5]]),
            np.array([[1.0]])
        ])
        nt.assert_array_almost_equal(gamma_encode(im, gamma=2).squeeze(), [0.0, 0.25, 1.0])

        # int image
        im = np.array([[0]], dtype='uint8')
        self.assertEqual(gamma_encode(im, gamma=0.5), 0)
        im = np.array([[255]], dtype='uint8')
        self.assertEqual(gamma_encode(im, gamma=0.5), 255)
        im = np.array([[64]], dtype='uint8')
        self.assertEqual(gamma_encode(im, gamma=0.5), 128)
        im = np.dstack([
            np.array([[0]], dtype='uint8'),
            np.array([[64]], dtype='uint8'), 
            np.array([[255]], dtype='uint8')
        ])
        nt.assert_array_almost_equal(gamma_encode(im, gamma=0.5).squeeze(), [0, 128, 255])

        im = np.array([[0]], dtype='uint8')
        self.assertEqual(gamma_decode(im, gamma=2), 0)
        im = np.array([[255]], dtype='uint8')
        self.assertEqual(gamma_decode(im, gamma=2), 255)
        im = np.array([[128]], dtype='uint8')
        self.assertEqual(gamma_encode(im, gamma=2), 64)
        im = np.dstack([
            np.array([[0]], dtype='uint8'),
            np.array([[128]], dtype='uint8'), 
            np.array([[255]], dtype='uint8')
        ])        
        nt.assert_array_almost_equal(gamma_encode(im, gamma=2).squeeze(), [0, 64, 255])

        ## sRGB

        # test the round trip
        im = np.array([[0.0005, 0.1, 0.5, 0.7, 1]])
        nt.assert_almost_equal(gamma_decode(gamma_encode(im, gamma='sRGB'), gamma='sRGB'), im)
        nt.assert_almost_equal(gamma_encode(gamma_decode(im, gamma='sRGB'), gamma='sRGB'), im)

        im = np.array([[0, 1, 50, 100, 200, 255]], dtype='uint8')
        self.assertTrue(np.all(abs(gamma_decode(gamma_encode(im, gamma='sRGB'), gamma='sRGB') - im) <= 1))


        # float image
        im = np.array([[0.0]])
        nt.assert_array_almost_equal(gamma_encode(im, gamma='sRGB'), 0)
        im = np.array([[1.0]])
        nt.assert_array_almost_equal(gamma_encode(im, gamma='sRGB'), 1.0)
        im = np.array([[0.25]])
        nt.assert_array_almost_equal(gamma_encode(im, gamma='sRGB'), 0.537099)
        im = np.dstack([
            np.array([[0.0]]),
            np.array([[0.25]]),
            np.array([[1.0]])
        ])
        nt.assert_array_almost_equal(gamma_encode(im, gamma='sRGB').squeeze(), [0.0, 0.537099, 1.0])

        im = np.array([[0.0]])
        nt.assert_array_almost_equal(gamma_decode(im, gamma='sRGB'), 0)
        im = np.array([[1.0]])

        # int image
        im = np.array([[0]], dtype='uint8')
        self.assertEqual(gamma_encode(im, gamma='sRGB'), 0)
        im = np.array([[255]], dtype='uint8')
        self.assertEqual(gamma_encode(im, gamma='sRGB'), 255)
        im = np.array([[64]], dtype='uint8')
        self.assertEqual(gamma_encode(im, gamma='sRGB'), 137)
        im = np.dstack([
            np.array([[0]], dtype='uint8'),
            np.array([[64]], dtype='uint8'), 
            np.array([[255]], dtype='uint8')
        ])
        nt.assert_array_almost_equal(gamma_encode(im, gamma='sRGB').squeeze(), [0, 137, 255])

        im = np.array([[0]], dtype='uint8')
        self.assertEqual(gamma_decode(im, gamma='sRGB'), 0)
        im = np.array([[255]], dtype='uint8')
        self.assertEqual(gamma_decode(im, gamma='sRGB'), 255)
        im = np.array([[128]], dtype='uint8')
        self.assertEqual(gamma_decode(im, gamma='sRGB'), 55)
        im = np.dstack([
            np.array([[0]], dtype='uint8'),
            np.array([[128]], dtype='uint8'), 
            np.array([[255]], dtype='uint8')
        ])        
        nt.assert_array_almost_equal(gamma_decode(im, gamma='sRGB').squeeze(), [0, 55, 255])

    def test_shadow(self):
        im, _ = iread('parks.png', gamma='sRGB', dtype='double')
        gs = shadow_invariant(im, 0.7)
        self.assertIsInstance(gs, np.ndarray)
        self.assertEqual(im.shape[:2], gs.shape)

# ----------------------------------------------------------------------------#
if __name__ == '__main__':

    unittest.main()

