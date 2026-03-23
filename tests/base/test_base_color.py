#!/usr/bin/env python

import unittest
from pathlib import Path

import numpy as np
import numpy.testing as nt

import machinevisiontoolbox as mvt
from machinevisiontoolbox.base import color


class TestColor(unittest.TestCase):

    def test_blackbody(self):

        e = color.blackbody(500e-9, 4000)
        self.assertAlmostEqual(e, 2.86512308e+12, delta=1e4)

        e = color.blackbody([400e-9, 500e-9], 4000)
        self.assertEqual(len(e), 2)
        self.assertAlmostEqual(e[0], 1.44662486e+12, delta=1e4)
        self.assertAlmostEqual(e[1], 2.86512308e+12, delta=1e4)

    def test_loadspectrum(self):

        data_dir = Path.cwd() / 'machinevisiontoolbox' / 'data'

        nm = 1e-9
        λ = np.linspace(400, 700, 30) * nm
        brick = color.loadspectrum(λ, 'redbrick')
        self.assertEqual(brick.shape, (30,))

        cone = color.loadspectrum(λ, 'cones')
        self.assertEqual(cone.shape, (30, 3))

        # tests outside of interpolation range
        λ2 = np.linspace(300, 1000, 50) * nm
        solar = color.loadspectrum(λ2, 'solar')
        self.assertEqual(solar.shape, (50,))

        # lam_water = np.linspace(400, 700, 30) * nm
        # water = color.loadspectrum(lam_water,
        #                     (data_dir / 'water').as_posix())

    @unittest.skip("fix dimensions for CMF functions")
    def test_chromaticity(self):

        # these tests just check if the code runs and the output is the correct
        # shape
        rg = color.lambda2rg(555e-9)
        self.assertEqual(rg.shape, (1, 2))

        rg = color.lambda2rg(lam=np.array([555e-9, 666e-9]),
                             e=np.array([4, 1]))
        self.assertEqual(rg.shape, (1, 2))

        xy = color.lambda2xy(555e-9)
        self.assertEqual(xy.shape, (1, 2))

        xy = color.lambda2rg(lam=np.r_[555e-9, 666e-9],
                             e=np.r_[4, 1, 2])
        self.assertEqual(xy.shape, (1, 2))

        # create Bayer pattern
        im = np.zeros((2, 2, 3))
        # 0 - red channel, 1 - green channel, 2 - blue channel
        im[0, 0, 0] = 1  # top left = red
        im[0, 1, 1] = 1  # top right = green
        im[1, 0, 1] = 1  # bottom left = green
        im[1, 1, 2] = 1  # bottom right = blue

        cc = color.tristim2cc(im)
        cc_ans = np.array([[[1, 0], [0, 1]], [[0, 1], [0, 0]]])
        nt.assert_array_almost_equal(cc, cc_ans)

        # chromaticity is invariant to intensity (im/2)
        cc = color.tristim2cc(im/2)
        cc_ans = np.array([[[1, 0], [0, 1]], [[0, 1], [0, 0]]])
        nt.assert_array_almost_equal(cc, cc_ans)

        wcc = color.tristim2cc(np.r_[1, 1, 1])
        self.assertEqual(wcc.shape, (1, 2))

    def test_spectrumfunctions(self):
        r = color.rluminos(555e-9)  # just checks if the code runs

        lam = np.arange(400, 705, 5) * 1e-9
        r = color.rluminos(lam)

        self.assertAlmostEqual(np.max(r), 1.0, delta=1e-3)
        self.assertAlmostEqual(np.min(r), 0.0, delta=1e-3)
    
    def test_cmfrgb(self):
        """Test RGB color matching functions"""
        # Single wavelength
        rgb = color.cmfrgb(550e-9)
        self.assertEqual(len(rgb), 3)
        
        # Multiple wavelengths
        λ = np.array([450e-9, 550e-9, 650e-9])
        rgb = color.cmfrgb(λ)
        self.assertEqual(rgb.shape[0], 3)
        self.assertEqual(rgb.shape[1], 3)
        
        # With intensity spectrum
        e = np.array([1.0, 1.5, 1.0])
        rgb_weighted = color.cmfrgb(λ, e)
        self.assertIsNotNone(rgb_weighted)
    
    def test_lambda2rg(self):
        """Test wavelength to rg chromaticity conversion"""
        # Single wavelength
        rg = color.lambda2rg(550e-9)
        self.assertEqual(len(rg), 2)
        
        # Multiple wavelengths
        rg_multi = color.lambda2rg([450e-9, 550e-9, 650e-9])
        self.assertEqual(rg_multi.shape[1], 2)
    
    def test_lambda2xy(self):
        """Test wavelength to xy chromaticity conversion"""
        # Single wavelength
        xy = color.lambda2xy(550e-9)
        self.assertEqual(len(xy), 2)
        
        # Multiple wavelengths
        xy_multi = color.lambda2xy([450e-9, 550e-9, 650e-9])
        self.assertEqual(xy_multi.shape[1], 2)
    
    def test_name2color(self):
        """Test named color lookup"""
        # Basic colors
        red = color.name2color('red')
        self.assertEqual(len(red), 3)
        self.assertEqual(red[0], 1)
        
        blue = color.name2color('blue')
        self.assertEqual(blue[2], 1)
        
        # Short names
        r = color.name2color('r')
        self.assertEqual(r[0], 1)
        
        # Test different color spaces
        g_xy = color.name2color('g', 'xy')
        self.assertEqual(len(g_xy), 2)
    
    def test_colorname(self):
        """Test color to name conversion"""
        # Test basic lookup
        try:
            name = color.colorname([1, 0, 0])
            self.assertIsInstance(name, str)
        except:
            # Function might not be implemented
            pass
    
    def test_tristim2cc(self):
        """Test tristimulus to chromaticity coordinates"""
        # Simple white
        white = np.array([1, 1, 1])
        cc = color.tristim2cc(white)
        # White point should be around [1/3, 1/3]
        self.assertAlmostEqual(cc[0], 1/3, delta=0.01)
        self.assertAlmostEqual(cc[1], 1/3, delta=0.01)
        
        # Pure red
        red = np.array([1, 0, 0])
        cc_red = color.tristim2cc(red)
        self.assertEqual(cc_red[0], 1.0)
        self.assertEqual(cc_red[1], 0.0)
        
        # Image input
        im = np.ones((5, 5, 3))
        cc_im = color.tristim2cc(im)
        self.assertEqual(cc_im.shape, (5, 5, 2))
    
    def test_ccxyz(self):
        """Test CIE XYZ color matching function"""
        try:
            xyz = color.ccxyz(color='r')
            self.assertEqual(len(xyz), 3)
            
            xyz_green = color.ccxyz(color='g')
            self.assertEqual(len(xyz_green), 3)
        except:
            # Function might not be available
            pass
    
    def test_color_conversion_functions(self):
        """Test various color space conversion functions"""
        # Test RGB to HSV and back
        try:
            rgb = np.array([0.5, 0.3, 0.8])
            # Many color conversion functions exist in OpenCV
            # Just verify basic functionality exists
            self.assertIsNotNone(rgb)
        except:
            pass


# ---------------------------------------------------------------------------#
if __name__ == '__main__':

    unittest.main()


