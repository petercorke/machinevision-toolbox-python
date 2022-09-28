#!/usr/bin/env python

import numpy as np
import numpy.testing as nt
import unittest

from machinevisiontoolbox import Image
from machinevisiontoolbox.base.color import *
class TestImageProcessingColor(unittest.TestCase):

    def test_color2name(self):
        pass

    def test_name2color(self):
        nt.assert_array_almost_equal(name2color('r'), (1,0,0))
        nt.assert_array_almost_equal(name2color('g', 'xyz'), (0.17879 , 0.35758 , 0.059597))
        nt.assert_array_almost_equal(name2color('g', 'xy'), (0.3, 0.6))

    @unittest.skip("broken")
    def test_showcolorspace(self):

        # test it runs and is the correct shape
        # may also be able to test values of specific coordinates?
        imcs = Image().showcolorspace('xy')
        self.assertEqual(imcs.shape, (451, 401, 3))

        imcs = Image().showcolorspace('ab')
        self.assertEqual(imcs.shape, (501, 501, 3))

    @unittest.skip("gamma has changed")
    def test_gamma(self):

        a = Image(np.array([[0.4]]))
        g = a.gamma_encode(0.5)
        nt.assert_array_almost_equal(g.image * g.image, a.image)

        a = Image(np.array([[64.0]]))
        g = a.gamma_encode(0.5)
        nt.assert_array_almost_equal(g.image * g.image, a.image)

        # test for shape
        g = a.gamma('srgb')
        self.assertEqual(g.shape, a.shape)

        a = Image(np.random.rand(5, 5))
        g = a.gamma(0.5)
        nt.assert_array_almost_equal(g.shape, a.shape)
        nt.assert_array_almost_equal(g.gamma(2).image, a.image)

        a = Image(np.random.rand(5, 5, 3))
        g = a.gamma(0.5)
        nt.assert_array_almost_equal(g.shape, a.shape)

    def test_colorize(self):

        im = np.array([[1, 2, 3],
                       [1, 2, 3],
                       [1, 3, 3]]) / 10
        im = Image(im)
        out = im.colorize(color=[0, 0, 1])

        # quick element teste
        self.assertAlmostEqual(out.A[0, 0, 0], 0)
        self.assertAlmostEqual(out.A[0, 0, 1], 0)
        self.assertAlmostEqual(out.A[0, 0, 2], 0.1)
        # TODO mask functionality not yet implemented

    def test_mono(self):
        # input an image that is not mono
        im = Image.Read('monalisa.png')
        imm = im.mono()
        self.assertEqual(imm.iscolor, False)
        self.assertEqual(imm.shape, im.shape[:2])
# ----------------------------------------------------------------------------#
if __name__ == '__main__':

    unittest.main()

    # import code
    # code.interact(local=dict(globals(), **locals()))
