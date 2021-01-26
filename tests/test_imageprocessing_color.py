#!/usr/bin/env python

import numpy as np
import numpy.testing as nt
import unittest

from machinevisiontoolbox.Image import Image


class TestImageProcessingColor(unittest.TestCase):

    def test_showcolorspace(self):

        # test it runs and is the correct shape
        # may also be able to test values of specific coordinates?
        imcs = Image().showcolorspace('xy')
        self.assertEqual(imcs.shape, (451, 401, 3))

        imcs = Image().showcolorspace('ab')
        self.assertEqual(imcs.shape, (501, 501, 3))

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

    def test_colorise(self):

        im = np.array([[1, 2, 3],
                       [1, 2, 3],
                       [1, 3, 3]]) / 10
        im = Image(im)
        out = im.colorise(c=[0, 0, 1])

        # quick element teste
        self.assertEqual(out.rgb[0, 0, 0], 0)
        self.assertEqual(out.rgb[0, 0, 1], 0)
        self.assertEqual(out.rgb[0, 0, 2], 0.1)
        # TODO mask functionality not yet implemented


# ----------------------------------------------------------------------------#
if __name__ == '__main__':

    unittest.main()

    # import code
    # code.interact(local=dict(globals(), **locals()))
