#!/usr/bin/env python

import numpy as np
import numpy.testing as nt
import unittest
from machinevisiontoolbox.base import color
import machinevisiontoolbox as mvt

from pathlib import Path


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


# ---------------------------------------------------------------------------#
if __name__ == '__main__':

    unittest.main()

