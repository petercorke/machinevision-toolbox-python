#!/usr/bin/env python

import numpy as np
import numpy.testing as nt
import unittest

from machinevisiontoolbox.Image import Image


class TestImageProcessingMorph(unittest.TestCase):

    def test_morph1(self):

        # test simple case
        im = Image(np.array([[1, 2],
                             [3, 4]]))
        se = 1
        nt.assert_array_almost_equal(im.morph(se, 'min').image, im.image)
        nt.assert_array_almost_equal(im.morph(se, 'max').image, im.image)
        nt.assert_array_almost_equal(im.morph(se,
                                              oper='min',
                                              opt='replicate').image, im.image)
        nt.assert_array_almost_equal(im.morph(se,
                                              oper='min',
                                              opt='none').image, im.image)

        # test different input formats
        nt.assert_array_almost_equal(im.int('uint8').morph(se, 'min').image,
                                     im.image)
        nt.assert_array_almost_equal(im.int('uint16').morph(se, 'min').image,
                                     im.image)
        nt.assert_array_almost_equal(im.float().morph(se, 'min').image * 255,
                                     im.image)

        im = np.array([[1, 0, 1],
                       [0, 1, 0],
                       [1, 1, 0]])
        im = Image(im.astype(bool))
        nt.assert_array_almost_equal(im.morph(se, 'min').image,
                                     im.image)


    # TODO
    # erode
    # dilate
    # morph
    # getse?
    # hitormiss
    # endpoint
    # triplepoint
    # iopen
    # iclose
    # thin
    # label
    # mpq
    # upq
    # npq
    # moments
    # humoments



# ----------------------------------------------------------------------- #
if __name__ == '__main__':

    unittest.main()
