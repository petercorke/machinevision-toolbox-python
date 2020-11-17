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

    def test_morph3(self):
        im = Image(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        out = np.array([[5, 6, 6], [8, 9, 9], [8, 9, 9]])
        nt.assert_array_almost_equal(im.morph(np.ones((3, 3)),
                                              oper='max',
                                              opt='none').image, out)

        out = np.array([[1, 1, 2], [1, 1, 2], [4, 4, 5]])
        nt.assert_array_almost_equal(im.morph(np.ones((3, 3)),
                                              oper='min',
                                              opt='replicate').image, out)

        # simple erosion
        im = Image(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8))
        out = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
        nt.assert_array_almost_equal(im.morph(se=np.ones((3, 3)),
                                              oper='min').image, out)

    def test_erode(self):
        im = np.array([[1, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0, 0]])
        im = Image(im)
        out = np.array([[0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]])
        nt.assert_array_almost_equal(im.erode(np.ones((3, 3))).image, out)

        im = np.array([[1, 1, 1, 0],
                       [1, 1, 1, 0],
                       [0, 0, 0, 0]])
        im = Image(im)
        out = np.array([[1, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])
        nt.assert_array_almost_equal(im.erode(np.ones((3, 3)),
                                              opt='replicate').image, out)

    def test_dilate(self):
        im = np.array([[0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0]])
        out = np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]])
        im = Image(im)
        nt.assert_array_almost_equal(im.dilate(np.ones((3, 3))).image, out)

        out = np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0]])
        nt.assert_array_almost_equal(im.dilate(np.ones((3, 3)), 2).image, out)

    def test_thin(self):
        im = np.array([[0, 0, 0, 0, 0, 1, 1, 1],
                       [0, 0, 0, 0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 0, 0],
                       [1, 1, 1, 1, 1, 0, 0, 0],
                       [1, 1, 1, 1, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0]])
        im = Image(im)
        out = np.array([[0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0]])
        nt.assert_array_almost_equal(im.thin().image, out)

    def test_triplepoint(self):

        im = np.array([[0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0],
                       [1, 1, 1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0]])
        im = Image(im)
        out = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]])
        nt.assert_array_almost_equal(im.triplepoint().image, out)

    def test_endpoint(self):
        im = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0]])
        im = Image(im)
        out = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]])
        nt.assert_array_almost_equal(im.endpoint().image, out)
    # TODO
    # getse?
    # iopen
    # iclose
    # label
    # mpq
    # upq
    # npq
    # moments
    # humoments



# ----------------------------------------------------------------------- #
if __name__ == '__main__':

    unittest.main()
