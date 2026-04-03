#!/usr/bin/env python

import unittest

import numpy as np
import numpy.testing as nt
import spatialmath.base.argcheck as argcheck

from machinevisiontoolbox import Image


class TestImageProcessingMorph(unittest.TestCase):

    def test_morph_scalar_se(self):

        # test simple case
        im = Image(np.array([[1, 2], [3, 4]]))
        se = 1
        nt.assert_array_almost_equal(im.morph(se, op="min").array, im.array)
        nt.assert_array_almost_equal(im.morph(se, op="max").array, im.array)
        nt.assert_array_almost_equal(
            im.morph(se, op="min", border="replicate").array, im.array
        )
        nt.assert_array_almost_equal(
            im.morph(se, op="min", border="none").array, im.array
        )

        # test different input formats
        nt.assert_array_almost_equal(
            im.astype("uint8").morph(se, op="min").array, im.array
        )
        nt.assert_array_almost_equal(
            im.astype("uint16").morph(se, op="min").array, im.array
        )
        nt.assert_array_almost_equal(
            im.astype("float32").morph(se, op="min").array, im.array
        )

        im = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        im = Image(im.astype(bool))
        nt.assert_array_almost_equal(im.morph(se, op="min").array, im.array)

    def test_morph_matrix_se(self):
        im = Image([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        out = np.array([[5, 6, 6], [8, 9, 9], [8, 9, 9]])
        nt.assert_array_almost_equal(
            im.morph(np.ones((3, 3)), op="max", border="none").array, out
        )

        out = np.array([[1, 1, 2], [1, 1, 2], [4, 4, 5]])
        nt.assert_array_almost_equal(
            im.morph(np.ones((3, 3)), op="min", border="replicate").array, out
        )

        # simple erosion
        im = Image(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8))
        out = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
        nt.assert_array_almost_equal(im.morph(se=np.ones((3, 3)), op="min").array, out)

    def test_erode(self):
        im = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype="uint8",
        )
        im = Image(im)
        out = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype="uint8",
        )
        nt.assert_array_almost_equal(im.erode(np.ones((3, 3))).array, out)

        im = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]], dtype="uint8")
        im = Image(im)
        out = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype="uint8")
        nt.assert_array_almost_equal(
            im.erode(np.ones((3, 3)), border="replicate").array, out
        )

    def test_dilate(self):
        im = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype="uint8",
        )
        out = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype="uint8",
        )
        im = Image(im)
        nt.assert_array_almost_equal(im.dilate(np.ones((3, 3))).array, out)

        out = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype="uint8",
        )
        nt.assert_array_almost_equal(im.dilate(np.ones((3, 3)), 2).array, out)

    def test_iclose(self):

        im = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype="uint8",
        )

        out = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0],  # note the border values
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype="uint8",
        )
        im = Image(im)
        se = np.ones((3, 3))
        nt.assert_array_almost_equal(im.close(se).array, out)

    def test_iopen(self):

        im = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype="uint8",
        )

        out = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype="uint8",
        )
        im = Image(im)
        se = np.ones((3, 3))
        nt.assert_array_almost_equal(im.open(se).array, out)

    def test_thin(self):
        im = np.array(
            [
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dtype="uint8",
        )
        im = Image(im)
        out = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dtype="uint8",
        )

        nt.assert_array_almost_equal(im.thin().array, out)

    def test_triplepoint(self):

        im = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype="uint8",
        )
        im = Image(im)
        out = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype="uint8",
        )
        nt.assert_array_almost_equal(im.triplepoint().array, out)

    def test_endpoint(self):
        im = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype="uint8",
        )
        im = Image(im)
        out = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype="uint8",
        )
        nt.assert_array_almost_equal(im.endpoint().array, out)

    def test_rank(self):
        im = np.array([[1, 2, 3], [3, 4, 5], [7, 8, 9]])
        se = np.ones((3, 3))
        out = np.array([[4, 5, 5], [8, 9, 9], [8, 9, 9]])
        im = Image(im)
        imr = im.rank(se, rank=0)
        nt.assert_array_almost_equal(imr.array, out)

        imr = im.rank(se, rank="max")
        nt.assert_array_almost_equal(imr.array, out)

    def test_humoments(self):

        im = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype="float32",
        )
        out = np.array(
            [
                0.184815794830043,
                0.004035534812971,
                0.000533013844814,
                0.000035606641461,
                0.000000003474073,
                0.000000189873096,
                -0.000000003463063,
            ]
        )
        im = Image(im)
        hu = im.humoments()
        nt.assert_array_almost_equal(hu, out, decimal=7)
        # np.assert

    # tc.assertEqual(humoments(im), out, 'absTol', 1e-8);

    def test_distance_transform(self):
        """Test distance transform"""
        img = Image(np.ones((10, 10), dtype="uint8"))
        img.array[5, 5] = 0
        dist = img.distance_transform()
        self.assertEqual(dist.shape, img.shape)
        self.assertGreater(dist.array[0, 0], 0)

    # TODO
    # getse?
    # label
    # mpq
    # upq
    # npq
    # moments


# ----------------------------------------------------------------------- #
if __name__ == "__main__":

    unittest.main()
