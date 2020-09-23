import numpy as np
import cv2 as cv
import numpy.testing as nt
import unittest
# import image as im
import machinevisiontoolbox as mvt

import pdb
from pathlib import Path


class TestImage(unittest.TestCase):

    # see ioTest.m
    def test_iread(self):
        # test image:
        img_name = 'wally.png'
        im = mvt.iread((Path('images') / img_name).as_posix())

    def test_idisp(self):
        # see ioTest.m
        # test image:
        im_name = 'monalisa.png'
        # read in image

        im = mvt.iread((Path('images') / im_name).as_posix())
        # im.idisp(img)

        # TODO figure out how to make figure not blocking
        #mvt.idisp(im, title='space')

        im2 = mvt.iread((Path('images') / im_name).as_posix())
        # mvt.idisp(im2, title='rover')

    def test_isimage(self):

        # create mini image (Bayer pattern)
        im = np.zeros((2, 2, 3))
        # 0 - red channel, 1 - green channel, 2 - blue channel
        im[0, 0, 0] = 1  # top left = red
        im[0, 1, 1] = 1  # top right = green
        im[1, 0, 1] = 1  # bottom left = green
        im[1, 1, 2] = 1  # bottom right = blue

        # a single grayscale image
        self.assertEqual(mvt.isimage(im[:, :, 0].astype(np.float)), True)

        # set type as float, then make sure isimage is true
        self.assertEqual(mvt.isimage(im.astype(np.float)), True)

        self.assertEqual(mvt.isimage(im.astype(np.int)), True)

        # we don't do complex values in images yet
        self.assertEqual(mvt.isimage(im.astype(np.complex)), False)

        # see utilityTest.m
        # im_name = 'longquechen-spacerover.jpg'
        # im = image.iread((Path('data') / im_name).as_posix())

    def test_iscolor(self):
        # TODO input color image, sequence of images
        # TODO input grayscale image

        # create mini image (Bayer pattern)
        im = np.zeros((2, 2, 3))
        # 0 - red channel, 1 - green channel, 2 - blue channel
        im[0, 0, 0] = 1  # top left = red
        im[0, 1, 1] = 1  # top right = green
        im[1, 0, 1] = 1  # bottom left = green
        im[1, 1, 2] = 1  # bottom right = blue

        self.assertEqual(mvt.iscolor(im), True)
        self.assertEqual(mvt.iscolor(im[:, :, 0]), False)

    def test_imono(self):
        im_name = 'shark1.png'
        im = mvt.iread((Path('images') / im_name).as_posix())

        immono = mvt.imono(im)
        # mvt.idisp(immono, title='space rover')

    def test_idouble(self):
        # test for uint8
        im = np.zeros((2, 2), np.uint8)
        nt.assert_array_almost_equal(
            mvt.idouble(im), np.zeros((2, 2), np.float32))
        im = 128 * np.ones((2, 2), np.uint8)
        nt.assert_array_almost_equal(mvt.idouble(
            im), (128.0/255.0 * np.ones((2, 2))))
        im = 255 * np.ones((2, 2), np.uint8)
        nt.assert_array_almost_equal(mvt.idouble(im), (np.ones((2, 2))))

        # test for uint16
        im = np.zeros((2, 2), np.uint16)
        nt.assert_array_almost_equal(
            mvt.idouble(im), np.zeros((2, 2), np.float32))
        im = 128 * np.ones((2, 2), np.uint16)
        nt.assert_array_almost_equal(mvt.idouble(
            im), (128.0/65535.0 * np.ones((2, 2))))
        im = 65535 * np.ones((2, 2), np.uint16)
        nt.assert_array_almost_equal(mvt.idouble(im), (np.ones((2, 2))))

        # test for sequence of images
        im = np.random.randint(
            low=1, high=255, size=(5, 8, 3, 4), dtype=np.uint8)
        b = mvt.idouble(im)
        nt.assert_array_almost_equal(b.shape, im.shape)
        nt.assert_array_almost_equal(b, im.astype(np.float32) / 255.0)

        im = np.random.randint(low=1, high=65535, size=(
            3, 10, 3, 7), dtype=np.uint16)
        b = mvt.idouble(im)
        nt.assert_array_almost_equal(b.shape, im.shape)
        nt.assert_array_almost_equal(b, im.astype(np.float32) / 65535.0)

    def test_iint(self):
        # test for uint8
        im = np.zeros((2, 2), np.float)
        nt.assert_array_almost_equal(mvt.iint(im), np.zeros((2, 2), np.uint8))
        im = np.ones((2, 2), np.float)
        nt.assert_array_almost_equal(
            mvt.iint(im), 255 * np.ones((2, 2)).astype(np.uint8))

        im = np.random.randint(1, 255, (3, 5), int)
        b = mvt.iint(im)
        nt.assert_array_almost_equal(b.shape, im.shape)

        im = np.random.randint(1, 255, (3, 5, 3), int)
        b = mvt.iint(im)
        nt.assert_array_almost_equal(b.shape, im.shape)

        im = np.random.randint(1, 255, (3, 5, 3, 10), int)
        b = mvt.iint(im)
        nt.assert_array_almost_equal(b.shape, im.shape)

    def test_imorph(self):
        # test simple case
        im = np.array([[1, 2], [3, 4]])
        se = 1
        nt.assert_array_almost_equal(mvt.imorph(im, se, 'min'), im)
        nt.assert_array_almost_equal(mvt.imorph(im, se, 'max'), im)
        nt.assert_array_almost_equal(
            mvt.imorph(im, se, 'min', opt='border'), im)
        nt.assert_array_almost_equal(mvt.imorph(im, se, 'min', opt='none'), im)
        nt.assert_array_almost_equal(mvt.imorph(im, se, 'min', opt='wrap'), im)

        # test different input formats
        nt.assert_array_almost_equal(mvt.imorph(
            im.astype(np.uint8), se, 'min'), im)
        nt.assert_array_almost_equal(mvt.imorph(
            im.astype(np.uint16), se, 'min'), im)
        nt.assert_array_almost_equal(mvt.imorph(
            im.astype(np.single), se, 'min'), im)

        im2 = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        nt.assert_array_almost_equal(
            mvt.imorph(im2.astype(bool), se, 'min'), im2)

        # test a SE that falls over the boundary
        # im3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # TODO 'valid' is basically the trim case, of which we have not found an
        # opencv equivalent
        # nt.assert_array_almost_equal(
        #    mvt.imorph(im3, se, 'min', opt='valid'), im3[1, 1])

    def test_icolor(self):

        print('todo')


# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':

    unittest.main()
