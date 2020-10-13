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

    def test_mono(self):
        im_name = 'shark1.png'
        im = mvt.iread((Path('images') / im_name).as_posix())

        immono = mvt.mono(im)
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

    def test_morph1(self):
        # test simple case
        im = np.array([[1, 2], [3, 4]])
        se = 1
        nt.assert_array_almost_equal(mvt.morph(im, se, 'min'), im)
        nt.assert_array_almost_equal(mvt.morph(im, se, 'max'), im)
        nt.assert_array_almost_equal(
            mvt.morph(im, se, 'min', opt='replicate'), im)
        nt.assert_array_almost_equal(mvt.morph(im, se, 'min', opt='none'), im)
        # nt.assert_array_almost_equal(mvt.morph(im, se, 'min', opt='wrap'), im)

        # test different input formats
        nt.assert_array_almost_equal(mvt.morph(
            im.astype(np.uint8), se, 'min'), im)
        nt.assert_array_almost_equal(mvt.morph(
            im.astype(np.uint16), se, 'min'), im)
        nt.assert_array_almost_equal(mvt.morph(
            im.astype(np.single), se, 'min'), im)

        im2 = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        nt.assert_array_almost_equal(
            mvt.morph(im2.astype(bool), se, 'min'), im2)

        # test a SE that falls over the boundary
        # im3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # TODO 'valid' is basically the trim case, of which we have not found an
        # opencv equivalent, so will eventually have to code our own
        # nt.assert_array_almost_equal(
        #    mvt.morph(im3, se, 'min', opt='valid'), im3[1, 1])

    """
    # wrap case is not supported by OpenCV for erode() and dilate()
    def test_morph2(self):
        # test wrap case
        im = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        nt.assert_array_almost_equal(mvt.morph(
                                     im, se=np.r_[0, 0, 1], oper='min',
                                     opt='wrap'), np.roll(im, -1, axis=1))
        nt.assert_array_almost_equal(mvt.morph(
                                     im, se=np.r_[1, 0, 0], oper='min',
                                     opt='wrap', np.roll(im, 1, axis=1))
        nt.assert_array_almost_equal(mvt.morph(
                                     im, se=np.r_[0, 0, 1], oper='min',
                                     opt='wrap', np.roll(im, -1, axis=0))
        nt.assert_array_almost_equal(mvt.morph(
                                     im, se=np.r_[1, 0, 0], oper='min',
                                     opt='wrap', np.roll(im, 1, axis=0))
    """

    def test_morph3(self):
        im = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        out = np.array([[5, 6, 6], [8, 9, 9], [8, 9, 9]])
        nt.assert_array_almost_equal(mvt.morph(im, np.ones((3, 3)),
                                               oper='max', opt='none'), out)

        out = np.array([[1, 1, 2], [1, 1, 2], [4, 4, 5]])
        nt.assert_array_almost_equal(mvt.morph(im, np.ones((3, 3)),
                                               oper='min', opt='replicate'), out)

        # simple erosion
        im = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
        out = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
        nt.assert_array_almost_equal(mvt.morph(im, se=np.ones((3, 3)),
                                               oper='min'), out)

    def test_erode(self):
        im = np.array([[1, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0, 0]])
        out = np.array([[0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]])
        nt.assert_array_almost_equal(mvt.erode(im, np.ones((3, 3))), out)

        im = np.array([[1, 1, 1, 0],
                       [1, 1, 1, 0],
                       [0, 0, 0, 0]])
        out = np.array([[1, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])
        nt.assert_array_almost_equal(mvt.erode(im, np.ones((3, 3)),
                                               opt='replicate'), out)

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
        nt.assert_array_almost_equal(mvt.dilate(im, np.ones((3, 3))), out)

        out = np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0]])
        nt.assert_array_almost_equal(mvt.dilate(im, np.ones((3, 3)), 2), out)

    def test_window(self):
        im = np.array([[3,     5,     8,    10,     9],
                       [7,    10,     3,     6,     3],
                       [7,     4,     6,     2,     9],
                       [2,     6,     7,     2,     3],
                       [2,     3,     9,     3,    10]])
        se = np.ones((1, 1))
        # se must be same number of dimensions as input image for scipy
        # TODO maybe detect this in mvt.window and perform this line?

        # test with different input formats
        nt.assert_array_almost_equal(mvt.window(im, se, np.sum), im)
        nt.assert_array_almost_equal(mvt.window(im.astype(np.uint8), se,
                                                np.sum), im)
        nt.assert_array_almost_equal(mvt.window(im.astype(np.uint16), se,
                                                np.sum), im)

        se = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        out = np.array([[43,    47,    57,    56,    59],
                        [46,    43,    51,    50,    57],
                        [45,   48,    40,    39,    31],
                        [33,    40,    35,    49,    48],
                        [22,    40,    36,    53,    44]])
        nt.assert_array_almost_equal(mvt.window(im, se, np.sum), out)

    # TODO ivar did not come up in the functions that required porting to Python
    # yet
    """
    function ivar_test(testCase)
    in = [
        0.7577    0.7060    0.8235    0.4387    0.4898
        0.7431    0.0318    0.6948    0.3816    0.4456
        0.3922    0.2769    0.3171    0.7655    0.6463
        0.6555    0.0462    0.9502    0.7952    0.7094
        0.1712    0.0971    0.0344    0.1869    0.7547
    ];
    out = [
        0.0564    0.0598    0.0675    0.0301    0.0014
        0.0720    0.0773    0.0719    0.0326    0.0163
        0.0787    0.1034    0.1143    0.0441    0.0233
        0.0508    0.0931    0.1261    0.0988    0.0345
        0.0552    0.1060    0.1216    0.1365    0.0618
    ];

    verifyEqual(testCase,  ivar(in, ones(3,3), 'var'), out, 'AbsTol', 1e-4);
    verifyEqual(testCase,  iwindow(in, ones(3,3), 'std').^2, out, 'AbsTol', 1e-4);
    """

    # TODO test_rank, need new test
    #def test_rank(self):
    #    im = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    #    se = np.ones((1, 1))

    def test_thin(self):
        im = np.array([[0, 0, 0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0]])
        out = np.array([[0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0]])
        nt.assert_array_almost_equal(mvt.thin(im), out)

    def test_triplepoint(self):

        im = np.array([[0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0]])
        out = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]])
        nt.assert_array_almost_equal(mvt.triplepoint(im), out)

    def test_endpoint(self):
        im = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]])
        out = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]])
        nt.assert_array_almost_equal(mvt.endpoint(im), out)


    def test_color(self):

        print('todo')


# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':

    unittest.main()
