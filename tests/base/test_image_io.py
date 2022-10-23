#!/usr/bin/env python

# test for Image input/output

import numpy as np
import os
import numpy.testing as nt
import matplotlib.pyplot as plt
import matplotlib as mpl

import unittest
from machinevisiontoolbox.base.imageio import *
from pathlib import Path
from collections.abc import Iterable

class TestImage(unittest.TestCase):

    def test_iread(self):
        im, filename = iread('street.png')
        self.assertIsInstance(filename, str)
        self.assertIsInstance(im, np.ndarray)
        self.assertIs(im.dtype.type, np.uint8)
        self.assertEqual(im.shape, (851, 1280))

        im, filename = iread('monalisa.png')
        self.assertIsInstance(filename, str)
        self.assertIsInstance(im, np.ndarray)
        self.assertIs(im.dtype.type, np.uint8)
        self.assertEqual(im.shape, (700, 677, 3))

    def test_iread_options(self):
        im, filename = iread('monalisa.png', mono=True)
        self.assertIsInstance(im, np.ndarray)
        self.assertIs(im.dtype.type, np.uint8)
        self.assertEqual(im.shape, (700, 677))

        im, filename = iread('monalisa.png', grey=True)
        self.assertIsInstance(im, np.ndarray)
        self.assertIs(im.dtype.type, np.uint8)
        self.assertEqual(im.shape, (700, 677)) 

        im, filename = iread('monalisa.png', gray=True)
        self.assertIsInstance(im, np.ndarray)
        self.assertIs(im.dtype.type, np.uint8)
        self.assertEqual(im.shape, (700, 677))

        im, filename = iread('monalisa.png')
        im2, filename = iread('monalisa.png', rgb=False)
        self.assertIsInstance(im2, np.ndarray)
        self.assertIs(im2.dtype.type, np.uint8)
        self.assertEqual(im2.shape, (700, 677, 3))
        nt.assert_array_equal(im[:,:,2], im2[:,:,0])
        nt.assert_array_equal(im[:,:,1], im2[:,:,1])
        nt.assert_array_equal(im[:,:,0], im2[:,:,2])

        im, filename = iread('monalisa.png', gamma='sRGB')
        self.assertIsInstance(im, np.ndarray)
        self.assertIs(im.dtype.type, np.uint8)
        self.assertEqual(im.shape, (700, 677, 3))

        im, filename = iread('monalisa.png', reduce=2)
        self.assertIsInstance(im, np.ndarray)
        self.assertIs(im.dtype.type, np.uint8)
        self.assertEqual(im.shape, (350, 339, 3))

        im, filename = iread('monalisa.png', roi=[100, 200, 300, 500])
        self.assertIsInstance(im, np.ndarray)
        self.assertIs(im.dtype.type, np.uint8)
        self.assertEqual(im.shape, (200, 100, 3))

        im, filename = iread('monalisa.png')
        im2, filename = iread('monalisa.png', dtype='uint16')
        self.assertIsInstance(im2, np.ndarray)
        self.assertIs(im2.dtype.type, np.uint16)
        self.assertEqual(im2.shape, (700, 677, 3))  
        self.assertEqual(im[350, 340, 2], round(255/65535*im2[350, 340, 2]))

        im2, filename = iread('monalisa.png', dtype='int16')
        self.assertIsInstance(im2, np.ndarray)
        self.assertIs(im2.dtype.type, np.int16)
        self.assertEqual(im2.shape, (700, 677, 3)) 
        self.assertEqual(im[350, 340, 2], round(255/32767*im2[350, 340, 2]))

        im, filename = iread('monalisa.png', dtype='float32')
        self.assertIsInstance(im, np.ndarray)
        self.assertIs(im.dtype.type, np.float32)
        self.assertEqual(im.shape, (700, 677, 3)) 

        im, filename = iread('monalisa.png', dtype='float64')
        self.assertIsInstance(im, np.ndarray)
        self.assertIs(im.dtype.type, np.float64)
        self.assertEqual(im.shape, (700, 677, 3))

        im2, filename = iread('monalisa.png', dtype='float64', maxintval=255*2)
        self.assertIsInstance(im, np.ndarray)
        self.assertIs(im.dtype.type, np.float64)
        self.assertEqual(im.shape, (700, 677, 3))
        self.assertAlmostEqual(im[350, 340, 2], 2*im2[350, 340, 2])

        im2, filename = iread('monalisa.png', dtype='float')
        self.assertIsInstance(im2, np.ndarray)
        self.assertIs(im2.dtype.type, np.float32)
        self.assertEqual(im2.shape, (700, 677, 3)) 
        self.assertAlmostEqual(im[350, 340, 2], im2[350, 340, 2])

        im2, filename = iread('monalisa.png', dtype='double')
        self.assertIsInstance(im2, np.ndarray)
        self.assertIs(im2.dtype.type, np.float64)
        self.assertEqual(im2.shape, (700, 677, 3))
        self.assertAlmostEqual(im[350, 340, 2], im2[350, 340, 2])

        im2, filename = iread('monalisa.png', dtype='half')
        self.assertIsInstance(im2, np.ndarray)
        self.assertIs(im2.dtype.type, np.float16)
        self.assertEqual(im2.shape, (700, 677, 3)) 
        self.assertAlmostEqual(im[350, 340, 2], im2[350, 340, 2], places=4)

    # def test_iread_url(self):
    #     im, filename = iread("https://webcam.dartmouth.edu/webcam/image.jpg")
    #     self.assertIsInstance(filename, str)

    #     self.assertIsInstance(im, np.ndarray)
    #     self.assertEqula(im.shape, (1,2,3))

    def test_iwrite(self):
        # int images
        filename = '/tmp/iwrite_test.png'
        im = np.random.randint(0, high=255, size=(100, 200), dtype='uint8')
        ret = iwrite(im, filename)
        self.assertTrue(ret)
        imr, _ = iread(filename)
        nt.assert_array_almost_equal(im, imr)

        im = np.random.randint(0, high=255, size=(100, 200, 3), dtype='uint8')
        ret = iwrite(im, filename)
        self.assertTrue(ret)
        imr, _ = iread(filename)
        nt.assert_array_almost_equal(im, imr)

        im = np.random.randint(0, high=255, size=(100, 200, 3), dtype='uint8')
        ret = iwrite(im, filename, rgb=False)
        self.assertTrue(ret)
        imr, _ = iread(filename, rgb=False)
        nt.assert_array_almost_equal(im, imr)

        # float images
        filename = '/tmp/iwrite_test.tiff'
        im = np.float32(np.random.rand(100, 200))
        ret = iwrite(im, filename)
        self.assertTrue(ret)
        imr, _ = iread(filename)
        nt.assert_array_almost_equal(im, imr)

        im = np.float32(np.random.rand(100, 200, 3))
        ret = iwrite(im, filename)
        self.assertTrue(ret)
        imr, _ = iread(filename)
        nt.assert_array_almost_equal(im, imr, decimal=1)

        filename = '/tmp/iwrite_test.exr'
        ret = iwrite(im, filename)
        self.assertTrue(ret)
        imr, _ = iread(filename)
        nt.assert_array_almost_equal(im, imr)

        os.remove('/tmp/iwrite_test.png')
        os.remove('/tmp/iwrite_test.tiff')
        os.remove('/tmp/iwrite_test.exr')


    def test_idisp(self):

        im, filename = iread('street.png')
        idisp(im)
        idisp(im, black=100)
        idisp(im, darken=True)
        idisp(im, darken=2)
        idisp(im, powernorm=2)
        idisp(im, gamma=2)
        idisp(im, vrange=[100, 200])

        idisp(im, grid=True)
        idisp(im, axes=False)
        idisp(im, gui=False)
        idisp(im, frame=False)
        idisp(im, plain=True)
        idisp(im, colormap='random') # custom colormap
        idisp(im, colormap='viridis') # MPL colormap
        viridis = mpl.colormaps['viridis']
        idisp(im, colormap=viridis)
        idisp(im, colormap='grey', badcolor='r', undercolor='g', overcolor='b')

        idisp(im, fps=10)
        idisp(im, square=False)
        idisp(im, width=4, height=4)
        idisp(im, ynormal=True)
        idisp(im, extent=[0, 1, 2, 3])
        idisp(im, title='different title')

        im, filename = iread('monalisa.png')
        idisp(im)
        idisp(im[:, :, ::-1], bgr=True)

        plt.close('all')

    # def test_isimage(self):

    #     # create mini image (Bayer pattern)
    #     im = np.zeros((2, 2, 3))
    #     # 0 - red channel, 1 - green channel, 2 - blue channel
    #     im[0, 0, 0] = 1  # top left = red
    #     im[0, 1, 1] = 1  # top right = green
    #     im[1, 0, 1] = 1  # bottom left = green
    #     im[1, 1, 2] = 1  # bottom right = blue

    #     # a single grayscale image
    #     img = Image(im)
    #     self.assertIsInstance(img, Image)
    #     self.assertEqual(img.shape, im.shape)
    #     self.assertEqual(img.dtype, np.float64)

    #     # set type as float, then make sure isimage is true
    #     img = Image(im.astype(np.float32))
    #     self.assertIsInstance(img, Image)
    #     self.assertEqual(img.shape, im.shape)
    #     self.assertEqual(img.dtype, np.float32)

    #     img = Image(im.astype(np.uint8))
    #     self.assertIsInstance(img, Image)
    #     self.assertEqual(img.shape, im.shape)
    #     self.assertEqual(img.dtype, np.uint8)

    # def test_str(self):
    #     # single color image as str
    #     # print('test_str')
    #     imname = 'monalisa.png'

    #     im = Image.Read(imname)
    #     # check attributes
    #     nt.assert_array_equal(im.shape, (700, 677, 3))
    #     self.assertEqual(os.path.split(im.name)[1], imname)
    #     self.assertEqual(im.iscolor, True)
    #     self.assertEqual(im.dtype, 'uint8')
    #     self.assertEqual(im.width, 677)
    #     self.assertEqual(im.height, 700)
    #     self.assertEqual(im.ndim, 3)
    #     self.assertEqual(im.colororder_str, 'R:G:B')
    #     self.assertEqual(im.nplanes, 3)




    # def test_image(self):
    #     # Image object
    #     # print('test_image')
    #     imname = 'shark1.png'
    #     im0 = Image.Read(imname)

    #     im1 = Image(im0)
    #     # TODO consider __eq__ to compare Image objects directly im0 == im1
    #     nt.assert_array_almost_equal(im1.A, im0.A)
    #     self.assertEqual(im1.shape, im0.shape)
    #     self.assertEqual(im1.iscolor, im0.iscolor)
    #     # ... for the rest of the attributes


    # def test_array(self):
    #     # test single numpy array
    #     # print('test_numpyarray')
    #     imarray = iread('walls-l.png')

    #     im = Image(imarray[0])
    #     self.assertEqual(im.shape, (2448, 3264, 3))
    #     self.assertEqual(im.iscolor, True)


    # def test_options(self):

    #     imname = 'monalisa.png'
    #     im = Image.Read(imname)

    #     # check predicatives
    #     self.assertFalse(im.isfloat)
    #     self.assertTrue(im.isint)
    #     self.assertIsInstance(im, Image)
    #     self.assertEqual(im.bgr.shape, im.shape)
    #     self.assertEqual(im.rgb.shape, im.shape)
    #     self.assertEqual(im.size, (677, 700))

    #     # check one element for rgb vs bgr ordering
    #     v = round(im.shape[0] / 2)  # rows
    #     u = round(im.shape[1] / 2)  # cols
    #     bgr = im.bgr[v, u, :]
    #     nt.assert_array_equal(im.rgb[v, u, :], bgr[::-1])

    #     self.assertTrue(im.isrgb)
    #     self.assertFalse(im.isbgr)

    #     self.assertTrue(im.iscolor)

    # TODO unit tests:
    # test_isimage - make sure Image rejects/fails with invalid input
    # test_imtypes - test Image works on different Image types?
    # test_getimage - make sure Image returns the same array but with valid
    # typing?
    # test_imwrite - test write/successfully save file?


# ------------------------------------------------------------------------ #
if __name__ == '__main__':

    unittest.main()
