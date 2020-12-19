#!/usr/bin/env python

# test for Image input/output

import numpy as np
import os
import numpy.testing as nt
import unittest
# import machinevisiontoolbox as mvt
from machinevisiontoolbox.Image import iread  # unsure how to get Image.iread
from machinevisiontoolbox.Image import Image

from pathlib import Path


class TestImage(unittest.TestCase):

    def test_iread(self):
        # see ioTest.m
        # test image:
        im = iread('wally.png')[0]
        self.assertEqual(isinstance(im, np.ndarray), True)
        self.assertEqual(im.ndim, 3)
        self.assertEqual(im.shape, (25, 21, 3))

    def test_isimage(self):

        # create mini image (Bayer pattern)
        im = np.zeros((2, 2, 3))
        # 0 - red channel, 1 - green channel, 2 - blue channel
        im[0, 0, 0] = 1  # top left = red
        im[0, 1, 1] = 1  # top right = green
        im[1, 0, 1] = 1  # bottom left = green
        im[1, 1, 2] = 1  # bottom right = blue

        # a single grayscale image
        self.assertEqual(Image.isimage(im[:, :, 0].astype(np.float)), True)

        # set type as float, then make sure isimage is true
        self.assertEqual(Image.isimage(im.astype(np.float32)), True)
        self.assertEqual(Image.isimage(im.astype(np.int)), True)

    def test_str(self):
        # single color image as str
        # print('test_str')
        imname = 'monalisa.png'

        im = Image(imname)
        # check attributes
        nt.assert_array_equal(im.shape, (700, 677, 3))
        self.assertEqual(os.path.split(im.filename)[1], imname)
        self.assertEqual(im.iscolor, True)
        self.assertEqual(im.dtype, 'uint8')
        self.assertEqual(im.width, 677)
        self.assertEqual(im.height, 700)
        self.assertEqual(im.issequence, False)
        self.assertEqual(im.ndim, 3)
        self.assertEqual(im.numimages, 1)
        self.assertEqual(im.colororder, 'BGR')
        self.assertEqual(im.numchannels, 3)

    def test_wildcardstr(self):
        # single str with wild card for folder of images
        # print('test_wildcardstr')
        imname = Image('campus/*.png')

        im = Image(imname)
        self.assertEqual(im.numimages, 20)
        self.assertEqual(im.issequence, True)
        self.assertEqual(im.shape, (426, 640, 3))
        self.assertEqual(im.dtype, 'uint8')
        self.assertEqual(im.colororder, 'BGR')
        self.assertEqual(im.numchannels, 3)

    def test_liststr(self):
        # list of image filenames
        # print('test_liststr')
        flowerlist = [str(('flowers' + str(i+1) + '.png')) for i in range(8)]

        im = Image(flowerlist)
        self.assertEqual(im.numimages, 8)
        self.assertEqual(im.issequence, True)
        imfilenamelist = [i.filename for i in im]
        self.assertTrue(all([os.path.split(x)[1] == y for x, y in zip(imfilenamelist, flowerlist)]))

    def test_image(self):
        # Image object
        # print('test_image')
        imname = 'shark1.png'
        im0 = Image(imname)

        im1 = Image(im0)
        # TODO consider __eq__ to compare Image objects directly im0 == im1
        nt.assert_array_almost_equal(im1.image, im0.image)
        self.assertEqual(im1.filename, im0.filename)
        self.assertEqual(im1.shape, im0.shape)
        self.assertEqual(im1.iscolor, im0.iscolor)
        # ... for the rest of the attributes

    def test_listimage(self):
        # list of Image objects
        # print('test_listimage')
        flowerlist = [str(('flowers' + str(i+1) + '.png')) for i in range(8)]
        imlist = [Image(flower) for flower in flowerlist]

        im = Image(imlist)

        imfilenamelist = [os.path.split(i.filename)[1] for i in im]
        # imfilenamelist == flowerlist
        self.assertEqual(imfilenamelist, flowerlist)
        self.assertEqual(im.issequence, True)

    def test_array(self):
        # test single numpy array
        # print('test_numpyarray')
        imarray = iread('walls-l.png')

        im = Image(imarray[0])
        self.assertEqual(im.shape, (2448, 3264, 3))
        self.assertEqual(im.iscolor, True)

    def test_listarray(self):
        # test list of arrays
        # print('test_listarray')
        flowerlist = [str(('flowers' + str(i+1) + '.png')) for i in range(8)]
        imlist = [iread(i)[0] for i in flowerlist]
        # concatenate list of images into a stack of images
        imlistexp = [np.expand_dims(imlist[i], axis=3)
                     for i in range(len(imlist))]
        imstack = imlistexp[0]
        for i in range(1, 8):
            imstack = np.concatenate((imstack, imlistexp[i]), axis=3)

        im = Image(imstack)
        self.assertEqual(im.numimages, 8)
        self.assertEqual(im.shape, (426, 640, 3))
        self.assertEqual(im.dtype, 'uint8')
        self.assertEqual(im.colororder, 'BGR')
        self.assertEqual(im.numchannels, 3)
        self.assertEqual(im.issequence, True)

    def test_options(self):

        imname = 'monalisa.png'
        im = Image(imname, colororder='BGR')

        # check predicatives
        self.assertEqual(im.issequence, False)
        self.assertEqual(im.isfloat, False)
        self.assertEqual(im.isint, True)
        self.assertEqual(isinstance(im, Image), True)
        nt.assert_array_equal(im.bgr.shape, im.shape)
        nt.assert_array_equal(im.rgb.shape, im.shape)
        self.assertEqual(im.size, (700, 677))

        # check one element for rgb vs bgr ordering
        v = round(im.shape[0] / 2)  # rows
        u = round(im.shape[1] / 2)  # cols
        bgr = im.bgr[v, u, :]
        nt.assert_array_equal(im.rgb[v, u, :], bgr[::-1])

        self.assertEqual(im.isrgb, False)
        self.assertEqual(im.isbgr, True)

        self.assertEqual(im.iscolor, True)

    # TODO unit tests:
    # test_isimage - make sure Image rejects/fails with invalid input
    # test_imtypes - test Image works on different Image types?
    # test_getimage - make sure Image returns the same array but with valid
    # typing?
    # test_imwrite - test write/successfully save file?


# ------------------------------------------------------------------------ #
if __name__ == '__main__':

    unittest.main()
