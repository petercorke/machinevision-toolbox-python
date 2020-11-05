#!/usr/bin/env python

# test for Image input/output

import numpy as np
import numpy.testing as nt
import unittest
import machinevisiontoolbox as mvtb
from machinevisiontoolbox.Image import *  # unsure how to get Image.iread
from machinevisiontoolbox.Image import Image

from pathlib import Path


class TestImage(unittest.TestCase):

    def test_str(self):
        # single color image as str
        print('test_str')
        imname = 'monalisa.png'

        im = Image(imname)
        # check attributes
        nt.assert_array_equal(im.shape, (700, 677, 3))
        self.assertEqual(im.filename, imname)
        self.assertEqual(im.iscolor, True)
        self.assertEqual(im.dtype, 'uint8')
        self.assertEqual(im.width, 677)
        self.assertEqual(im.height, 700)
        self.assertEqual(im.numimages, 1)
        self.assertEqual(im.colororder, 'BGR')
        self.assertEqual(im.numchannels, 3)

    def test_wildcardstr(self):
        # single str with wild card for folder of images
        print('test_wildcardstr')
        imname = Image('images/campus/*.png')

        im = Image(imname)
        self.assertEqual(im.numimages, 20)
        self.assertEqual(im.shape, (426, 640, 3))
        self.assertEqual(im.dtype, 'uint8')
        self.assertEqual(im.colororder, 'BGR')
        self.assertEqual(im.numchannels, 3)

    def test_liststr(self):
        # list of image filenames
        print('test_liststr')
        flowerlist = [str(('flowers' + str(i+1) + '.png')) for i in range(8)]

        im = Image(flowerlist)
        self.assertEqual(im.numimages, 8)
        imfilenamelist = [i.filename for i in im]
        # imfilenamelist == flowerlist
        self.assertEqual(imfilenamelist, flowerlist)

    def test_image(self):
        # Image object
        print('test_image')
        imname = 'images/shark1.png'
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
        print('test_listimage')
        flowerlist = [str(('flowers' + str(i+1) + '.png')) for i in range(8)]
        imlist = [Image(flower) for flower in flowerlist]

        im = Image(imlist)

        imfilenamelist = [i.filename for i in im]
        # imfilenamelist == flowerlist
        self.assertEqual(imfilenamelist, flowerlist)

    def test_array(self):
        # test single numpy array
        print('test_numpyarray')
        imarray = iread('images/walls-l.png')

        im = Image(imarray)
        self.assertEqual(im.shape, (2448, 3264, 3))
        self.assertEqual(im.iscolor, True)

    def test_listarray(self):
        # test list of arrays
        print('test_listarray')
        flowerlist = [str(('flowers' + str(i+1) + '.png')) for i in range(8)]
        imlist = [iread(('images/' + i)) for i in flowerlist]
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

    def test_options(self):
        # TODO
        # iscolor
        # isrgb
        # etc
        pass


# ------------------------------------------------------------------------ #
if __name__ == '__main__':
    unittest.main()