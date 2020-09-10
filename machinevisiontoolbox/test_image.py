import numpy as np
import cv2 as cv
import numpy.testing as nt
import unittest
import image as image

from pathlib import Path

class TestColor(unittest.TestCase):

    # see ioTest.m
    # def test_iread(self):
        # test image:
    #    img_name = 'longquechen-spacerover.jpg'
    #    img = im.iread((Path('data')/img_name).as_posix())

    def test_idisp(self):
        # see ioTest.m
        # test image:
        im_name = 'longquechen-spacerover.jpg'
        # read in image

        im = image.iread((Path('data') / im_name).as_posix())
        # im.idisp(img)

        # TODO figure out how to make figure not blocking
        # im.idisp(img, title='space rover 2020')

    def test_isimage(self):
        # see utilityTest
        im_name = 'longquechen-spacerover.jpg'
        im = image.iread((Path('data') / im_name).as_posix())

        print('is image: ', image.isimage(im))

    # see utilityTest
    def test_iscolor(self):
        # TODO input color image, sequence of images
        # TODO input grayscale image
        print('to do')

    def test_imono(self):
        im_name = 'longquechen-spacerover.jpg'
        im = image.iread((Path('data') / im_name).as_posix())

        immono = image.imono(im)
        image.idisp(immono)

# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':

    unittest.main()
