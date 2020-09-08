import numpy as np
import cv2 as cv
import numpy.testing as nt
import unittest
import image as im

from pathlib import Path

class TestColor(unittest.TestCase):

    # def test_iread(self):
        # test image:
    #    img_name = 'longquechen-spacerover.jpg'
    #    img = im.iread((Path('data')/img_name).as_posix())


    def test_idisp(self):

        # test image:
        img_name = 'longquechen-spacerover.jpg'
        # read in image

        img = im.iread((Path('data') / img_name).as_posix())
        # im.idisp(img)

        im.idisp(img, title='space rover 2020')


# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':

    unittest.main()