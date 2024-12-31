from unittest.case import skip
import numpy as np
import numpy.testing as nt
import unittest
from machinevisiontoolbox import mvtb_path_to_datafile, iwrite
import os

from pathlib import Path

# test mvtb_path_to_datafile() which is used by a number of other functions
# to locate files in the MVTB data package


class TestDataFiles(unittest.TestCase):

    def test_datafile_local(self):

        # test for exception if file not found
        # test that exception raised if not present
        with self.assertRaises(ValueError):
            mvtb_path_to_datafile("notthere.png")

        # create a local test image file
        im = iwrite(np.zeros((5, 5)), "./test-image.png")
        path = mvtb_path_to_datafile("./test-image.png")
        self.assertTrue(os.path.isfile(path))
        # remove the test image file
        os.remove("./test-image.png")

    def test_datafile_mvtb(self):
        path = mvtb_path_to_datafile("images", "shark1.png")
        self.assertTrue(os.path.isfile(path))

        with self.assertRaises(ValueError):
            mvtb_path_to_datafile("images", "__shark100.png")

        path = mvtb_path_to_datafile("data", "bunny.ply")
        self.assertTrue(os.path.isfile(path))

        with self.assertRaises(ValueError):
            mvtb_path_to_datafile("data", "__bunny100.png")

    def tearDown(self):
        # remove the test image file
        try:
            os.remove("./test-image.png")
        except FileNotFoundError:
            pass


# ------------------------------------------------------------------------ #
if __name__ == "__main__":

    unittest.main()
