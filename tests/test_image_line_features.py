#!/usr/bin/env python

import unittest

import numpy as np
import numpy.testing as nt

from machinevisiontoolbox import Image


class TestImageLineFeatures(unittest.TestCase):
    # new test
    def test_hough(self):
        """Test Hough transform for line detection"""
        # Create simple image with lines
        im = Image.Zeros(size=(100, 100), dtype="uint8")
        # Draw a line (diagonal)
        # TODO: Add actual line drawing
        # hough_result = im.Hough()
        # self.assertIsNotNone(hough_result)
        pass

    # new test
    def test_accumulator(self):
        """Test Hough accumulator access"""
        # TODO: Perform Hough transform and access accumulator
        pass

    # new test
    def test_lines(self):
        """Test standard Hough line detection"""
        # Create image with known lines
        im = Image.Zeros(size=(100, 100), dtype="uint8")
        # TODO: Draw lines and detect them
        pass

    def test_lines_p(self):
        """Test probabilistic Hough line detection

        Regression: cv2.HoughLinesP returns (N,1,4) on OpenCV 4 but a
        true (N,4) on OpenCV 5 -- lines_p() must not assume the old
        singleton middle dimension.
        """
        arr = np.zeros((100, 100), dtype="uint8")
        arr[50, 10:90] = 255
        im = Image(arr)

        lines = im.Hough().lines_p(minvotes=20)
        self.assertEqual(lines.ndim, 2)
        self.assertEqual(lines.shape[1], 4)
        self.assertGreater(lines.shape[0], 0)

    # new test
    def test_plot_accumulator(self):
        """Test plotting Hough accumulator"""
        # TODO: Test accumulator visualization
        pass

    # new test
    def test_plot_lines(self):
        """Test plotting detected lines"""
        # TODO: Test line visualization
        pass


# ----------------------------------------------------------------------- #
if __name__ == "__main__":
    unittest.main()
