#!/usr/bin/env python

import unittest
import unittest.mock

import cv2
import numpy as np

from machinevisiontoolbox import Image


class TestImageLineFeatures(unittest.TestCase):
    # new test
    def test_hough(self):
        """Test Hough transform for line detection"""
        # Create simple image with lines
        Image.Zeros(size=(100, 100), dtype="uint8")
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
        Image.Zeros(size=(100, 100), dtype="uint8")
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


class TestHoughNoLines(unittest.TestCase):
    """OpenCV 4 and 5.0 return None from HoughLines when nothing is found,
    OpenCV 5.1 returns an empty array."""

    def setUp(self):
        arr = np.zeros((60, 60), dtype="uint8")
        arr[30, 10:50] = 255
        self.hough = Image(arr).Hough()

    @staticmethod
    def as_opencv_5_1(function):
        """Wrap a cv2 Hough function so that "no lines" is an empty array."""
        calls = []

        def wrapper(*args, **kwargs):
            calls.append(1)
            if len(calls) > 1000:
                raise RuntimeError("the loop did not terminate")
            result = function(*args, **kwargs)
            if result is None:
                shape = (0, 1, 4) if function is cv2.HoughLinesP else (0, 1, 2)
                return np.zeros(shape, dtype=np.float32)
            return result

        return wrapper

    def test_lines_with_an_empty_array(self):
        import cv2

        with unittest.mock.patch.object(
            cv2, "HoughLines", self.as_opencv_5_1(cv2.HoughLines)
        ):
            self.assertEqual(self.hough.lines(minvotes=100).shape, (0, 2))

    def test_lines_p_with_an_empty_array(self):
        import cv2

        with unittest.mock.patch.object(
            cv2, "HoughLinesP", self.as_opencv_5_1(cv2.HoughLinesP)
        ):
            self.assertEqual(self.hough.lines_p(minvotes=100).shape, (0, 4))


class TestHoughAccumulator(unittest.TestCase):
    """The accumulator is computed by voting, so it can be checked exactly."""

    def setUp(self):
        arr = np.zeros((100, 100), dtype="uint8")
        arr[50, 10:90] = 255  # horizontal, theta = pi/2, rho = 50, 80 votes
        arr[20:80, 30] = 255  # vertical, theta = 0, rho = 30, 60 votes
        self.arr = arr
        self.hough = Image(arr).Hough()
        self.hough.accumulator()

    def test_matches_voting_one_pixel_at_a_time(self):
        arr = np.zeros((6, 8), dtype="uint8")
        arr[1, 2] = arr[4, 5] = arr[3, 0] = 255
        h = Image(arr).Hough(ntheta=12)
        h.accumulator()
        nrho = (6 + 8) * 2 + 1
        expected = np.zeros((nrho, 12), dtype=int)
        for v, u in zip(*np.nonzero(arr)):
            for n in range(12):
                theta = n * (np.pi / 12)
                rho = int(round(u * np.cos(theta) + v * np.sin(theta)))
                expected[rho + (nrho - 1) // 2, n] += 1
        np.testing.assert_array_equal(h.A, expected)

    def test_every_set_pixel_votes_once_per_angle(self):
        self.assertEqual(self.hough.nz, 80 + 60 - 1)
        self.assertEqual(self.hough.A.sum(), self.hough.nz * 180)
        self.assertEqual(self.hough.A.shape, (401, 180))

    def test_the_peaks_are_the_lines(self):
        A = self.hough.A
        centre = (A.shape[0] - 1) // 2
        self.assertEqual(A[centre + 50, 90], 80)  # theta = 90 degrees
        self.assertEqual(A[centre + 30, 0], 60)  # theta = 0
        self.assertEqual(np.sort(A.ravel())[-1], 80)

    def test_votes_count_the_lines_at_each_threshold(self):
        import cv2

        h = self.hough
        for t in (0, 10, 30, 59, 60, 61, 79, 80):
            lines = cv2.HoughLines(h.image, h.drho, h.dtheta, t)
            expected = 0 if lines is None else len(lines)
            n = h.votes[t] if t < len(h.votes) else 0
            self.assertEqual(n, expected, f"threshold {t}")

    def test_votes_fall_with_threshold_and_t_is_the_last_with_a_line(self):
        votes = np.array(self.hough.votes)
        self.assertTrue((np.diff(votes) <= 0).all())
        self.assertGreater(votes[-1], 0)
        self.assertEqual(self.hough.t, len(votes) - 1)
        self.assertEqual(self.hough.t, 79)  # the longest line has 80 votes

    def test_empty_image(self):
        h = Image(np.zeros((20, 20), dtype="uint8")).Hough()
        h.accumulator()
        self.assertEqual(h.A.sum(), 0)
        self.assertEqual(h.votes, [])
        self.assertEqual(h.t, -1)

    def test_quantization_options(self):
        h = Image(self.arr).Hough(ntheta=90, drho=2)
        h.accumulator()
        self.assertEqual(h.A.shape, (200, 90))

    def test_extent_gives_the_edges_of_the_array(self):
        left, right, bottom, top = self.hough.extent
        self.assertAlmostEqual(left, -np.pi / 360)
        self.assertAlmostEqual(right, np.pi - np.pi / 360)
        # rho = 0 is the middle row, so the range is symmetric
        self.assertAlmostEqual(bottom, -top)
        self.assertAlmostEqual(top - bottom, self.hough.A.shape[0])

    def test_skip_is_no_longer_accepted(self):
        with self.assertRaises(TypeError):
            self.hough.accumulator(skip=2)


# ----------------------------------------------------------------------- #
if __name__ == "__main__":
    unittest.main()
