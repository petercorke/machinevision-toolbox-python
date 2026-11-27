#!/usr/bin/env python
"""Tests of pickpoints().

A real click cannot be automated, so Matplotlib's ``ginput`` (which waits for clicks)
and OpenCV's mouse functions are replaced.  Everything else is real: the Agg backend,
the pyplot module, and the conversion of the clicked points to an array.
"""

import contextlib
import io
import unittest
from unittest.mock import patch

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt

from machinevisiontoolbox.base import imageio
from machinevisiontoolbox.base.imageio import pickpoints


class TestPickpointsMatplotlib(unittest.TestCase):
    def setUp(self):
        self._prev_backend = matplotlib.get_backend()
        plt.switch_backend("Agg")

    def tearDown(self):
        plt.switch_backend(self._prev_backend)
        plt.close("all")

    def test_points_are_returned_one_per_column(self):
        clicks = [(1.0, 2.0), (3.5, 4.5), (5.0, 6.0)]
        with patch.object(plt, "ginput", return_value=clicks) as ginput:
            points = pickpoints(None)
        ginput.assert_called_once_with(None)
        self.assertEqual(points.shape, (2, 3))
        nt.assert_array_equal(points, np.array(clicks).T)

    def test_number_of_points_is_passed_on(self):
        with patch.object(plt, "ginput", return_value=[(1.0, 2.0)]) as ginput:
            points = pickpoints(None, n=1)
        ginput.assert_called_once_with(1)
        self.assertEqual(points.shape, (2, 1))


class TestPickpointsOpenCV(unittest.TestCase):
    def test_click_prints_the_position_and_nothing_is_returned(self):
        callbacks = []

        with (
            patch.object(
                imageio.cv2,
                "setMouseCallback",
                lambda name, f: callbacks.append((name, f)),
            ),
            patch.object(imageio.cv2, "waitKey", return_value=-1) as wait,
        ):
            result = pickpoints(None, matplotlib=False)

        self.assertIsNone(result)
        wait.assert_called_once_with(0)
        ((name, click),) = callbacks
        self.assertEqual(name, "image")

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            click(cv2.EVENT_LBUTTONDOWN, 12, 34, 0, None)  # left button down
            click(cv2.EVENT_MOUSEMOVE, 50, 60, 0, None)  # other events are ignored
        self.assertEqual(stream.getvalue().split(), ["12", "34"])


if __name__ == "__main__":
    unittest.main()
