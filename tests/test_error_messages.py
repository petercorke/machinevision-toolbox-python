#!/usr/bin/env python
"""Error messages should say what was received and what is allowed."""

import numpy as np
import pytest

from machinevisiontoolbox import CentralCamera, Image, mkgrid
from machinevisiontoolbox.base.color import color2name


class TestMessagesNameTheValue:
    def test_unknown_colorspace(self):
        with pytest.raises(ValueError, match="unknown colorspace 'foo'"):
            color2name([1, 0, 0], "foo")

    def test_wrong_side_length(self):
        with pytest.raises(ValueError, match="1 or 2 elements, got 3"):
            mkgrid(side=[1, 2, 3])

    def test_convolve_mode(self):
        with pytest.raises(ValueError, match="mode must be one of.*got 'wrong'"):
            Image.Squares(1, size=20).convolve(np.ones((3, 3)), mode="wrong")

    def test_planes_mismatch(self):
        a = Image(np.zeros((4, 4, 3), dtype=np.uint8), colororder="RGB")
        b = Image(np.zeros((4, 4, 2), dtype=np.uint8))
        with pytest.raises(ValueError, match="3 and 2"):
            a + b

    def test_fundamental_matrix_needs_a_camera_or_pose(self):
        camera = CentralCamera.Default()
        with pytest.raises(ValueError, match="CentralCamera or SE3, got int"):
            camera.F(5)
        with pytest.raises(ValueError, match="fundamental matrix, got str"):
            camera.E("a")

    def test_integer_arguments_report_the_value(self):
        with pytest.raises(ValueError, match="n must be greater than 0, got 0"):
            Image.Squares(1, size=20).erode(np.ones((3, 3)), n=0)

    def test_base_class_step_names_the_class(self):
        from machinevisiontoolbox.VisualServo import VisualServo

        with pytest.raises(NotImplementedError, match="must implement step"):
            VisualServo.step(object.__new__(VisualServo), 0.0)
