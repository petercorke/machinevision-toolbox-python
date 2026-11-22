#!/usr/bin/env python
"""Tests of the colorspace name -> OpenCV conversion code table, and of the Lab
chromaticity-diagram background image."""

import cv2
import numpy as np
import pytest

from machinevisiontoolbox.base.color import _ab_chromaticity_diagram, _convertflag

# (source, destination, OpenCV constant) for every conversion the table supports
FROM_RGB = [
    ("grey", "COLOR_RGB2GRAY"),
    ("gray", "COLOR_RGB2GRAY"),
    ("xyz", "COLOR_RGB2XYZ"),
    ("xyz_709", "COLOR_RGB2XYZ"),
    ("ycrcb", "COLOR_RGB2YCrCb"),
    ("hsv", "COLOR_RGB2HSV"),
    ("hls", "COLOR_RGB2HLS"),
    ("lab", "COLOR_RGB2Lab"),
    ("l*a*b*", "COLOR_RGB2Lab"),
    ("luv", "COLOR_RGB2Luv"),
    ("l*u*v*", "COLOR_RGB2Luv"),
]
FROM_BGR = [(dst, name.replace("RGB2", "BGR2")) for dst, name in FROM_RGB]
TO_RGB_OR_BGR = [
    ("xyz", "XYZ"),
    ("xyz_709", "XYZ"),
    ("ycrcb", "YCrCb"),
    ("hsv", "HSV"),
    ("hls", "HLS"),
    ("lab", "Lab"),
    ("l*a*b*", "Lab"),
    ("luv", "Luv"),
    ("l*u*v*", "Luv"),
]


class TestConvertFlag:
    @pytest.mark.parametrize("dst, name", FROM_RGB)
    def test_from_rgb(self, dst, name):
        assert _convertflag("rgb", dst) == getattr(cv2, name)

    @pytest.mark.parametrize("dst, name", FROM_BGR)
    def test_from_bgr(self, dst, name):
        assert _convertflag("bgr", dst) == getattr(cv2, name)

    @pytest.mark.parametrize("src, space", TO_RGB_OR_BGR)
    def test_to_rgb(self, src, space):
        assert _convertflag(src, "rgb") == getattr(cv2, f"COLOR_{space}2RGB")

    @pytest.mark.parametrize("src, space", TO_RGB_OR_BGR)
    def test_to_bgr(self, src, space):
        assert _convertflag(src, "bgr") == getattr(cv2, f"COLOR_{space}2BGR")

    def test_names_ignore_case_and_colons(self):
        assert _convertflag("RGB", "HSV") == cv2.COLOR_RGB2HSV
        assert _convertflag("R:G:B", "Lab") == cv2.COLOR_RGB2Lab
        assert _convertflag("B:G:R", "Gray") == cv2.COLOR_BGR2GRAY

    def test_unknown_destination_from_rgb(self):
        with pytest.raises(ValueError, match="not known"):
            _convertflag("rgb", "cmyk")

    @pytest.mark.parametrize(
        "src, dst",
        [
            ("bgr", "cmyk"),  # unknown destination from bgr
            ("hsv", "lab"),  # only conversions to rgb and bgr exist
            ("lab", "hsv"),
            ("grey", "rgb"),  # no way back from grey
            ("cmyk", "rgb"),  # unknown source
            ("xyz", "ycrcb"),
        ],
    )
    def test_unsupported_conversions_raise(self, src, dst):
        with pytest.raises(ValueError, match="unknown conversion"):
            _convertflag(src, dst)


class TestAbChromaticityDiagram:
    def test_shape_and_range(self):
        img = _ab_chromaticity_diagram(L=70, N=64)
        assert img.shape == (64, 64, 3)
        assert img.min() >= 0 and img.max() <= 1

    def test_outside_the_gamut_circle_is_white(self):
        img = _ab_chromaticity_diagram(N=64)
        for corner in (img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]):
            np.testing.assert_array_equal(corner, [1, 1, 1])

    def test_centre_is_neutral(self):
        # a = b = 0 is grey: equal red, green and blue
        img = _ab_chromaticity_diagram(L=50, N=65)
        centre = img[32, 32]
        assert np.ptp(centre) < 0.05

    def test_a_axis_runs_left_to_right(self):
        # green (negative a) on the left, red (positive a) on the right
        img = _ab_chromaticity_diagram(L=60, N=65)
        left, right = img[32, 20], img[32, 44]
        assert left[1] > left[0]
        assert right[0] > right[1]

    def test_b_axis_runs_bottom_to_top(self):
        # blue (negative b) at the bottom, yellow (positive b) at the top: the
        # image is flipped so that b increases upwards
        img = _ab_chromaticity_diagram(L=60, N=65)
        top, bottom = img[20, 32], img[44, 32]
        assert top[2] < top[0]
        assert bottom[2] > bottom[0]
