"""A missing OpenCV contrib module gives a clear error, and a present one is untouched."""

import types

import pytest

from machinevisiontoolbox import _cv2_contrib as contrib


def plain_cv2() -> types.ModuleType:
    """A cv2 without any contrib module, as in JupyterLite."""
    cv2 = types.ModuleType("cv2")
    cv2.SIFT_create = lambda: "sift"
    return cv2


def test_a_missing_module_is_replaced_by_a_stand_in():
    cv2 = plain_cv2()
    added = contrib.install(cv2)
    assert "ximgproc" in added and "xfeatures2d" in added
    assert hasattr(cv2, "ximgproc")


def test_using_a_stand_in_says_why():
    cv2 = plain_cv2()
    contrib.install(cv2)
    with pytest.raises(contrib.ContribNotAvailableError) as info:
        cv2.ximgproc.segmentation.createGraphSegmentation()
    message = str(info.value)
    assert "ximgproc" in message and "not installed" in message
    assert "pip install opencv-contrib-python" in message
    assert "JupyterLite" in message


def test_the_error_is_an_attribute_error_so_hasattr_still_works():
    cv2 = plain_cv2()
    contrib.install(cv2)
    assert not hasattr(cv2.xfeatures2d, "BriefDescriptorExtractor_create")
    with pytest.raises(AttributeError):
        cv2.xfeatures2d.BriefDescriptorExtractor_create


def test_a_module_that_is_present_is_left_alone():
    cv2 = plain_cv2()
    real = types.ModuleType("ximgproc")
    real.thinning = lambda: "thin"
    cv2.ximgproc = real
    added = contrib.install(cv2)
    assert "ximgproc" not in added
    assert cv2.ximgproc is real and cv2.ximgproc.thinning() == "thin"


def test_main_opencv_is_not_touched():
    cv2 = plain_cv2()
    contrib.install(cv2)
    assert cv2.SIFT_create() == "sift"
    assert not hasattr(cv2, "aruco")  # in main OpenCV, so never stood in for


def test_installing_twice_adds_nothing_more():
    cv2 = plain_cv2()
    contrib.install(cv2)
    assert contrib.install(cv2) == []


def test_the_real_cv2_still_works_after_import():
    import cv2

    __import__("machinevisiontoolbox")

    assert hasattr(cv2, "SIFT_create")
