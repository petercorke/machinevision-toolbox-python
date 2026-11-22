#!/usr/bin/env python
"""More tests for ``ImageRegionFeatures``: MSER indexing and printing, and OCR.

OpenCV 5 finds far fewer MSERs than OpenCV 4 on the same image (Issue 54), so
the indexing tests build an ``MSERFeature`` by hand with known points and
bounding boxes, which behaves the same on both.  The OCR tests use the real
``pytesseract`` where it and the ``tesseract`` program are installed, and a
stand-in for ``pytesseract`` for the code that filters and plots the words.
"""

import shutil

import matplotlib
import numpy as np
import numpy.testing as nt
import pytest

matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt

from machinevisiontoolbox import Image
from machinevisiontoolbox import ImageRegionFeatures as irf
from machinevisiontoolbox.ImageRegionFeatures import MSERFeature, OCRWord

# --------------------------------------------------------------------------- #
# MSER


@pytest.fixture
def msers():
    """Three regions with 3, 2 and 1 points, each a 2 x N array of (u, v)."""
    m = MSERFeature()
    m._points = [
        np.array([[1, 3, 5], [2, 4, 6]]),
        np.array([[7, 9], [8, 10]]),
        np.array([[0], [0]]),
    ]
    m._bboxes = np.array([[1, 2, 5, 6], [7, 8, 9, 10], [0, 0, 1, 1]])
    return m


class TestMSERIndexing:
    def test_length(self, msers):
        assert len(msers) == 3

    def test_empty(self):
        m = MSERFeature()
        assert len(m) == 0
        assert m.bbox.shape == (0, 4)
        assert m.points == []

    def test_slice(self, msers):
        sub = msers[1:]
        assert len(sub) == 2
        nt.assert_array_equal(sub.bbox, [[7, 8, 9, 10], [0, 0, 1, 1]])

    @pytest.mark.parametrize(
        "index",
        [
            [0, 2],
            (0, 2),
            np.array([0, 2]),
            np.array([True, False, True]),
        ],
        ids=["list", "tuple", "int array", "bool array"],
    )
    def test_sequence_of_indices(self, msers, index):
        sub = msers[index]
        assert len(sub) == 2
        nt.assert_array_equal(sub.bbox, [[1, 2, 5, 6], [0, 0, 1, 1]])
        nt.assert_array_equal(sub.points[0], [[1, 3, 5], [2, 4, 6]])
        nt.assert_array_equal(sub.points[1], [[0], [0]])

    def test_single_region_gives_arrays(self, msers):
        # a one-region object returns the array itself, not a list of one
        one = msers[2:]
        assert len(one) == 1
        nt.assert_array_equal(one.bbox, [0, 0, 1, 1])
        nt.assert_array_equal(one.points, [[0], [0]])

    def test_several_regions_give_lists(self, msers):
        assert len(msers.points) == 3
        assert msers.bbox.shape == (3, 4)

    @pytest.mark.xfail(
        strict=True,
        reason="an int index leaves the region's 2 x N point array in _points, "
        "so len() of the result is 2, not 1",
    )
    def test_integer_index_is_one_region(self, msers):
        assert len(msers[0]) == 1


class TestMSERPrinting:
    def test_several(self, msers):
        assert str(msers) == "MSER features, 3 regions"
        assert repr(msers) == str(msers)

    def test_single(self, msers):
        assert str(msers[2:]) == "MSER feature: u: 0 - 1, v: 0 - 1"

    @pytest.mark.xfail(
        strict=True,
        raises=IndexError,
        reason="str() of an object with no regions looks up the first bounding box",
    )
    def test_empty(self):
        str(MSERFeature())


class TestMSERDetection:
    def test_blank_image_finds_nothing(self):
        # covers the empty-result handling, which differs in OpenCV 4 and 5
        m = Image(np.zeros((50, 50), dtype=np.uint8)).MSER()
        assert len(m) == 0
        assert m.bbox.shape == (0, 4)

    def test_real_image(self):
        m = Image.Read("castle.png").MSER()
        if len(m) == 0:
            pytest.skip("OpenCV 5 finds no MSERs on this image, see Issue 54")
        bbox = m.bbox
        assert bbox.shape == (len(m), 4)
        # lrtb: left < right and top < bottom for every region
        assert (bbox[:, 2] > bbox[:, 0]).all()
        assert (bbox[:, 3] > bbox[:, 1]).all()
        # each point array is 2 x N, inside its own bounding box
        for p, b in zip(m.points, bbox):
            assert p.shape[0] == 2
            assert p[0].min() >= b[0] and p[0].max() <= b[2]
            assert p[1].min() >= b[1] and p[1].max() <= b[3]


# --------------------------------------------------------------------------- #
# OCR

WORD = {
    "text": ["", "HELLO", "low", "maybe"],
    "conf": ["-1", 96, 12, 70],
    "left": [0, 10, 40, 80],
    "top": [0, 20, 30, 40],
    "width": [100, 50, 20, 25],
    "height": [60, 15, 10, 12],
}


class TestOCRWord:
    def test_properties(self):
        w = OCRWord(WORD, 1)
        assert w.text == "HELLO"
        assert w.conf == 96
        assert (w.l, w.t, w.w, w.h) == (10, 20, 50, 15)
        assert w.ltrb == [10, 20, 60, 35]
        assert str(w) == "HELLO (96%)"
        assert repr(w) == str(w)

    def test_plot(self):
        plt.figure()
        OCRWord(WORD, 1).plot()
        ax = plt.gca()
        assert len(ax.patches) == 1  # the label box
        plt.close("all")

    def test_plot_falls_back_when_linestyle_is_not_accepted(self, monkeypatch):
        calls = []

        def fake(text, lb, wh, color, linestyle=None):
            calls.append(linestyle)
            if linestyle is not None:
                raise TypeError("linestyle passed twice")

        monkeypatch.setattr(irf, "plot_labelbox", fake)
        OCRWord(WORD, 1).plot()
        assert calls == ["--", None]


class FakePytesseract:
    """Stand-in that returns the dictionary WORD, as image_to_data would."""

    class Output:
        DICT = "dict"

    @staticmethod
    def image_to_data(image, output_type):
        assert output_type == "dict"
        return WORD


class TestOCRFiltering:
    @pytest.fixture(autouse=True)
    def fake(self, monkeypatch):
        monkeypatch.setattr(irf, "_pytesseract", FakePytesseract)
        monkeypatch.setattr(irf, "_pytesseract_available", True)

    def test_words_above_the_confidence_threshold(self):
        words = Image.Zeros(size=(100, 60), dtype="uint8").ocr(minconf=50)
        # "-1" marks a non-word row and "low" is under the threshold
        assert [w.text for w in words] == ["HELLO", "maybe"]

    def test_lower_threshold_keeps_more(self):
        words = Image.Zeros(size=(100, 60), dtype="uint8").ocr(minconf=10)
        assert [w.text for w in words] == ["HELLO", "low", "maybe"]

    def test_plot(self):
        plt.figure()
        Image.Zeros(size=(100, 60), dtype="uint8").ocr(minconf=50, plot=True)
        assert len(plt.gca().patches) == 2
        plt.close("all")


def test_ocr_without_pytesseract(monkeypatch, capsys):
    monkeypatch.setattr(irf, "_pytesseract_available", False)
    words = Image.Zeros(size=(10, 10), dtype="uint8").ocr()
    assert words == []
    assert "pytesseract is required" in capsys.readouterr().out


@pytest.mark.skipif(
    not irf._pytesseract_available or shutil.which("tesseract") is None,
    reason="pytesseract and the tesseract program are needed",
)
def test_ocr_reads_rendered_text():
    img = np.full((60, 300), 255, dtype=np.uint8)
    cv2.putText(img, "HELLO", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
    words = Image(img).ocr(minconf=10)
    assert "HELLO" in [w.text for w in words]
    word = next(w for w in words if w.text == "HELLO")
    # the bounding box lies on the drawn text
    assert 0 <= word.l < 30 and 0 <= word.t < 30
    assert word.ltrb[2] > 100
