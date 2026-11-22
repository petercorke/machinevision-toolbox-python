#!/usr/bin/env python
"""Tests for the stereo and rectification methods of ``ImageMultiview``.

The stereo pairs are synthetic: the right image is the left image shifted by a
known number of pixels, so the true disparity is known exactly.  Nothing is
mocked except the matches object given to ``rectify_homographies``.
"""

from types import SimpleNamespace

import numpy as np
import numpy.testing as nt
import pytest

from machinevisiontoolbox import Image

DISPARITY = 8  # true disparity of the synthetic pair, in pixels


@pytest.fixture(scope="module")
def pair():
    """Random-texture stereo pair, the right image shifted left by DISPARITY."""
    rng = np.random.default_rng(0)
    left = rng.integers(0, 255, size=(40, 96), dtype=np.uint8)
    right = np.roll(left, -DISPARITY, axis=1)
    return Image(left), Image(right)


def interior(a, hw, margin):
    """Columns and rows where the disparity is well defined."""
    return a[hw + margin : -(hw + margin), DISPARITY + hw + margin : -(hw + margin)]


class TestStereoSimple:
    # stereo_simple takes float images here: given a uint8 image it pads with
    # NaN, which numpy casts to 0 with a RuntimeWarning.
    @pytest.fixture(scope="class")
    def result(self, pair):
        left, right = pair
        left = Image(left.array.astype(np.float32))
        right = Image(right.array.astype(np.float32))
        return left.stereo_simple(right, hw=2, drange=[4, 12])

    def test_recovers_disparity(self, result):
        disparity, similarity, dsi = result
        # well inside the image every window sees the true shift
        nt.assert_array_equal(disparity.array[10:30, 30:80], DISPARITY)
        # a perfect match has a ZNCC of 1
        nt.assert_allclose(similarity.array[10:30, 30:80], 1.0, atol=1e-5)

    def test_dsi_peak_is_at_the_true_disparity(self, result):
        _, _, dsi = result
        assert dsi.shape[2] == 8
        peak = np.argmax(dsi[10:30, 30:80, :], axis=2) + 4
        nt.assert_array_equal(peak, DISPARITY)

    def test_border_is_nan(self, result):
        disparity, _, _ = result
        assert np.isnan(disparity.array[:, :4]).all()

    @pytest.mark.xfail(
        strict=True,
        reason="the docstring says the outputs are the same size as the inputs "
        "but the window stack makes them larger by 2*hw in each direction",
    )
    def test_output_is_the_size_of_the_input(self, pair, result):
        left, _ = pair
        disparity, similarity, dsi = result
        assert disparity.shape == left.shape
        assert similarity.shape == left.shape
        assert dsi.shape[:2] == left.shape

    @pytest.mark.parametrize("drange", [[0, 12], (0, 12)])
    def test_zero_minimum_disparity(self, pair, drange):
        # a minimum of zero used to give the empty slice right_arr[:, :-0]
        left, right = pair
        left = Image(left.array.astype(np.float32))
        right = Image(right.array.astype(np.float32))
        disparity, similarity, dsi = left.stereo_simple(right, hw=2, drange=drange)
        assert dsi.shape[2] == 12
        nt.assert_array_equal(disparity.array[10:30, 30:80], DISPARITY)
        nt.assert_allclose(similarity.array[10:30, 30:80], 1.0, atol=1e-5)

    def test_zero_minimum_agrees_with_a_larger_one(self, pair):
        left, right = pair
        left = Image(left.array.astype(np.float32))
        right = Image(right.array.astype(np.float32))
        a, _, _ = left.stereo_simple(right, hw=2, drange=[0, 12])
        b, _, _ = left.stereo_simple(right, hw=2, drange=[4, 12])
        nt.assert_array_equal(a.array[10:30, 30:80], b.array[10:30, 30:80])


class TestDSIRefine:
    def test_parabola_gives_subpixel_peak(self):
        # every fibre is an exact parabola peaking at 3.3, so the
        # three-point fit must recover 3.3
        k = np.arange(7)
        dsi = np.tile(-((k - 3.3) ** 2), (4, 5, 1))
        refined, curvature = Image.DSI_refine(dsi, drange=(0, 6))
        assert refined.shape == (4, 5)
        nt.assert_allclose(refined.A, 3.3)
        # second difference of -(x^2) is -2
        nt.assert_allclose(curvature.A, -2.0)

    def test_default_range_is_the_span_of_the_peaks(self):
        # the peak is at index 3 everywhere, so the default range is (3, 3)
        # and no pixel is strictly inside it: all NaN
        k = np.arange(7)
        dsi = np.tile(-((k - 3.3) ** 2), (2, 2, 1))
        refined, _ = Image.DSI_refine(dsi)
        assert np.isnan(refined.A).all()

    def test_peaks_at_the_ends_are_nan(self):
        k = np.arange(7)
        dsi = np.zeros((1, 3, 7))
        dsi[0, 0] = -((k - 0.2) ** 2)  # peak at 0, no neighbour below
        dsi[0, 1] = -((k - 3.3) ** 2)  # interior peak
        dsi[0, 2] = -((k - 5.9) ** 2)  # peak at 6, no neighbour above
        refined, _ = Image.DSI_refine(dsi, drange=(0, 6))
        assert np.isnan(refined.A[0, 0])
        assert np.isnan(refined.A[0, 2])
        nt.assert_allclose(refined.A[0, 1], 3.3)


@pytest.mark.parametrize("method", ["stereo_BM", "stereo_SGBM"])
class TestOpenCVStereo:
    def test_recovers_disparity(self, pair, method):
        left, right = pair
        disparity = getattr(left, method)(right, hw=3, drange=[0, 16])
        assert disparity.shape == left.shape
        inner = interior(disparity.A, 3, 3)
        # the matcher agrees with the known shift for most interior pixels
        assert np.median(inner) == pytest.approx(DISPARITY, abs=0.5)
        assert np.mean(np.abs(inner - DISPARITY) < 1) > 0.9

    def test_integer_range_means_zero_to_n(self, pair, method):
        left, right = pair
        a = getattr(left, method)(right, hw=3, drange=16)
        b = getattr(left, method)(right, hw=3, drange=(0, 16))
        nt.assert_array_equal(a.A, b.A)

    def test_speckle_filter_is_accepted(self, pair, method):
        left, right = pair
        disparity = getattr(left, method)(right, hw=3, drange=[0, 16], speckle=(50, 2))
        assert disparity.shape == left.shape

    def test_small_block_is_an_error(self, pair, method):
        left, right = pair
        with pytest.raises(ValueError, match="block size too small"):
            getattr(left, method)(right, hw=1, drange=[0, 16])


class TestRectifyHomographies:
    def test_pure_horizontal_translation(self):
        # For a pair related by a horizontal translation the fundamental matrix
        # is the skew-symmetric matrix of the x axis, and rectification must
        # leave corresponding points on the same row.
        rng = np.random.default_rng(1)
        n = 20
        u1 = rng.uniform(10, 150, n)
        v = rng.uniform(10, 110, n)
        p1 = np.vstack((u1, v))
        p2 = np.vstack((u1 - 12 - 0.05 * v, v))  # shifted along the row
        F = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        matches = SimpleNamespace(inliers=SimpleNamespace(p1=p1, p2=p2))

        im = Image.Zeros(size=(160, 120), dtype="uint8")
        H1, H2 = im.rectify_homographies(matches, F)
        assert H1.shape == (3, 3)
        assert H2.shape == (3, 3)

        def warp(H, p):
            q = H @ np.vstack((p, np.ones(n)))
            return q[:2] / q[2]

        q1 = warp(H1, p1)
        q2 = warp(H2, p2)
        nt.assert_allclose(q1[1], q2[1], atol=1e-3)
