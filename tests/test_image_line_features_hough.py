#!/usr/bin/env python
"""Tests for the Hough line methods of ``ImageLineFeatures``.

The test image has one horizontal line (v = 50, 80 pixels long) and one
vertical line (u = 30, 60 pixels long), so the line parameters are known:
a horizontal line is (theta, rho) = (pi/2, 50) and a vertical one is (0, 30),
using u cos(theta) + v sin(theta) = rho.  Nothing is mocked.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt
import pytest

from machinevisiontoolbox import Image


@pytest.fixture
def lines_image():
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[50, 10:90] = 255  # horizontal, 80 pixels
    arr[20:80, 30] = 255  # vertical, 60 pixels
    return Image(arr)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def nearest(lines, theta, rho, tol_theta=0.03, tol_rho=1.5):
    """True if some row of ``lines`` is within tolerance of (theta, rho)."""
    return bool(
        np.any(
            (np.abs(lines[:, 0] - theta) < tol_theta)
            & (np.abs(lines[:, 1] - rho) < tol_rho)
        )
    )


class TestLines:
    def test_finds_both_lines(self, lines_image):
        lines = lines_image.Hough().lines(minvotes=55)
        assert lines.ndim == 2 and lines.shape[1] == 2
        assert nearest(lines, np.pi / 2, 50)
        assert nearest(lines, 0, 30)

    def test_threshold_selects_the_stronger_line(self, lines_image):
        # only the 80 pixel line has more than 70 votes
        lines = lines_image.Hough().lines(minvotes=70)
        assert nearest(lines, np.pi / 2, 50)
        assert not nearest(lines, 0, 30)

    def test_no_lines_gives_an_empty_array(self):
        empty = Image(np.zeros((50, 50), dtype=np.uint8))
        assert empty.Hough().lines(minvotes=5).shape == (0, 2)

    def test_float_image_is_converted(self, lines_image):
        floating = Image(lines_image.array.astype(np.float32) / 255)
        lines = floating.Hough().lines(minvotes=55)
        assert nearest(lines, np.pi / 2, 50)

    def test_ntheta_sets_the_angle_step(self, lines_image):
        h = lines_image.Hough(ntheta=90)
        assert h.dtheta == pytest.approx(np.pi / 90)
        assert nearest(h.lines(minvotes=55), np.pi / 2, 50, tol_theta=0.05)


class TestLinesP:
    def test_segment_end_points(self, lines_image):
        segs = lines_image.Hough().lines_p(minvotes=40, minlinelength=30, seed=1)
        assert segs.ndim == 2 and segs.shape[1] == 4
        horizontal = segs[(segs[:, 1] == 50) & (segs[:, 3] == 50)]
        assert len(horizontal) >= 1
        # the segment spans most of the 80 pixel line
        assert horizontal[:, [0, 2]].min() <= 12
        assert horizontal[:, [0, 2]].max() >= 87

    def test_seed_makes_the_result_repeatable(self, lines_image):
        h = lines_image.Hough()
        a = h.lines_p(minvotes=40, seed=7)
        b = h.lines_p(minvotes=40, seed=7)
        nt.assert_array_equal(a, b)

    def test_min_length_rejects_short_segments(self, lines_image):
        segs = lines_image.Hough().lines_p(minvotes=20, minlinelength=70, seed=1)
        # only the 80 pixel line is long enough
        assert len(segs) >= 1
        length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        assert (length >= 70).all()

    def test_no_segments_gives_an_empty_array(self):
        empty = Image(np.zeros((50, 50), dtype=np.uint8))
        assert empty.Hough().lines_p(minvotes=5).shape == (0, 4)


class TestAccumulator:
    @pytest.fixture
    def h(self, lines_image):
        h = lines_image.Hough()
        h.accumulator()
        return h

    def test_attributes(self, h):
        assert h.A.ndim == 2
        assert len(h.extent) == 4
        # theta extent is in radians, rho extent in pixels
        assert -h.dtheta <= h.extent[0] < h.extent[1] <= np.pi + h.dtheta
        assert h.extent[2] < h.extent[3]
        # the highest threshold with any line is the number of pixels of the
        # longest line, and h.t records the last threshold that found one
        assert h.t == 80 or h.t == 79

    def test_votes_never_increase_with_the_threshold(self, h):
        votes = np.array(h.votes)
        assert (np.diff(votes) <= 0).all()
        assert votes[-1] >= 1

    def test_peak_is_the_longest_line(self, h):
        rho_index, theta_index = np.unravel_index(np.argmax(h.A), h.A.shape)
        theta_bins = np.linspace(h.extent[0], h.extent[1], h.A.shape[1] + 1)
        rho_bins = np.linspace(h.extent[2], h.extent[3], h.A.shape[0] + 1)
        theta = (theta_bins[theta_index] + theta_bins[theta_index + 1]) / 2
        rho = (rho_bins[rho_index] + rho_bins[rho_index + 1]) / 2
        # the accumulator counts how many thresholds each line survives, so
        # the longest line is the strongest cell
        assert theta == pytest.approx(np.pi / 2, abs=0.05)
        assert rho == pytest.approx(50, abs=1.5)

    def test_skip_gives_a_coarser_vote_list(self, lines_image):
        fine = lines_image.Hough()
        fine.accumulator(skip=1)
        coarse = lines_image.Hough()
        coarse.accumulator(skip=10)
        assert len(coarse.votes) < len(fine.votes)


class TestPlotting:
    def test_plot_hough_lines(self, lines_image):
        h = lines_image.Hough()
        lines = h.lines(minvotes=55)
        plt.figure()
        plt.axis([0, 100, 100, 0])
        before = len(plt.gca().lines)
        h.plot_lines(lines, "r")
        assert len(plt.gca().lines) > before

    def test_plot_segments(self, lines_image):
        h = lines_image.Hough()
        segs = h.lines_p(minvotes=40, seed=1)
        plt.figure()
        h.plot_lines(segs, "b")
        ax = plt.gca()
        assert len(ax.lines) == len(segs)
        nt.assert_array_equal(ax.lines[0].get_xdata(), segs[0, [0, 2]])
        nt.assert_array_equal(ax.lines[0].get_ydata(), segs[0, [1, 3]])

    def test_plot_accumulator_computes_it_if_needed(self, lines_image):
        h = lines_image.Hough()
        assert h.A is None
        h.plot_accumulator()
        assert h.A is not None
        ax = plt.gcf().axes[0]
        assert len(ax.images) == 1
        assert ax.get_xlabel().startswith(r"$\theta$")
        assert ax.get_ylabel().startswith(r"$\rho$")
        assert ax.get_xlim() == (0, np.pi)
        assert plt.gcf().canvas.manager.get_window_title() == "Hough Accumulator"

    def test_plot_accumulator_into_given_axes(self, lines_image):
        h = lines_image.Hough()
        fig, ax = plt.subplots()
        h.plot_accumulator(ax=ax, cmap="gray")
        assert len(ax.images) == 1
        assert ax.images[0].get_cmap().name == "gray"
        # no default title when the axes are supplied
        assert plt.gcf().canvas.manager.get_window_title() != "Hough Accumulator"

    def test_plot_accumulator_with_title(self, lines_image):
        h = lines_image.Hough()
        fig, ax = plt.subplots()
        h.plot_accumulator(ax=ax, title="my accumulator")
        assert fig.canvas.manager.get_window_title() == "my accumulator"
