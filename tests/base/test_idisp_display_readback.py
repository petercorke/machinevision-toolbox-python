"""
Display-correctness readback tests for idisp()'s two backends.

Unlike the "doesn't crash" style coverage in test_io.py's TestIdisp* classes,
these tests verify the actual *content* handed to (or rendered by) each
backend, across a matrix of pixel dtypes -- exactly the class of bug found
2026-08 (Image() silently corrupting dtype-converted data before it ever
reached idisp(); see test_dtype_resolution.py's TestImageConstructorMaxintval).
idisp()/cv2.imshow's own display conversion was independently verified
correct at the time (real on-screen pixel readback via screenshot +
template matching); these tests pin that down as an automated regression.

Matplotlib path: genuinely reads back rendered pixels via the Agg backend's
off-screen buffer_rgba() -- headless-safe and deterministic on every
platform, no virtual display required.

OpenCV path: there is no portable, cross-platform Python API to read back
the actual rendered content of a native HighGUI window (Linux/macOS/Windows
each differ, and CI runs all three). Instead we intercept the array handed
to cv2.imshow() -- since cv2.imshow's own internal conversion is already
independently verified correct (see the fix/image-ctor-drops-maxintval PR
description), asserting the *correct* array reaches it is equivalent to
asserting the *correct* image is displayed.
"""

import unittest
from unittest.mock import patch

import matplotlib
import numpy as np

from machinevisiontoolbox.base import imageio
from machinevisiontoolbox.base.imageio import idisp


def _mpl_pixel_rgba(fig, ax, col: float, row: float) -> np.ndarray:
    """Read back the actual rendered RGBA pixel at image data coordinate (col, row)."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    x_disp, y_disp = ax.transData.transform((col, row))
    px = int(round(x_disp))
    # canvas buffer row 0 is the TOP of the image; matplotlib display
    # coordinates have y increasing upward from the bottom of the figure.
    py = buf.shape[0] - 1 - int(round(y_disp))
    return buf[py, px]


class TestIdispMatplotlibReadback(unittest.TestCase):
    """Matplotlib backend: verify actual rendered pixel colors, not just 'no crash'."""

    def setUp(self):
        self._prev_backend = matplotlib.get_backend()
        matplotlib.pyplot.switch_backend("Agg")

    def tearDown(self):
        matplotlib.pyplot.switch_backend(self._prev_backend)
        matplotlib.pyplot.close("all")

    def _assert_min_black_max_white(self, im: np.ndarray, msg: str):
        # 4x4 greyscale, min value at (0,0), max value at (3,3), everything
        # else at the midpoint -- default vrange (data min/max) + default
        # 'gray' colormap must render min as black, max as white.
        fig, ax = matplotlib.pyplot.subplots()
        try:
            idisp(im, matplotlib=True, fig=fig, ax=ax, block=None)
            lo = _mpl_pixel_rgba(fig, ax, 0, 0)
            hi = _mpl_pixel_rgba(fig, ax, 3, 3)
            self.assertTrue(np.all(lo[:3] <= 10), f"{msg}: min-value pixel not black: {lo}")
            self.assertTrue(np.all(hi[:3] >= 245), f"{msg}: max-value pixel not white: {hi}")
        finally:
            matplotlib.pyplot.close(fig)

    def test_uint8_full_range(self):
        im = np.full((4, 4), 128, dtype=np.uint8)
        im[0, 0] = 0
        im[3, 3] = 255
        self._assert_min_black_max_white(im, "uint8")

    def test_uint16_full_range(self):
        im = np.full((4, 4), 32768, dtype=np.uint16)
        im[0, 0] = 0
        im[3, 3] = 65535
        self._assert_min_black_max_white(im, "uint16 full-range")

    def test_uint16_12bit_narrow_range(self):
        # the enpeda/bridge scenario: real data only spans 0..4095, not the
        # full uint16 range. Matplotlib's auto vrange (actual data min/max)
        # must still stretch this to black..white -- unlike OpenCV's fixed
        # /256 conversion, which would render this almost entirely black.
        im = np.full((4, 4), 2048, dtype=np.uint16)
        im[0, 0] = 0
        im[3, 3] = 4095
        self._assert_min_black_max_white(im, "uint16 12-bit narrow-range")

    def test_float32_unit_range(self):
        im = np.full((4, 4), 0.5, dtype=np.float32)
        im[0, 0] = 0.0
        im[3, 3] = 1.0
        self._assert_min_black_max_white(im, "float32")

    def test_bool(self):
        im = np.zeros((4, 4), dtype=bool)
        im[3, 3] = True
        self._assert_min_black_max_white(im, "bool")


class TestIdispOpenCVReadback(unittest.TestCase):
    """OpenCV backend: verify the exact array handed to cv2.imshow()."""

    def _captured_imshow_arg(self, im: np.ndarray, **kwargs) -> np.ndarray:
        with patch.object(imageio.cv2, "imshow") as mock_imshow:
            idisp(im, matplotlib=False, **kwargs)
        self.assertEqual(mock_imshow.call_count, 1)
        _title, array = mock_imshow.call_args[0]
        return array

    def test_uint8_passthrough(self):
        im = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        got = self._captured_imshow_arg(im)
        self.assertEqual(got.dtype, np.uint8)
        np.testing.assert_array_equal(got, im)

    def test_uint16_full_range_passthrough(self):
        # OpenCV's own imshow does the 16-bit->8-bit scaling internally
        # (documented, independently verified); MVTB must hand it the
        # untouched original array, not a pre-truncated/pre-scaled one.
        im = np.random.randint(0, 65536, (10, 10), dtype=np.uint16)
        got = self._captured_imshow_arg(im)
        self.assertEqual(got.dtype, np.uint16)
        np.testing.assert_array_equal(got, im)

    def test_uint16_12bit_narrow_range_passthrough(self):
        im = np.random.randint(0, 4096, (10, 10), dtype=np.uint16)
        got = self._captured_imshow_arg(im)
        self.assertEqual(got.dtype, np.uint16)
        np.testing.assert_array_equal(got, im)

    def test_float32_passthrough(self):
        im = np.random.rand(10, 10).astype(np.float32)
        got = self._captured_imshow_arg(im)
        self.assertEqual(got.dtype, np.float32)
        np.testing.assert_array_equal(got, im)

    def test_color_rgb_reordered_to_bgr(self):
        # distinct per-channel values so a channel-order bug is unmissable
        im = np.zeros((5, 5, 3), dtype=np.uint8)
        im[..., 0] = 10  # R
        im[..., 1] = 20  # G
        im[..., 2] = 30  # B
        got = self._captured_imshow_arg(im, colororder="RGB")
        self.assertEqual(got.shape, (5, 5, 3))
        np.testing.assert_array_equal(got[..., 0], 30)  # B first
        np.testing.assert_array_equal(got[..., 1], 20)  # G
        np.testing.assert_array_equal(got[..., 2], 10)  # R last

    def test_color_already_bgr_unchanged(self):
        im = np.zeros((5, 5, 3), dtype=np.uint8)
        im[..., 0] = 30  # B
        im[..., 1] = 20  # G
        im[..., 2] = 10  # R
        got = self._captured_imshow_arg(im, colororder="BGR")
        np.testing.assert_array_equal(got, im)


if __name__ == "__main__":
    unittest.main()
