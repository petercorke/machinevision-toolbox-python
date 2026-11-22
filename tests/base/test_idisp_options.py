#!/usr/bin/env python
"""Tests of idisp()'s colormap, colorbar and pixel-readout options, and of the
OpenCV window helpers.

The Matplotlib tests use the off-screen Agg backend and check what idisp() hands to
Matplotlib (the colormap, the normalisation, the displayed values and the text shown
for a pixel), not just that the call returned.  The clipboard and OpenCV window tests
use stand-ins for ``pyclip`` and ``cv2``, since a real clipboard and real windows
depend on the machine the tests run on.
"""

import contextlib
import io
import unittest
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt
from matplotlib.backend_bases import KeyEvent

from machinevisiontoolbox.base import imageio
from machinevisiontoolbox.base.imageio import cv_destroy_window, idisp


class _Agg(unittest.TestCase):
    def setUp(self):
        self._prev_backend = matplotlib.get_backend()
        plt.switch_backend("Agg")

    def tearDown(self):
        plt.switch_backend(self._prev_backend)
        plt.close("all")

    def show(self, im, **kwargs):
        return idisp(im, matplotlib=True, block=None, **kwargs)


class TestColormaps(_Agg):
    def setUp(self):
        super().setUp()
        self.signed = np.array([[-2.0, -1.0], [1.0, 4.0]])

    def test_grey_and_invert(self):
        im = np.arange(16, dtype=np.uint8).reshape(4, 4)
        self.assertEqual(self.show(im, colormap="grey").get_cmap().name, "gray")
        self.assertEqual(self.show(im, colormap="invert").get_cmap().name, "Greys")

    def test_named_colormap_is_passed_on(self):
        im = np.arange(16, dtype=np.uint8).reshape(4, 4)
        self.assertEqual(self.show(im, colormap="viridis").get_cmap().name, "viridis")

    def test_random_colormap(self):
        im = np.arange(16, dtype=np.uint8).reshape(4, 4)
        h = self.show(im, colormap="random")
        # a different set of colors each time
        a = h.get_cmap()(np.linspace(0, 1, 10))
        b = self.show(im, colormap="random").get_cmap()(np.linspace(0, 1, 10))
        self.assertFalse(np.allclose(a, b))

    def test_signed_is_symmetric_about_zero(self):
        h = self.show(self.signed, colormap="signed")
        self.assertEqual(h.get_cmap().name, "bwr_r")
        self.assertIsInstance(h.norm, matplotlib.colors.CenteredNorm)
        # zero maps to the middle of the range: white
        nt.assert_allclose(h.get_cmap()(0.5)[:3], (1, 1, 1), atol=0.02)

    def test_signed_with_powernorm(self):
        h = self.show(self.signed, colormap="signed", powernorm=True)
        self.assertIsInstance(h.norm, matplotlib.colors.PowerNorm)

    def test_invsigned_zero_is_black(self):
        h = self.show(self.signed, colormap="invsigned")
        self.assertEqual(h.get_cmap().name, "signed")
        nt.assert_allclose(h.get_cmap()(0.5)[:3], (0, 0, 0), atol=0.02)
        # the larger magnitude here is positive, so the range runs from -max to max
        self.assertEqual(h.norm.vmax, 4.0)
        self.assertEqual(h.norm.vmin, -4.0)

    def test_invsigned_negative_larger(self):
        h = self.show(-self.signed, colormap="invsigned")
        self.assertEqual(h.norm.vmin, -4.0)
        self.assertEqual(h.norm.vmax, 4.0)

    def test_invsigned_ncolors_and_powernorm(self):
        h = self.show(self.signed, colormap="invsigned", ncolors=6)
        self.assertEqual(h.get_cmap().N, 6)
        h = self.show(self.signed, colormap="invsigned", powernorm=True)
        self.assertIsInstance(h.norm, matplotlib.colors.PowerNorm)


class TestBlack(_Agg):
    """``black`` lifts the displayed level of black pixels."""

    def test_float_image(self):
        im = np.array([[0.0, 0.5], [0.5, 1.0]])
        h = self.show(im, black=0.2)
        shown = np.asarray(h.get_array())
        nt.assert_allclose(shown, 0.8 * im + 0.2)
        self.assertEqual((h.norm.vmin, h.norm.vmax), (0, 1))

    def test_integer_image(self):
        im = np.array([[0, 100], [200, 255]], dtype=np.uint8)
        h = self.show(im, black=0.2)
        shown = np.asarray(h.get_array())
        # black (0) goes to 20% of full scale, and white stays white
        self.assertEqual(shown[0, 0], round(0.2 * 255))
        self.assertEqual(shown[1, 1], 255)
        self.assertEqual((h.norm.vmin, h.norm.vmax), (0, 255))

    def test_bool_image(self):
        im = np.array([[True, False], [False, True]])
        h = self.show(im, black=0.2)
        self.assertEqual((h.norm.vmin, h.norm.vmax), (0, 1))


class TestColorbar(_Agg):
    def setUp(self):
        super().setUp()
        self.im = np.arange(16, dtype=np.uint8).reshape(4, 4)

    def colorbar_axes(self, h):
        return [a for a in h.axes.figure.axes if a is not h.axes]

    def test_colorbar(self):
        h = self.show(self.im, colorbar=True)
        self.assertEqual(len(self.colorbar_axes(h)), 1)

    def test_no_colorbar_by_default(self):
        h = self.show(self.im)
        self.assertEqual(len(self.colorbar_axes(h)), 0)

    def test_colorbar_with_ncolors(self):
        h = self.show(self.im, colorbar=True, ncolors=4, colormap="viridis")
        self.assertEqual(len(self.colorbar_axes(h)), 1)

    def test_colorbar_with_labelled_ticks(self):
        labels = ["zero", "one", "two"]
        h = self.show(self.im, colorbar=dict(xticks=[[0, 7, 15], labels]))
        (cax,) = self.colorbar_axes(h)
        texts = [t.get_text() for t in cax.get_yticklabels()]
        self.assertEqual(texts, labels)

    def test_colorbar_options_are_passed_on(self):
        h = self.show(self.im, colorbar=dict(orientation="horizontal"))
        (cax,) = self.colorbar_axes(h)
        # a horizontal colorbar is wider than it is tall
        box = cax.get_position()
        self.assertGreater(box.width, box.height)


class TestPixelReadout(_Agg):
    """The text Matplotlib shows in the toolbar for the pixel under the cursor."""

    def readout(self, im, u, v, **kwargs):
        h = self.show(im, **kwargs)
        return h.axes.format_coord(u, v)

    def test_integer_monochrome(self):
        im = np.arange(16, dtype=np.uint8).reshape(4, 4)
        self.assertEqual(self.readout(im, 2, 1), "(2, 1): 6")

    def test_float_monochrome(self):
        im = np.linspace(0, 1, 16).reshape(4, 4)
        self.assertEqual(self.readout(im, 3, 3), "(3, 3): 1.000")

    def test_bool_monochrome(self):
        im = np.zeros((4, 4), dtype=bool)
        im[1, 2] = True
        self.assertEqual(self.readout(im, 2, 1), "(2, 1): True")

    def test_color_integer(self):
        im = np.zeros((4, 4, 3), dtype=np.uint8)
        im[1, 2] = (10, 20, 30)
        self.assertEqual(self.readout(im, 2, 1), "(2, 1): [10, 20, 30] RGB, uint8")

    def test_color_float(self):
        im = np.zeros((4, 4, 3), dtype=np.float32)
        im[1, 2] = (0.25, 0.5, 0.75)
        self.assertEqual(
            self.readout(im, 2, 1), "(2, 1): [0.250, 0.500, 0.750] RGB, float32"
        )

    def test_bgr_color_is_shown_in_bgr_order(self):
        im = np.zeros((4, 4, 3), dtype=np.uint8)
        im[1, 2] = (10, 20, 30)
        self.assertIn("[30, 20, 10] BGR", self.readout(im, 2, 1, colororder="BGR"))

    def test_outside_the_image_is_blank(self):
        im = np.zeros((4, 4), dtype=np.uint8)
        self.assertEqual(self.readout(im, 10, 10), "")

    def test_custom_format_replaces_the_default(self):
        im = np.zeros((4, 4), dtype=np.uint8)
        h = self.show(im, coordformat=lambda u, v: f"here {u:.0f},{v:.0f}")
        self.assertEqual(h.axes.format_coord(1, 2), "here 1,2")


class _FakeClipboard:
    """Stands in for ``pyclip``: copy() stores text, paste() returns it."""

    def __init__(self):
        self.text = ""

    def copy(self, text):
        self.text = text

    def paste(self):
        return self.text.encode()


class TestClipboardKeys(_Agg):
    def setUp(self):
        super().setUp()
        self.clip = _FakeClipboard()
        patches = [
            patch.object(imageio, "_pyclip", True),
            patch.object(imageio, "pyclip", self.clip, create=True),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        self.im = np.arange(16, dtype=np.uint8).reshape(4, 4)
        self.h = self.show(self.im)
        self.ax = self.h.axes

    def press(self, key, u=2, v=1):
        canvas = self.ax.figure.canvas
        x, y = self.ax.transData.transform((u, v))
        canvas.callbacks.process(
            "key_press_event", KeyEvent("key_press_event", canvas, key, x, y)
        )

    def test_c_copies_the_pixel_position(self):
        self.press("c", 2, 1)
        self.assertEqual(self.clip.text, "2,1")

    def test_C_appends_a_position(self):
        self.press("c", 2, 1)
        self.press("C", 3, 3)
        self.assertEqual(self.clip.text, "2,1\n3,3")

    def test_x_clears_the_clipboard(self):
        self.press("c")
        self.press("x")
        self.assertEqual(self.clip.text, "")

    def test_v_and_V_with_a_monochrome_image(self):
        # the value is a plain number
        self.press("v", 2, 1)
        self.assertEqual(self.clip.text, "2,1,6")
        self.press("V", 3, 3)
        self.assertEqual(self.clip.text, "2,1,6\n3,3,15")

    def test_v_with_a_float_image(self):
        plt.close("all")
        im = np.full((4, 4), 0.25, dtype=np.float32)
        self.ax = self.show(im).axes
        self.press("v", 2, 1)
        self.assertEqual(self.clip.text, "2,1,0.25")

    def test_v_and_V_with_a_color_image(self):
        plt.close("all")
        im = np.zeros((4, 4, 3), dtype=np.uint8)
        im[1, 2] = (10, 20, 30)
        im[3, 3] = (1, 2, 3)
        self.ax = self.show(im).axes
        self.press("v", 2, 1)
        self.assertEqual(self.clip.text, "2,1,(10, 20, 30)")
        self.press("V", 3, 3)
        self.assertEqual(self.clip.text, "2,1,(10, 20, 30)\n3,3,(1, 2, 3)")

    def test_other_keys_and_outside_clicks_do_nothing(self):
        self.press("z")
        self.assertEqual(self.clip.text, "")
        canvas = self.ax.figure.canvas
        canvas.callbacks.process(
            "key_press_event", KeyEvent("key_press_event", canvas, "c", -50, -50)
        )
        self.assertEqual(self.clip.text, "")


class TestKeyHelp(_Agg):
    """The ``?`` key prints the list of keys, which depends on whether pyclip is installed."""

    def press(self, key, x=None, y=None):
        h = self.show(np.zeros((4, 4), dtype=np.uint8))
        canvas = h.axes.figure.canvas
        if x is None:
            x, y = h.axes.transData.transform((2, 1))
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            canvas.callbacks.process(
                "key_press_event", KeyEvent("key_press_event", canvas, key, x, y)
            )
        return stream.getvalue()

    def with_clipboard(self, available):
        patches = [
            patch.object(imageio, "_pyclip", available),
            patch.object(
                imageio, "pyclip", _FakeClipboard() if available else None, create=True
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_help_with_clipboard_keys(self):
        self.with_clipboard(True)
        text = self.press("?")
        for line in (
            "?  show this help",
            "q  close the window",
            "c  copy",
            "C  append",
            "v  copy",
            "V  append",
            "x  clear",
        ):
            self.assertIn(line, text)
        self.assertNotIn("need pyclip", text)

    def test_help_without_clipboard_keys(self):
        self.with_clipboard(False)
        text = self.press("?")
        self.assertIn("?  show this help", text)
        self.assertIn("need pyclip", text)
        # the individual clipboard keys are not listed
        self.assertNotIn("append the pixel position", text)

    def test_help_works_with_the_cursor_outside_the_image(self):
        self.with_clipboard(True)
        self.assertIn("idisp keys:", self.press("?", -50, -50))

    def test_other_keys_print_nothing(self):
        self.with_clipboard(True)
        self.assertEqual(self.press("z"), "")

    def test_help_text_is_concise(self):
        for available in (True, False):
            lines = imageio._idisp_key_help(available).splitlines()
            self.assertLessEqual(len(lines), 12)
            self.assertTrue(all(len(line) < 80 for line in lines))

    def test_help_lists_matplotlib_keys(self):
        self.assertIn("h home", imageio._idisp_key_help(False))


class TestOpenCVWindows(unittest.TestCase):
    """The OpenCV window functions, with cv2 replaced by a recorder."""

    def setUp(self):
        self.calls = []
        self.keys = []
        calls = self.calls
        keys = self.keys

        class FakeCV2:
            WINDOW_AUTOSIZE = 1

            def namedWindow(self, *a):
                calls.append(("namedWindow", *a))

            def imshow(self, title, im):
                calls.append(("imshow", title, im.shape))

            def waitKey(self, delay=0):
                calls.append(("waitKey", delay))
                return keys.pop(0) if keys else -1

            def destroyWindow(self, title):
                calls.append(("destroyWindow", title))

            def destroyAllWindows(self):
                calls.append(("destroyAllWindows",))

        p = patch.object(imageio, "cv2", FakeCV2())
        p.start()
        self.addCleanup(p.stop)

    def names(self):
        return [c[0] for c in self.calls]

    def test_destroy_all(self):
        cv_destroy_window("all")
        self.assertEqual(self.names(), ["destroyAllWindows", "waitKey"])

    def test_destroy_named_window_waits_for_q(self):
        self.keys[:] = [ord("a"), ord("q")]
        cv_destroy_window("view", block=True)
        # keys other than q are ignored
        self.assertEqual(
            self.names(), ["waitKey", "waitKey", "destroyWindow", "waitKey"]
        )
        self.assertIn(("destroyWindow", "view"), self.calls)

    def test_destroy_without_blocking(self):
        cv_destroy_window("view", block=False)
        self.assertEqual(self.names(), ["destroyWindow", "waitKey"])

    def test_destroy_without_a_title(self):
        self.keys[:] = [ord("q")]
        cv_destroy_window(None, block=True)
        self.assertNotIn("destroyWindow", self.names())

    def test_color_image_is_shown_in_bgr_order(self):
        im = np.zeros((3, 4, 3), dtype=np.uint8)
        idisp(im, matplotlib=False, title="t", fps=10)
        self.assertIn(("imshow", "t", (3, 4, 3)), self.calls)
        # fps waits one frame time
        self.assertIn(("waitKey", 100), self.calls)

    def test_blocking_display_waits_for_q_then_closes(self):
        im = np.zeros((3, 4), dtype=np.uint8)
        self.keys[:] = [-1, ord("q")]
        idisp(im, matplotlib=False, title="t", block=True)
        self.assertIn(("destroyWindow", "t"), self.calls)


if __name__ == "__main__":
    unittest.main()
