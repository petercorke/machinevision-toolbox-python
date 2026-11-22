#!/usr/bin/env python
"""More tests for ``ImageIO``: reading, EXIF metadata, pixel and window display,
stereo display and anaglyphs.

Images are synthetic or written to a temporary folder, with known values.  The
only stand-in is for the native file-chooser dialogue in ``Image.Read()``.
"""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt
import PIL.Image
import pytest
from matplotlib.backend_bases import MouseEvent

from machinevisiontoolbox import Image

DATA = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------------------- #
# Read


class TestRead:
    def test_no_filename_asks_the_file_chooser(self, monkeypatch):
        from machinevisiontoolbox import ImageIO

        monkeypatch.setattr(ImageIO, "_pick_imagefile", lambda: DATA / "stripes.png")
        im = Image.Read()
        assert im.shape == (400, 1000, 3)

    @pytest.mark.parametrize("bad", [5, 1.5, ["a.png"], object()])
    def test_filename_must_be_text_or_a_path(self, bad):
        with pytest.raises(ValueError, match="string or path"):
            Image.Read(bad)

    def test_wildcard_that_matches_nothing_is_refused(self):
        with pytest.raises(ValueError, match="wildcard"):
            Image.Read("no-such-images-*.png")

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="iread() returns (list of images, list of paths) for a wildcard, "
        "which Image.Read() mistakes for a single image, so its own "
        "'wildcard read not supported' ValueError is never reached",
    )
    def test_wildcard_that_matches_files_is_refused(self):
        with pytest.raises(ValueError, match="wildcard"):
            Image.Read(str(DATA / "stripes*.png"))

    def test_alpha_plane_is_dropped_unless_asked_for(self):
        assert Image.Read(DATA / "stripes_a.png", alpha=False).nplanes == 3
        assert Image.Read(DATA / "stripes_a.png", alpha=True).nplanes == 4

    def test_color_order(self):
        assert Image.Read(DATA / "stripes.png").colororder_str == "R:G:B"
        assert Image.Read(DATA / "stripes.png", rgb=False).colororder_str == "B:G:R"
        assert (
            Image.Read(DATA / "stripes_a.png", alpha=True).colororder_str == "R:G:B:A"
        )

    def test_mono_option_gives_one_plane(self):
        assert Image.Read(DATA / "stripes.png", mono=True).nplanes == 1


# --------------------------------------------------------------------------- #
# metadata


@pytest.fixture
def exif_file(tmp_path):
    img = PIL.Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    exif = img.getexif()
    exif[0x0112] = 6  # Orientation, an int
    exif[0x010F] = "Acme"  # Make, text
    exif[0x0131] = "1.5"  # Software, text that is a number
    exif[0x0132] = "2024"  # DateTime, text that is an integer
    exif[0x011A] = 300.0  # XResolution, a float
    path = tmp_path / "exif.jpg"
    img.save(path, exif=exif)
    return path


class TestMetadata:
    def test_all_items(self, exif_file):
        meta = Image.Read(exif_file).metadata()
        assert meta["Orientation"] == 6
        assert meta["Make"] == "Acme"
        assert meta["XResolution"] == 300.0

    def test_int(self, exif_file):
        assert Image.Read(exif_file).metadata("Orientation") == 6

    def test_text_stays_text(self, exif_file):
        assert Image.Read(exif_file).metadata("Make") == "Acme"

    def test_text_that_is_a_number_is_converted(self, exif_file):
        im = Image.Read(exif_file)
        assert im.metadata("Software") == 1.5
        assert im.metadata("DateTime") == 2024
        assert isinstance(im.metadata("DateTime"), int)

    def test_float(self, exif_file):
        assert Image.Read(exif_file).metadata("XResolution") == pytest.approx(300.0)

    def test_missing_key(self, exif_file):
        with pytest.raises(KeyError):
            Image.Read(exif_file).metadata("FocalLength")

    def test_image_with_no_file_has_none(self):
        assert Image(np.zeros((4, 4), dtype=np.uint8)).metadata() is None


# --------------------------------------------------------------------------- #
# showpixels and showwindow


def texts(ax):
    return [t.get_text() for t in ax.texts]


class TestShowPixels:
    def test_int_values(self):
        im = Image(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8))
        ax = im.showpixels()
        assert texts(ax) == ["1", "2", "3", "4", "5", "6"]
        assert ax.get_xlabel() == "u (pixels)" and ax.get_ylabel() == "v (pixels)"
        # the text for pixel (u=2, v=1) is at that pixel
        assert ax.texts[5].get_position() == (2, 1)

    def test_float_values_have_two_decimals(self):
        im = Image(np.array([[0.5, 1.0], [0.25, 0.125]]))
        assert texts(im.showpixels()) == ["0.50", "1.00", "0.25", "0.12"]

    def test_bool_values(self):
        im = Image(np.array([[True, False], [False, True]]))
        assert texts(im.showpixels()) == ["1", "0", "0", "1"]

    def test_format_string(self):
        im = Image(np.array([[1, 2]], dtype=np.uint8))
        assert texts(im.showpixels(fmt="<{:03d}>")) == ["<001>", "<002>"]

    def test_nan_pixels_have_no_label(self):
        im = Image(np.array([[1.0, np.nan], [3.0, 4.0]]))
        assert len(texts(im.showpixels())) == 3

    def test_grid_draws_a_line_between_pixels(self):
        im = Image(np.zeros((3, 4), dtype=np.uint8))
        ax = im.showpixels(grid=1)
        # (width - 1) vertical and (height - 1) horizontal lines
        assert len(ax.lines) == 3 + 2

    def test_grid_is_on_by_default_and_can_be_turned_off(self):
        im = Image(np.zeros((3, 4), dtype=np.uint8))
        assert len(im.showpixels().lines) == 5
        plt.close("all")
        assert len(im.showpixels(grid=None).lines) == 0

    def test_colormap_by_name_and_bad_color(self):
        im = Image(np.array([[1.0, np.nan]]))
        ax = im.showpixels(cmap="gray", badcolor="red")
        assert ax.images[0].get_cmap().name == "gray"

    def test_default_colormap(self):
        ax = Image(np.zeros((2, 2), dtype=np.uint8)).showpixels()
        assert ax.images[0].get_cmap().name == "midnight"

    def test_given_axes(self):
        fig, ax = plt.subplots()
        out = Image(np.zeros((2, 2), dtype=np.uint8)).showpixels(ax=ax)
        assert out is ax

    def test_text_options(self):
        ax = Image(np.zeros((1, 1), dtype=np.uint8)).showpixels(
            color="yellow", fontsize=12, linewidth=0
        )
        assert ax.texts[0].get_color() == "yellow"
        assert ax.texts[0].get_fontsize() == 12


class TestShowWindow:
    @pytest.fixture
    def im(self):
        return Image(np.arange(100, dtype=np.uint8).reshape(10, 10))

    def test_window_moves_and_returns_its_pixels(self, im):
        plt.figure()
        plt.imshow(im.array)
        win = im.showwindow(h=1)
        patch = win.move(5, 4)
        # a 3 x 3 window centred on u=5, v=4: rows 3-5, columns 4-6
        nt.assert_array_equal(patch, im.array[3:6, 4:7])
        assert win.patch.get_x() == pytest.approx(5 - 1 - 0.5)
        assert win.patch.get_y() == pytest.approx(4 - 1 - 0.5)
        assert win.patch.get_width() == 3

    def test_window_into_given_axes(self, im):
        fig, ax = plt.subplots()
        win = im.showwindow(h=2, ax=ax)
        assert win.patch in ax.patches
        assert win.patch.get_width() == 5

    def test_visibility(self, im):
        plt.figure()
        win = im.showwindow(h=1)
        win.visible(False)
        assert not win.patch.get_visible()
        win.visible()
        assert win.patch.get_visible()

    def test_window_style_can_be_overridden(self, im):
        plt.figure()
        win = im.showwindow(h=1, alpha=0.3, facecolor="green")
        assert win.patch.get_alpha() == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# disp with an image domain


def test_disp_with_a_domain_uses_it_for_the_axes_extent():
    u = np.linspace(-1, 1, 6)
    v = np.linspace(10, 20, 4)
    im = Image(np.zeros((4, 6), dtype=np.uint8), domain=(u, v))
    im.disp(title=False)
    extent = plt.gca().images[0].get_extent()
    # left, right, bottom, top
    assert extent == pytest.approx((-1, 1, 20, 10))


# --------------------------------------------------------------------------- #
# anaglyph


class TestAnaglyph:
    @pytest.fixture
    def pair(self):
        left = Image(np.full((6, 8), 100, dtype=np.uint8))
        right = Image(np.full((6, 8), 200, dtype=np.uint8))
        return left, right

    def test_size_must_match(self, pair):
        left, _ = pair
        with pytest.raises(ValueError, match="same size"):
            left.anaglyph(Image(np.zeros((5, 8), dtype=np.uint8)))

    def test_left_is_red_and_right_is_cyan(self, pair):
        left, right = pair
        out = left.anaglyph(right, colors="rc")
        assert out.nplanes == 3
        px = out.array[0, 0]
        # the red channel comes from the left image, green and blue from the right
        assert px[0] == 100 and px[1] == 200 and px[2] == 200

    @pytest.mark.parametrize("disp", [2, -2])
    def test_disparity_trims_the_width(self, pair, disp):
        left, right = pair
        out = left.anaglyph(right, disp=disp)
        assert out.width == 8 - 2

    def test_zero_disparity_keeps_the_width(self, pair):
        left, right = pair
        assert left.anaglyph(right).width == 8


# --------------------------------------------------------------------------- #
# stereo display


class TestStdisp:
    @pytest.fixture
    def pair(self):
        rng = np.random.default_rng(0)
        left = Image(rng.integers(0, 255, (20, 30), dtype=np.uint8))
        right = Image(rng.integers(0, 255, (20, 30), dtype=np.uint8))
        return left, right

    def test_shows_two_images_side_by_side(self, pair):
        left, right = pair
        left.stdisp(right, interactive=False)
        fig = plt.gcf()
        assert len(fig.axes) == 2
        assert all(len(ax.images) == 1 for ax in fig.axes)
        assert fig.canvas.manager.get_window_title() == "Stereo Pair"

    def test_title_from_the_file_names(self, tmp_path, pair):
        left, right = pair
        left.write(tmp_path / "l.png")
        right.write(tmp_path / "r.png")
        a = Image.Read(tmp_path / "l.png")
        b = Image.Read(tmp_path / "r.png")
        a.stdisp(b, interactive=False)
        assert plt.gcf().canvas.manager.get_window_title() == "l.png / r.png"

    def test_given_title(self, pair):
        left, right = pair
        left.stdisp(right, interactive=False, title="rocks")
        assert plt.gcf().canvas.manager.get_window_title() == "rocks"

    @staticmethod
    def event(fig, name, ax, u, v):
        x, y = ax.transData.transform((u, v))
        return MouseEvent(name, fig.canvas, x, y, button=1)

    @pytest.mark.xfail(
        strict=True,
        raises=RuntimeError,
        reason="the click handler passes a scalar to Line2D.set_xdata/set_ydata, "
        "which Matplotlib 3.11 refuses: 'y must be a sequence'",
    )
    def test_clicking_places_the_cross_hairs_and_dragging_measures_disparity(
        self, pair, monkeypatch
    ):
        # stdisp() waits in plt.show(block=True) while the user clicks, so the
        # stand-in for that call plays the part of the user
        from machinevisiontoolbox import ImageIO

        seen = {}

        def user(block=False):
            fig = plt.gcf()
            ax1, ax2 = fig.axes
            fig.canvas.draw()
            process = fig.canvas.callbacks.process
            # move before clicking: nothing happens
            process(
                "motion_notify_event",
                self.event(fig, "motion_notify_event", ax2, 12, 8),
            )
            seen["before"] = [t.get_text() for t in ax2.texts]
            # click at u=10, v=8 in the left image
            process(
                "button_press_event", self.event(fig, "button_press_event", ax1, 10, 8)
            )
            seen["vertical"] = [ln.get_xdata()[0] for ln in ax1.lines]
            seen["horizontal"] = [ln.get_ydata()[0] for ln in ax1.lines]
            # a click in the right image is ignored
            process(
                "button_press_event", self.event(fig, "button_press_event", ax2, 20, 3)
            )
            seen["after_right_click"] = [ln.get_xdata()[0] for ln in ax1.lines]
            # drag over the right image to u=14: disparity is 10 - 14
            process(
                "motion_notify_event",
                self.event(fig, "motion_notify_event", ax2, 14, 8),
            )
            seen["after"] = [t.get_text() for t in ax2.texts]
            seen["dashed"] = ax2.lines[-1].get_xdata()[0]
            # moving outside both images changes nothing
            process(
                "motion_notify_event",
                MouseEvent("motion_notify_event", fig.canvas, 1, 1),
            )

        monkeypatch.setattr(ImageIO, "safe_plt_show", user)
        left, right = pair
        left.stdisp(right, interactive=True)

        assert seen["before"] == [""]
        assert 10 in [pytest.approx(v) for v in seen["vertical"]]
        assert pytest.approx(8) in seen["horizontal"]
        assert seen["after_right_click"] == seen["vertical"]
        assert seen["after"] == ["d=-4.00"]
        assert seen["dashed"] == pytest.approx(14)

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="stdisp() keeps its cursor object only in a local variable, and "
        "Matplotlib holds callbacks weakly, so when plt.show() does not block "
        "(a notebook, or a non-interactive backend) the clicks do nothing",
    )
    def test_cursor_survives_when_show_does_not_block(self, pair):
        left, right = pair
        left.stdisp(right, interactive=True)
        fig = plt.gcf()
        ax1, _ = fig.axes
        fig.canvas.draw()
        fig.canvas.callbacks.process(
            "button_press_event", self.event(fig, "button_press_event", ax1, 10, 8)
        )
        assert ax1.lines[0].get_ydata()[0] == pytest.approx(8)
