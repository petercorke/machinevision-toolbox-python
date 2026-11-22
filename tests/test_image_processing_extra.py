#!/usr/bin/env python
"""More tests for ``ImageProcessing``: thresholds, blending, pasting and the
pixel-wise operations.

Test images are small synthetic arrays and the expected values are worked out
directly with NumPy.  The interactive thresholds are driven with simulated mouse
events; nothing else is mocked.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt
import pytest
from matplotlib.backend_bases import MouseEvent

from machinevisiontoolbox import Image
from machinevisiontoolbox import ImageProcessing as ip


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def ramp():
    """4 x 5 uint8 image with values 0, 50, 100, ... 950 clipped to 255."""
    return Image(np.array([[0, 50, 100, 150, 200]] * 4, dtype=np.uint8))


@pytest.fixture
def bimodal():
    """Half the pixels at 40, half at 200."""
    arr = np.full((10, 10), 40, dtype=np.uint8)
    arr[:, 5:] = 200
    return Image(arr)


# --------------------------------------------------------------------------- #
# pixel-wise operations


class TestApply:
    def test_apply(self, ramp):
        out = ramp.apply(lambda a: a // 2)
        nt.assert_array_equal(out.array, ramp.array // 2)

    def test_apply_vectorized(self, ramp):
        def half(x):
            return x // 2  # works on one pixel at a time

        out = ramp.apply(half, vectorize=True)
        nt.assert_array_equal(out.array, ramp.array // 2)

    def test_apply2(self, ramp):
        out = ramp.apply2(ramp, lambda a, b: a // 2 + b // 2)
        nt.assert_array_equal(out.array, 2 * (ramp.array // 2))

    def test_apply2_vectorized(self, ramp):
        out = ramp.apply2(ramp, lambda a, b: max(int(a), int(b)), vectorize=True)
        nt.assert_array_equal(out.array, ramp.array)

    def test_apply2_sizes_must_match(self, ramp):
        with pytest.raises(ValueError, match="same size"):
            ramp.apply2(Image(np.zeros((3, 3), dtype=np.uint8)), lambda a, b: a)


class TestSimpleOperations:
    def test_clip(self, ramp):
        out = ramp.clip(60, 160)
        assert out.array.min() == 60 and out.array.max() == 160

    def test_roll(self):
        im = Image(np.arange(12, dtype=np.uint8).reshape(3, 4))
        nt.assert_array_equal(im.roll(ru=1).array, np.roll(im.array, 1, axis=1))
        nt.assert_array_equal(im.roll(rv=1).array, np.roll(im.array, 1, axis=0))
        nt.assert_array_equal(
            im.roll(dx=2, dy=1).array, np.roll(im.array, (1, 2), axis=(0, 1))
        )

    def test_invert_int_and_float(self, ramp):
        nt.assert_array_equal(ramp.invert().array, 255 - ramp.array)
        f = Image(np.array([[0.0, 0.25, 1.0]]))
        nt.assert_allclose(f.invert().array, [[1.0, 0.75, 0.0]])

    def test_normhist_spreads_the_histogram(self):
        arr = np.tile(np.array([100, 110, 120, 130], dtype=np.uint8), (8, 8))
        out = Image(arr).normhist()
        assert out.array.min() < 100 and out.array.max() > 130

    def test_LUT_on_a_mono_image(self, ramp):
        lut = np.arange(256)[::-1]
        nt.assert_array_equal(ramp.LUT(lut).array, 255 - ramp.array)

    def test_LUT_with_a_color_table_makes_a_color_image(self, ramp):
        table = np.zeros((256, 3), dtype=np.uint8)
        table[:, 0] = np.arange(256)  # red follows the value
        table[:, 2] = 255 - np.arange(256)  # blue falls
        out = ramp.LUT(table, colororder="RGB")
        assert out.shape == (4, 5, 3)
        nt.assert_array_equal(out.array[0, 2], [100, 0, 155])


class TestStretch:
    def test_to_the_full_range(self):
        im = Image(np.array([[10.0, 20.0, 30.0]]))
        nt.assert_allclose(im.stretch().array, [[0, 0.5, 1]])

    def test_scaled_maximum(self):
        im = Image(np.array([[10.0, 20.0, 30.0]]))
        nt.assert_allclose(im.stretch(max=255).array, [[0, 127.5, 255]])

    def test_given_range_is_clipped(self):
        im = Image(np.array([[0.0, 10.0, 20.0, 30.0]]))
        out = im.stretch(range=[10, 20])
        nt.assert_allclose(out.array, [[0, 0, 1, 1]])

    def test_given_range_without_clipping(self):
        im = Image(np.array([[0.0, 10.0, 20.0, 30.0]]))
        out = im.stretch(range=[10, 20], clip=False)
        nt.assert_allclose(out.array, [[-1, 0, 1, 2]])


# --------------------------------------------------------------------------- #
# thresholds


class TestThreshold:
    def test_binary(self, ramp):
        out = ramp.threshold(100)
        assert out.dtype == np.uint8
        nt.assert_array_equal(out.array[0], [0, 0, 0, 255, 255])

    def test_binary_as_bool(self, ramp):
        out = ramp.threshold(100, as_bool=True)
        assert out.dtype == bool
        nt.assert_array_equal(out.array[0], [False, False, False, True, True])

    @pytest.mark.parametrize("method", ["binary_inv", "/binary"])
    def test_inverse_binary(self, ramp, method):
        nt.assert_array_equal(
            ramp.threshold(100, method=method).array[0], [255, 255, 255, 0, 0]
        )

    def test_truncate(self, ramp):
        nt.assert_array_equal(
            ramp.threshold(100, method="truncate").array[0], [0, 50, 100, 100, 100]
        )

    def test_tozero(self, ramp):
        nt.assert_array_equal(
            ramp.threshold(100, method="tozero").array[0], [0, 0, 0, 150, 200]
        )

    @pytest.mark.parametrize("method", ["tozero_inv", "/tozero"])
    def test_inverse_tozero(self, ramp, method):
        nt.assert_array_equal(
            ramp.threshold(100, method=method).array[0], [0, 50, 100, 0, 0]
        )

    def test_unknown_method(self, ramp):
        with pytest.raises(ValueError, match="unknown threshold method"):
            ramp.threshold(100, method="wobble")

    def test_threshold_is_required(self, ramp):
        with pytest.raises(ValueError, match="must be specified"):
            ramp.threshold()

    def test_threshold_must_be_a_scalar(self, ramp):
        with pytest.raises(ValueError, match="scalar"):
            ramp.threshold([1, 2])

    @pytest.mark.parametrize("name", ["otsu", "triangle", "percentile"])
    def test_automatic_thresholds_return_the_value_used(self, bimodal, name):
        out, t = bimodal.threshold(name)
        assert 40 <= t <= 200
        # the value splits the two populations
        nt.assert_array_equal(out.array[:, :5], 0)
        nt.assert_array_equal(out.array[:, 5:], 255)

    def test_percentile(self, ramp):
        out, t = ramp.threshold("percentile", p=60)
        assert t == pytest.approx(np.percentile(ramp.array, 60))

    def test_automatic_names_are_not_case_sensitive(self, bimodal):
        out, _ = bimodal.threshold("OTSU")
        assert out.array[:, 5:].min() == 255

    @pytest.mark.parametrize("name", ["otsu", "triangle"])
    def test_automatic_methods_need_a_grey_image(self, name):
        color = Image(np.zeros((4, 4, 3), dtype=np.uint8), colororder="RGB")
        with pytest.raises(ValueError, match="greyscale"):
            color.threshold(name)

    def test_unknown_automatic_method(self, ramp):
        with pytest.raises(ValueError, match="unknown automatic"):
            ramp.threshold("wobble")

    def test_t_is_deprecated(self, ramp):
        with pytest.warns(DeprecationWarning, match="t="):
            out = ramp.threshold(t=100)
        nt.assert_array_equal(out.array, ramp.threshold(100).array)

    def test_t_and_threshold_together(self, ramp):
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError, match="not both"):
                ramp.threshold(100, t=50)

    def test_opt_is_deprecated(self, ramp):
        with pytest.warns(DeprecationWarning, match="method="):
            out = ramp.threshold(100, opt="binary_inv")
        nt.assert_array_equal(out.array[0], [255, 255, 255, 0, 0])

    def test_opt_and_method_together(self, ramp):
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError, match="not both"):
                ramp.threshold(100, opt="binary_inv", method="tozero")

    def test_thresh_is_deprecated(self, ramp):
        with pytest.warns(DeprecationWarning, match="threshold"):
            out = ramp.thresh(t=100, opt="binary_inv")
        nt.assert_array_equal(out.array[0], [255, 255, 255, 0, 0])


class TestOtsuAndTriangle:
    def test_otsu_separates_two_levels(self, bimodal):
        t = bimodal.otsu()
        assert 40 <= t < 200

    def test_otsu_of_a_constant_image(self):
        assert Image(np.full((4, 4), 7, dtype=np.uint8)).otsu() == 7

    def test_otsu_uses_the_grey_version_of_a_color_image(self, bimodal):
        color = Image(np.dstack([bimodal.array] * 3), colororder="RGB")
        assert 30 <= color.otsu() < 200

    def test_otsu_bins(self, bimodal):
        with pytest.raises(ValueError, match="nbins"):
            bimodal.otsu(nbins=1)

    def test_triangle_of_a_constant_image(self):
        assert Image(np.full((4, 4), 7, dtype=np.uint8)).triangle() == 7

    def test_triangle_bins(self, bimodal):
        with pytest.raises(ValueError, match="nbins"):
            bimodal.triangle(nbins=1)

    def test_triangle_with_a_long_tail_to_the_right(self):
        # a big peak at 10 and a tail trailing off to the right
        values = np.repeat(
            [10, 12, 14, 16, 18, 20, 24, 30], [200, 60, 30, 20, 12, 8, 4, 2]
        )
        im = Image(values.astype(np.uint8).reshape(1, -1))
        t = im.triangle()
        assert 10 < t < 30

    def test_triangle_with_a_long_tail_to_the_left(self):
        values = np.repeat(
            [10, 16, 20, 22, 24, 26, 28, 30], [2, 4, 8, 12, 20, 30, 60, 200]
        )
        im = Image(values.astype(np.uint8).reshape(1, -1))
        t = im.triangle()
        assert 10 < t < 30

    def test_triangle_with_a_single_populated_bin(self):
        # one level plus a lone different pixel: the peak is at one end
        arr = np.full((5, 5), 100, dtype=np.uint8)
        arr[0, 0] = 101
        assert 100 <= Image(arr).triangle() <= 101


class TestAdaptiveThreshold:
    @pytest.fixture
    def gradient(self):
        # dark on the left, bright on the right, with the edge at column 5
        arr = np.full((12, 12), 50, dtype=np.uint8)
        arr[:, 5:] = 200
        return Image(arr)

    @pytest.mark.parametrize("method", ["mean", "gaussian"])
    def test_methods(self, gradient, method):
        out = gradient.threshold_adaptive(C=5, h=3, method=method).array
        assert set(np.unique(out)) <= {0, 255}
        # a pixel is set if it is brighter than its local mean less C
        assert out[6, 4] == 0  # dark side of the edge, below its neighbourhood
        assert out[6, 5] == 255  # bright side of the edge
        assert out[6, 0] == 255  # flat region: equal to its mean, above mean - 5

    def test_blocksize(self, gradient):
        a = gradient.threshold_adaptive(blocksize=7)
        b = gradient.threshold_adaptive(h=3)
        nt.assert_array_equal(a.array, b.array)

    def test_color_image_is_refused(self):
        with pytest.raises(ValueError, match="greyscale"):
            Image(np.zeros((8, 8, 3), dtype=np.uint8)).threshold_adaptive()

    def test_unknown_method(self, gradient):
        with pytest.raises(KeyError):
            gradient.threshold_adaptive(method="median")

    def test_adaptive_threshold_is_deprecated(self, gradient):
        with pytest.warns(DeprecationWarning, match="threshold_adaptive"):
            out = gradient.adaptive_threshold(h=3)
        nt.assert_array_equal(out.array, gradient.threshold_adaptive(h=3).array)


class TestInteractiveThreshold:
    @staticmethod
    def drag_slider(value):
        """Stand-in for plt.show(): moves the slider the way a user would."""

        def show(block=False):
            fig = plt.gcf()
            slider_ax = fig.axes[-1]
            fig.canvas.draw()
            x, y = slider_ax.transData.transform((value, 0.5))
            for name in ("button_press_event", "button_release_event"):
                fig.canvas.callbacks.process(
                    name, MouseEvent(name, fig.canvas, x, y, button=1)
                )

        return show

    def test_initial_threshold_is_mid_range(self, ramp):
        t = ramp.threshold_interactive()
        assert t == 100  # (0 + 200) / 2

    def test_moving_the_slider_changes_the_threshold_and_the_image(
        self, ramp, monkeypatch
    ):
        monkeypatch.setattr(ip, "safe_plt_show", self.drag_slider(160))
        t = ramp.threshold_interactive()
        assert t == pytest.approx(160, abs=3)
        shown = plt.gcf().axes[0].images[0].get_array()
        nt.assert_array_equal(np.asarray(shown)[0], [0, 0, 0, 0, 1])

    def test_tozero_mode(self, ramp, monkeypatch):
        monkeypatch.setattr(ip, "safe_plt_show", self.drag_slider(120))
        t = ramp.threshold_interactive(mode="tozero")
        assert t == pytest.approx(120, abs=3)
        shown = np.asarray(plt.gcf().axes[0].images[0].get_array())
        nt.assert_array_equal(shown[0], [0, 0, 0, 150, 200])

    def test_window_title(self, ramp):
        ramp.threshold_interactive(title="pick one")
        assert plt.gcf().canvas.manager.get_window_title() == "pick one"

    def test_flat_image_returns_its_value(self):
        assert Image(np.full((3, 3), 9, dtype=np.uint8)).threshold_interactive() == 9

    def test_color_image_is_refused(self):
        with pytest.raises(ValueError, match="greyscale"):
            Image(np.zeros((3, 3, 3), dtype=np.uint8)).threshold_interactive()

    def test_unknown_mode(self, ramp):
        with pytest.raises(ValueError, match="unknown threshold mode"):
            ramp.threshold_interactive(mode="wobble")

    def test_ithresh_with_a_value_thresholds(self, ramp):
        out = ramp.ithresh(100)
        nt.assert_array_equal(out.array[0], [0, 0, 0, 255, 255])

    def test_ithresh_without_a_value_is_deprecated(self, ramp):
        with pytest.warns(DeprecationWarning, match="threshold_interactive"):
            t = ramp.ithresh()
        assert t == 100

    def test_notebook_widget_backend(self, ramp, monkeypatch):
        widgets = pytest.importorskip("ipywidgets")
        shown = []
        import IPython.display

        monkeypatch.setattr(plt, "get_backend", lambda: "module://ipympl.backend_nbagg")
        monkeypatch.setattr(IPython.display, "display", lambda w: shown.append(w))
        t = ramp.threshold_interactive()
        assert t == 100
        assert len(shown) == 1 and isinstance(shown[0], widgets.FloatSlider)
        # moving the slider updates the displayed image
        shown[0].value = 160
        image = np.asarray(plt.gcf().axes[0].images[0].get_array())
        nt.assert_array_equal(image[0], [0, 0, 0, 0, 1])


# --------------------------------------------------------------------------- #
# blend and choose


class TestBlend:
    def test_default_weights(self):
        a = Image(np.full((3, 3), 100, dtype=np.uint8))
        b = Image(np.full((3, 3), 200, dtype=np.uint8))
        assert (a.blend(b, 0.25).array == 175).all()  # 0.25*100 + 0.75*200

    def test_explicit_beta_and_gamma(self):
        a = Image(np.full((3, 3), 100, dtype=np.uint8))
        b = Image(np.full((3, 3), 200, dtype=np.uint8))
        assert (a.blend(b, 0.5, beta=0.25, gamma=10).array == 110).all()

    @pytest.mark.parametrize("dtype", ["float", "float32"])
    def test_single_precision_output(self, dtype):
        a = Image(np.full((3, 3), 100.0, dtype=np.float32))
        b = Image(np.full((3, 3), 200.0, dtype=np.float32))
        assert a.blend(b, 0.5, dtype=dtype).dtype == np.float32

    @pytest.mark.xfail(
        strict=True,
        reason="blend() builds the result with Image(out), which narrows float64 "
        "to float32, so dtype='double' and float64 inputs lose their precision",
    )
    @pytest.mark.parametrize("dtype", ["double", "float64"])
    def test_double_precision_output(self, dtype):
        a = Image(np.full((3, 3), 100.0), dtype="float64")
        b = Image(np.full((3, 3), 200.0), dtype="float64")
        assert a.blend(b, 0.5, dtype=dtype).dtype == np.float64

    def test_bad_output_type(self):
        a = Image(np.zeros((3, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="dtype must be"):
            a.blend(a, 0.5, dtype="int")

    def test_shapes_must_match(self):
        a = Image(np.zeros((3, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="not the same size"):
            a.blend(Image(np.zeros((4, 4), dtype=np.uint8)), 0.5)

    def test_int_and_float_cannot_be_mixed(self):
        a = Image(np.zeros((3, 3), dtype=np.uint8))
        b = Image(np.zeros((3, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="both int or both float"):
            a.blend(b, 0.5)


class TestChoose:
    @pytest.fixture
    def base(self):
        return Image(np.full((4, 4), 10, dtype=np.uint8))

    @pytest.fixture
    def mask(self):
        m = np.zeros((4, 4), dtype=bool)
        m[:, 2:] = True
        return m

    def test_from_another_image(self, base, mask):
        other = Image(np.full((4, 4), 99, dtype=np.uint8))
        out = base.choose(other, mask)
        nt.assert_array_equal(out.array[0], [10, 10, 99, 99])

    def test_mask_may_be_an_image(self, base, mask):
        other = Image(np.full((4, 4), 99, dtype=np.uint8))
        out = base.choose(other, Image(mask.astype(np.uint8) * 255))
        nt.assert_array_equal(out.array[0], [10, 10, 99, 99])

    def test_scalar(self, base, mask):
        out = base.choose(77, mask)
        nt.assert_array_equal(out.array[0], [10, 10, 77, 77])

    def test_color_name_makes_a_color_image(self, base, mask):
        out = base.choose("red", mask)
        assert out.shape == (4, 4, 3)
        assert tuple(out.array[0, 3]) == (255, 0, 0)
        assert tuple(out.array[0, 0]) == (10, 10, 10)

    def test_three_vector(self, base, mask):
        out = base.choose([1, 2, 3], mask)
        assert tuple(out.array[0, 3]) == (1, 2, 3)

    def test_bgr_image_gets_the_color_reversed(self, mask):
        bgr = Image(np.zeros((4, 4, 3), dtype=np.uint8), colororder="BGR")
        out = bgr.choose([1, 2, 3], mask)
        assert tuple(out.array[0, 3]) == (3, 2, 1)

    def test_bad_color(self, base, mask):
        with pytest.raises(ValueError, match="scalar, string or 3-vector"):
            base.choose([1, 2], mask)

    def test_mask_type(self, base):
        with pytest.raises(ValueError, match="bad type for mask"):
            base.choose(5, "mask")

    def test_mask_size(self, base):
        with pytest.raises(ValueError, match="mask must be same size"):
            base.choose(5, np.zeros((3, 3), dtype=bool))

    def test_image_size(self, base, mask):
        with pytest.raises(ValueError, match="image2 must be same size"):
            base.choose(Image(np.zeros((4, 4, 3), dtype=np.uint8)), mask)


# --------------------------------------------------------------------------- #
# paste


class TestPaste:
    @pytest.fixture
    def canvas(self):
        return Image(np.full((8, 10), 50, dtype=np.uint8))

    @pytest.fixture
    def pattern(self):
        return Image(np.full((2, 3), 200, dtype=np.uint8))

    def test_set_at_top_left(self, canvas, pattern):
        out = canvas.paste(pattern, (4, 2))
        expected = np.full((8, 10), 50, dtype=np.uint8)
        expected[2:4, 4:7] = 200
        nt.assert_array_equal(out.array, expected)
        # the canvas is not changed
        assert (canvas.array == 50).all()

    def test_centred(self, canvas, pattern):
        out = canvas.paste(pattern, (5, 4), position="centre")
        # a 3 wide, 2 high pattern centred on (5, 4): columns 4-6, rows 3-4
        assert (out.array[3:5, 4:7] == 200).all()
        assert out.array[3, 3] == 50 and out.array[5, 4] == 50

    def test_american_spelling_of_centre(self, canvas, pattern):
        a = canvas.paste(pattern, (5, 4), position="center")
        b = canvas.paste(pattern, (5, 4), position="centre")
        nt.assert_array_equal(a.array, b.array)

    def test_one_based_offset(self, canvas, pattern):
        out = canvas.paste(pattern, (4, 2), zero=False)
        assert (out.array[3:5, 5:8] == 200).all()

    def test_ndarray_pattern(self, canvas):
        out = canvas.paste(np.full((2, 2), 9, dtype=np.uint8), (0, 0))
        assert (out.array[:2, :2] == 9).all()

    def test_add(self, canvas, pattern):
        small = Image(np.full((2, 3), 20, dtype=np.uint8))
        out = canvas.paste(small, (1, 1), method="add")
        assert (out.array[1:3, 1:4] == 70).all()

    def test_mean(self, canvas):
        pat = Image(np.full((2, 2), 150.0, dtype=np.float32))
        cvs = Image(np.full((8, 10), 50.0, dtype=np.float32))
        out = cvs.paste(pat, (0, 0), method="mean")
        assert (out.array[:2, :2] == 100).all()

    def test_blend_averages_where_both_are_set(self):
        cvs = Image(np.array([[100, 100, 0, 0]], dtype=np.uint8))
        pat = Image(np.array([[200, 0, 200, 0]], dtype=np.uint8))
        out = cvs.paste(pat, (0, 0), method="blend")
        # both set: mean; only the canvas: canvas; only the pattern: pattern; neither: 0
        nt.assert_array_equal(out.array, [[150, 100, 200, 0]])

    def test_color_pattern_on_a_mono_canvas_makes_a_color_image(self, canvas):
        pat = Image(np.full((2, 2, 3), 200, dtype=np.uint8), colororder="RGB")
        out = canvas.paste(pat, (0, 0))
        assert out.shape == (8, 10, 3)
        assert out.colororder_str == "R:G:B"
        assert tuple(out.array[0, 0]) == (200, 200, 200)
        assert tuple(out.array[5, 5]) == (50, 50, 50)

    def test_mono_pattern_on_a_color_canvas(self):
        cvs = Image(np.full((6, 6, 3), 50, dtype=np.uint8), colororder="RGB")
        pat = Image(np.full((2, 2), 200, dtype=np.uint8))
        out = cvs.paste(pat, (1, 1))
        assert tuple(out.array[1, 1]) == (200, 200, 200)
        assert tuple(out.array[0, 0]) == (50, 50, 50)

    def test_add_with_color(self):
        cvs = Image(np.full((6, 6, 3), 50, dtype=np.uint8), colororder="RGB")
        pat = Image(np.full((2, 2, 3), 20, dtype=np.uint8), colororder="RGB")
        out = cvs.paste(pat, (0, 0), method="add")
        assert tuple(out.array[0, 0]) == (70, 70, 70)

    def test_mean_with_color(self):
        cvs = Image(np.full((6, 6, 3), 50.0, dtype=np.float32), colororder="RGB")
        pat = Image(np.full((2, 2, 3), 150.0, dtype=np.float32), colororder="RGB")
        out = cvs.paste(pat, (0, 0), method="mean")
        assert tuple(out.array[0, 0]) == (100, 100, 100)

    def test_falls_off_the_bottom(self, canvas, pattern):
        with pytest.raises(ValueError, match="bottom edge"):
            canvas.paste(pattern, (0, 7))

    def test_falls_off_the_right(self, canvas, pattern):
        with pytest.raises(ValueError, match="right edge"):
            canvas.paste(pattern, (8, 0))

    def test_bad_position(self, canvas, pattern):
        with pytest.raises(ValueError, match="bad position"):
            canvas.paste(pattern, (0, 0), position="middle")

    def test_bad_method(self, canvas, pattern):
        with pytest.raises(ValueError, match="method is not valid"):
            canvas.paste(pattern, (0, 0), method="xor")
