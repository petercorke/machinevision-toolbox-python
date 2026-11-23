import io
import json
import unittest
from pathlib import Path
from unittest.case import skip

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt

from machinevisiontoolbox import Blobs, Image

# Number of blobs produced by Image.Squares(2, ...): a 2×2 grid = 4 squares.
_N_MULTI = 4


class TestBlobs(unittest.TestCase):
    def test_blob_2x2_square(self):
        # 1 blob: 2x2
        im = Image.String(
            r"""
                    ..........
                    ..........
                    ..........
                    ....##....
                    ....##....
                    ..........
                    ..........
                    ..........
                    ..........
                    ..........
                   """,
            binary=True,
        )
        blobs = im.blobs()

        self.assertEqual(len(blobs), 1)
        self.assertEqual(blobs[0].moments.m00, 1)
        self.assertEqual(blobs[0].uc, 4.5)
        self.assertEqual(blobs[0].vc, 3.5)
        self.assertEqual(blobs[0].perimeter_length, 3)

    def test_blob_single_pixel(self):
        # 1 blob: single pixel
        im = Image.String(
            r"""
                    ..........
                    ..........
                    ..........
                    ....#.....
                    ..........
                    ..........
                    ..........
                    ..........
                    ..........
                    ..........
                   """,
            binary=True,
        )
        blobs = im.blobs()

        self.assertEqual(len(blobs), 1)
        self.assertEqual(blobs[0].area, 1)
        self.assertEqual(blobs[0].uc, 4.0)
        self.assertEqual(blobs[0].vc, 3.0)

    def test_blob_diagonal_line(self):
        # 1 blob: single pixel width line
        # should be zero blobs
        im = Image.String(
            r"""
                    ..........
                    ..........
                    .....#....
                    ....#.....
                    ...#......
                    ..........
                    ..........
                    ..........
                    ..........
                    ..........
                   """,
            binary=True,
        )
        blobs = im.blobs()

        self.assertEqual(len(blobs), 1)
        self.assertEqual(blobs[0].area, 4)
        self.assertEqual(blobs[0].uc, 4.0)
        self.assertEqual(blobs[0].vc, 3.0)

    def test_two_blobs_different_sizes(self):
        # 2 blobs: 3x3 and 4x4
        im = Image.String(
            r"""
                    ..........
                    ..###.....
                    ..###.....
                    ..###.....
                    ..........
                    ....####..
                    ....####..
                    ....####..
                    ....####..
                    ..........
                   """,
            binary=True,
        )
        blobs = im.blobs()

        self.assertEqual(len(blobs), 2)
        self.assertEqual(blobs[0].area, 9)
        self.assertEqual(blobs[0].perimeter_length, 11)

        self.assertEqual(blobs[1].area, 4)
        self.assertEqual(blobs[1].perimeter_length, 7)

    def test_orientation_square_blob_degenerate_moments(self):
        # A square (or any blob with equal mu20/mu02 and mu11≈0) has a
        # moment matrix that is a scalar multiple of the identity: its two
        # eigenvalues are exactly equal. np.linalg.eig (general, non-
        # symmetric solver) can return complex128 for this case due to
        # floating-point noise even though the matrix is real-symmetric and
        # the eigenvalues are mathematically real, which then makes
        # np.arctan2() raise TypeError. np.linalg.eigh (symmetric solver)
        # does not have this failure mode. See ImageBlobs.py's equivalent
        # ellipse / orientation calculation.
        im = Image.Squares(1, size=61, fg=255, bg=0)
        blobs = im.blobs()

        orientation = blobs[0].orientation
        self.assertIsInstance(orientation, float)
        self.assertFalse(np.iscomplexobj(orientation))


# ============================================================================ #
#  Fixtures
#  - blobs1: Image.Squares(1, ...) → 1 white square blob
#  - blobs4: Image.Squares(2, ...) → 2×2 grid of white squares = 4 blobs
# ============================================================================ #


def _make_blobs():
    """Return (blobs1, blobs4) fixtures."""
    blobs1 = Image.Squares(1, size=61, fg=255, bg=0).blobs()
    blobs4 = Image.Squares(2, size=61, fg=255, bg=0).blobs()
    assert len(blobs1) == 1, f"Expected 1 blob, got {len(blobs1)}"
    assert len(blobs4) == _N_MULTI, f"Expected {_N_MULTI} blobs, got {len(blobs4)}"
    return blobs1, blobs4


# ============================================================================ #
#  Scalar properties
#  @scalar_result: 1 blob → scalar, N blobs → ndarray(N)
# ============================================================================ #


class TestBlobScalarProperties(unittest.TestCase):
    """
    For every @scalar_result property:
      - blobs1.prop          → Python scalar (1 blob in Blobs)
      - blobs4[0].prop       → Python scalar (single element of multi-blob)
      - blobs4.prop          → ndarray of length _N_MULTI
    """

    def setUp(self):
        self.blobs1, self.blobs4 = _make_blobs()

    def _check(self, prop: str) -> None:
        # 1-blob Blobs → scalar
        val = getattr(self.blobs1, prop)
        self.assertTrue(
            np.isscalar(val),
            f"{prop}: blobs1 (1 blob) should return scalar, got {type(val).__name__}",
        )
        # indexed single element from multi → scalar
        val = getattr(self.blobs4[0], prop)
        self.assertTrue(
            np.isscalar(val),
            f"{prop}: blobs4[0] should return scalar, got {type(val).__name__}",
        )
        # multi-blob Blobs → ndarray of length _N_MULTI
        val = getattr(self.blobs4, prop)
        self.assertIsInstance(
            val,
            np.ndarray,
            f"{prop}: blobs4 should return ndarray, got {type(val).__name__}",
        )
        self.assertEqual(len(val), _N_MULTI)

    def test_area(self):
        self._check("area")

    def test_uc(self):
        self._check("uc")

    def test_vc(self):
        self._check("vc")

    def test_umin(self):
        self._check("umin")

    def test_umax(self):
        self._check("umax")

    def test_vmin(self):
        self._check("vmin")

    def test_vmax(self):
        self._check("vmax")

    def test_bboxarea(self):
        self._check("bboxarea")

    def test_fillfactor(self):
        self._check("fillfactor")

    def test_a(self):
        self._check("a")

    def test_b(self):
        self._check("b")

    def test_aspect(self):
        self._check("aspect")

    def test_orientation(self):
        self._check("orientation")

    def test_touch(self):
        self._check("touch")

    def test_level(self):
        self._check("level")

    def test_color(self):
        self._check("color")

    def test_parent(self):
        self._check("parent")

    def test_id(self):
        self._check("id")

    def test_perimeter_length(self):
        self._check("perimeter_length")

    def test_circularity(self):
        self._check("circularity")

    # Sanity-check values for a square blob
    def test_area_value(self):
        # Squares(1) has a single square; area should be a reasonable pixel count
        area = self.blobs1.area
        self.assertGreater(area, 0)

    def test_aspect_value(self):
        # A square blob has aspect ratio ≈ 1 (b/a ≤ 1)
        asp = self.blobs1.aspect
        self.assertGreater(asp, 0.9)
        self.assertLessEqual(asp, 1.0)

    def test_touch_false(self):
        # Squares(1) on a 61×61 canvas does not touch the edge
        self.assertFalse(self.blobs1.touch)

    def test_level_zero(self):
        # Top-level blobs have level == 0
        self.assertEqual(self.blobs1.level, 0)

    def test_parent_minus_one(self):
        # Top-level blobs have no parent (returned as -1)
        self.assertEqual(self.blobs1.parent, -1)


# ============================================================================ #
#  Array properties
#  @array_result: 1 blob → single object, N blobs → list of N objects
# ============================================================================ #


class TestBlobArrayProperties(unittest.TestCase):
    """
    For every @array_result property:
      - blobs1.prop          → single object (not wrapped in a list)
      - blobs4[0].prop       → single object
      - blobs4.prop          → list of length _N_MULTI
    """

    def setUp(self):
        self.blobs1, self.blobs4 = _make_blobs()

    # --- centroid ----------------------------------------------------------- #

    def test_centroid_single(self):
        val = self.blobs1.centroid
        self.assertIsInstance(val, tuple)
        self.assertEqual(len(val), 2)

    def test_centroid_indexed(self):
        val = self.blobs4[0].centroid
        self.assertIsInstance(val, tuple)
        self.assertEqual(len(val), 2)

    def test_centroid_multi(self):
        val = self.blobs4.centroid
        self.assertIsInstance(val, list)
        self.assertEqual(len(val), _N_MULTI)
        for item in val:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    # --- p ------------------------------------------------------------------ #

    def test_p_single(self):
        val = self.blobs1.p
        self.assertIsInstance(val, tuple)
        self.assertEqual(len(val), 2)

    def test_p_multi(self):
        val = self.blobs4.p
        self.assertIsInstance(val, list)
        self.assertEqual(len(val), _N_MULTI)

    # --- bbox --------------------------------------------------------------- #

    def test_bbox_single(self):
        val = self.blobs1.bbox
        # OpenCV boundingRect returns (x, y, w, h)
        self.assertEqual(len(val), 4)
        self.assertNotIsInstance(val, list)

    def test_bbox_multi(self):
        val = self.blobs4.bbox
        self.assertIsInstance(val, list)
        self.assertEqual(len(val), _N_MULTI)
        for item in val:
            self.assertEqual(len(item), 4)

    # --- perimeter ---------------------------------------------------------- #

    def test_perimeter_single(self):
        val = self.blobs1.perimeter
        self.assertIsInstance(val, np.ndarray)
        self.assertEqual(val.shape[0], 2)  # 2 × N_points

    def test_perimeter_multi(self):
        val = self.blobs4.perimeter
        self.assertIsInstance(val, list)
        self.assertEqual(len(val), _N_MULTI)
        for item in val:
            self.assertIsInstance(item, np.ndarray)
            self.assertEqual(item.shape[0], 2)

    # --- moments ------------------------------------------------------------ #

    def test_moments_single(self):
        val = self.blobs1.moments
        self.assertFalse(isinstance(val, list))
        self.assertTrue(hasattr(val, "m00"))
        self.assertGreater(val.m00, 0)

    def test_moments_multi(self):
        val = self.blobs4.moments
        self.assertIsInstance(val, list)
        self.assertEqual(len(val), _N_MULTI)
        for item in val:
            self.assertTrue(hasattr(item, "m00"))

    # --- MEC ---------------------------------------------------------------- #

    def test_MEC_single(self):
        val = self.blobs1.MEC
        self.assertIsInstance(val, np.ndarray)
        self.assertEqual(val.shape, (3,))  # (uc, vc, radius)

    def test_MEC_multi(self):
        val = self.blobs4.MEC
        self.assertIsInstance(val, list)
        self.assertEqual(len(val), _N_MULTI)
        for item in val:
            self.assertEqual(item.shape, (3,))

    # --- MER ---------------------------------------------------------------- #

    def test_MER_single(self):
        val = self.blobs1.MER
        self.assertIsInstance(val, np.ndarray)
        self.assertEqual(val.shape, (5,))  # (uc, vc, w, h, theta)

    def test_MER_multi(self):
        val = self.blobs4.MER
        self.assertIsInstance(val, list)
        self.assertEqual(len(val), _N_MULTI)
        for item in val:
            self.assertEqual(item.shape, (5,))

    # --- children ----------------------------------------------------------- #

    def test_children_single(self):
        # A top-level square has no children
        val = self.blobs1.children
        self.assertIsInstance(val, list)

    def test_children_multi(self):
        val = self.blobs4.children
        self.assertIsInstance(val, list)
        self.assertEqual(len(val), _N_MULTI)
        for item in val:
            self.assertIsInstance(item, list)


# ============================================================================ #
#  Non-plot methods
# ============================================================================ #


class TestBlobMethods(unittest.TestCase):
    """humoments, perimeter_approx, perimeter_hull, polar, filter, sort."""

    def setUp(self):
        self.blobs1, self.blobs4 = _make_blobs()

    # --- humoments ---------------------------------------------------------- #

    def test_humoments_single_blob(self):
        h = self.blobs1.humoments()
        self.assertIsInstance(h, np.ndarray)
        self.assertEqual(h.shape, (7,))

    def test_humoments_indexed(self):
        h = self.blobs4[0].humoments()
        self.assertIsInstance(h, np.ndarray)
        self.assertEqual(h.shape, (7,))

    def test_humoments_multi(self):
        h = self.blobs4.humoments()
        self.assertIsInstance(h, np.ndarray)
        self.assertEqual(h.shape, (_N_MULTI, 7))

    # --- perimeter_approx --------------------------------------------------- #

    def test_perimeter_approx_single(self):
        p = self.blobs1.perimeter_approx(5)
        self.assertIsInstance(p, np.ndarray)
        self.assertEqual(p.shape[0], 2)
        # Approximation should not exceed the full perimeter point count
        n_full = self.blobs1.perimeter.shape[1]
        self.assertLessEqual(p.shape[1], n_full)

    def test_perimeter_approx_indexed(self):
        p = self.blobs4[0].perimeter_approx(5)
        self.assertIsInstance(p, np.ndarray)
        self.assertEqual(p.shape[0], 2)

    def test_perimeter_approx_multi(self):
        p = self.blobs4.perimeter_approx(5)
        self.assertIsInstance(p, list)
        self.assertEqual(len(p), _N_MULTI)
        for pi in p:
            self.assertIsInstance(pi, np.ndarray)
            self.assertEqual(pi.shape[0], 2)

    # --- perimeter_hull ----------------------------------------------------- #

    def test_perimeter_hull_single(self):
        h = self.blobs1.perimeter_hull()
        self.assertIsInstance(h, np.ndarray)
        self.assertEqual(h.shape[0], 2)

    def test_perimeter_hull_multi(self):
        h = self.blobs4.perimeter_hull()
        self.assertIsInstance(h, list)
        self.assertEqual(len(h), _N_MULTI)
        for hi in h:
            self.assertEqual(hi.shape[0], 2)

    # --- polar -------------------------------------------------------------- #

    def test_polar_single(self):
        r, theta = self.blobs1.polar()
        self.assertEqual(r.shape, (400,))
        self.assertEqual(theta.shape, (400,))

    def test_polar_multi(self):
        result = self.blobs4.polar()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), _N_MULTI)
        for r, theta in result:
            self.assertEqual(r.shape, (400,))
            self.assertEqual(theta.shape, (400,))

    # --- filter ------------------------------------------------------------- #

    def test_filter_returns_blobs(self):
        result = self.blobs4.filter(area=0)
        self.assertIsInstance(result, Blobs)

    def test_filter_area_minimum_keeps_all(self):
        area = self.blobs4[0].area
        kept = self.blobs4.filter(area=area * 0.5)
        self.assertEqual(len(kept), _N_MULTI)

    def test_filter_area_minimum_removes_all(self):
        area = self.blobs4[0].area
        removed = self.blobs4.filter(area=area * 2)
        self.assertEqual(len(removed), 0)

    def test_filter_area_range(self):
        area = self.blobs4[0].area
        kept = self.blobs4.filter(area=[area * 0.5, area * 2])
        self.assertEqual(len(kept), _N_MULTI)

    def test_filter_touch(self):
        # None of the blobs touch the edge
        kept = self.blobs4.filter(touch=False)
        self.assertEqual(len(kept), _N_MULTI)
        removed = self.blobs4.filter(touch=True)
        self.assertEqual(len(removed), 0)

    def test_filter_area_single_blob_keeps_match(self):
        # Regression for GH issue #17: a single-blob Blobs mask came out
        # 2D (from np.array(mask).all(axis=0)) and indexing self[m] with
        # it dropped the blob even though it matched the filter.
        area = self.blobs1[0].area
        kept = self.blobs1.filter(area=area * 0.5)
        self.assertEqual(len(kept), 1)

    def test_filter_area_single_blob_removes_non_match(self):
        area = self.blobs1[0].area
        removed = self.blobs1.filter(area=area * 2)
        self.assertEqual(len(removed), 0)

    # --- sort --------------------------------------------------------------- #

    def test_sort_returns_blobs(self):
        result = self.blobs4.sort()
        self.assertIsInstance(result, Blobs)
        self.assertEqual(len(result), _N_MULTI)

    def test_sort_area_ascending(self):
        areas = self.blobs4.sort(by="area").area
        self.assertTrue(np.all(areas[:-1] <= areas[1:]))

    def test_sort_area_descending(self):
        areas = self.blobs4.sort(by="area", reverse=True).area
        self.assertTrue(np.all(areas[:-1] >= areas[1:]))

    def test_sort_circularity(self):
        result = self.blobs4.sort(by="circularity")
        self.assertIsInstance(result, Blobs)

    def test_sort_perimeter(self):
        result = self.blobs4.sort(by="perimeter")
        self.assertIsInstance(result, Blobs)

    def test_sort_aspect(self):
        result = self.blobs4.sort(by="aspect")
        self.assertIsInstance(result, Blobs)

    def test_sort_invalid_key(self):
        with self.assertRaises(ValueError):
            self.blobs4.sort(by="nonsense")

    # --- graph -------------------------------------------------------------- #

    def test_graph_dot_returns_string(self):
        text = self.blobs4.graph()
        self.assertIsInstance(text, str)
        self.assertIn("digraph", text)

    def test_graph_dot_and_dotfile_match(self):
        stream = io.StringIO()
        self.blobs4.dotfile(filename=stream)
        dotfile_text = stream.getvalue()
        graph_text = self.blobs4.graph(format="dot")
        self.assertEqual(graph_text, dotfile_text)

    def test_graph_mermaid_returns_string(self):
        text = self.blobs4.graph(format="mermaid")
        self.assertIn("flowchart", text)

    def test_graph_mermaid_fenced_returns_fenced_markdown(self):
        text = self.blobs4.graph(format="mermaid_fenced")
        self.assertTrue(text.startswith("```mermaid\n"))
        self.assertTrue(text.endswith("```\n"))

    def test_graph_removed_mermaid_aliases_raise(self):
        with self.assertRaises(ValueError):
            self.blobs4.graph(format="mermaid_md")
        with self.assertRaises(ValueError):
            self.blobs4.graph(format="mermaid-markdown")

    def test_graph_graphml_returns_xml(self):
        text = self.blobs4.graph(format="graphml")
        self.assertTrue(text.startswith("<?xml"))
        self.assertIn("<graphml", text)

    def test_graph_elk_returns_json(self):
        text = self.blobs4.graph(format="elk")
        data = json.loads(text)
        self.assertIn("children", data)
        self.assertIn("edges", data)

    def test_graph_writes_filename(self):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".dot", delete=False) as f:
            out = Path(f.name)
        try:
            text = self.blobs4.graph(format="dot", filename=str(out))
            self.assertEqual(out.read_text(), text)
        finally:
            if out.exists():
                out.unlink()

    def test_graph_writes_stream(self):
        stream = io.StringIO()
        text = self.blobs4.graph(format="mermaid", filename=stream)
        self.assertEqual(stream.getvalue(), text)

    def test_graph_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            self.blobs4.graph(format="badformat")


# ============================================================================ #
#  Plot methods
#  Strategy: count matplotlib artists before/after each call.
#  For line-producing methods  → count ax.lines.
#  For patch-producing methods → count ax.patches.
#  Verify: multi-blob call adds exactly _N_MULTI × single-blob count.
#
#  plot_box() is skipped — a positional fmt argument is broken in the current
#  release; an upstream PR has been submitted but not yet merged.
#  TODO: remove skips once the PR is folded in.
# ============================================================================ #


class TestBlobPlotMethods(unittest.TestCase):
    """Plot methods produce the right number of artists; accept fmt and kwargs."""

    def setUp(self):
        self.blobs1, self.blobs4 = _make_blobs()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _fresh_ax(self):
        fig, ax = plt.subplots()
        return fig, ax

    def _test_plot_ratio(self, method_name: str, count_fn, *args, **kwargs):
        """
        Call method on single blob and on multi-blob, verify the multi count
        is exactly _N_MULTI times the single count (with default args).
        """
        fig, ax = self._fresh_ax()
        n_before = count_fn(ax)
        getattr(self.blobs1, method_name)(*args, **kwargs)
        n_single = count_fn(ax) - n_before
        self.assertGreater(
            n_single, 0, f"{method_name}: expected artists for single blob, got 0"
        )
        plt.close(fig)

        fig, ax = self._fresh_ax()
        n_before = count_fn(ax)
        getattr(self.blobs4, method_name)(*args, **kwargs)
        n_multi = count_fn(ax) - n_before
        plt.close(fig)

        self.assertEqual(
            n_multi,
            _N_MULTI * n_single,
            f"{method_name}: expected {_N_MULTI}×{n_single}={_N_MULTI * n_single} "
            f"artists for {_N_MULTI} blobs, got {n_multi}",
        )

    def _test_plot_adds(self, method_name: str, count_fn, *args, **kwargs):
        """Call method and verify at least 1 artist is added (no crash)."""
        fig, ax = self._fresh_ax()
        n_before = count_fn(ax)
        getattr(self.blobs1, method_name)(*args, **kwargs)
        self.assertGreater(
            count_fn(ax) - n_before, 0, f"{method_name}: expected at least 1 artist"
        )
        plt.close(fig)

    @staticmethod
    def _lines(ax):
        return len(ax.lines)

    @staticmethod
    def _patches(ax):
        return len(ax.patches)

    # ------------------------------------------------------------------ #
    # plot_box and plot_labelbox  (one Rectangle patch per blob)
    # ------------------------------------------------------------------ #

    def test_plot_box_ratio(self):
        self._test_plot_ratio("plot_box", self._patches)

    def test_plot_box_fmt(self):
        self._test_plot_adds("plot_box", self._patches, "r--")

    def test_plot_box_kwargs(self):
        self._test_plot_adds("plot_box", self._patches, color="yellow")

    def _box_patch(self, method, *args, **kwargs):
        fig, ax = self._fresh_ax()
        getattr(self.blobs1, method)(*args, **kwargs)
        patch = ax.patches[0]
        plt.close(fig)
        return patch

    def test_box_format_strings(self):
        for method in ("plot_box", "plot_labelbox"):
            for fmt, color, linestyle in [
                ("r--", (1, 0, 0), "--"),
                ("g", (0, 0.5, 0), "solid"),
                ("C1:", mcolors.to_rgb("C1"), ":"),
                ("orange", mcolors.to_rgb("orange"), "solid"),
                ("--", mcolors.to_rgb("yellow"), "--"),  # the default color
            ]:
                with self.subTest(method=method, fmt=fmt):
                    patch = self._box_patch(method, fmt)
                    self.assertEqual(tuple(patch.get_edgecolor()[:3]), tuple(color))
                    self.assertEqual(patch.get_linestyle(), linestyle)

    def test_box_with_a_marker_in_the_format_string_says_so(self):
        for method in ("plot_box", "plot_labelbox"):
            for fmt in ("o", "r."):
                with self.subTest(method=method, fmt=fmt):
                    with self.assertRaisesRegex(ValueError, "cannot draw"):
                        self._box_patch(method, fmt)

    def _labelbox(self, *args, **kwargs):
        """Draw one labelled box; return (box patch, label text)."""
        fig, ax = self._fresh_ax()
        self.blobs1.plot_labelbox(*args, **kwargs)
        patch, text = ax.patches[0], ax.texts[0]
        plt.close(fig)
        return patch, text

    @staticmethod
    def _hex(color):
        return mcolors.to_hex(color)

    def test_labelbox_default_label_background_is_the_box_edge_color(self):
        patch, text = self._labelbox()
        self.assertEqual(self._hex(text.get_bbox_patch().get_facecolor()), "#ffff00")
        self.assertEqual(self._hex(text.get_color()), "#000000")

    def test_labelbox_label_background_follows_however_the_edge_was_set(self):
        # was: only the color option was followed, so ec= left a yellow label
        for args, kwargs in [
            ((), dict(color="red")),
            ((), dict(edgecolor="red")),
            ((), dict(ec="red")),
            (("r--",), {}),
        ]:
            with self.subTest(args=args, kwargs=kwargs):
                patch, text = self._labelbox(*args, **kwargs)
                self.assertEqual(self._hex(patch.get_edgecolor()), "#ff0000")
                self.assertEqual(
                    self._hex(text.get_bbox_patch().get_facecolor()), "#ff0000"
                )

    def test_labelbox_shorthands(self):
        patch, text = self._labelbox(textcolor="white", labelcolor="blue")
        self.assertEqual(self._hex(text.get_color()), "#ffffff")
        self.assertEqual(self._hex(text.get_bbox_patch().get_facecolor()), "#0000ff")
        self.assertEqual(self._hex(patch.get_edgecolor()), "#ffff00")  # box unchanged

    def test_labelbox_text_style(self):
        patch, text = self._labelbox(
            text_style=dict(color="white", fontsize=17, fontweight="bold")
        )
        self.assertEqual(self._hex(text.get_color()), "#ffffff")
        self.assertEqual(text.get_fontsize(), 17)
        self.assertEqual(text.get_fontweight(), "bold")
        self.assertEqual(self._hex(patch.get_edgecolor()), "#ffff00")  # box unchanged

    def test_labelbox_text_style_accepts_short_names(self):
        # size and c are aliases of fontsize and color
        patch, text = self._labelbox(text_style=dict(size=15, c="red", weight="bold"))
        self.assertEqual(text.get_fontsize(), 15)
        self.assertEqual(self._hex(text.get_color()), "#ff0000")
        self.assertEqual(text.get_fontweight(), "bold")

    def test_labelbox_text_style_wins_over_the_shorthands(self):
        patch, text = self._labelbox(
            textcolor="white",
            labelcolor="blue",
            text_style=dict(color="red", backgroundcolor="green"),
        )
        self.assertEqual(self._hex(text.get_color()), "#ff0000")
        self.assertEqual(self._hex(text.get_bbox_patch().get_facecolor()), "#008000")

    def test_labelbox_bbox_in_text_style_is_merged_over_the_default(self):
        patch, text = self._labelbox(
            labelcolor="blue", text_style=dict(bbox=dict(linewidth=2, edgecolor="k"))
        )
        bbox = text.get_bbox_patch()
        self.assertEqual(self._hex(bbox.get_facecolor()), "#0000ff")  # kept
        self.assertEqual(bbox.get_linewidth(), 2)
        self.assertEqual(self._hex(bbox.get_edgecolor()), "#000000")
        # and a facecolor in bbox is the most specific of all
        patch, text = self._labelbox(
            labelcolor="blue", text_style=dict(bbox=dict(facecolor="red"))
        )
        self.assertEqual(self._hex(text.get_bbox_patch().get_facecolor()), "#ff0000")

    def test_labelbox_position(self):
        positions = {}
        for position in ("topleft", "topright", "bottomleft", "bottomright"):
            patch, text = self._labelbox(position=position)
            positions[position] = (text.get_ha(), text.get_va())
        self.assertEqual(positions["topleft"], ("left", "bottom"))
        self.assertEqual(positions["topright"], ("right", "bottom"))
        self.assertEqual(positions["bottomleft"], ("left", "top"))
        self.assertEqual(positions["bottomright"], ("right", "top"))

    def test_labelbox_bad_position(self):
        with self.assertRaisesRegex(ValueError, "position must be"):
            self._labelbox(position="middle")

    def test_labelbox_text_style_is_not_changed_by_drawing(self):
        style = dict(size=15, c="red")
        self._labelbox(text_style=style)
        self.assertEqual(style, dict(size=15, c="red"))

    def test_labelbox_label_text(self):
        patch, text = self._labelbox(label="shark {:d}")
        self.assertTrue(text.get_text().startswith("shark "))

    def test_plot_labelbox_ratio(self):
        self._test_plot_ratio("plot_labelbox", self._patches)

    # ------------------------------------------------------------------ #
    # plot_centroid  (default: marker=["bx","bo"] → 2 Line2D per blob)
    # ------------------------------------------------------------------ #

    def test_plot_centroid_ratio(self):
        self._test_plot_ratio("plot_centroid", self._lines)

    def test_plot_centroid_fmt(self):
        self._test_plot_adds("plot_centroid", self._lines, "r+")

    def test_plot_centroid_kwargs(self):
        self._test_plot_adds("plot_centroid", self._lines, marker="x", color="red")

    def test_plot_centroid_text_style(self):
        fig, ax = self._fresh_ax()
        self.blobs1.plot_centroid(
            label=True, text_style=dict(color="white", fontsize=17)
        )
        (text,) = ax.texts
        self.assertEqual(text.get_color(), "white")
        self.assertEqual(text.get_fontsize(), 17)
        plt.close(fig)

    def test_plot_centroid_label_without_text_style(self):
        fig, ax = self._fresh_ax()
        self.blobs1.plot_centroid(label=True)
        self.assertEqual(len(ax.texts), 1)
        plt.close(fig)

    def test_plot_centroid_text_style_without_label_adds_no_text(self):
        fig, ax = self._fresh_ax()
        self.blobs1.plot_centroid(text_style=dict(color="white"))
        self.assertEqual(len(ax.texts), 0)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # plot_perimeter  (plt.plot → 1 Line2D per blob)
    # ------------------------------------------------------------------ #

    def test_plot_perimeter_ratio(self):
        self._test_plot_ratio("plot_perimeter", self._lines)

    def test_plot_perimeter_fmt(self):
        self._test_plot_adds("plot_perimeter", self._lines, "c--")

    def test_plot_perimeter_kwargs(self):
        self._test_plot_adds("plot_perimeter", self._lines, color="cyan")

    # ------------------------------------------------------------------ #
    # plot_ellipse  (smb.plot_ellipse → 1 Line2D per blob)
    # ------------------------------------------------------------------ #

    def test_plot_ellipse_ratio(self):
        self._test_plot_ratio("plot_ellipse", self._lines)

    def test_plot_ellipse_fmt(self):
        self._test_plot_adds("plot_ellipse", self._lines, "g-")

    def test_plot_ellipse_kwargs(self):
        self._test_plot_adds("plot_ellipse", self._lines, color="green")

    # ------------------------------------------------------------------ #
    # plot_axes  (plt.plot × 2 per blob: major + minor axis)
    # ------------------------------------------------------------------ #

    def test_plot_axes_ratio(self):
        self._test_plot_ratio("plot_axes", self._lines)

    def test_plot_axes_fmt(self):
        self._test_plot_adds("plot_axes", self._lines, "b-")

    def test_plot_axes_kwargs(self):
        self._test_plot_adds("plot_axes", self._lines, color="blue")

    def _axes_lines(self, *args, **kwargs):
        """Plot the axes of one blob; return its (major, minor) Line2D."""
        fig, ax = self._fresh_ax()
        self.blobs1.plot_axes(*args, **kwargs)
        major, minor = ax.lines[-2:]
        plt.close(fig)
        return major, minor

    def test_plot_axes_default_colors(self):
        major, minor = self._axes_lines()
        self.assertEqual(major.get_color(), "red")
        self.assertEqual(minor.get_color(), "green")

    def test_plot_axes_kwargs_apply_to_both(self):
        major, minor = self._axes_lines(color="blue", linestyle="--")
        for line in (major, minor):
            self.assertEqual(line.get_color(), "blue")
            self.assertEqual(line.get_linestyle(), "--")

    def test_plot_axes_major_style_only(self):
        major, minor = self._axes_lines(major_style=dict(color="orange", linewidth=5))
        self.assertEqual(major.get_color(), "orange")
        self.assertEqual(major.get_linewidth(), 5)
        self.assertEqual(minor.get_color(), "green")  # untouched default
        self.assertEqual(minor.get_linewidth(), 2)

    def test_plot_axes_minor_style_only(self):
        major, minor = self._axes_lines(minor_style=dict(linestyle=":"))
        self.assertEqual(minor.get_linestyle(), ":")
        self.assertEqual(major.get_linestyle(), "-")
        self.assertEqual(minor.get_color(), "green")

    def test_plot_axes_style_beats_kwargs_and_fmt(self):
        major, minor = self._axes_lines(
            "c--", color="blue", major_style=dict(color="orange")
        )
        self.assertEqual(major.get_color(), "orange")  # style wins over kwargs
        self.assertEqual(major.get_linestyle(), "--")  # fmt still applies
        self.assertEqual(minor.get_color(), "blue")  # kwargs wins over fmt
        self.assertEqual(minor.get_linestyle(), "--")

    def test_plot_axes_style_and_kwargs_may_use_different_spellings(self):
        # lw in kwargs and linewidth in the style used to be an alias clash
        major, minor = self._axes_lines(lw=3, major_style=dict(linewidth=5))
        self.assertEqual(major.get_linewidth(), 5)
        self.assertEqual(minor.get_linewidth(), 3)
        major, minor = self._axes_lines(minor_style=dict(lw=4, c="k"))
        self.assertEqual(minor.get_linewidth(), 4)
        self.assertEqual(minor.get_color(), "k")

    def test_plot_axes_does_not_change_the_defaults(self):
        from machinevisiontoolbox.ImageBlobs import MAJOR_AXIS_STYLE, MINOR_AXIS_STYLE

        before = dict(MAJOR_AXIS_STYLE), dict(MINOR_AXIS_STYLE)
        self._axes_lines(
            "r--",
            color="blue",
            major_style=dict(color="k"),
            minor_style=dict(linewidth=4),
        )
        self.assertEqual((MAJOR_AXIS_STYLE, MINOR_AXIS_STYLE), before)

    # ------------------------------------------------------------------ #
    # plot_aligned_box  (smb.plot_polygon → 1 Line2D per blob)
    # ------------------------------------------------------------------ #

    def test_plot_aligned_box_ratio(self):
        self._test_plot_ratio("plot_aligned_box", self._lines)

    def test_plot_aligned_box_kwargs(self):
        self._test_plot_adds("plot_aligned_box", self._lines, color="yellow")

    # the format string is applied once, by mpl_styling: what each line looks like
    _FMT_RESULTS = [
        # fmt, color, linestyle, marker
        (None, "yellow", "-", "None"),
        ("g", "g", "-", "None"),
        ("r--", "r", "--", "None"),
        ("--", "yellow", "--", "None"),
        ("r:", "r", ":", "None"),
        ("o", "yellow", "None", "o"),  # markers only, as for plot()
        ("r.", "r", "None", "."),
        ("C1", "C1", "-", "None"),
        ("orange", "orange", "-", "None"),
    ]

    def test_polygon_and_circle_methods_apply_the_format_string_once(self):
        for method in ("plot_aligned_box", "plot_MEC", "plot_MER"):
            for fmt, color, linestyle, marker in self._FMT_RESULTS:
                with self.subTest(method=method, fmt=fmt):
                    fig, ax = self._fresh_ax()
                    getattr(self.blobs1, method)(*([] if fmt is None else [fmt]))
                    line = ax.lines[0]
                    plt.close(fig)
                    self.assertEqual(line.get_color(), color)
                    self.assertEqual(line.get_linestyle(), linestyle)
                    self.assertEqual(line.get_marker(), marker)
                    self.assertEqual(line.get_linewidth(), 2)

    # ------------------------------------------------------------------ #
    # plot_MEC  (smb.plot_circle → 1 artist per blob)
    # ------------------------------------------------------------------ #

    def test_plot_MEC_ratio(self):
        # smb.plot_circle may add a Line2D or a Patch; use combined count
        def _combined(ax):
            return len(ax.lines) + len(ax.patches)

        self._test_plot_ratio("plot_MEC", _combined)

    def test_plot_MEC_kwargs(self):
        def _combined(ax):
            return len(ax.lines) + len(ax.patches)

        self._test_plot_adds("plot_MEC", _combined, color="cyan")

    # ------------------------------------------------------------------ #
    # plot_MER  (smb.plot_polygon → 1 Line2D per blob)
    # ------------------------------------------------------------------ #

    def test_plot_MER_ratio(self):
        self._test_plot_ratio("plot_MER", self._lines)

    def test_plot_MER_kwargs(self):
        self._test_plot_adds("plot_MER", self._lines, color="magenta")


# ------------------------------------------------------------------------ #
if __name__ == "__main__":
    unittest.main()
