import unittest
import io
import json
from pathlib import Path
from unittest.case import skip

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt

from machinevisiontoolbox import Image, Blobs

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
    # plot_box  — SKIPPED; positional fmt arg is broken, upstream PR pending
    # TODO: remove skips once the PR is merged into the release
    # ------------------------------------------------------------------ #

    @unittest.skip(
        "plot_box(): passing a positional fmt argument is broken in the current "
        "release. An upstream PR has been submitted but not yet merged. "
        "Revisit and remove this skip once the fix is distributed."
    )
    def test_plot_box_ratio(self):
        self._test_plot_ratio("plot_box", self._patches)

    @unittest.skip("plot_box(): see test_plot_box_ratio for details")
    def test_plot_box_fmt(self):
        self._test_plot_adds("plot_box", self._patches, "r--")

    @unittest.skip("plot_box(): see test_plot_box_ratio for details")
    def test_plot_box_kwargs(self):
        self._test_plot_adds("plot_box", self._patches, color="yellow")

    # ------------------------------------------------------------------ #
    # plot_centroid  (default: marker=["bx","bo"] → 2 Line2D per blob)
    # ------------------------------------------------------------------ #

    def test_plot_centroid_ratio(self):
        self._test_plot_ratio("plot_centroid", self._lines)

    def test_plot_centroid_fmt(self):
        self._test_plot_adds("plot_centroid", self._lines, "r+")

    def test_plot_centroid_kwargs(self):
        self._test_plot_adds("plot_centroid", self._lines, marker="x", color="red")

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

    # ------------------------------------------------------------------ #
    # plot_aligned_box  (smb.plot_polygon → 1 Line2D per blob)
    # ------------------------------------------------------------------ #

    def test_plot_aligned_box_ratio(self):
        self._test_plot_ratio("plot_aligned_box", self._lines)

    def test_plot_aligned_box_kwargs(self):
        self._test_plot_adds("plot_aligned_box", self._lines, color="yellow")

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
