#!/usr/bin/env python

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "machinevisiontoolbox"
    / "blocks"
    / "Icons"
    / "make_icons.py"
)


def load_make_icons():
    # the Icons folder is a data folder, not a package, so load the script by path
    spec = importlib.util.spec_from_file_location("make_icons", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMakeIcons(unittest.TestCase):
    def test_icons_are_square_black_ink_on_transparent(self):
        """Every icon is a 250x250 RGBA image of black ink, centered, with
        a transparent border"""
        make_icons = load_make_icons()
        self.assertGreater(len(make_icons.ICONS), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            for name, tex in make_icons.ICONS.items():
                path = Path(tmpdir) / f"{name}.png"
                make_icons.tex_icon(tex, path)

                im = Image.open(path)
                self.assertEqual(im.size, (250, 250), name)
                self.assertEqual(im.mode, "RGBA", name)

                rgba = np.asarray(im)
                ink = rgba[..., 3] > 0
                self.assertTrue(ink.any(), f"{name}: no ink")
                # black ink, wherever there is any
                self.assertEqual(rgba[ink][:, :3].max(), 0, name)

                # the ink fits inside the margin, and the border is transparent
                ys, xs = np.nonzero(ink)
                self.assertGreaterEqual(min(xs.min(), ys.min()), 10, name)
                self.assertLessEqual(max(xs.max(), ys.max()), 239, name)
                for edge in (rgba[:10], rgba[-10:], rgba[:, :10], rgba[:, -10:]):
                    self.assertEqual(edge[..., 3].max(), 0, f"{name}: border not clear")

                # glyphs are thin strokes: a solid square (e.g. an opaque
                # background mistaken for ink) would cover most of the icon
                coverage = ink.mean()
                self.assertGreater(coverage, 0.01, f"{name}: almost empty")
                self.assertLess(coverage, 0.40, f"{name}: ink covers {coverage:.0%}")

                # some pixels are partially transparent, i.e. anti-aliased
                self.assertTrue(((rgba[..., 3] > 0) & (rgba[..., 3] < 255)).any(), name)

    def test_unknown_tex_is_an_error(self):
        """A malformed expression must raise rather than write a blank icon"""
        make_icons = load_make_icons()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                make_icons.tex_icon(r"\notacommand{x}", Path(tmpdir) / "bad.png")


if __name__ == "__main__":
    unittest.main()
