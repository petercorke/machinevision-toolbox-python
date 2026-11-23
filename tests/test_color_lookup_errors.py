import unittest

from machinevisiontoolbox.base.color import (
    XYZ2RGBxform,
    name2color,
    plot_spectral_locus,
)


class TestColorLookupErrors(unittest.TestCase):
    def test_unknown_white_lists_supported_values(self):
        """Report supported white points once, separated by commas."""
        with self.assertRaises(ValueError) as raised:
            XYZ2RGBxform(white="unknown")
        self.assertEqual(
            str(raised.exception), "unknown white value, must be one of d65, e"
        )
        self.assertIsInstance(raised.exception.__cause__, KeyError)

    def test_unknown_primaries_lists_supported_values(self):
        """Report supported primaries once, separated by commas."""
        with self.assertRaises(ValueError) as raised:
            XYZ2RGBxform(primaries="unknown")
        self.assertEqual(
            str(raised.exception),
            "unknown primary value, must be one of itu-709, cie, srgb",
        )
        self.assertIsInstance(raised.exception.__cause__, KeyError)


class TestName2ColorErrors(unittest.TestCase):
    def test_unknown_colorspace_raises(self):
        """An unknown colorspace is an error, not the None of an unknown name."""
        with self.assertRaisesRegex(ValueError, "unknown colorspace 'hsv'"):
            name2color("red", "hsv")

    def test_unknown_name_is_none(self):
        self.assertIsNone(name2color("not a color"))
        self.assertIsNone(name2color("not a color", "xy"))

    def test_colorspace_is_not_case_sensitive(self):
        self.assertEqual(name2color("red", "RGB").tolist(), [1, 0, 0])
        self.assertEqual(name2color("red", "XYZ").shape, (3,))
        self.assertEqual(name2color("red", "xy").shape, (2,))

    def test_wildcard_with_unknown_colorspace_is_a_list(self):
        # the wildcard branch returns names and does not use the colorspace
        self.assertIsInstance(name2color(".*burnt.*", "xy"), list)


class TestPlotSpectralLocusErrors(unittest.TestCase):
    def test_unknown_colorspace_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown colorspace 'lab'"):
            plot_spectral_locus("lab")

    def test_a_substring_of_rg_is_not_rg(self):
        # `colorspace in ("rg")` is a substring test, so "r" and "" used to pass
        for bad in ("r", "g", ""):
            with self.assertRaises(ValueError):
                plot_spectral_locus(bad)
