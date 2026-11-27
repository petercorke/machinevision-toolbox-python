import unittest

from machinevisiontoolbox.base.color import (
    XYZ2RGBxform,
    color_names,
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

    def test_wildcard_is_an_error_that_names_the_replacement(self):
        for name in (".*burnt.*", "?burnt", "*coral"):
            with self.assertRaisesRegex(ValueError, "color_names"):
                name2color(name)


class TestColorNames(unittest.TestCase):
    def test_no_pattern_lists_every_name(self):
        names = color_names()
        self.assertIsInstance(names, list)
        self.assertIn("red", names)
        self.assertGreater(len(names), 100)
        self.assertEqual(len(names), len(set(names)))

    def test_pattern_is_a_regular_expression(self):
        names = color_names(".*burnt.*")
        self.assertGreater(len(names), 0)
        self.assertTrue(all("burnt" in n for n in names))

    def test_the_whole_name_must_match(self):
        # "coral" alone is not a match for "light coral" or "coral"-containing names
        self.assertEqual(color_names("coral"), ["coral"])
        self.assertIn("lightcoral", color_names(".*coral"))
        self.assertNotIn("lightcoral", color_names("coral"))

    def test_alternatives(self):
        self.assertEqual(
            sorted(color_names("(red|green|blue)")), ["blue", "green", "red"]
        )

    def test_no_match_is_an_empty_list(self):
        self.assertEqual(color_names("zzzz.*"), [])

    def test_every_name_is_known_to_name2color(self):
        for name in color_names("[a-z]+"):
            self.assertEqual(name2color(name).shape, (3,))

    def test_shell_style_wildcard_is_an_error_that_says_so(self):
        for bad in ("*coral*", "?burnt"):
            with self.assertRaisesRegex(ValueError, "not shell style wildcards"):
                color_names(bad)


class TestPlotSpectralLocusErrors(unittest.TestCase):
    def test_unknown_colorspace_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown colorspace 'lab'"):
            plot_spectral_locus("lab")

    def test_a_substring_of_rg_is_not_rg(self):
        # `colorspace in ("rg")` is a substring test, so "r" and "" used to pass
        for bad in ("r", "g", ""):
            with self.assertRaises(ValueError):
                plot_spectral_locus(bad)
