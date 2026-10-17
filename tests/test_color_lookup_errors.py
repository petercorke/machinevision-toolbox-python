import unittest

from machinevisiontoolbox.base.color import XYZ2RGBxform


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
