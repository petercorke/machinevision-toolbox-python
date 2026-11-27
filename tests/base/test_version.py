#!/usr/bin/env python
import unittest
from contextlib import redirect_stdout
from io import StringIO

import cv2
import numpy as np
import spatialmath as sm

import machinevisiontoolbox as mvt
from machinevisiontoolbox.base.version import mvtb_version


class TestVersion(unittest.TestCase):
    def test_mvtb_version_reports_versions(self):
        stream = StringIO()
        with redirect_stdout(stream):
            mvtb_version()
        text = stream.getvalue()

        # one line for each package, each with the version that is installed
        self.assertIn(
            f"Machine Vision Toolbox for Python: version {mvt.__version__}", text
        )
        self.assertIn(sm.__version__, text)
        self.assertIn(np.__version__, text)
        self.assertIn(cv2.__version__, text)

    def test_mvtb_version_includes_opencv_build_information(self):
        stream = StringIO()
        with redirect_stdout(stream):
            mvtb_version()
        text = stream.getvalue()

        self.assertIn("OpenCV build information:", text)
        # the build information is indented as a quotation
        self.assertIn("> General configuration for OpenCV", text)


if __name__ == "__main__":
    unittest.main()
