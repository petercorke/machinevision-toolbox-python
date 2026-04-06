#!/usr/bin/env python

import unittest

import numpy as np
import numpy.testing as nt

from machinevisiontoolbox import Image


class TestImageFiducials(unittest.TestCase):
    # new test
    def test_create(self):
        """Test fiducial marker creation"""
        # TODO: Test creating ArUco or AprilTag markers
        pass

    # new test
    def test_fiducial(self):
        """Test fiducial marker detection"""
        # TODO: Create a test image with known fiducial marker
        im = Image.Zeros(size=(200, 200), dtype="uint8")
        # Detect fiducials
        # result = im.fiducial(dict='DICT_4X4_50')
        pass

    # new test
    def test_chart(self):
        """Test loading calibration chart"""
        # TODO: Test loading a checkerboard or circle grid chart
        pass

    # new test
    def test_draw(self):
        """Test drawing detected fiducials"""
        # TODO: Create image with detected fiducials and test drawing
        pass

    # new test
    def test_estimatePose(self):
        """Test pose estimation from fiducials"""
        # TODO: Test pose estimation with known camera parameters
        pass

    # new test
    def test_matchImagePoints(self):
        """Test matching image points to object points"""
        # TODO: Test point matching for pose estimation
        pass

    # new test
    def test_fiducial_properties(self):
        """Test fiducial id and pose properties"""
        # TODO: Create fiducial detection result and test properties
        pass


# ----------------------------------------------------------------------- #
if __name__ == "__main__":
    unittest.main()
