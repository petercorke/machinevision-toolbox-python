#!/usr/bin/env python

import unittest

import numpy as np
import numpy.testing as nt

from machinevisiontoolbox import ArUcoBoard, Fiducial, Image


class TestImageFiducials(unittest.TestCase):
    @staticmethod
    def _single_marker_scene(
        dict_name: str, marker_id: int, side: int, canvas: int
    ) -> Image:
        marker = Fiducial.create(dict_name, marker_id, side)
        img = np.full((canvas, canvas), 255, dtype=np.uint8)
        offset = (canvas - side) // 2
        img[offset : offset + side, offset : offset + side] = marker.array
        return Image(img)

    def test_create(self):
        """Test fiducial marker creation"""
        marker = Fiducial.create("4x4_50", id=7, sidelength=120)

        self.assertIsInstance(marker, Image)
        self.assertEqual(marker.shape, (120, 120))
        self.assertEqual(marker.dtype, np.uint8)
        self.assertGreaterEqual(marker.array.min(), 0)
        self.assertLessEqual(marker.array.max(), 255)

    def test_fiducial(self):
        """Test fiducial marker detection"""
        marker_id = 17
        im = self._single_marker_scene(
            "4x4_50", marker_id=marker_id, side=160, canvas=320
        )

        result = im.fiducial(dict="4x4_50")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, marker_id)
        self.assertEqual(result[0].corners.shape, (2, 4))
        self.assertIsNone(result[0].pose)

    def test_chart(self):
        """Test loading calibration chart"""
        board = ArUcoBoard(
            layout=(2, 3), sidelength=0.04, separation=0.01, dict="4x4_50"
        )
        chart = board.chart(dpi=100)

        self.assertIsInstance(chart, Image)

        dpm = 100 * 1000 / 25.4
        expected_width = int((2 * (0.04 + 0.01) - 0.01) * dpm)
        expected_height = int((3 * (0.04 + 0.01) - 0.01) * dpm)
        self.assertEqual(chart.shape[1], expected_width)
        self.assertEqual(chart.shape[0], expected_height)

    def test_draw(self):
        """Test drawing detected fiducials"""
        im = self._single_marker_scene("4x4_50", marker_id=5, side=180, canvas=360)
        K = np.array([[300.0, 0.0, 180.0], [0.0, 300.0, 180.0], [0.0, 0.0, 1.0]])
        markers = im.fiducial(dict="4x4_50", K=K, side=0.05)

        self.assertEqual(len(markers), 1)

        canvas = Image(np.zeros((360, 360, 3), dtype=np.uint8), colororder="BGR")
        before = canvas.array.copy()
        markers[0].draw(canvas, length=50, thick=2)
        self.assertFalse(np.array_equal(before, canvas.array))

    def test_estimatePose(self):
        """Test pose estimation from fiducials"""
        im = self._single_marker_scene("4x4_50", marker_id=9, side=160, canvas=320)
        K = np.array([[320.0, 0.0, 160.0], [0.0, 320.0, 160.0], [0.0, 0.0, 1.0]])

        markers = im.fiducial(dict="4x4_50", K=K, side=0.06)

        self.assertEqual(len(markers), 1)
        self.assertIsNotNone(markers[0].pose)
        self.assertEqual(markers[0].rvec.shape, (1, 3))
        self.assertEqual(markers[0].tvec.shape, (1, 3))

    def test_matchImagePoints(self):
        """Test matching image points to object points"""
        board = ArUcoBoard(
            layout=(2, 1), sidelength=0.04, separation=0.01, dict="4x4_50", firsttag=0
        )
        cornerss = [
            np.array([[10, 40, 40, 10], [10, 10, 40, 40]], dtype=np.float32),
            np.array([[60, 90, 90, 60], [10, 10, 40, 40]], dtype=np.float32),
        ]

        obj_points, img_points = board.matchImagePoints(cornerss, ids=[0, 1])

        self.assertEqual(obj_points.shape, (8, 1, 3))
        self.assertEqual(img_points.shape, (8, 1, 2))
        nt.assert_array_equal(img_points[:, 0, :], np.vstack([c.T for c in cornerss]))

    def test_fiducial_properties(self):
        """Test fiducial id and pose properties"""
        corners = np.array([[1, 2, 2, 1], [3, 3, 4, 4]], dtype=np.float32)
        f0 = Fiducial(id=42, corners=corners)
        self.assertEqual(f0.id, 42)
        self.assertIsNone(f0.pose)
        self.assertIn("id=42", str(f0))

        K = np.array([[250.0, 0.0, 100.0], [0.0, 250.0, 100.0], [0.0, 0.0, 1.0]])
        rvec = np.array([[0.01, 0.02, 0.03]], dtype=np.float64)
        tvec = np.array([[0.1, 0.2, 0.8]], dtype=np.float64)
        f1 = Fiducial(id=7, corners=corners, K=K, rvec=rvec, tvec=tvec)
        self.assertEqual(f1.id, 7)
        self.assertIsNotNone(f1.pose)
        self.assertIn("Fiducial(id=7", repr(f1))


# ----------------------------------------------------------------------- #
if __name__ == "__main__":
    unittest.main()
