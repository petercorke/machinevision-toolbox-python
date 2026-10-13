#!/usr/bin/env python

import unittest

import numpy as np
from spatialmath import SE3

from machinevisiontoolbox import BundleAdjust, CentralCamera, mkcube


class TestBundleAdjust(unittest.TestCase):
    def test_optimize(self):
        """A two-view problem with a deliberately wrong initial pose guess
        for the second view must converge back towards the true pose."""
        cam = CentralCamera.Default()
        P = mkcube(0.2, pose=SE3.Tz(1.0))

        pose0 = SE3()
        pose1_true = SE3.Tx(0.1)

        p0 = cam.project_point(P, pose=pose0)
        p1 = cam.project_point(P, pose=pose1_true)

        ba = BundleAdjust(cam)
        v0 = ba.add_view(pose0, fixed=True)
        v1 = ba.add_view(SE3())  # wrong initial guess, true pose is Tx(0.1)

        for k in range(P.shape[1]):
            landmark = ba.add_landmark(P[:, k])
            ba.add_projection(v0, landmark, p0[:, k])
            ba.add_projection(v1, landmark, p1[:, k])

        x_new, err = ba.optimize(iterations=20)
        self.assertLess(err, 1.0)

    def test_optimize_empty_graph_raises(self):
        """optimize() on a graph with no landmark projections must raise a
        clear error rather than silently reporting err=nan (previously
        err = sqrt(enew / self.g.ne) divided by zero with no landmarks
        added, eg. every candidate landmark got discarded upstream)."""
        cam = CentralCamera.Default()
        ba = BundleAdjust(cam)
        ba.add_view(SE3(), fixed=True)
        ba.add_view(SE3())

        with self.assertRaises(ValueError):
            ba.optimize(iterations=5)


    def test_fixed_landmark(self):
        """A fixed landmark is recorded as fixed, excluded from the variable
        states, and not moved by the optimization (previously add_landmark
        raised NameError for fixed=True)."""
        cam = CentralCamera.Default()
        P = mkcube(0.2, pose=SE3.Tz(1.0))
        p0 = cam.project_point(P, pose=SE3())
        p1 = cam.project_point(P, pose=SE3.Tx(0.1))

        ba = BundleAdjust(cam)
        v0 = ba.add_view(SE3(), fixed=True)
        v1 = ba.add_view(SE3())
        landmarks = []
        for k in range(P.shape[1]):
            landmark = ba.add_landmark(P[:, k], fixed=(k == 0))
            landmarks.append(landmark)
            ba.add_projection(v0, landmark, p0[:, k])
            ba.add_projection(v1, landmark, p1[:, k])

        self.assertTrue(landmarks[0].isfixed)
        self.assertFalse(landmarks[1].isfixed)
        self.assertEqual(ba.fixedlandmarks, [landmarks[0]])
        self.assertEqual(ba.nvarstates, 6 + 3 * (P.shape[1] - 1))

        x_new, err = ba.optimize(iterations=20)
        self.assertLess(err, 1.0)
        np.testing.assert_array_equal(landmarks[0].P, P[:, 0])

    def test_no_pgraph_deprecation_warning(self):
        """Importing BundleAdjust must not trigger a pgraph DeprecationWarning
        (pgraph 1.0 deprecates UVertex)."""
        import subprocess
        import sys

        r = subprocess.run(
            [
                sys.executable,
                "-W",
                "error::DeprecationWarning",
                "-c",
                "import machinevisiontoolbox.BundleAdjust",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)


# ----------------------------------------------------------------------- #
if __name__ == "__main__":
    unittest.main()
