#!/usr/bin/env python

import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from spatialmath import SE3

from machinevisiontoolbox import BundleAdjust, CentralCamera, mkcube


def _problem(poses=(SE3(), SE3.Tx(0.1), SE3.Ty(0.1)), start=None):
    """Multi-view problem on a cube.  The first view is fixed; the others are
    initialised at ``start`` (identity if None) rather than their true pose."""
    cam = CentralCamera.Default()
    P = mkcube(0.2, pose=SE3.Tz(1.0))
    ba = BundleAdjust(cam)
    views = []
    for i, pose in enumerate(poses):
        views.append(ba.add_view(pose if i == 0 else (start or SE3()), fixed=(i == 0)))
    landmarks = []
    for k in range(P.shape[1]):
        landmark = ba.add_landmark(P[:, k])
        landmarks.append(landmark)
        for view, pose in zip(views, poses):
            ba.add_projection(view, landmark, cam.project_point(P[:, k], pose=pose))
    return ba, views, landmarks, P, cam


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

    # --- scene graph structure (the pgraph boundary) ------------------------ #

    def test_graph_structure(self):
        ba, views, landmarks, P, cam = _problem()
        g = ba.graph  # property or method, both must give the pgraph graph
        g = g() if callable(g) else g
        self.assertIs(g, ba.g)
        self.assertEqual(ba.nviews, 3)
        self.assertEqual(ba.nlandmarks, 8)
        self.assertEqual(g.n, 3 + 8)
        self.assertEqual(g.ne, 3 * 8)

        # every edge joins one view and one landmark and carries the observation
        for e in g.edges():
            self.assertIsInstance(
                e,
                (
                    type(g.edges()[0])
                    if isinstance(g.edges(), list)
                    else type(next(iter(g.edges())))
                ),
            )
            self.assertEqual(len(e.p), 2)
            self.assertEqual(
                {type(e.v1), type(e.v2)}, {type(views[0]), type(landmarks[0])}
            )

    def test_vertex_and_edge_names(self):
        ba, views, landmarks, P, cam = _problem()
        self.assertEqual([v.name for v in views], ["view#0", "view#1", "view#2"])
        self.assertEqual(landmarks[0].name, "landmark#0")
        names = {e.name for e in ba.g.edges()}
        self.assertIn("view#1--landmark#3", names)
        self.assertEqual(len(names), ba.g.ne)  # all distinct

    def test_back_reference_and_connectivity(self):
        ba, views, landmarks, P, cam = _problem()
        for v in views + landmarks:
            self.assertIs(v.ba, ba)
        # each view sees all 8 landmarks, each landmark is seen by all 3 views
        self.assertEqual(list(ba.g.connectivity(ba.views)), [8, 8, 8])
        self.assertEqual(list(ba.g.connectivity(ba.landmarks)), [3] * 8)

    def test_observation_values(self):
        ba, views, landmarks, P, cam = _problem()
        expect = cam.project_point(P[:, 2], pose=SE3.Tx(0.1))
        edges = [
            e
            for e in ba.g.edges()
            if landmarks[2] in (e.v1, e.v2) and views[1] in (e.v1, e.v2)
        ]
        self.assertEqual(len(edges), 1)
        np.testing.assert_allclose(edges[0].p, expect.flatten())

    def test_add_projection_bad_uv(self):
        ba, views, landmarks, P, cam = _problem()
        with self.assertRaises(ValueError):
            ba.add_projection(views[0], landmarks[0], np.r_[1.0, 2.0, 3.0])

    # --- state vector and bookkeeping ---------------------------------------- #

    def test_state_counts(self):
        ba, views, landmarks, P, cam = _problem()
        self.assertEqual(ba.nstates, 3 * 6 + 8 * 3)
        self.assertEqual(ba.nvarstates, 2 * 6 + 8 * 3)  # first view fixed
        self.assertEqual(ba.getstate().shape, (ba.nstates,))

    def test_setstate_respects_fixed(self):
        ba, views, landmarks, P, cam = _problem()
        x0 = ba.getstate()
        ba.setstate(x0 + 1.0)
        x1 = ba.getstate()
        np.testing.assert_array_equal(x1[:6], x0[:6])  # fixed view unchanged
        np.testing.assert_allclose(x1[6:], x0[6:] + 1.0)

    def test_residual_zero_at_truth(self):
        ba, views, landmarks, P, cam = _problem(start=None)
        # set the free views to their true poses via the state vector
        for view, pose in zip(views[1:], (SE3.Tx(0.1), SE3.Ty(0.1))):
            view.coord = np.r_[pose.t, np.zeros(3)]
        r = ba.getresidual()
        self.assertLess(np.abs(r).max(), 1e-6)

    # --- optimization, multi-view ------------------------------------------- #

    def test_optimize_three_views(self):
        ba, views, landmarks, P, cam = _problem()
        x, err = ba.optimize(iterations=30)
        self.assertLess(err, 1.0)
        # optimize() returns the state vector, it does not write it back
        np.testing.assert_array_equal(views[1].pose.t, 0)
        ba.setstate(x)
        np.testing.assert_allclose(views[1].pose.t, [0.1, 0, 0], atol=1e-3)
        np.testing.assert_allclose(views[2].pose.t, [0, 0.1, 0], atol=1e-3)

    # --- reporting and plotting ---------------------------------------------- #

    def test_str_and_repr(self):
        ba, views, landmarks, P, cam = _problem()
        s = str(ba)
        self.assertIn("3 views", s)
        self.assertIn("8 landmarks", s)
        self.assertIn("24 projections", s)
        self.assertIn("48 equations", s)
        self.assertIn("landmarks per view", s)
        self.assertIn("views per landmark", s)
        r = repr(ba)
        self.assertIn("views=3", r)
        self.assertIn("landmarks=8", r)
        self.assertIn("projections=24", r)

    def test_str_with_fixed_landmarks(self):
        cam = CentralCamera.Default()
        ba = BundleAdjust(cam)
        ba.add_view(SE3(), fixed=True)
        ba.add_landmark(np.r_[0, 0, 1.0], fixed=True)
        self.assertIn("1 locked landmarks", str(ba))
        self.assertIn("1 locked views", str(ba))

    def test_plot(self):
        ba, views, landmarks, P, cam = _problem()
        try:
            ba.plot(block=False)
        finally:
            plt.close("all")

    # --- SBA file loader ------------------------------------------------------ #

    def test_load_sba(self):
        """Round trip: write a synthetic SBA problem to disk and read it back."""
        cam = CentralCamera.Default()
        P = mkcube(0.2, pose=SE3.Tz(1.0))
        poses = [SE3(), SE3.Tx(0.1)]
        with tempfile.TemporaryDirectory() as d:
            cams, pts, calib = (os.path.join(d, n) for n in ("c.txt", "p.txt", "k.txt"))
            np.savetxt(cams, np.array([np.r_[1, 0, 0, 0, p.t] for p in poses]))
            np.savetxt(calib, cam.K)
            with open(pts, "w") as f:
                f.write("# comment line\n")
                for k in range(P.shape[1]):
                    row = [*P[:, k], len(poses)]
                    for i, pose in enumerate(poses):
                        uv = cam.project_point(P[:, k], pose=pose).flatten()
                        row += [i, *uv]
                    f.write(" ".join(str(x) for x in row) + "\n")

            ba = BundleAdjust.load_SBA(cams, pts, calib, imagesize=cam.imagesize)

        self.assertEqual(ba.nviews, 2)
        self.assertEqual(ba.nlandmarks, 8)
        self.assertEqual(ba.g.ne, 16)
        self.assertEqual(ba.g.n, 10)

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

    def test_vertex_base_class(self):
        """ViewPoint and Landmark subclass pgraph.Vertex where it exists
        (pgraph 1.0 deprecates UVertex, which warns on access)."""
        import pgraph

        mod = __import__(BundleAdjust.__module__, fromlist=["ViewPoint"])
        if not hasattr(pgraph, "Vertex"):
            self.skipTest("pgraph < 1.0 has only UVertex")
        self.assertIs(mod._Vertex, pgraph.Vertex)
        self.assertTrue(issubclass(mod.ViewPoint, pgraph.Vertex))
        self.assertTrue(issubclass(mod.Landmark, pgraph.Vertex))


# ----------------------------------------------------------------------- #
if __name__ == "__main__":
    unittest.main()
