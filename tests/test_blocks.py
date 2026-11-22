#!/usr/bin/env python
"""Tests for the bdsim blocks defined in MVTB (``machinevisiontoolbox.blocks``).

bdsim is an optional dependency, so these tests are skipped when it is not installed.
Only blocks defined in this package are tested here, not bdsim's own blocks.
"""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import SE3

pytest.importorskip("bdsim")

from machinevisiontoolbox import CentralCamera, mkcube  # noqa: E402
from machinevisiontoolbox.blocks import Camera, EstPose_p, Visjac_p  # noqa: E402


class TestCameraBlock:
    """Camera projects world points through a camera model, from a camera pose."""

    def setup_method(self):
        self.cam = CentralCamera.Default()
        self.block = Camera(self.cam)
        self.P = np.array([1, 2, 5])

    def test_camera_at_the_origin(self):
        p = self.cam.project_point(self.P, pose=SE3())
        nt.assert_array_almost_equal(self.block.test_output(self.P, SE3())[0], p)

    def test_camera_moved(self):
        T = SE3.Trans(0.2, 0.3, 0.4)
        p = self.cam.project_point(self.P, pose=T)
        nt.assert_array_almost_equal(self.block.test_output(self.P, T)[0], p)

    def test_many_points(self):
        P = mkcube(0.2) + np.array([[0], [0], [2]])
        T = SE3.Trans(0.1, 0, 0)
        p = self.cam.project_point(P, pose=T)
        out = self.block.test_output(P, T)[0]
        assert out.shape == (2, 8)
        nt.assert_array_almost_equal(out, p)

    def test_no_camera_raises(self):
        with pytest.raises(ValueError):
            Camera(None)


class TestVisjacBlock:
    """Visjac_p gives the point-feature interaction matrix for image points."""

    def setup_method(self):
        self.cam = CentralCamera.Default()
        self.P = np.array([1, 2, 5])
        self.p = self.cam.project_point(self.P)

    def test_depth(self):
        block = Visjac_p(self.cam, 5)
        J = self.cam.visjac_p(self.p, 5)
        nt.assert_array_almost_equal(block.test_output(self.p)[0], J)

    def test_default_depth_is_one(self):
        J = self.cam.visjac_p(self.p, 1)
        nt.assert_array_almost_equal(Visjac_p(self.cam).test_output(self.p)[0], J)

    def test_jacobian_has_two_rows_per_point(self):
        p = self.cam.project_point(mkcube(0.2) + np.array([[0], [0], [2]]))
        J = Visjac_p(self.cam, 2).test_output(p)[0]
        assert J.shape == (16, 6)

    def test_no_camera_raises(self):
        with pytest.raises(ValueError):
            Visjac_p(None)


class TestEstPoseFrame:
    """EstPose_p passes its ``frame`` argument on to ``CentralCamera.estpose``."""

    def setup_method(self):
        # a camera that is not at the world origin, so that the two frames differ
        self.cam = CentralCamera.Default(pose=SE3.Trans(0.1, -0.2, -1.0))
        self.P = mkcube(0.2)
        self.T_world = SE3.Trans(0.1, 0.2, 0.5) * SE3.RPY(0.1, 0.2, 0.3)
        self.p = self.cam.project_point(self.P, objpose=self.T_world)

    def test_default_is_world(self):
        block = EstPose_p(self.cam, self.P)
        T = block.test_output(self.p)[0]
        nt.assert_array_almost_equal(T.A, self.T_world.A, decimal=4)

    def test_camera_frame(self):
        block = EstPose_p(self.cam, self.P, frame="camera")
        T = block.test_output(self.p)[0]
        T_camera = self.cam.pose.inv() * self.T_world
        nt.assert_array_almost_equal(T.A, T_camera.A, decimal=4)

    def test_frames_differ(self):
        T_w = EstPose_p(self.cam, self.P, frame="world").test_output(self.p)[0]
        T_c = EstPose_p(self.cam, self.P, frame="camera").test_output(self.p)[0]
        assert not np.allclose(T_w.A, T_c.A)

    def test_bad_frame_raises(self):
        with pytest.raises(ValueError):
            EstPose_p(self.cam, self.P, frame="robot")
