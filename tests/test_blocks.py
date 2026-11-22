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
from machinevisiontoolbox.blocks import EstPose_p  # noqa: E402


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
