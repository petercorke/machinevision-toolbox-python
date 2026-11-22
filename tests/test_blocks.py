#!/usr/bin/env python
"""Tests for the bdsim blocks defined in MVTB (``machinevisiontoolbox.blocks``).

bdsim is an optional dependency, so these tests are skipped when it is not installed.
Only blocks defined in this package are tested here, not bdsim's own blocks.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import SE3

pytest.importorskip("bdsim")
import bdsim  # noqa: E402

from machinevisiontoolbox import CentralCamera, mkcube  # noqa: E402
from machinevisiontoolbox.blocks import (  # noqa: E402
    Camera,
    EstPose_p,
    ImagePlane,
    Visjac_p,
)


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


class TestEstPoseBlock:
    """EstPose_p estimates the pose of a set of points from their image projection."""

    def setup_method(self):
        self.cam = CentralCamera.Default()
        self.P = mkcube(0.2)
        self.T_unknown = SE3.Trans(0.1, 0.2, 1.5) * SE3.RPY(0.1, 0.2, 0.3)
        self.p = self.cam.project_point(self.P, objpose=self.T_unknown)

    def test_matches_the_camera_method(self):
        T_est = self.cam.estpose(self.P, self.p)
        T = EstPose_p(self.cam, self.P).test_output(self.p)[0]
        nt.assert_array_almost_equal(T.A, T_est.A)

    def test_recovers_the_true_pose(self):
        T = EstPose_p(self.cam, self.P).test_output(self.p)[0]
        nt.assert_array_almost_equal(T.A, self.T_unknown.A, decimal=4)

    def test_method_is_passed_on(self):
        T_est = self.cam.estpose(self.P, self.p, method="epnp")
        T = EstPose_p(self.cam, self.P, method="epnp").test_output(self.p)[0]
        nt.assert_array_almost_equal(T.A, T_est.A)

    def test_no_camera_raises(self):
        with pytest.raises(ValueError):
            EstPose_p(None, self.P)


@pytest.fixture
def agg_backend():
    """Run with the off-screen backend, and close the figures afterwards."""
    previous = matplotlib.get_backend()
    plt.switch_backend("Agg")
    yield
    plt.close("all")
    plt.switch_backend(previous)


class TestImagePlaneConstruction:
    """ImagePlane is a graphics block, but its parameters can be checked without drawing."""

    def setup_method(self):
        self.cam = CentralCamera.Default()

    def test_no_camera_raises(self):
        with pytest.raises(ValueError):
            ImagePlane(None)

    def test_default_style_is_markers_without_a_line(self):
        block = ImagePlane(self.cam)
        assert block.args == []
        assert block.kwargs["linestyle"] == "none"
        assert block.kwargs["marker"] == "o"

    def test_style_dict_overrides_the_default(self):
        block = ImagePlane(self.cam, style=dict(marker="x", color="red"))
        assert block.kwargs["marker"] == "x"
        assert block.kwargs["color"] == "red"
        # the rest of the default style is kept
        assert block.kwargs["linestyle"] == "none"

    def test_style_string_is_a_format_string(self):
        block = ImagePlane(self.cam, style="r*")
        assert block.args == ["r*"]
        assert block.kwargs == {}

    def test_bad_style_raises(self):
        with pytest.raises(ValueError):
            ImagePlane(self.cam, style=["r", "b"])

    def test_init_must_be_callable(self):
        with pytest.raises(AssertionError):
            ImagePlane(self.cam, init="not a function")

    def test_options_are_stored(self):
        init = lambda camera: None  # noqa: E731
        block = ImagePlane(self.cam, retain=True, watch=True, init=init)
        assert block.retain is True
        assert block.watch is True
        assert block.init is init
        assert block.camera is self.cam


class TestImagePlaneInASimulation:
    """Run a small diagram, constant point -> CAMERA -> IMAGEPLANE, in a real bdsim run."""

    P = np.array([1.0, 2.0, 5.0])

    def run_diagram(self, **imageplane_args):
        cam = CentralCamera.Default()
        sim = bdsim.BDSim(
            animation=False, graphics=True, hold=False, progress=False, verbose=False
        )
        bd = sim.blockdiagram()
        point = bd.CONSTANT(self.P)
        pose = bd.CONSTANT(SE3())
        camera = bd.CAMERA(cam)
        imageplane = bd.IMAGEPLANE(cam, **imageplane_args)
        bd.connect(point, camera[0])
        bd.connect(pose, camera[1])
        bd.connect(camera, imageplane)
        bd.compile()
        sim.run(bd, T=0.5, dt=0.1)
        return cam, imageplane

    def test_point_is_drawn_where_the_camera_projects_it(self, agg_backend):
        cam, imageplane = self.run_diagram()
        u, v = cam.project_point(self.P).ravel()
        nt.assert_allclose(imageplane.line.get_xdata(), [u])
        nt.assert_allclose(imageplane.line.get_ydata(), [v])

    def test_the_axes_are_titled_with_the_block_name(self, agg_backend):
        _, imageplane = self.run_diagram()
        assert imageplane.ax.get_title() == imageplane.name_tex

    def test_style_reaches_the_line(self, agg_backend):
        _, imageplane = self.run_diagram(style=dict(marker="s", markersize=9))
        assert imageplane.line.get_marker() == "s"
        assert imageplane.line.get_markersize() == 9

    def test_without_retain_only_the_latest_point_is_kept(self, agg_backend):
        _, imageplane = self.run_diagram(retain=False)
        assert np.size(imageplane.u_data) == 1

    def test_with_retain_every_step_is_kept(self, agg_backend):
        cam, imageplane = self.run_diagram(retain=True)
        u, v = cam.project_point(self.P).ravel()
        assert len(imageplane.u_data) > 1
        nt.assert_allclose(np.ravel(imageplane.u_data), u)
        nt.assert_allclose(np.ravel(imageplane.v_data), v)

    def test_init_function_is_called_once_with_the_camera(self, agg_backend):
        calls = []
        cam, _ = self.run_diagram(init=calls.append)
        assert calls == [cam]


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
