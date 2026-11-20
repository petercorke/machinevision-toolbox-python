"""Plotting functions that create a window give it a sensible title."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from machinevisiontoolbox import Image


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def window_title(fig=None):
    fig = fig or plt.gcf()
    return fig.canvas.manager.get_window_title()


def test_histogram_default_title():
    Image.Random(size=(8, 6)).hist().plot()
    assert window_title() == "Image Histogram"


def test_histogram_given_title():
    Image.Random(size=(8, 6)).hist().plot(title="mine")
    assert window_title() == "mine"


def test_threshold_interactive_default_title():
    Image.Random(size=(8, 6), dtype="uint8").threshold_interactive(block=False)
    assert window_title() == "Interactive Image Threshold"


def test_threshold_interactive_given_title():
    Image.Random(size=(8, 6), dtype="uint8").threshold_interactive(
        title="mine", block=False
    )
    assert window_title() == "mine"


def test_camera_image_plane_title():
    from machinevisiontoolbox import CentralCamera

    camera = CentralCamera.Default(name="mycam")
    camera.plot_point([[0], [0], [1]], "o")
    assert window_title() == "Camera Simulator: mycam Image Plane"


def test_visual_servo_plot_titles():
    import numpy as np
    from spatialmath import SE3

    from machinevisiontoolbox import CentralCamera, IBVS

    camera = CentralCamera.Default(pose=SE3.Trans(1, 1, -3) * SE3.Rz(0.6))
    P = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1], [0, 0, 0, 0]])
    p_star = np.array([[462, 562, 562, 462], [362, 362, 662, 662]])
    ibvs = IBVS(camera, P=P, p_d=p_star, graphics=False)
    ibvs.run(20)

    expected = {
        "plot_p": "Image Feature Trajectories",
        "plot_vel": "Camera Velocity",
        "plot_pose": "Camera Pose",
        "plot_jcond": "Image Jacobian Condition",
        "plot_error": "Feature Error",
    }
    for method, title in expected.items():
        plt.figure()
        getattr(ibvs, method)()
        assert window_title() == title, method


def test_kernel_disp3d_titles():
    from machinevisiontoolbox import Kernel

    Kernel.Gauss(2).disp3d()
    assert window_title() == "Image Kernel: Gaussian σ=2"
    Kernel.Gauss(2).disp3d(title="mine")
    assert window_title() == "mine"
    Kernel(Kernel.Gauss(2).K).disp3d()
    assert window_title() == "Image Kernel"


def test_stdisp_titles():
    left = Image.Random(size=(8, 6))
    right = Image.Random(size=(8, 6))
    left.stdisp(right, interactive=False)
    assert window_title() == "Stereo Pair"
    left.stdisp(right, interactive=False, title="mine")
    assert window_title() == "mine"
    left._name = "/some/dir/left.png"
    right._name = "/some/dir/right.png"
    left.stdisp(right, interactive=False)
    assert window_title() == "left.png / right.png"
