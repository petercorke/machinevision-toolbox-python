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


def test_chromaticity_titles():
    from machinevisiontoolbox import plot_chromaticity_diagram, plot_spectral_locus

    plot_chromaticity_diagram(N=20, block=None)
    assert window_title() == "Chromaticity Diagram"
    # drawing the locus onto the diagram leaves the window title alone
    plot_spectral_locus(ax=plt.gca())
    assert window_title() == "Chromaticity Diagram"

    plt.figure()
    plot_spectral_locus()
    assert window_title() == "Spectral Locus"
    plt.figure()
    plot_spectral_locus(title="mine")
    assert window_title() == "mine"


@pytest.mark.parametrize("block", [True, False])
def test_pointcloud_disp_passes_title_to_open3d(block, monkeypatch):
    import numpy as np
    import open3d as o3d

    from machinevisiontoolbox import PointCloud

    names = []
    monkeypatch.setattr(
        o3d.visualization,
        "draw_geometries",
        lambda geoms, window_name="Open3D", **kw: names.append(window_name),
    )

    class FakeVis:
        def create_window(self, window_name="Open3D", **kw):
            names.append(window_name)

        def __getattr__(self, name):
            return lambda *a, **kw: None

    monkeypatch.setattr(o3d.visualization, "Visualizer", FakeVis)

    pc = PointCloud(np.random.rand(3, 10))
    pc.disp(block=block)
    pc.disp(block=block, title="mine")
    assert names == ["Point Cloud", "mine"]


def test_hough_accumulator_titles():
    import numpy as np

    from machinevisiontoolbox import Image

    a = np.zeros((40, 40), dtype="uint8")
    a[10:30, 20] = 255
    im = Image(a)
    h = im.Hough()
    h.plot_accumulator()
    assert window_title() == "Hough Accumulator"
    h.plot_accumulator(title="mine")
    assert window_title() == "mine"
    # an existing axes is drawn into, and its window left alone
    fig, ax = plt.subplots()
    h.plot_accumulator(ax=ax)
    assert ax.get_xlabel() != ""
    assert window_title(fig) != "Hough Accumulator"


def test_thin_animate_default_title():
    import numpy as np

    from machinevisiontoolbox import Image

    a = np.zeros((20, 20), dtype="uint8")
    a[5:15, 5:15] = 1
    im = Image(a)
    im.thin_animate(delay=0)
    assert window_title() == "Animation of Morphological Skeletonization"


def test_feature_match_plot_passes_title():
    from machinevisiontoolbox import Image

    im = Image.Read("eiffel-1.png")
    sf = im.SIFT()
    matches = sf.match(sf)
    matches.plot(title="mine")
    assert window_title() == "mine"
