"""
Machine Vision blocks:
- have inputs and outputs
- are a subclass of ``FunctionBlock`` |rarr| ``Block`` for kinematics and have no states
- are a subclass of ``TransferBlock`` |rarr| ``Block`` for dynamics and have states

"""

from typing import Any, Callable

import numpy as np
from bdsim.components import FunctionBlock, SimulationState
from bdsim.graphics import GraphicsBlock

from machinevisiontoolbox.Camera import CameraBase, CentralCamera

# The constructor of each class ``MyClass`` with a ``@block`` decorator becomes a method ``MYCLASS()`` of the BlockDiagram instance.


# ------------------------------------------------------------------------ #
class Camera(FunctionBlock):
    """
    :blockname:`CAMERA`

    Camera projection block.
    """

    nin = 2
    nout = 1
    inlabels = ("P", "ξ")
    outlabels = ("p",)

    def __init__(
        self,
        camera: CameraBase | None = None,
        args: dict[str, Any] = {},
        **blockargs: Any,
    ) -> None:
        """
        :param camera: camera model, a subclass of :class:`~machinevisiontoolbox.Camera.CameraBase`
        :param args: not used
        :param blockargs: Additional options for the block, passed as a dictionary
        :raises ValueError: if ``camera`` is None
        :return: a CAMERA block
        :rtype: Camera instance

        Camera projection model.

        **Block ports**

            :input P: world points as ndarray(3,N)
            :input ξ: camera pose as an SE3 object

            :output p: image plane points as ndarray(2,N)
        """
        if camera is None:
            raise ValueError("camera is not defined")

        super().__init__(**blockargs)

        self.camera = camera

    def output(self, t: float, inports: list[Any], x: np.ndarray) -> list[Any]:
        return [self.camera.project_point(inports[0], pose=inports[1])]


# ------------------------------------------------------------------------ #


class Visjac_p(FunctionBlock):
    """
    :blockname:`VISJAC_P`

    Interaction-matrix block for image points.
    """

    nin = 1
    nout = 1
    inlabels = ("p",)
    outlabels = ("J",)

    def __init__(
        self,
        camera: CentralCamera,
        depth: float | np.ndarray = 1,
        depthest: bool = False,
        **blockargs: Any,
    ) -> None:
        """
        :param camera: camera model
        :param depth: point depth, a scalar or one value per point
        :type depth: float or ndarray(N)
        :param depthest: use depth estimation, not implemented so it must be
            False (the default).  Give the depth with ``depth``.
        :param blockargs: Additional options for the block, passed as a dictionary
        :raises ValueError: if ``camera`` is None
        :raises NotImplementedError: if ``depthest`` is True
        :return: a VISJAC_P block
        :rtype: Visjac_p instance

        Computes the image Jacobian (interaction matrix) of the image plane
        points on its input, using the same ``depth`` for every point, see
        :meth:`~machinevisiontoolbox.Camera.CentralCamera.visjac_p`.

        **Block ports**

            :input p: image plane points as ndarray(2,N)

            :output J: image Jacobian as ndarray(2N,6)

        .. versionchanged:: 2.5.0
            ``depthest=True`` raises :class:`NotImplementedError`.  It was
            accepted but did nothing, so the Jacobian was computed with ``depth``.
        """

        # TODO allow a depth input

        if camera is None:
            raise ValueError("camera is not defined")
        if depthest:
            raise NotImplementedError(
                "depth estimation is not implemented by the VISJAC_P block, "
                "give the depth with the depth argument"
            )

        super().__init__(**blockargs)

        self.camera = camera
        self.depthest = depthest
        self.depth = depth

    def output(self, t: float, inports: list[Any], x: np.ndarray) -> list[Any]:
        J = self.camera.visjac_p(inports[0], self.depth)
        return [J]


# ------------------------------------------------------------------------ #


class EstPose_p(FunctionBlock):
    """
    :blockname:`ESTPOSE_P`

    Pose-estimation block for image points.
    """

    nin = 1
    nout = 1
    inlabels = ("p",)
    outlabels = ("ξ",)

    def __init__(
        self,
        camera: CentralCamera,
        P: np.ndarray,
        frame: str = "world",
        method: str = "iterative",
        **blockargs: Any,
    ) -> None:
        """
        :param camera: camera model
        :param P: world point coordinates
        :type P: ndarray(3,N)
        :param frame: return pose of points with respect to reference frame which is one of: 'world' [default] or 'camera'
        :param method: pose estimation algorithm one of: 'iterative' [default], 'epnp', 'p3p', 'ap3p', 'ippe', 'ippe-square'
        :param blockargs: Additional options for the block, passed as a dictionary
        :return: a ESTPOSE_P block
        :rtype: EstPose_p instance
        :raises ValueError: if ``camera`` is None or ``frame`` is not ``"world"`` or ``"camera"``
        :raises RuntimeError: when the diagram is evaluated, if the pose estimation finds
            no solution for the points.  bdsim prints the block, the time and the input,
            and stops the run with a "Fatal failure" error caused by this one.

        .. versionchanged:: 2.5.0
            ``frame`` is now passed to :meth:`~machinevisiontoolbox.Camera.CentralCamera.estpose`.
            Previously it was accepted but ignored, so the pose was always given with
            respect to the world frame.

        .. versionchanged:: 2.5.0
            A pose estimate that fails raises :class:`RuntimeError`.  It used to
            output ``None``, which the next block received as its input.
        """
        if camera is None:
            raise ValueError("camera is not defined")
        if frame not in ("world", "camera"):
            raise ValueError(f"bad frame value {frame}, must be 'world' or 'camera'")

        super().__init__(**blockargs)

        self.camera = camera
        self.P = P
        self.frame = frame
        self.method = method

    def output(self, t: float, inports: list[Any], x: np.ndarray) -> list[Any]:
        p = inports[0]
        T = self.camera.estpose(self.P, p, method=self.method, frame=self.frame)
        if T is None:
            # bdsim prints the block, the time and the input, then stops the run
            # with a "Fatal failure" error caused by this one
            raise RuntimeError(
                f"pose estimation failed: method {self.method!r} found no pose "
                f"for {p.shape[1]} points"
            )
        return [T]


# ------------------------------------------------------------------------ #


class ImagePlane(GraphicsBlock):
    """
    :blockname:`IMAGEPLANE`

    Graphics block that displays points on the image plane.
    """

    nin = 1
    nout = 0

    def __init__(
        self,
        camera: CameraBase,
        style: str | dict[str, Any] | None = None,
        labels: Any = None,
        grid: bool | list[Any] | tuple[Any, ...] = True,
        retain: bool = False,
        watch: bool = False,
        init: Callable[[CameraBase], None] | None = None,
        **blockargs: Any,
    ) -> None:
        """
        Create a block that plots image plane coordinates.

        :param camera: camera model, which draws the image plane
        :param style: style for the points to be plotted, a dict of Matplotlib
            line properties which is added to the default style, or a format
            string such as ``"r*"``
        :param labels: not used
        :param grid: draw a grid, defaults to True.  Can be a bool or a tuple of
            options for ``grid()``
        :param retain: keep previous image plane points, defaults to False
        :param watch: add the input signal to the watchlist, defaults to False
        :param init: function to initialize the graphics, called once with the
            camera when the simulation starts, defaults to None
        :param blockargs: Additional options for the block, passed as a dictionary
        :raises ValueError: if ``camera`` is None or ``style`` is not a str or dict
        :return: An IMAGEPLANE block
        :rtype: ImagePlane instance

        Create a block that plots points on a camera object's virtual image plane.

        **Block ports**

            :input p: image plane points as ndarray(2,N)

        Examples::

            IMAGEPLANE(camera)
            IMAGEPLANE(camera, style="r*")
            IMAGEPLANE(camera, style=dict(marker="s", markersize=9), retain=True)
            IMAGEPLANE(camera, grid=False)

        .. versionchanged:: 2.5.0
            Works with the graphics block interface of bdsim 1.5: ``watch=True``
            no longer fails, only one figure is opened, and nothing is drawn when
            graphics are turned off.
        """
        if camera is None:
            raise ValueError("camera is not defined")

        self.camera = camera

        if style is None:
            style = {}
        if isinstance(style, dict):
            default_style = dict(
                linestyle="none",
                marker="o",
                markersize=4,
                markeredgecolor="black",
                markerfacecolor="black",
            )
            self.kwargs: dict[str, Any] = {**default_style, **style}
            self.args: list[str] = []
        elif isinstance(style, str):
            self.args = [style]
            self.kwargs = {}
        else:
            raise ValueError("bad style, must be str or dict")

        if init is not None:
            assert callable(init), "graphics init function must be callable"
        self.init = init
        self.retain = retain

        # watch is passed to the base class, which adds the input signals to the
        # watchlist when the simulation starts
        super().__init__(nin=1, watch=watch, **blockargs)

        self.grid = grid

        # TODO, wire width
        # inherit names from wires, block needs to be able to introspect

    def start(self, simstate: SimulationState) -> None:
        # the base class creates the figure and axes, and registers the input
        # signals for watching if watch=True
        super().start(simstate)

        # init the arrays that hold the data
        self.u_data: Any = []
        self.v_data: Any = []
        self.t_data: list[float] = []

        if not self._enabled:
            # graphics are turned off for this run
            return

        self.camera._init_imageplane(ax=self.ax)

        self.ax.set_title(self.name_tex)

        (self.line,) = self.ax.plot(self.u_data, self.v_data, *self.args, **self.kwargs)

        # grid control
        if self.grid is True:
            self.ax.grid(self.grid)
        elif isinstance(self.grid, (list, tuple)):
            self.ax.grid(True, *self.grid)

        if self.init is not None:
            self.init(self.camera)

    def record(self, t: float, inports: list[Any]) -> None:
        # called for every accepted sample, whereas step() may be called less often
        # just to refresh the display
        if not self._enabled:
            return
        self.t_data.append(t)
        u, v = inports[0]

        if self.retain:
            self.u_data.append(u)
            self.v_data.append(v)
        else:
            self.u_data = u
            self.v_data = v

    def step(self, t: float, inports: list[Any]) -> None:
        if not self._enabled:
            return

        self.line.set_data(self.u_data, self.v_data)

        super().step(t, inports)
