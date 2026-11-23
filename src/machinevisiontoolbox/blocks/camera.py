from bdsim.components import FunctionBlock
from bdsim.graphics import GraphicsBlock

"""
Machine Vision blocks:
- have inputs and outputs
- are a subclass of ``FunctionBlock`` |rarr| ``Block`` for kinematics and have no states
- are a subclass of ``TransferBlock`` |rarr| ``Block`` for dynamics and have states

"""
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

    def __init__(self, camera=None, args={}, **blockargs):
        """
        :param camera: Camera model, defaults to None
        :type camera: Camera subclass, optional
        :param blockargs: Additional options for the block, passed as a dictionary
        :type blockargs: dict
        :return: a CAMERA block
        :rtype: Camera instance

        Camera projection model.

        **Block ports**

            :input pose: Camera pose as an SE3 object.
            :input P: world points as ndarray(3,N)

            :output p: image plane points as ndarray(2,N)
        """
        if camera is None:
            raise ValueError("camera is not defined")

        super().__init__(**blockargs)

        self.camera = camera

    def output(self, t, inports, x):
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

    def __init__(self, camera, depth=1, depthest=False, **blockargs):
        """
        :param camera: Camera model, defaults to None
        :type camera: Camera subclass, optional
        :param depth: Point depth
        :type depth: float or ndarray
        :param depthest: use depth estimation, not implemented so it must be
            False (the default).  Give the depth with ``depth``.
        :type depthest: bool, optional
        :param blockargs: Additional options for the block, passed as a dictionary
        :type blockargs: dict
        :return: a VISJAC_P block
        :rtype: Visjac_p instance
        :raises ValueError: if ``camera`` is None
        :raises NotImplementedError: if ``depthest`` is True

        Computes the image Jacobian (interaction matrix) of the image plane
        points on its input, using the same ``depth`` for every point.

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

    def output(self, t, inports, x):
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

    def __init__(self, camera, P, frame="world", method="iterative", **blockargs):
        """
        :param camera: Camera model, defaults to None
        :type camera: Camera subclass, optional
        :param P: World point coordinates
        :type P: ndarray(2,N)
        :param frame: return pose of points with respect to reference frame which is one of: 'world' [default] or 'camera'
        :type frame: str, optional
        :param method: pose estimation algorithm one of: 'iterative' [default], 'epnp', 'p3p', 'ap3p', 'ippe', 'ippe-square'
        :type method: str, optional
        :param blockargs: Additional options for the block, passed as a dictionary
        :type blockargs: dict
        :return: a ESTPOSE_P block
        :rtype: EstPose_p instance
        :raises ValueError: if ``camera`` is None or ``frame`` is not ``"world"`` or ``"camera"``

        .. versionchanged:: 2.5.0
            ``frame`` is now passed to :meth:`~machinevisiontoolbox.Camera.CentralCamera.estpose`.
            Previously it was accepted but ignored, so the pose was always given with
            respect to the world frame.

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

    def output(self, t, inports, x):
        p = inports[0]
        T = self.camera.estpose(self.P, p, method=self.method, frame=self.frame)
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
        camera,
        style=None,
        labels=None,
        grid=True,
        retain=False,
        watch=False,
        init=None,
        **blockargs,
    ):
        """
        Create a block that plots image plane coordinates.

        :param camera: a camera model
        :type camera: Camera instance
        :param style: styles for each point to be plotted
        :type style: str or dict, list of strings or dicts; one per line, optional
        :param grid: draw a grid, defaults to True. Can be boolean or a tuple of
                     options for grid()
        :type grid: bool or sequence
        :param retain: keep previous image plane points, defaults to False
        :type retain: bool, optional
        :param watch: add these signals to the watchlist, defaults to False
        :type watch: bool, optional
        :param init: function to initialize the graphics, defaults to None
        :type init: callable, optional
        :param blockargs: Additional options for the block, passed as a dictionary
        :type blockargs: dict
        :return: An IMAGEPLANE block
        :rtype: ImagePlane instance

        Create a block that plots points on a camera object's virtual image plane.

        Examples::

            SCOPE()
            SCOPE(nin=2)
            SCOPE(nin=2, scale=[-1,2])
            SCOPE(styles='k--')
            SCOPE(styles=[{'color': 'blue'}, {'color': 'red', 'linestyle': '--'}])
            SCOPE(styles=['k', 'r--'])


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
            self.kwargs = {**default_style, **style}
            self.args = []
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

    def start(self, simstate):
        # the base class creates the figure and axes, and registers the input
        # signals for watching if watch=True
        super().start(simstate)

        # init the arrays that hold the data
        self.u_data = []
        self.v_data = []
        self.t_data = []

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

    def record(self, t, inports):
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

    def step(self, t, inports):
        if not self._enabled:
            return

        self.line.set_data(self.u_data, self.v_data)

        super().step(t, inports)
