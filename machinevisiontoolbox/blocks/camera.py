import numpy as np
import matplotlib.pyplot as plt
import time
from spatialmath import base, SE3

from bdsim.components import TransferBlock, FunctionBlock, SourceBlock
from bdsim.graphics import GraphicsBlock

from machinevisiontoolbox import Camera, mkgrid

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

    .. table::
       :align: left

    +------------+---------+---------+
    | inputs     | outputs |  states |
    +------------+---------+---------+
    | 2          | 1       | 0       |
    +------------+---------+---------+
    | SE3        | ndarray |         |
    | ndarray    |         |         |
    +------------+---------+---------+
    """

    nin = 2
    nout = 1
    inlabels = ('P', 'ξ')
    outlabels = ('p',)

    def __init__(self, camera=None, args={}, **blockargs):
        """
        :param camera: Camera model, defaults to None
        :type camera: Camera subclass, optional
        :param blockargs: |BlockOptions|
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
            raise ValueError('camera is not defined')

        super().__init__(**blockargs)
        self.type = "camera"

        self.camera = camera

    def output(self, t=None):
        return [self.camera.project_point(self.inputs[0], pose=self.inputs[1])]


# ------------------------------------------------------------------------ #

class Visjac_p(FunctionBlock):
    """
    :blockname:`VISJAC_P`

    .. table::
       :align: left

    +------------+---------+---------+
    | inputs     | outputs |  states |
    +------------+---------+---------+
    | 1          | 2       | 0       |
    +------------+---------+---------+
    | ndarray    | ndarray |         |
    |            | float   |         |
    +------------+---------+---------+
    """

    nin = 1
    nout = 1
    inlabels = ('p',)
    outlabels = ()

    def __init__(self, camera, depth=1, depthest=False, **blockargs):
        """
        :param camera: Camera model, defaults to None
        :type camera: Camera subclass, optional
        :param depth: Point depth
        :type depth: float or ndarray
        :param depthest: Use depth estimation, defaults to True
        :type depthest: bool, optional
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        :return: a VISJAC_P block
        :rtype: Visjac_p instance

        If the Jacobian 


        """
        if camera is None:
            raise ValueError('camera is not defined')
            
        super().__init__(**blockargs)
        self.type = "visjac_p"

        self.camera = camera
        self.depthest = depthest
        self.depth = depth


    def output(self, t=None):
        # do depth estimation here

        J = self.camera.visjac_p(self.inputs[0], self.depth)
        return [J]


# ------------------------------------------------------------------------ #


class EstPose_p(FunctionBlock):
    """
    :blockname:`ESTPOSE_P`

    .. table::
       :align: left

    +------------+---------+---------+
    | inputs     | outputs |  states |
    +------------+---------+---------+
    | 1          | 1       | 0       |
    +------------+---------+---------+
    | ndarray    | SE3     |         |
    +------------+---------+---------+
    """

    nin = 1
    nout = 1
    inlabels = ('p',)
    outlabels = ('ξ',)

    def __init__(self, camera, P, frame='world', method='iterative', **blockargs):
        """
        :param camera: Camera model, defaults to None
        :type camera: Camera subclass, optional
        :param P: World point coordinates
        :type P: ndarray(2,N)
        :param frame: return pose of points with respect to reference frame which is one of: 'world' [default] or 'camera'
        :type frame: str, optional
        :param method: pose estimation algorithm one of: 'iterative' [default], 'epnp', 'p3p', 'ap3p', 'ippe', 'ippe-square'
        :type method: str, optional
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        :return: a ESTPOSE_P block
        :rtype: EstPose_p instance

        """
        if camera is None:
            raise ValueError('camera is not defined')

        super().__init__(**blockargs)
        self.type = "estpose_p"

        self.camera = camera
        self.P = P
        self.method = method

    def output(self, t=None):
        p = self.inputs[0]
        T = self.camera.estpose(self.P, p, method=self.method)
        return [T]


# ------------------------------------------------------------------------ #


class ImagePlane(GraphicsBlock):
    """
    :blockname:`IMAGEPLANE`
    
    .. table::
       :align: left
    
       +--------------+---------+---------+
       | inputs       | outputs |  states |
       +--------------+---------+---------+
       | 1            | 0       | 0       |
       +--------------+---------+---------+
       | ndarray(2,N) |         |         | 
       +--------------+---------+---------+
    """
    
    nin = 1
    nout = 0

    def __init__(self, camera, style=None, labels=None, grid=True, retain=False, watch=False, init=None, **blockargs):
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
        :param blockargs: |BlockOptions|
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

            
        .. figure:: ../../figs/Figure_1.png
           :width: 500px
           :alt: example of generated graphic

           Example of scope display.
        """
        if camera is None:
            raise ValueError('camera is not defined')

        self.camera = camera

        if style is None:
            style = {}
        if isinstance(style, dict):
            default_style = dict(linestyle='none', marker='o', markersize=4, markeredgecolor='black', markerfacecolor='black')
            self.kwargs = {**default_style, **style}
            self.args = []
        elif isinstance(style, str):
            self.args = [style]
            self.kwargs = {}
        else:
            raise ValueError('bad style, must be str or dict')

        if init is not None:
            assert callable(init), 'graphics init function must be callable'
        self.init = init
        self.retain = retain
        
        super().__init__(nin=1, **blockargs)
        
        self.grid = grid

        self.watch = watch

        # TODO, wire width
        # inherit names from wires, block needs to be able to introspect
        
    def start(self, state=None):        
        # init the arrays that hold the data
        self.u_data = []
        self.v_data = []
        self.t_data = []

        # create the figures
        self.fig = self.create_figure(state)
        self.ax = self.fig.add_subplot(111)
        self.camera._init_imageplane(ax=self.ax)

        self.ax.set_title(self.name_tex)

        print('@@@@@@', self.args, self.kwargs)
        self.line, = self.ax.plot(self.u_data, self.v_data, *self.args, **self.kwargs)

        # grid control
        if self.grid is True:
            self.ax.grid(self.grid)
        elif isinstance(self.grid, (list, tuple)):
            self.ax.grid(True, *self.grid)

        if self.init is not None:
            self.init(self.camera)
         
        if self.watch:
            for wire in self.inports:
                plug = wire.start  # start plug for input wire

                # append to the watchlist, bdsim.run() will do the rest
                state.watchlist.append(plug)
                state.watchnamelist.append(str(plug))

        super().start()
        
    def step(self, state=None):
        # inputs are set
        self.t_data.append(state.t)
        u, v = self.inputs[0]

        if self.retain:
            self.u_data.append(u)
            self.v_data.append(v)
        else:
            self.u_data = u
            self.v_data = v

        self.line.set_data(self.u_data, self.v_data)
        
        if self.bd.options.animation:
            self.fig.canvas.flush_events()

    
        super().step(state=state)
        
