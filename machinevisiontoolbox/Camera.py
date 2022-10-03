#!/usr/bin/env python
"""
Camera class
@author: Dorian Tsai
@author: Peter Corke
"""
from math import cos, pi, sqrt, sin, tan
from abc import ABC, abstractmethod
import copy
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import scipy

import cv2 as cv
from spatialmath import base, Line3, SO3
from machinevisiontoolbox.ImagePointFeatures import FeatureMatch
from machinevisiontoolbox.base import idisp


# from machinevisiontoolbox.classes import Image
# import CameraVisualizer as CamVis

# from mpl_toolkits.mplot3d import Axes3D, art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# from collections import namedtuple
from spatialmath import SE3
import spatialmath.base as smbase

from machinevisiontoolbox import Image

class CameraBase(ABC):

    # list of attributes
    _name = None      # camera  name (string)
    _camtype = None   # camera type (string)

    _imagesize = None        # number of pixels (horizontal, vertical)
    _pp = None        # principal point (horizontal, vertical)
    _rhou = None      # pixel imagesize (single pixel) horizontal
    _rhov = None      # pixel imagesize (single pixel) vertical
    _image = None     # image (TODO image class?), for now, just numpy array

    _T = []         # camera pose (homogeneous transform, SE3 class)

    _ax = []        # for plotting, axes handle

    def __init__(self,
                 name=None,
                 camtype='central',
                 rho=1,
                 imagesize=None,
                 sensorsize=None,
                 pp=None,
                 noise=None,
                 pose=None,
                 limits=None,
                 labels=None,
                 seed=None):
        """Abstract camera base class

        :param name: camera instance name, defaults to None
        :type name: str, optional
        :param camtype: camera projection type, defaults to 'central'
        :type camtype: str, optional
        :param rho: pixel size, defaults to 1
        :type rho: scalar or array_like(2), optional
        :param imagesize: image dimension in pixels, defaults to None
        :type imagesize: int or array_like(2), optional
        :param sensorsize: image sensor size, defaults to None
        :type sensorsize: array_like(2), optional
        :param pp: principal point position, defaults to None
        :type pp: array_like(2), optional
        :param noise: standard deviation of image plane projection noise, defaults to None
        :type noise: float, optional
        :param pose: camera pose, defaults to None
        :type pose: :class:`~spatialmath..pose3d.SE3`, optional
        :param limits: bounds of virtual image plane [umin, umax, vmin, vmax], defaults to None
        :type limits: array_like(4), optional
        :param labels: axis labels for virtual image plane, defaults to ('u', 'v')
        :type labels: 2-tuple of str, optional
        :param seed: random number seed for projection noise, defaults to None
        :type seed: int, optional
        :raises TypeError: name must be a string
        :raises TypeError: camtype must be a string
        :raises ValueError: rho must be a 1- or 2-element vector

        This abstract class is the base for all camera projection model
        classes.  All baseclass constructors support these options.
        """
        if name is None:
            self._name = camtype
        else:
            if not isinstance(name, str):
                raise TypeError(name, 'name must be a string')
            self._name = name

        if not isinstance(camtype, str):
            raise TypeError(camtype, 'camtype must be a string')
        self._camtype = camtype

        if imagesize is None:
            if pp is None:
                self.pp = (0, 0)
            else:
                self.pp = pp
        else:
            self.imagesize = imagesize
            if pp is None:
                self.pp = [x / 2 for x in self.imagesize]
            else:
                self.pp = pp

        if sensorsize is not None:
            self._rhou = sensorsize[0] / self.imagesize[1]
            self._rhov = sensorsize[1] / self.imagesize[0]
        else:
            rho = base.getvector(rho)
            if len(rho) == 1:
                self._rhou = rho[0]
                self._rhov = rho[0]
            elif len(rho) == 2:
                self._rhou = rho[0]
                self._rhov = rho[1]
            else:
                raise ValueError(rho, 'rho must be a 1- or 2-element vector')

        if noise is not None:
            self._noise = noise

        self._random = np.random.default_rng(seed)
            
        if pose is None:
            self._pose = SE3()
        else:
            self._pose = SE3(pose)

        self.pose0 = self.pose

        self._noise = noise

        self._image = None

        self._ax = None

        self._distortion = None
        self.labels = labels
        self.limits = limits

    def reset(self):
        """
        Reset camera pose (base method)

        Restore camera to a copy of the pose given to the constructor.  The copy
        means that the camera pose can be modified freely, without destroying
        the initial pose value.
        """
        self.pose = self.pose0.copy()

    def __str__(self):
        """
        String representation of camera parameters (base method)

        :return: string representation
        :rtype: str

        Multi-line string representation of camera intrinsic and extrinsic
        parameters.
        """
        # TODO, imagesize should be integers
        s = ''
        self.fmt = '{:>15s}: {}\n'
        s += self.fmt.format('Name', self.name + ' [' + self.__class__.__name__ + ']')
        s += self.fmt.format('pixel size', ' x '.join([str(x) for x in self.rho]))
        if self.imagesize is not None:
            s += self.fmt.format('image size', ' x '.join([str(x) for x in self.imagesize]))
        s += self.fmt.format('pose', self.pose.strline(fmt="{:.3g}", orient="rpy/yxz"))
        return s

    def __repr__(self):
        """
        Readable representatio of camera parameters (base method)

        :return: string representation
        :rtype: str

        Multi-line string representation of camera intrinsic and extrinsic
        parameters.
        """
        return str(self)
        
    @abstractmethod
    def project_point(self, P, **kwargs):
        pass

    def project_line(self, *args, **kwargs):
        raise NotImplementedError('not implemented for this camera model')

    def project_conic(self, *args, **kwargs):
        raise NotImplementedError('not implemented for this camera model')

    @property
    def name(self):
        """
        Set/get camera name (base method)

        A camera has a string-valued name that can be read and written.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera();
            >>> camera.name
            >>> camera.name = "foo"
            >>> camera.name
        """
        return self._name

    @name.setter
    def name(self, newname):
        """
        Set camera name

        :param newname: camera name
        :type newname: str
        """
        if isinstance(newname, str):
            self._name = newname
        else:
            raise TypeError(newname, 'name must be a string')

    @property
    def camtype(self):
        """
        Set/get camera type (base method)

        A camera has a string-valued type that can be read and written.  This
        is unique to the camera subclass and projection model.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera();
            >>> camera.camtype
            >>> camera.camtype = "foo"
            >>> camera.camtype
        """
        return self._camtype

    @camtype.setter
    def camtype(self, newcamtype):
        """
        Set camera type

        :param newcamtype: camera projection type
        :type newcamtype: str
        """
        if isinstance(newcamtype, str):
            self._camtype = newcamtype
        else:
            raise TypeError(newcamtype, 'camtype must be a string')

    @property
    def imagesize(self):
        """
        Set/get size of virtual image plane (base method)

        The dimensions of the virtual image plane is a 2-tuple, width and
        height, that can be read or written.  For writing the size must be an
        iterable of length 2.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default();
            >>> camera.imagesize
            >>> camera.imagesize = (500, 500)
            >>> camera.imagesize

        .. note:: If the principal point is not set, then setting imagesize
            sets the principal point to the centre of the image plane.

        :seealso: :meth:`width` :meth:`height` :meth:`nu` :meth:`nv`
        """
        return self._imagesize

    @imagesize.setter
    def imagesize(self, npix):
        """
        Set image plane size

        :param npix: [description]
        :type npix: array_like(2)
        :raises ValueError: bad value

        Sets the size of the virtual image plane.
        
        .. note:: If the principle point is not set, then it
            is set to the centre of the image plane.

        :seealso: :meth:`width` :meth:`height` :meth:`nu` :meth:`nv`
        """
        npix = base.getvector(npix, dtype='int')
        if len(npix) == 1:
            self._imagesize = np.r_[npix[0], npix[0]]
        elif len(npix) in (2, 3):
            # ignore color dimension in case it is given
            self._imagesize = npix[:2]
        else:
            raise ValueError(
                npix, 'imagesize must be a 1- or 2-element vector')
        if self._pp is None:
            self._pp = self._imagesize / 2

    @property
    def nu(self):
        """
        Get image plane width (base method)

        :return: width
        :rtype: int

        Number of pixels in the u-direction (width)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default();
            >>> camera.nu

        :seealso: :meth:`nv` :meth:`width` :meth:`imagesize`
        """
        return self._imagesize[0]

    @property
    def nv(self):
        """
        Get image plane height (base method)

        :return: height
        :rtype: int

        Number of pixels in the v-direction (height)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default();
            >>> camera.nv

        :seealso: :meth:`nu` :meth:`height`  :meth:`imagesize`
        """
        return self._imagesize[1]

    @property
    def width(self):
        """
        Get image plane width (base method)

        :return: width
        :rtype: int

        Image plane height, number of pixels in the v-direction

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default();
            >>> camera.width

        :seealso: :meth:`nu` :meth:`height`
        """
        return self._imagesize[0]

    @property
    def height(self):
        """
        Get image plane height (base method)

        :return: height
        :rtype: int

        Image plane width, number of pixels in the u-direction

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default();
            >>> camera.height

        :seealso: :meth:`nv` :meth:`width`
        """
        return self._imagesize[1]

    @property
    def pp(self):
        """
        Set/get principal point coordinate (base method)

        The principal point is the coordinate of the point where
        the optical axis pierces the image plane.  It is a 2-tuple which can
        be read or written.  For writing the size must be an
        iterable of length 2.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default();
            >>> camera.pp

        :seealso: :meth:`u0` :meth:`v0`
        """
        return self._pp

    @pp.setter
    def pp(self, pp):
        """
        Set principal point coordinate

        :param pp: principal point
        :type pp: array_like(2)

        :seealso: :meth:`pp` :meth:`u0` :meth:`v0`
        """
        pp = base.getvector(pp)
        if len(pp) == 1:
            self._pp = np.r_[pp[0], pp[0]]
        elif len(pp) == 2:
            self._pp = pp
        else:
            raise ValueError(pp, 'pp must be a 1- or 2-element vector')

    @property
    def u0(self):
        """
        Get principal point: horizontal coordinate (base method)

        :return: horizontal component of principal point
        :rtype: float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera();
            >>> camera.u0

        :seealso: :meth:`v0` :meth:`pp`
        """
        return self._pp[0]

    @property
    def v0(self):
        """
        Get principal point: vertical coordinate (base method)

        :return: vertical component of principal point
        :rtype: float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera();
            >>> camera.v0

        :seealso: :meth:`u0` :meth:`pp`
        """
        return self._pp[1]

    @property
    def rhou(self):
        """
        Get pixel width (base method)

        :return: horizontal pixel size
        :rtype: float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default();
            >>> camera.rhou

        :seealso: :meth:`rhov` :meth:`rho`
        """
        return self._rhou

    # this is generally the centre of the image, has special meaning for
    # perspective camera
    
    @property
    def rhov(self):
        """
        Get pixel width (base method)

        :return: vertical pixel size
        :rtype: float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default();
            >>> camera.rhov

        :seealso: :meth:`rhov` :meth:`rho`
        """
        return self._rhov

    @property
    def rho(self):
        """
        Get pixel dimensions (base method)

        :return: horizontal pixel size
        :rtype: ndarray(2)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera();
            >>> camera.rhov

        :seealso: :meth:`rhou` :meth:`rhov`
        """

        return np.array([self._rhou, self._rhov])

    # @property
    # def image(self):
    #     return self._image

    # @image.setter
    # def image(self, newimage):
    #     """


    #     :param newimage: [description]
    #     :type newimage: [type]
    #     """
    #     self._image = Image(newimage)

    @property
    def pose(self):
        """
        Set/get camera pose (base method)

        The camera pose with respect to the global frame can be read or written
        as an :class:`~spatialmath..pose3d.SE3` instance.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> from spatialmath import SE3
            >>> camera = CentralCamera();
            >>> camera.pose
            >>> camera.pose = SE3.Trans(1, 2, 3)
            >>> camera

        .. note:: Changes the pose of the current camera instance, whereas
            :meth:`move` clones the camera instance with a new pose.

        :seealso: :meth:`move`
        """
        return self._pose

    @pose.setter
    def pose(self, newpose):
        """
        Set camera pose

        :param newpose: pose of camera frame
        :type newpose: :class:`~spatialmath..pose3d.SE3` or ndarray(4,4)

        :seealso: :meth:`move`
        """
        self._pose = SE3(newpose)

    @property
    def noise(self):
        """
        Set/Get projection noise (base method)

        :return: standard deviation of noise added to projected image plane points
        :rtype: float

        The noise parameter is set by the object constructor.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default();
            >>> camera.project_point([0, 0, 3])
            >>> camera.noise = 2
            >>> camera.project_point([0, 0, 2])
            >>> camera.project_point([0, 0, 2])

        :seealso: :meth:`project`
        """
        return self._noise

    @noise.setter
    def noise(self, noise):
        self._noise = noise

    def move(self, T, name=None, relative=False):
        """
        Move camera (base method)

        :param T: pose of camera frame
        :type T: :class:`~spatialmath..pose3d.SE3`
        :param relative: move relative to pose of original camera, defaults to False
        :type relative: bool, optional
        :return: new camera object
        :rtype: :class:`CameraBase` subclass

        Returns a copy of the camera object with pose set to ``T``.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> from spatialmath import SE3
            >>> camera = CentralCamera();
            >>> camera.move(SE3.Trans(0.1, 0.2, 0.3))
            >>> camera

        .. note:: The ``plot`` method of this cloned camera will create a new
            window.

        :seealso: :meth:`pose`
        """
        newcamera = copy.copy(self)
        if name is not None:
            newcamera.name = name
        else:
            newcamera.name = self.name + "-moved"
        newcamera._ax = None
        if relative:
            newcamera.pose = self.pose * T
        else:
            newcamera.pose = T
        return newcamera

    # ----------------------- plotting ----------------------------------- #

    def _new_imageplane(self, fig=None, ax=None):
        """
        Create a new virtual image plane if required

        :param fig: Matplotlib figure number, defaults to None
        :type fig: int, optional
        :param ax: Matplotlob axes, defaults to None
        :type ax: :class:`matplotlib.axes`, optional
        :return: existing image plane
        :rtype: bool

        If this camera already has a virtual image plane, return True.
        Otherwise, create an axes, and optionally a  figure, and return False.
        """
        if self._ax is not None:
            return True

        if (fig is None) and (ax is None):
            # create our own handle for the figure/plot
            # print('creating new figure and axes for camera')
            fig, ax = plt.subplots()  # TODO *args, **kwargs?
        # TODO elif ax is a plot handle, else raise ValueError
        self._ax = ax
        self._fig = fig
        return False

    def _init_imageplane(self, fig=None, ax=None):
        """
        Create plot window for camera image plane

        :param fig: figure to plot into, defaults to None
        :type fig: figure handle, optional
        :param ax: axis to plot into, defaults to None
        :type ax: 2D axis handle, optional
        :return: figure and axis
        :rtype: (fig, axis)

        Creates a 2D axis that represents the image plane of the virtual
        camera.

        :seealso: :meth:`plot` :meth:`mesh`
        """

        if self._new_imageplane(fig, ax):
            return self._ax
        ax = self._ax

        if self._image is not None:
            # if camera has an image, display said image
            idisp(self._image,
                      fig=fig,
                      ax=ax,
                      title=self._name,
                      drawonly=True)
        else:
            if self.limits is None:
                ax.set_xlim(0, self.nu)
                ax.set_ylim(0, self.nv)
            else:
                ax.set_xlim(self.limits[0], self.limits[1])
                ax.set_ylim(self.limits[2], self.limits[3])
            ax.autoscale(False)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.grid(True)
            if self.labels is None:
                ax.set_xlabel('u (pixels)')
                ax.set_ylabel('v (pixels)')
            else:
                ax.set_xlabel(self.labels[0])
                ax.set_ylabel(self.labels[1])
            ax.set_title(self.name)
            ax.set_facecolor('lightyellow')
                
            try:
                ax.figure.canvas.manager.set_window_title('Machine Vision Toolbox for Python')
            except AttributeError:
                # can happen during unit test without GUI
                pass

        # TODO figure out axes ticks, etc
        return ax  # likely this return is not necessary

    def clf(self):
        """
        Clear the virtual image plane (base method)

        Every camera object has a virtual image plane drawn using Matplotlib.
        Remove all points and lines from the image plane.
        
        :seealso: :meth:`plot_point` :meth:`plot_line` :meth:`disp`
        """
        if self._ax is not None:
            for artist in self._ax.get_children():
                try:
                    artist.remove()
                except:
                    pass

    def plot_point(self, P, *fmt, return_artist=False, objpose=None, pose=None, ax=None, **kwargs):
        """
        Plot points on virtual image plane (base method)

        :param P: 3D world point or points, or 2D image plane point or points
        :type P: ndarray(3,), ndarray(3,N), or ndarray(2,), ndarray(2,N)
        :param objpose: transformation for the 3D points, defaults to None
        :type objpose: :class:`~spatialmath..pose3d.SE3`, optional
        :param pose: pose of the camera, defaults to None
        :type pose: :class:`~spatialmath..pose3d.SE3`, optional
        :param ax: axes to plot into
        :type ax: :class:`matplotlib.axes`
        :param kwargs: additional arguments passed to :obj:`~matplotlib.pyplot.plot`
        :return: Matplotlib line objects
        :rtype: list of :class:`~matplotlib.lines.Line2d`

        3D world points are first projected to the image plane and then 
        plotted on the camera's virtual image plane.
        Points are organized as columns of the arrays.

        Example::

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default()
            >>> camera.plot_point([0.2, 0.3, 2])
            >>> camera.plot_point([0.2, 0.3, 2], 'r*')
            >>> camera.plot_point([0.2, 0.3, 2], pose=SE3(0.1, 0, 0))

        .. plot::

            from machinevisiontoolbox import CentralCamera
            camera = CentralCamera.Default()
            camera.plot_point([0.2, 0.3, 2])
            camera.plot_point([0.2, 0.3, 2], 'r*')
            camera.plot_point([0.2, 0.3, 2], pose=SE3(0.1, 0, 0))

        .. note::
            - Successive calls add items to the virtual image plane.
            - This method is common to all ``CameraBase`` subclasses, but it
              invokes a camera-specific projection method.

        :seealso: :meth:`plot_line2` :meth:`plot_line3` :meth:`plot_wireframe` :meth:`clf`
        """
        self._init_imageplane(ax)

        if not isinstance(P, np.ndarray):
            P = base.getvector(P)

        if P.shape[0] == 3:
            # plot world points
            p = self.project_point(P, pose=pose, objpose=objpose)
        else:
            # plot image plane points
            p = P

        if p.shape[0] != 2:
            raise ValueError('p must have be (2,), (3,), (2,n), (3,n)')

        defaults = dict(markersize=6, color='k')
        if len(fmt) == 0:
            fmt = ['o']
            kwargs = {**defaults, **kwargs}

        artist = self._ax.plot(p[0, :], p[1, :], *fmt, **kwargs)
        plt.show(block=False)

        if return_artist:
            return p, artist[0]
        else:
            return p

    def plot_line2(self, l, *args, **kwargs):
        r"""
        Plot 2D line on virtual image plane (base method)

        :param l: homogeneous line
        :type l: array_like(3)
        :param kwargs: arguments passed to ``plot``

        Plot the homogeneous line on the camera's virtual image plane. The line
        is expressed in the form
        
        .. math:: \ell_0 u + \ell_1 v + \ell_2 = 0

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default()
            >>> camera.plot_line2([1, 0.2, -500])

        .. note::
            - Successive calls add items to the virtual image plane.
            - This method is common to all ``CameraBase`` subclasses, but it
              invokes a camera-specific projection method.

        :seealso: :meth:`plot_point` :meth:`plot_line3` :meth:`clf`
        """
        # get handle for this camera image plane
        self._init_imageplane()
        plt.autoscale(False)

        base.plot_homline(l, *args, ax=self._ax, **kwargs)


    # def plot_line3(self, L, nsteps=21, **kwargs):
    #     """
    #     Plot 3D line on virtual image plane (base method)

    #     :param L: 3D line or lines in Plucker coordinates
    #     :type L: :class:`~spatialmath..geom3d.Line3`
    #     :param kwargs: arguments passed to ``plot``

    #     The Plucker lines are projected to the camera's virtual image plane and
    #     plotted.  Each line is approximated by ``nsteps`` points, each of which
    #     is projected, allowing straight lines to appear curved after projection.

    #     Example:

    #     .. runblock:: pycon

    #         >>> from machinevisiontoolbox import CentralCamera, mkcube
    #         >>> from spatialmath import Line3
    #         >>> camera = CentralCamera()
    #         >>> line = Line3.Join((-1, -2, -3), (4, 5, 6))
    #         >>> camera.plot_line3(line, 'k--')

    #     .. note::
    #         - Successive calls add items to the virtual image plane.
    #         - This method is common to all ``CameraBase`` subclasses, but it
    #           invokes a camera-specific projection method.
              
    #     :seealso: :meth:`plot_point` :meth:`plot_line2` :meth:`plot_wireframe` :meth:`clf`
    #     """
    #     # draw 3D line segments
    #     s = np.linspace(0, 1, nsteps)

        # this is actually pretty tricky
        #  - how to determine which part of the 3D line is visible
        #  - if highly curved it may be in two or more segments
        # for line in L:
        #     l = self.project_line(line)

        #     # straight world lines are not straight, plot them piecewise
        #     P = (1 - s) * P0[:, np.newaxis] + s * P2[:, np.newaxis]
        #     uv = self.project_point(P, pose=pose)

    def plot_wireframe(self, X, Y, Z, *fmt, objpose=None, pose=None, nsteps=21, **kwargs):
        """
        Plot 3D wireframe in virtual image plane (base method)

        :param X: world X coordinates
        :type X: ndarray(N,M)
        :param Y: world Y coordinates
        :type Y: ndarray(N,M)
        :param Z: world Z coordinates
        :type Z: ndarray(N,M)
        :param objpose: transformation for the wireframe points, defaults to None
        :type objpose: :class:`~spatialmath..pose3d.SE3`, optional
        :param pose: pose of the camera, defaults to None
        :type pose: :class:`~spatialmath..pose3d.SE3`, optional
        :param nsteps: number of points for each wireframe segment, defaults to 21
        :type nsteps: int, optional
        :param kwargs: arguments passed to ``plot``

        The 3D wireframe is projected to the camera's virtual image plane.  Each
        wire link in the wireframe is approximated by ``nsteps`` points, each of
        which is projected, allowing straight edges to appear curved.

        Example::

            >>> from machinevisiontoolbox import CentralCamera, mkcube
            >>> from spatialmath import SE3
            >>> camera = CentralCamera.Default()
            >>> X, Y, Z = mkcube(0.2, pose=SE3(0, 0, 1), edge=True)
            >>> camera.plot_wireframe(X, Y, Z, 'k--')

        .. plot::

            from machinevisiontoolbox import CentralCamera, mkcube
            from spatialmath import SE3
            camera = CentralCamera.Default()
            X, Y, Z = mkcube(0.2, pose=SE3(0, 0, 1), edge=True)
            camera.plot_wireframe(X, Y, Z, 'k--')

        :seealso: :func:`mkcube` :obj:`spatialmath.base.cylinder` :obj:`spatialmath.base.sphere` :obj:`spatialmath.base.cuboid`
        """

        # self._ax.plot_surface(X, Y, Z)
        # plt.show()
   
        # check that mesh matrices conform
        if X.shape != Y.shape or X.shape != Z.shape:
            raise ValueError('matrices must be the same shape')
        
        if pose is None:
            pose = self.pose
        if objpose is not None:
            pose = objpose.inv() * pose
        
        # get handle for this camera image plane
        self._init_imageplane()
        plt.autoscale(False)
        
        # draw 3D line segments
        s = np.linspace(0, 1, nsteps)

        # c.clf
        # holdon = c.hold(1);
        
        for i in range(X.shape[0]-1):      # i=1:numrows(X)-1
            for j in range(X.shape[1]-1):  # j=1:numcols(X)-1
                P0 = np.r_[X[i, j], Y[i, j], Z[i, j]]
                P1 = np.r_[X[i+1, j], Y[i+1, j], Z[i+1, j]]
                P2 = np.r_[X[i, j+1], Y[i, j+1], Z[i, j+1]]
                
                if self.camtype == 'perspective':
                    # straight world lines are straight on the image plane
                    uv = self.project_point(np.c_[P0, P1], pose=pose)
                else:
                    # straight world lines are not straight, plot them piecewise
                    P = (1 - s) * P0[:, np.newaxis] + s * P1[:, np.newaxis]
                    uv = self.project_point(P, pose=pose)

                self._ax.plot(uv[0, :], uv[1, :], *fmt, **kwargs)
                
                if self.camtype == 'perspective':
                    # straight world lines are straight on the image plane
                    uv = self.project_point(np.c_[P0, P2], pose=pose)
                else:
                    # straight world lines are not straight, plot them piecewise
                    P = (1 - s) * P0[:, np.newaxis] + s * P2[:, np.newaxis]
                    uv = self.project_point(P, pose=pose)

                self._ax.plot(uv[0, :], uv[1, :], *fmt, **kwargs)

        
        for j in range(X.shape[1]-1):  # j=1:numcols(X)-1
            P0 = [X[-1,j],   Y[-1,j],   Z[-1,j]]
            P1 = [X[-1,j+1], Y[-1,j+1], Z[-1,j+1]]
            
            # if c.perspective
                # straight world lines are straight on the image plane
            uv = self.project_point(np.c_[P0, P1], pose=pose);
            # else
            #     # straight world lines are not straight, plot them piecewise
            #     P = bsxfun(@times, (1-s), P0) + bsxfun(@times, s, P1);
            #     uv = c.project(P, 'setopt', opt);
            self._ax.plot(uv[0,:], uv[1,:], *fmt, **kwargs)
        
        # c.hold(holdon); # turn hold off if it was initially off

        plt.draw()

    def disp(self, im, **kwargs):
        """
        Display image on virtual image plane (base method)

        :param im: image to display
        :type im: :class:`Image` instance
        :param kwargs: options to :func:`~machinevisiontoolbox.base.imageio.idisp()`

        An image is displayed on camera's the virtual image plane.  
        
        .. note: The dimensions of the image plane should match the dimensions of the image.

        :seealso: :func:`machinevisiontoolbox.base.idisp()`
        """
        self.imagesize = (im.shape[1], im.shape[0])
        self._init_imageplane()
        im.disp(ax=self._ax, title=False, **kwargs)

        plt.autoscale(False)

    def plot(self=None, pose=None, scale=1, shape='camera', label=True,
                    alpha=1, solid=False, color='r', projection='ortho', frame=False,
                    ax=None):
        """
        Plot 3D camera icon in world view (base method)

        :param pose: camera pose
        :type pose: :class:`~spatialmath..pose3d.SE3`
        :param scale: scale factor, defaults to 1
        :type scale: float
        :param shape: icon shape: 'frustum' [default], 'camera'
        :type shape: str, optional
        :param label: show camera name, defaults to True
        :type label: bool, optional
        :param alpha: transparency of icon, defaults to 1
        :type alpha: float, optional
        :param solid: icon comprises solid faces, defaults to False
        :type solid: bool, optional
        :param color: icon color, defaults to 'r'
        :type color: str, optional
        :param projection: projection model for new axes, defaults to 'ortho'
        :type projection: str, optional
        :param ax: axes to draw in, defaults to current 3D axes
        :type ax: :class:`~matplotlib.Axes3D`, optional
        :return: axes drawn into
        :rtype: :class:`~matplotlib.Axes3D`

        Plot a 3D icon representing the pose of a camera into a 3D Matplotlib
        plot.  Two icons are supported: the traditional frustum, and a
        simplistic camera comprising a box and cylinder.

        .. note:: If ``pose`` is not given it defaults to the pose of the
            instance.
        """

        # if (fig is None) and (ax is None):
        #     # create our own handle for the figure/plot
        #     print('creating new figure and axes for camera')
        #     fig = plt.figure()
        #     ax = fig.gca(projection='3d')
        #     # ax.set_aspect('equal')

        """[summary]
        face order -x, +y, +x, -y
        """
        # get axes to draw in
        ax = smbase.axes_logic(ax, 3, projection=projection)

        if pose is None:
            pose = self.pose

        # draw camera-like object:
        if shape == 'frustum':
            # TODO make this kwargs or optional inputs
            # side colors:
            #  +x red
            #  -y red
            #  +y green
            #  -y yellow
            length = scale
            widthb = scale/10
            widtht = scale
            widthb /= 2
            widtht /= 2
            b0 = np.array([-widthb, -widthb, 0, 1])
            b1 = np.array([-widthb, widthb, 0, 1])
            b2 = np.array([widthb, widthb, 0, 1])
            b3 = np.array([widthb, -widthb, 0, 1])
            t0 = np.array([-widtht, -widtht, length, 1])
            t1 = np.array([-widtht, widtht, length, 1])
            t2 = np.array([widtht, widtht, length, 1])
            t3 = np.array([widtht, -widtht, length, 1])

            # bottom/narrow end
            T = pose.A
            b0 = (T @ b0)[:-1]
            b1 = (T @ b1)[:-1]
            b2 = (T @ b2)[:-1]
            b3 = (T @ b3)[:-1]

            # wide/top end
            t0 = (T @ t0)[:-1]
            t1 = (T @ t1)[:-1]
            t2 = (T @ t2)[:-1]
            t3 = (T @ t3)[:-1]

            # Each set of four points is a single side of the Frustrum
            # points = np.array([[b0, b1, t1, t0], [b1, b2, t2, t1], [
            #                   b2, b3, t3, t2], [b3, b0, t0, t3]])
            points = [
                np.array([b0, b1, t1, t0]),  # -x face
                np.array([b1, b2, t2, t1]),  # +y face
                np.array([b2, b3, t3, t2]),  # +x face
                np.array([b3, b0, t0, t3])   # -y face
            ]
            poly = Poly3DCollection(points,
                                    facecolors=['r', 'g', 'r', 'y'],
                                    alpha=alpha)
            ax.add_collection3d(poly)

        elif shape == 'camera':

            # the box is centred at the origin and its centerline parallel to the
            # z-axis.  Its z-extent is -bh/2 to bh/2.
            W = 0.5       # width & height of the box
            L = 1.2       # length of the box
            cr = 0.2       # cylinder radius
            ch = 0.4       # cylinder height
            cn = 12        # number of facets of cylinder
            a = 3          # length of axis line segments

            # draw the box part of the camera
            smbase.plot_cuboid(sides=np.r_[W, W, L] * scale, pose=pose, filled=solid, color=color, alpha=0.5 * alpha if solid else alpha, ax=ax)

            # draw the lens
            smbase.plot_cylinder(radius=cr * scale, height=np.r_[L / 2, L / 2 + ch] * scale, resolution=cn, pose=pose, filled=solid, color=color, alpha=0.5 * alpha, ax=ax)

            if label:
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

        if frame is True:
            self.pose.plot(length=scale*1.5, style='line', color=color, flo=(0.07, 0, -0.01))
        elif frame is not False:
            self.pose.plot(**frame)

        return ax

    def _add_noise_distortion(self, uv):
        """
        Add noise to pixel coordinates

        :param uv: image plane point coordinates
        :type uv: ndarray(2,N)
        :return: point coordinates with additive noise
        :rtype: ndarray(2,N)

        Model noise in the image process by adding zero-mean Gaussian noise
        to the coordinates of projected world points.  The noise has a
        standard deviation specified by the camera constructor.

        :seealso: :meth:`noise`
        """
        # distort the pixels
        
        # add Gaussian noise with specified standard deviation
        if self.noise is not None:
            uv += self._random.normal(0.0, self.noise, size=uv.shape)
        return uv 

class CentralCamera(CameraBase):
    """
    .. inheritance-diagram:: machinevisiontoolbox.Camera.CentralCamera
        :top-classes: machinevisiontoolbox.Camera.Camera
        :parts: 1
    """

    def __init__(self,
                 f=1,
                 distortion=None,
                 **kwargs):
        """
        Create central camera projection model

        :param f: focal length, defaults to 8mm
        :type f: float, optional
        :param distortion: camera distortion parameters, defaults to None
        :type distortion: array_like(5), optional
        :param kwargs: arguments passed to :class:`CameraBase` constructor

        A camera object contains methods for projecting 3D points and lines
        to the image plane, as well as supporting a virtual image plane onto
        which 3D points and lines can be drawn.

        :references: 
            - Robotics, Vision & Control for Python, Section 13.1, P. Corke, Springer 2023.

        :seealso: :class:`CameraBase` :class:`FishEyeCamera` :class:`SphericalCamera`
        """

        super().__init__(camtype='perspective', **kwargs)
        # TODO some of this logic to f and pp setters
        self.f = f

        self._distortion = distortion

    @classmethod
    def Default(cls, **kwargs):
        r"""
        Set default central camera parameters

        :return: central camera model
        :rtype: :class:`CentralCamera` instance

        Initialize a central camera with: focal length of 8mm, :math:`10\mu`m pixels,
        image size of :math:`1000 \times 1000` with principal point at (500, 500).

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default(name='camera1')
            >>> camera

        :references: 
            - Robotics, Vision & Control for Python, Section 13.1, P. Corke, Springer 2023.

        :seealso: :class:`CentralCamera`
        """
        default = {
            'f': 0.008, 
            'rho': 10e-6,
            'imagesize': 1000, 
            'pp': (500,500),
            'name': 'default perspective camera'
        }

        return CentralCamera(**{**default, **kwargs})
        
    def __str__(self):
        """
        String representation of central camera parameters

        :return: string representation
        :rtype: str

        Multi-line string representation of camera intrinsic and extrinsic
        parameters.
        """
        s = super().__str__()
        s += self.fmt.format('principal pt', self.pp)
        s += self.fmt.format('focal length', self.f)
        return s


    def project_point(self, P, pose=None, objpose=None, behind=True, visibility=False, retinal=False, **kwargs):
        r"""
        Project 3D points to image plane

        :param P: 3D world point or points in Euclidean or homogeneous form
        :type P: array_like(3), array_like(3,n), array_like(4), array_like(4,n)
        :param pose: camera pose with respect to the world frame, defaults to
            camera's ``pose`` attribute
        :type pose: :class:`~spatialmath..pose3d.SE3`, optional
        :param objpose:  3D point reference frame, defaults to world frame
        :type objpose: :class:`~spatialmath..pose3d.SE3`, optional
        :param behind: points behind the camera indicated by NaN, defaults to True
        :type behind: bool, optional
        :param visibility: return visibility array, defaults to False
        :type visibility: bool
        :param retinal: transform to retinal coordinates, defaults to False
        :type retinal: bool, optional
        :return: image plane points, optional visibility vector
        :rtype: ndarray(2,n), ndarray(n)

        Project a 3D point to the image plane

        .. math::

            \hvec{p} = \mat{C} \hvec{P}

        where :math:`\mat{C}` is the camera calibration matrix and
        :math:`\hvec{p}` and :math:`\hvec{P}` are the image plane and world
        frame coordinates respectively, in homogeneous form. 
        
        World points are given as a 1D array or the columns of a 2D array of
        Euclidean or homogeneous coordinates. The computed image plane
        coordinates are Euclidean or homogeneous and given as a 1D array or the
        corresponding columns of a 2D array.

        If ``pose`` is specified it is used for the camera frame pose, otherwise
        the attribute ``pose`` is used.  The object's ``pose`` attribute is not
        updated if ``pose`` is specified.

        A single point can be specified as a 3-vector, multiple points as an
        array with three rows and each column is the 3D point coordinate (X, Y,
        Z).

        The points ``P`` are by default with respect to the world frame, but 
        they can be transformed by specifying ``objpose``.
        
        If world points are behind the camera and ``behind`` is True then the
        image plane coordinates are set to NaN.
        
        if ``visibility`` is True then each projected point is checked to ensure
        it lies in the bounds of the image plane.  In this case there are two
        return values: the image plane coordinates and an array of booleans
        indicating if the corresponding point is visible.

        If ``retinal`` is True then project points in retinal coordinates, 
        in units of metres with respect to the principal point.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default()
            >>> camera.project_point((0.3, 0.4, 2))

        :references: 
            - Robotics, Vision & Control for Python, Section 13.1, P. Corke, Springer 2023.

        :seealso: :meth:`C` :meth:`project_point` :meth:`project_line` :meth:`project_quadric`
        """
        if pose is None:
            pose = self.pose

        C = self.C(pose, retinal=retinal)

        if isinstance(P, np.ndarray):
            if P.ndim == 1:
                P = P.reshape((-1, 1))  # make it a column
        else:
            P = base.getvector(P, out='col')

        if P.shape[0] == 3:
            P = base.e2h(P)  # make it homogeneous
            euclidean = True
        else:
            euclidean = False

        # project 3D points

        if objpose is not None:
            P = objpose.A @ P

        x = C @ P

        if behind:
            x[2, x[2, :] < 0] = np.nan  # points behind the camera are set to NaN

        if euclidean:
            # Euclidean points given, return Euclidean points
            x = base.h2e(x)

            # add Gaussian noise and distortion
            if self._distortion:
                x = self._distort(x)
            x = self._add_noise_distortion(x)

            #  do visibility check if required
            if visibility:
                visible = ~np.isnan(x[0,:]) \
                    & (x[0, :] >= 0) \
                    & (x[1, :] >= 0) \
                    & (x[0, :] < self.nu) \
                    & (x[1, :] < self.nv)
                
                return x, visible
            else:
                return x
        else:
            # homogeneous points given, return homogeneous points
            return x

    def project_line(self, lines):
        r"""
        Project 3D lines to image plane

        :param lines: Plucker line or lines
        :type lines: :class:`~spatialmath..geom3d.Line3` instance with N values
        :return: 2D homogeneous lines, one per column
        :rtype: ndarray(3,N)

        The :class:`~spatialmath..geom3d.Line3` object can contain multiple lines.  The result array has one
        column per line, and each column is a vector describing the image plane
        line in homogeneous form :math:`\ell_0 u + \ell_1 v + \ell_2 = 0`.

        The projection is

        .. math::

            \ell = \vex{\mat{C} \sk{\vec{L}} \mat{C}^{\top}}

        where :math:`\mat{C}` is the camera calibration matrix and :math:`\sk{\vec{L}}`
        is the skew matrix representation of the Plucker line.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> from spatialmath import Line3
            >>> line = Line3.Join((-3, -4, 5), (5, 2, 6))
            >>> line
            >>> camera = CentralCamera()
            >>> camera.project_line(line)

        :references: 
            - Robotics, Vision & Control for Python, Section 13.7.1, P. Corke, Springer 2023.

        :seealso: :meth:`C` :class:`~spatialmath..geom3d.Line3` :meth:`project_point` :meth:`project_quadric`
        """
        if not isinstance(lines, Line3):
            raise ValueError('expecting Line3 lines')
        # project Plucker lines

        lines2d = []
        C = self.C()
        for line in lines:
            l = base.vex(C  @ line.skew() @ C.T)
            x = l / np.max(np.abs(l))  # normalize by largest element
            lines2d.append(x)
        return np.column_stack(lines2d)

    def project_quadric(self, Q):
        r"""
        Project 3D quadric to image plane

        :param Q: quadric matrix
        :type Q: ndarray(4,4)
        :return: image plane conic
        :rtype: ndarray(3,3)

        Quadrics, short for quadratic surfaces, are a rich family of
        3-dimensional surfaces. There are 17 standard types including spheres,
        ellipsoids, hyper- boloids, paraboloids, cylinders and cones all
        described by points :math:`\vec{x} \in \mathbb{P}^3` such that
        
        .. math::
        
            \hvec{x}^{\top} \mat{Q} \hvec{x} = 0

        The outline of the quadric is projected to a conic section on the image
        plane

        .. math::

            c^* = \mat{C} \mat{Q}^* \mat{C}^{\top}

        where :math:`(\mat{X})^* = det(\mat{X}) \mat{X}^{-1}` is the adjugate
        operator.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> from spatialmath import SE3
            >>> T_cam = SE3.Trans(0.2, 0.1, -5) * SE3.Rx(0.2)
            >>> Q = np.diag([1, 1, 1, -1])  # unit sphere at origin
            >>> camera = CentralCamera.Default(f=0.015, pose=T_cam);
            >>> camera.project_quadric(Q)

        :references:
            - Robotics, Vision & Control for Python, Section 13.7.1, P. Corke, Springer 2023.

        :seealso: :meth:`C` :meth:`project_point` :meth:`project_line`
        """
        if not smbase.ismatrix(Q, (4,4)):
            raise ValueError('expecting 4x4 conic matrix')

        return self.C() @ Q @ self.C().T

    def epiline(self, p, camera2):
        r"""
        Compute epipolar line

        :param p: image plane point or points
        :type p: array_like(2) or ndarray(2,N)
        :param camera2: second camera
        :type camera2: :class:`CentralCamera` instance
        :return: epipolar line or lines in homogeneous form
        :rtype: ndarray(3), ndarray(3,N)
    
        Compute the epipolar line in ``camera2`` induced by the image plane
        points ``p`` in the current camera.  Each line is given by

        .. math::

            \ell = \mat{F} {}^1 \hvec{p}

        which is in homogeneous form :math:`\ell_0 u + \ell_1 v + \ell_2 = 0`
        and the conjugate point :math:`{}^2 \vec{p}` lies on this line.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> from spatialmath import SE3
            >>> camera1 = CentralCamera.Default(name='camera1')
            >>> camera2 = CentralCamera.Default(pose=SE3(0.1, 0.05, 0), name='camera2')
            >>> P = [-0.2, 0.3, 5]  # world point
            >>> p1 = camera1.project_point(P)  # project to first camera
            >>> camera1.epiline(p1, camera2)   # epipolar line in second camera

        :references:
            - Robotics, Vision & Control for Python, Section 14.2.1, P. Corke, Springer 2023.

        :seealso: :meth:`plot_epiline` :meth:`CentralCamera.F`
        """
        # p is 3 x N, result is 3 x N
        return self.F(camera2) @ base.e2h(p)

    def plot_epiline(self, F, p, *fmt, **kwargs):
        r"""
        Plot epipolar line

        :param F: fundamental matrix
        :type F: ndarray(3,3)
        :param p: image plane point or points
        :type p: array_like(2) or ndarray(2,N)
        :param fmt: line style argument passed to ``plot``
        :param kwargs: additional line style arguments passed to ``plot``
    
        Plot the epipolar line induced by the image plane points ``p`` in the
        camera's virtual image plane.  Each line is given by

        .. math::

            \ell = \mat{F} {}^1 \hvec{p}

        which is in homogeneous form :math:`\ell_0 u + \ell_1 v + \ell_2 = 0`
        and the conjugate point :math:`{}^2 \vec{p}` lies on this line.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> from spatialmath import SE3
            >>> camera1 = CentralCamera.Default(name='camera1')
            >>> camera2 = CentralCamera.Default(pose=SE3(0.1, 0.05, 0), name='camera2')
            >>> P = [-0.2, 0.3, 5]  # world point
            >>> p1 = camera1.project_point(P)  # project to first camera
            >>> camera2.plot_point(P, 'kd') # project and display in second camera
            >>> camera2.plot_epiline(camera1.F(camera2), p1) # plot epipolar line in second camera

        :references:
            - Robotics, Vision & Control for Python, Section 14.2.1, P. Corke, Springer 2023.

        :seealso: :meth:`plot_point` :meth:`epiline` :meth:`CentralCamera.F`
        """
        # p is 3 x N, result is 3 x N
        self.plot_line2(F @ base.e2h(p), *fmt, **kwargs)

    def plot_line3(self, L, **kwargs):
        """
        Plot 3D line on virtual image plane (base method)

        :param L: 3D line in Plucker coordinates
        :type L: :class:`~spatialmath..geom3d.Line3`
        :param kwargs: arguments passed to ``plot``

        The Plucker line is projected to the camera's virtual image plane and
        plotted.

        .. note::
            - Successive calls add items to the virtual image plane.
            - This method is common to all ``CameraBase`` subclasses, but it
              invokes a camera-specific projection method.
              
        :seealso: :meth:`plot_point` :meth:`plot_line2` :meth:`plot_wireframe` :meth:`clf`
        """

        l = self.project_line(L)
        for hl in l.T:
            self.plot_line2(hl, **kwargs)

    def ray(self, points, pose=None):
        """
        Project image plane points to a ray

        :param points: image plane points
        :type points: ndarray(2,N)
        :param pose: camera pose, defaults to None
        :type pose: :class:`~spatialmath..pose3d.SE3`, optional
        :return: corresponding Plucker lines
        :rtype: :class:`~spatialmath..geom3d.Line3`

        For each image plane point compute the equation of a Plucker line
        that represents the 3D ray from the camera origin through the image
        plane point.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default()
            >>> line = camera.ray((100, 200))
            >>> line

        :reference:
            - "Multiview Geometry", Hartley & Zisserman, p.162
            - Robotics, Vision & Control for Python, Section 14.3, P. Corke, Springer 2023.

        :seealso: :class:`~spatialmath..geom3d.Line3`
        """
        # define Plucker line in terms of point (centre of camera) and direction
        C = self.C(pose=pose)
        Mi = np.linalg.inv(C[:3, :3])
        v = C[:, 3]
        lines = []
        for point in base.getmatrix(points, (2, None)).T:
            lines.append(Line3.PointDir(-Mi @ v, Mi @ smbase.e2h(point)))
        return Line3(lines)

    @property
    def centre(self):
        """
        Position of camera frame

        :return: Euclidean coordinate of the camera frame's origin
        :rtype: ndarray(3)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> from spatialmath import SE3
            >>> camera1 = CentralCamera.Default(name='camera1', pose=SE3.Trans(1,2,3))
            >>> camera1
            >>> camera1.centre
        """
        return self.pose.t

    @property
    def center(self):
        """
        Position of camera frame

        :return: Euclidean coordinate of the camera frame's origin
        :rtype: ndarray(3)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> from spatialmath import SE3
            >>> camera1 = CentralCamera.Default(name='camera1', pose=SE3.Trans(1,2,3))
            >>> camera1
            >>> camera1.center
        """
        return self.pose.t

    def fov(self):
        """
        Camera field-of-view angles

        :return: field of view angles in radians
        :rtype: ndarray(2)
        
        Computes the field of view angles (2x1) in radians for the camera
        horizontal and vertical directions.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera1 = CentralCamera.Default(name='camera1')
            >>> camera1.fov()

        :references:
            - Robotics, Vision & Control for Python, Section 13.1.4, P. Corke, 
              Springer 2023.
        """
        try:
            return 2 * np.arctan(np.r_[self.imagesize] / 2 * np.r_[self.rho] / self.f)
        except:
            raise ValueError('imagesize or rho properties not set')

    def distort(self, points):
        """
        Compute distorted coordinate

        :param points: image plane points
        :type points: ndarray(2,n)
        :returns: distorted image plane coordinates
        :rtype: ndarray(2,n)
        
        Compute the image plane coordinates after lens distortion has been
        applied.  The lens distortion model is initialized at constructor time.
        """
        if self._distortion is None:
            return points

        # convert to normalized image coordinates
        X = np.linalg.inv(self.K) * smbase.e2h(points)

        # unpack coordinates
        u = X[0, :]
        v = X[1, :]

        # unpack distortion vector
        k = self._distortion[:3]
        p = self._distortion[3:]

        r = np.sqrt(u ** 2 + v ** 2) # distance from principal point
        
        # compute the shift due to distortion
        delta_u = u * (k[0] * r ** 2 + k[1] * r ** 4 + k[2] * r ** 6) + \
            2 * p[0] * u * v + p[1] * (r ** 2 + 2 * u ** 2)
        delta_v = v  * (k[0] * r ** 2 + k[1] * r ** 4 + k[2] * r ** 6) + \
            p[0] * (r ** 2 + 2 * v ** 2) + 2  *p[1] * u * v
        
        # distorted coordinates
        ud = u + delta_u
        vd = v + delta_v
        
        return self.K * smbase.e2h( np.r_[ud, vd] ) # convert to pixel coords

    @property
    def fu(self):
        """
        Get focal length in horizontal direction

        :return: focal length in horizontal direction
        :rtype: float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default(name='camera1')
            >>> camera.fu

        :seealso: :meth:`fv` :meth:`f`
        """
        return self._fu

    @property
    def fv(self):
        """
        Get focal length in vertical direction

        :return: focal length in vertical direction
        :rtype: float

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default(name='camera1')
            >>> camera.fv

        :seealso: :meth:`fu` :meth:`f`
        """
        return self._fv

    @property
    def f(self):
        """
        Set/get focal length

        :return: focal length in horizontal and vertical directions
        :rtype: ndarray(2)

        Return focal length in horizontal and vertical directions.  

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default(name='camera1')
            >>> camera.f
            >>> camera.f = 0.015
            >>> camera.f
            >>> camera.f = [0.015, 0.020]
            >>> camera.f

        .. note:: These are normally identical but will differ if the sensor
            has non-square pixels or the frame grabber is changing the aspect 
            ratio of the image.

        :seealso: :meth:`fu` :meth:`fv`
        """
        return np.r_[self._fu, self._fv]

    @f.setter
    def f(self, f):
        """[summary]

        :param f: focal length
        :type f: scalar or array_like(2)
        :raises ValueError: incorrect length of ``f``
        """
        f = base.getvector(f)

        if len(f) == 1:
            self._fu = f[0]
            self._fv = f[0]
        elif len(f) == 2:
            self._fu = f[0]
            self._fv = f[1]
        else:
            raise ValueError(f, 'f must be a 1- or 2-element vector')

    @property
    def fpix(self):
        """
        Get focal length in pixels

        :return: focal length in horizontal and vertical directions in pixels
        :rtype: ndarray(2)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default(name='camera1')
            >>> camera.fpix

        :seealso: :meth:`f`
        """
        return np.r_[self._fu / self._rhou, self._fv / self._rhov]

    @property
    def K(self):
        """
        Intrinsic matrix of camera

        :return: intrinsic matrix
        :rtype: ndarray(3,3)

        Return the camera intrinsic matrix.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default(name='camera1')
            >>> camera.K

        :references: 
            - Robotics, Vision & Control for Python, Section 13.1, P. Corke, Springer 2023.

        :seealso: :meth:`C` :meth:`pp` :meth:`rho`
        """
        # fmt: off
        K = np.array([[self.fu / self.rhou, 0,                   self.u0],
                      [ 0,                  self.fv / self.rhov, self.v0],
                      [ 0,                  0,                    1]
                      ], dtype=np.float64)
        # fmt: on
        return K

    # =================== camera calibration =============================== #
    def C(self, pose=None, retinal=False):
        """
        Camera projection matrix

        :param T: camera pose with respect to world frame, defaults to pose from camera object
        :type T: :class:`~spatialmath..pose3d.SE3`, optional
        :param retinal: transform to retinal coordinates, default False
        :type retinal: bool, optional
        :return: camera projection/calibration matrix
        :rtype: ndarray(3,4)

        Return the camera matrix which projects 3D points to the image plane.
        It is a function of the camera's intrinsic and extrinsic parameters.

        If ``retinal`` is True then project points in retinal coordinates, 
        in units of metres with respect to the principal point.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default(name='camera1')
            >>> camera.C()
            >>> camera.C(SE3.Trans(0.1, 0, 0))
            >>> camera.move(SE3(0.1, 0, 0)).C()

        :references: 
            - Robotics, Vision & Control for Python, Section 13.1, P. Corke, Springer 2023.

        :seealso: :meth:`project_point` :meth:`K` :meth:`decomposeC`
        """
        P0 = np.eye(3, 4)
        if pose is None:
            pose = self.pose
        if retinal:
            K = np.diag([self.fu, self.fv, 1])
        else:
            K = self.K
        return K @ P0 @ pose.inv().A

    @staticmethod
    def points2C(P, p):
        r"""
        Estimate camera matrix from data points

        :param P: calibration points in world coordinate frame
        :type P: ndarray(3,N)
        :param p: calibration points in image plane
        :type p: ndarray(2,N)
        :return: camera calibration matrix and residual
        :rtype: ndarray(3,4), float

        Estimate the camera matrix :math:`\mat{C}` determined by least
        squares from corresponding world ``P`` and image-plane ``p`` points.
        Corresponding points are represented by corresponding columns of ``P``
        and ``p``.  Also returns the residual which is:

        .. math::

            \max | \mat{C}\mat{P} - \mat{p} |

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera, mkcube
            >>> P = mkcube(0.2)
            >>> camera_unknown = CentralCamera(f=0.015, rho=10e-6, imagesize=[1280, 1024], noise=0.05, seed=0)
            >>> T_unknown = SE3.Trans(0.1, 0.2, 1.5) * SE3.RPY(0.1, 0.2, 0.3)
            >>> p = camera_unknown.project_point(P, objpose=T_unknown)
            >>> C, resid = CentralCamera.points2C(P, p)
            >>> C
            >>> camera_unknown.C()
            >>> resid

        .. note:: This method assumes no lens distortion affecting the image plane
            coordinates.

        :references:
            - Robotics, Vision & Control for Python, Section 13.2.1, P. Corke, 
              Springer 2023.

        :seealso: :meth:`C` :meth:`images2C` :meth:`decomposeC`
        """
        z4 = np.zeros((4,))

        A = np.empty(shape=(0,11))
        b = np.empty(shape=(0,))
        for uv, X in zip(p.T, P.T):
            u, v = uv
            # fmt: off
            row = np.array([
                    np.r_[ X, 1, z4, -u * X],
                    np.r_[z4, X,  1, -v * X]
                ])
            # fmt: on
            A = np.vstack((A, row))
            b = np.r_[b, uv]

        # solve Ax = b where c is 11 elements of camera matrix
        c, *_ = scipy.linalg.lstsq(A, b)

        # compute and print the residual
        r = np.max(np.abs((A @ c - b)))

        c = np.r_[c, 1]   # append a 1
        C = c.reshape((3,4))  # make a 3x4 matrix

        return C, r

    @classmethod
    def images2C(self, images, gridshape=(7,6), squaresize=0.025):
        """
        Calibrate camera from checkerboard images

        :param images: an iterator that returns :class:`~machinevisiontoolbox.ImageCore.Image` objects
        :type images: :class:`~machinevisiontoolbox.Sources.ImageSource`
        :param gridshape: number of grid squares in each dimension, defaults to (7,6)
        :type gridshape: tuple, optional
        :param squaresize: size of the grid squares in units of length, defaults to 0.025
        :type squaresize: float, optional
        :return: camera calibration matrix, distortion parameters, image frames
        :rtype: ndarray(3,4), ndarray(5), list of named tuples

        The distortion coefficients are in the order :math:`(k_1, k_2, p_1, p_2, k_3)`
        where :math:`k_i` are radial distortion coefficients and :math:`p_i` are
        tangential distortion coefficients.

        Image frames that were successfully processed are returned as a list of
        named tuples ``CalibrationFrame`` with elements:

        =======  =====================  ===========================================================
        element  type                   description
        =======  =====================  ===========================================================
        image    :class:`Image`         calibration image with overlaid annotation
        pose     :class:`SE3` instance  pose of the camera with respect to the origin of this image
        id       int                    sequence number of this image in ``images``
        =======  =====================  ===========================================================

        .. note:: The units used for ``squaresize`` must match the units used
            for defining 3D points in space.

        :references:
            - Robotics, Vision & Control for Python, Section 13.7, P. Corke, 
              Springer 2023.

        :seealso: :meth:`C` :meth:`points2C` :meth:`decomposeC` :class:`~spatialmath..pose3d.SE3`
        """

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # create set of feature points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # these all have Z=0 since they are relative to the calibration target frame
        objp = np.zeros((gridshape[0] * gridshape[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:gridshape[0], 0:gridshape[1]].T.reshape(-1, 2) * squaresize

        # lists to store object points and image points from all the images
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        corner_images = []
        valid = []

        for i, image in enumerate(images):

            gray = image.mono().A
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, gridshape, None)
            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
                # Draw the corners
                image = Image(image, copy=True)
                if not image.iscolor:
                    image = image.colorize()
                corner_images.append(cv.drawChessboardCorners(image.A, gridshape, corners2, ret))
                valid.append(i)

        ret, C, distortion, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        CalibrationFrame = namedtuple("CalibrationFrame", "image pose id")
        if ret:
            frames = []
            for rvec, tvec, corner_image, id in zip(rvecs, tvecs, corner_images, valid):
                frame = CalibrationFrame(
                    Image(corner_image, colororder="BGR"), 
                    (SE3(tvec) * SE3.EulerVec(rvec.flatten())).inv(),
                    id)
                frames.append(frame)
            return C, distortion[0], frames
        else:
            return None

    @classmethod
    def decomposeC(cls, C):
        r"""
        Decompose camera calibration matrix

        :param C: camera calibration matrix
        :type C: ndarray(3,4)
        :return: camera model parameters
        :rtype: :class:`CentralCamera`

        Decompose a :math:`3\times 4` camera calibration matrix ``C`` to
        determine feasible intrinsic and extrinsic parameters. The result is a
        ``CentralCamera`` instance with the following parameters set:

        ================  ====================================
        Parameter         Meaning
        ================  ====================================
        ``f``             focal length in pixels
        ``sx``, ``sy``    pixel size where ``sx`` =1
        (``u0``, ``v0``)  principal point
        ``pose``          pose of the camera frame wrt world
        ================  ====================================

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> from spatialmath import SE3
            >>> camera = CentralCamera(name='camera1')
            >>> C = camera.C(SE3(0.1, 0, 0))
            >>> CentralCamera.decomposeC(C)

        .. note:: Since only :math:`f s_x` and :math:`f s_y` can be estimated we
            set :math:`s_x = 1`.

        :reference:
            - Multiple View Geometry, Hartley&Zisserman, p 163-164
            - Robotics, Vision & Control for Python, Section 13.2.3, P. Corke, 
              Springer 2023.

        :seealso: :meth:`C` :meth:`points2C`
        """
        def rq(S):
            # from vgg_rq.m
            # [R,Q] = vgg_rq(S)  Just like qr but the other way around.
            # If [R,Q] = vgg_rq(X), then R is upper-triangular, Q is orthogonal, and X==R*Q.
            # Moreover, if S is a real matrix, then det(Q)>0.
            # By awf

            S = S.T
            Q, U = np.linalg.qr(S[::-1, ::-1])
            Q = Q.T
            Q = Q[::-1, ::-1]
            U = U.T
            U = U[::-1, ::-1]

            if np.linalg.det(Q) < 0:
                U[:, 0] = -U[:, 0]
                Q[0, :] = -Q[0, :]
            return U, Q


        if not C.shape == (3,4):
            raise ValueError('argument is not a 3x4 matrix')

        u, s, v = np.linalg.svd(C)
        v = v.T

        # determine camera position
        t = v[:, 3]  # last column
        t = t / t[3]
        t = t[:3]

        # determine camera orientation
        M = C[:3, :3]
        # K, R = rq(M)
        K, R = scipy.linalg.rq(M)

        # deal with K having negative elements on the diagonal
        # make a matrix to fix this, K*C has positive diagonal
        C = np.diag(np.sign(np.diag(K)))
        
        # now  K*R = (K*C) * (inv(C)*R), so we need to check C is a proper rotation
        # matrix.  If isn't then the situation is unfixable
        
        if not np.isclose(np.linalg.det(C), 1):
            raise RuntimeError('cannot correct signs in the intrinsic matrix')
        
        # all good, let's fix it
        K = K @ C
        R = C.T @ R
        
        # normalize K so that lower left is 1
        K = K / K[2, 2]
        
        # pull out focal length and scale factors
        f = K[0, 0]
        s = np.r_[1, K[1,1] / K[0, 0]]

        # build an equivalent camera model
        return cls(name='invC',
            f=f, pp=K[:2, 2], rho=s, pose=SE3.Rt(R.T, t))

# https://docs.opencv.org/3.1.0/d9/d0c/group__calib3d.html#gaaae5a7899faa1ffdf268cd9088940248

    # ======================= homography =============================== #

    def H(self, T, n, d):
        """
        Compute homography from plane and camera pose

        :param T: relative camera motion
        :type T: :class:`~spatialmath..pose3d.SE3`
        :param n: plane normal with respect to world frame
        :type n: array_like(3)
        :param d: plane offset from world frame origin
        :type d: float
        :return: homography matrix
        :rtype: ndarray(3,3)

        Computes the homography matrix for the camera observing points on a
        plane from two viewpoints. The first view is from the current camera
        pose (``self.pose``), and the second is after a relative motion
        represented by the rigid-body motion ``T``. The plane has normal ``n``
        and at distance ``d`` with respect to the world frame.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> from spatialmath import SE3
            >>> camera = CentralCamera.Default(name='camera1') # looking along z-axis
            >>> plane = [0, 1, 1]
            >>> H = camera.H(SE3.Tx(0.2), plane, 5)
            >>> H

        :seealso: :meth:`points2H` :meth:`decomposeH`
        """
        if d < 0:
            raise ValueError(d, 'plane distance d must be > 0')

        n = base.getvector(n)
        if n[2] < 0:
            raise ValueError(n, 'normal must be away from camera (n[2] >= 0)')

        # T transform view 1 to view 2
        T = SE3(T).inv()

        HH = T.R + 1.0 / d * T.t @ n  # need to ensure column then row = 3x3

        # apply camera intrinsics
        HH = self.K @ HH @ np.linalg.inv(self.K)

        return HH / HH[2, 2]  # normalised

    @staticmethod
    def points2H(p1, p2, method='leastsquares', seed=None, **kwargs):
        """
        Estimate homography from corresponding points

        :param p1: image plane points from first camera
        :type p1: ndarray(2,N)
        :param p2: image plane points from second camera
        :type p2: ndarray(2,N)
        :param method: algorithm: 'leastsquares' [default], 'ransac', 'lmeds', 'prosac'
        :type method: str
        :param kwargs: optional arguments as required for ransac' and 'lmeds'
            methods
        :return: homography, residual and optional inliers
        :rtype: ndarray(3,3), float, ndarray(N,bool)

        Compute a homography from two sets of corresponding image plane points
        whose world points lie on a plane.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera, mkgrid
            >>> from spatialmath import SE3
            >>> camera1 = CentralCamera(name="camera 1", f=0.002, imagesize=1000, rho=10e-6, pose=SE3.Tx(-0.1)*SE3.Ry(0.4))
            >>> camera2 = CentralCamera(name="camera 2", f=0.002, imagesize=1000, rho=10e-6, pose=SE3.Tx(0.1)*SE3.Ry(-0.4))
            >>> T_grid = SE3.Tz(1) * SE3.Rx(0.1) * SE3.Ry(0.2)
            >>> P = mkgrid(3, 1.0, pose=T_grid)
            >>> p1 = camera1.project_point(P)
            >>> p2 = camera2.project_point(P);
            >>> H, resid = CentralCamera.points2H(p1, p2)
            >>> H
            >>> resid

        .. note:: If the method is 'ransac' or 'lmeds' then a boolean array
            of inliers is also returned, True means the corresponding input
            point pair is an inlier.

        :reference:
            - Robotics, Vision & Control for Python, Section 14.2.4, P. Corke, 
              Springer 2023.

        :seealso: :meth:`H` :meth:`decomposeH` 
            `opencv.findHomography <https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780>`_
        """

        points2H_dict = {
            'leastsquares': 0,
            'ransac': cv.RANSAC,
            'lmeds': cv.LMEDS,
            'prosac': cv.RHO 
        }
        if seed is not None:
            cv.setRNGSeed(seed)

        H, mask = cv.findHomography(
            srcPoints=p1.T,
            dstPoints=p2.T,
            method=points2H_dict[method],
            **kwargs)

        mask = mask.ravel().astype(np.bool)
        e = base.homtrans(H, p1[:, mask]) - p2[:, mask]
        resid = np.linalg.norm(e)

        if method in ('ransac', 'lmeds'):
            return H, resid, mask
        else:
            return H, resid

    # https://docs.opencv.org/3.1.0/d9/d0c/group__calib3d.html#ga7f60bdff78833d1e3fd6d9d0fd538d92
    def decomposeH(self, H, K=None):
        """
        Decompose homography matrix

        :param H: homography matrix
        :type H: ndarray(3,3)
        :param K: camera intrinsics, defaults to parameters from object
        :type K: ndarray(3,3), optional
        :return: camera poses, plane normals
        :rtype: :class:`~spatialmath..pose3d.SE3`, list of ndarray(3,1)

        Decomposes the homography matrix into the camera motion and the normal
        to the plane. In practice, there are multiple solutions. The translation
        not to scale.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera, mkgrid
            >>> from spatialmath import SE3
            >>> camera1 = CentralCamera(name="camera 1", f=0.002, imagesize=1000, rho=10e-6, pose=SE3.Tx(-0.1)*SE3.Ry(0.4))
            >>> camera2 = CentralCamera(name="camera 2", f=0.002, imagesize=1000, rho=10e-6, pose=SE3.Tx(0.1)*SE3.Ry(-0.4))
            >>> T_grid = SE3.Tz(1) * SE3.Rx(0.1) * SE3.Ry(0.2)
            >>> P = mkgrid(3, 1.0, pose=T_grid)
            >>> p1 = camera1.project_point(P)
            >>> p2 = camera2.project_point(P);
            >>> H, resid = CentralCamera.points2H(p1, p2)
            >>> T, normals = camera1.decomposeH(H)
            >>> T.printline(orient="camera")
            >>> normals

        :reference:
            - Robotics, Vision & Control for Python, Section 14.2.4, P. Corke, 
              Springer 2023.

        :seealso: :meth:`points2H` :meth:`H`
            `opencv.decomposeHomographyMat <>`_
        """

        retval, rotations, translations, normals = cv.decomposeHomographyMat(H, self.K)

        T = SE3.Empty()
        for R, t in zip(rotations, translations):
            # we normalize the rotation matrix, those returned by openCV can
            # not quite proper SO(3) values
            pose = SE3.Rt(smbase.trnorm(R), t).inv()
            T.append(pose)
        return T, normals

        # if K is None:
        #     K = np.identity(3)
        #     # also have K = self.K

        # H = np.linalg.inv(K) @ H @ K

        # # normalise so that the second singular value is one
        # U, S, V = np.linalg.svd(H, compute_uv=True)
        # H = H / S[1]

        # # compute the SVD of the symmetric matrix H'*H = VSV'
        # U, S, V = np.linalg.svd(np.transpose(H) @ H)

        # # ensure V is right-handed
        # if np.linalg.det(V) < 0:
        #     print('det(V) was < 0')
        #     V = -V

        # # get squared singular values
        # s0 = S[0]
        # s2 = S[2]

        # # v0 = V[0:, 0]
        # # v1 = V[0:, 1]
        # # v2 = V[0:, 2]

        # # pure rotation - where all singular values == 1
        # if np.abs(s0 - s2) < (100 * np.spacing(1)):
        #     print('Warning: Homography due to pure rotation')
        #     if np.linalg.det(H) < 0:
        #         H = -H
        #     # sol = namedtuple('T', T, ''
        # # TODO finish from invhomog.m
        # print('Unfinished')
        # return False


    # =================== fundamental matrix =============================== #

    def F(self, other):
        """
        Fundamental matrix

        :param other: second camera view
        :type other: :class:`CentralCamera`, :class:`~spatialmath..pose3d.SE3`
        :return: fundamental matrix
        :rtype: numpy(3,3)

        Compute the fundamental matrix relating two camera views.  The
        first view is defined by the instance.  The second
        is defined by:
        
        * another :class:`CentralCamera` instance
        * an SE3 pose describing the pose of the second view with respect to 
          the first, assuming the same camera intrinsic parameters.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> from spatialmath import SE3
            >>> camera1 = CentralCamera(name="camera 1", f=0.002, imagesize=1000, rho=10e-6, pose=SE3.Tx(-0.1)*SE3.Ry(0.4))
            >>> camera2 = CentralCamera(name="camera 2", f=0.002, imagesize=1000, rho=10e-6, pose=SE3.Tx(0.1)*SE3.Ry(-0.4))
            >>> F = camera1.F(camera2)
            >>> F
            >>> F = camera1.F(SE3.Tx(0.2))
            >>> F

        :reference:
            - Y.Ma, J.Kosecka, S.Soatto, S.Sastry, "An invitation to 3D",
              Springer, 2003. p.177
            - Robotics, Vision & Control for Python, Section 14.2.1, P. Corke, 
              Springer 2023.
        
        :seealso: :meth:`points2F` :meth:`E`
        """

        if isinstance(other, SE3):
            E = self.E(other)
            K = self.K
            return np.linalg.inv(K).T @ E @ np.linalg.inv(K)

        elif isinstance(other, CentralCamera):
            # use relative pose and camera parameters of 
            E = self.E(other)
            K1 = self.K
            K2 = other.K
            return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

        else:
            raise ValueError('bad type')

    @staticmethod
    def points2F(
                    p1,
                    p2,
                    method='8p',
                    residual=True,
                    seed=None,
                    **kwargs):
        """
        Estimate fundamental matrix from corresponding points

        :param p1: image plane points from first camera
        :type p1: ndarray(2,N)
        :param p2: image plane points from second camera
        :type p2: ndarray(2,N)
        :param method: algorithm '7p', '8p' [default], 'ransac', 'lmeds'
        :type method: str, optional
        :param kwargs: optional arguments as required for ransac', 'lmeds'
            methods
        :return: fundamental matrix and residual
        :rtype: ndarray(3,3), float

        Computes the fundamental matrix from two sets of corresponding
        image-plane points. Corresponding points are given by corresponding
        columns of ``p1`` and ``p2``.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera, mkgrid
            >>> from spatialmath import SE3
            >>> camera1 = CentralCamera(name="camera 1", f=0.002, imagesize=1000, rho=10e-6, pose=SE3.Tx(-0.1)*SE3.Ry(0.4))
            >>> camera2 = CentralCamera(name="camera 2", f=0.002, imagesize=1000, rho=10e-6, pose=SE3.Tx(0.1)*SE3.Ry(-0.4))
            >>> T_grid = SE3.Tz(1) * SE3.Rx(0.1) * SE3.Ry(0.2)
            >>> P = mkgrid(3, 1.0, pose=T_grid)
            >>> p1 = camera1.project_point(P)
            >>> p2 = camera2.project_point(P);
            >>> F, resid = CentralCamera.points2F(p1, p2)
            >>> F
            >>> resid

        :seealso: :meth:`F` :meth:`E`
            `opencv.findFundamentalMat <https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga59b0d57f46f8677fb5904294a23d404a>`_
        """
        
        points2F_dict = {
                '7p': cv.FM_7POINT,
                '8p': cv.FM_8POINT,
                'ransac': cv.FM_RANSAC,
                'lmeds': cv.FM_LMEDS}

        if seed is not None:
            cv.setRNGSeed(seed)

        F, mask = cv.findFundamentalMat(p1.T, p2.T,
                                        method=points2F_dict[method],
                                        **kwargs)

        mask = mask.ravel().astype(np.bool)

        # add various return values
        retval = [F]
        if residual:
            e = base.e2h(p2[:, mask]).T @ F @ base.e2h(p1[:, mask])
            resid = np.linalg.norm(np.diagonal(e))
            retval.append(resid)
        if method in ('ransac', 'lmeds'):
            retval.append(mask)

        return retval
        # elines = base.e2h(p2).T @ F # homog lines, one per row
        # p1h = base.e2h(p1)
        # residuals = []
        # for i, line in enumerate(elines):
        #     d = line @ p1h[:, i] / np.sqrt(line[0] ** 2 + line[1] ** 2)
        #     residuals.append(d)
        # resid = np.array(residuals).mean()

        
    @staticmethod
    def epidist(F, p1, p2):
        """
        Epipolar distance

        :param F: fundamental matrix
        :type F: ndarray(3,3)
        :param p1: image plane point or points from first camera
        :type p1: ndarray(2) or ndarray(2,N)
        :param p2: image plane point or points from second camera
        :type p2: ndarray(2) or ndarray(2,M)
        :return: distance matrix
        :rtype: ndarray(N,M)

        Computes the distance of the points ``p2`` from the 
        epipolar lines induced by points ``p1``.  Element [i,j] of the return
        value is the istance of point j in camera 2 from the epipolar line
        induced by point i in camera 1.
        """
        if p1.ndim == 1:
            p1 = np.c_[p1]
        if p2.ndim == 1:
            p2 = np.c_[p2]

        D = np.empty((p1.shape[1], p2.shape[1]))

        # compute epipolar lines corresponding to p1
        l = F @ base.e2h(p1)
        for i in range(p1.shape[1]):
            for j in range(p2.shape[1]):
                D[i, j] = np.abs(l[0, i] * p2[0,j] + l[1, i] * p2[1, j] + l[2, i]) \
                    / np.sqrt(l[0, i]**2 + l[1, i]**2)
        return D

    # ===================== essential matrix =============================== #

    def E(self, other):
        """
        Essential matrix from two camera views

        :param other: second camera view, camera pose or fundamental matrix
        :type other: :class:`CentralCamera`, :class:`~spatialmath..pose3d.SE3`, ndarray(3,3)
        :return: essential matrix
        :rtype: ndarray(3,3)

        Compute the essential matrix relating two camera views. The first view
        is defined by the instance, and second view is specified by:

        * a camera instance represented by a :class:`CentralCamera`. Assumes the
          cameras have the same intrinsics.
        * a relative motion represented by a :class:`~spatialmath..pose3d.SE3`
        * a fundamental matrix
        
        :reference:
            - Y.Ma, J.Kosecka, S.Soatto, S.Sastry, "An invitation to 3D",
              Springer, 2003. p.177

        :seealso: :meth:`F` :meth:`decomposeE` :meth:`points2E`
        """

        if isinstance(other, np.ndarray) and other.shape == (3,3):
            # essential matrix from F matrix and intrinsics
            return self.K.T @ other @ self.K

        elif isinstance(other, CentralCamera):
            # camera relative pose
            T21 = other.pose.inv() * self.pose

        elif isinstance(other, SE3):
            # relative pose given explicitly
            T21 = other.inv()

        else:
            raise ValueError('bad type')
        
        return base.skew(T21.t) @ T21.R

    def points2E(self,
                    p1,
                    p2,
                    method=None,
                    K=None,
                    **kwargs):
        """
        Essential matrix from points

        :param P1: image plane points
        :type P1: ndarray(2,N)
        :param P2: image plane points
        :type P2: ndarray(2,N)
        :param method: method, can be 'ransac' or 'lmeds'
        :type method: str
        :param K: camera intrinsic matrix, defaults to that of camera object
        :type K: ndarray(3,3), optional
        :param kwargs: additional arguments required for 'ransac' or 'lmeds'
            options
        :return: essential matrix and optional inlier vevtor
        :rtype: ndarray(3,3), ndarray(N, bool)

        Compute the essential matrix from two sets of corresponding points.
        Each set of points is represented by the columns of the array ``p1``
        or ``p2``.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera, mkgrid
            >>> from spatialmath import SE3
            >>> camera1 = CentralCamera(name="camera 1", f=0.002, imagesize=1000, rho=10e-6, pose=SE3.Tx(-0.1)*SE3.Ry(0.4))
            >>> camera2 = CentralCamera(name="camera 2", f=0.002, imagesize=1000, rho=10e-6, pose=SE3.Tx(0.1)*SE3.Ry(-0.4))
            >>> T_grid = SE3.Tz(1) * SE3.Rx(0.1) * SE3.Ry(0.2)
            >>> P = mkgrid(3, 1.0, pose=T_grid)
            >>> p1 = camera1.project_point(P)
            >>> p2 = camera2.project_point(P);
            >>> E, inliers = camera1.points2E(p1, p2)
            >>> E
            >>> inliers

        .. note:: If the method is 'ransac' or 'lmeds' then a boolean array
            of inliers is also returned, True means the corresponding input
            point pair is an inlier.

        :seealso: :meth:`E` :meth:`decomposeE` 
            `opencv.findEssentialMat <https://docs.opencv.org/master/d9/d0c/group__calib3d.html#gad245d60e64d0c1270dbfd0520847bb87>`_

        """
        if K is None:
            K = self.K

        points2E_dict = {
            'ransac': cv.RANSAC,
            'lmeds': cv.LMEDS
        }
        if method is not None:
            method = points2E_dict[method]

        E, mask = cv.findEssentialMat(p1.T, p2.T, cameraMatrix=K, method=method, **kwargs)
        if mask is not None:
            mask = mask.flatten().astype(np.bool)
            return E, mask
        else:
            return E

    def decomposeE(self, E, P=None):
        """
        Decompose essential matrix

        :param E: essential matrix
        :type E: ndarray(3,3)
        :param P: world point or feature match object to resolve ambiguity
        :type P: array_like(3), :class:`~machinevisiontoolbox.ImagePointFeatures.FeatureMatch`
        :return: camera relative pose
        :rtype: :class:`~spatialmath..pose3d.SE3`

        Determines relative pose from essential matrix. This operation has
        multiple solutions which is resolved by passing in:

        - a single 3D world point in front of the camera
        - a :class:`~machinevisiontoolbox.ImagePointFeatures.FeatureMatch` object

        :reference:
        - OpenCV: 
        
        :seealso: :meth:`E` :meth:`points2E`
            :class:`~machinevisiontoolbox.ImagePointFeatures.FeatureMatch`
            `opencv.decomposeEssentialMat <https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga54a2f5b3f8aeaf6c76d4a31dece85d5d>`_
            `opencv.recoverPose <https://docs.opencv.org/master/d9/d0c/group__calib3d.html#gadb7d2dfcc184c1d2f496d8639f4371c0>`_
        """
        if isinstance(P, FeatureMatch):
            # passed a Match object
            match = P

            retval, R, t, mask = cv.recoverPose(
                    E=E,
                    points1=match.p1.T,
                    points2=match.p2.T,
                    cameraMatrix=self.C()[:3, :3]
                    )
            # not explicitly stated, but seems that this returns (R, t) from 
            # camera to world

            return SE3.Rt(R, t).inv()

        else:
        
            R1, R2, t = cv.decomposeEssentialMat(E=E)
            # not explicitly stated, but seems that this returns (R, t) from 
            # camera to world

            possibles = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

            if base.isvector(P, 3):
                for Rt in possibles:
                    pose = SE3.Rt(Rt[0], Rt[1]).inv()
                    p = self.project_point(P, pose=pose, behind=True)
                    # check if point is projected behind the camera, indicated
                    # by nan values
                    if not np.isnan(p[0]):
                        # return the first good one
                        return pose
            else:
                T = SE3.Empty()
                for Rt in possibles:
                    pose = SE3.Rt(Rt[0], Rt[1]).inv()
                    T.append(pose)
                return T


    # ===================== image plane motion ============================= #

    def visjac_p(self, uv, depth):
        r"""
        Visual Jacobian for point features

        :param p: image plane point or points
        :type p: array_like(2), ndarray(2,N)
        :param depth: point depth
        :type depth: float, array_like(N)
        :return: visual Jacobian matrix
        :rtype: ndarray(2,6), ndarray(2N,6)

        Compute the image Jacobian :math:`\mat{J}` which maps

        .. math::

            \dvec{p} = \mat{J}(\vec{p}, z) \vec{\nu}

        camera spatial velocity :math:`\vec{\nu}` to the image plane velocity
        :math:`\dvec{p}` of the point.

        If ``p`` describes multiple points then return a stack of these 
        :math:`2\times 6` matrices, one per point.
        
        Depth is the z-component of the point's coordinate in the camera frame.
        If ``depth`` is a scalar then it is the depth for all points. 

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> from spatialmath import SE3
            >>> camera = CentralCamera.Default()
            >>> camera.visjac_p((200, 300), 2)

        :references:
            - A tutorial on Visual Servo Control, Hutchinson, Hager & Corke, 
              IEEE Trans. R&A, Vol 12(5), Oct, 1996, pp 651-670.
            - Robotics, Vision & Control for Python, Section 15.2.1, P. Corke, 
              Springer 2023.
        
        :seealso: :meth:`flowfield` :meth:`visjac_p_polar` :meth:`visjac_l` :meth:`visjac_e`
        """

        uv = base.getmatrix(uv, (2, None))
        Z = depth

        Z = base.getvector(Z)
        if len(Z) == 1:
            Z = np.repeat(Z, uv.shape[1])
        elif len(Z) != uv.shape[1]:
                raise ValueError('Z must be a scalar or have same number of columns as uv')
            
        L = np.empty((0, 6))  # empty matrix

        K = self.K
        Kinv = np.linalg.inv(K)
        
        for z, p in zip(Z, uv.T):  # iterate over each column (point)

            # convert to normalized image-plane coordinates
            xy = Kinv @ base.e2h(p)
            x = xy[0,0]
            y = xy[1,0]

            # 2x6 Jacobian for this point
            # fmt: off
            Lp = K[:2,:2] @ np.array(
                [ [-1/z,  0,     x/z, x * y,      -(1 + x**2), y],
                  [ 0,   -1/z,   y/z, (1 + y**2), -x*y,       -x] ])
            # fmt: on
            # stack them vertically
            L = np.vstack([L, Lp])

        return L

    def visjac_p_polar(self, p, Z):
        r"""
        Visual Jacobian for point features in polar coordinates

        :param p: image plane point or points
        :type p: array_like(2), ndarray(2,N)
        :param depth: point depth
        :type depth: float, array_like(N)
        :return: visual Jacobian matrix in polar coordinates
        :rtype: ndarray(2,6), ndarray(2N,6)

        Compute the image Jacobian :math:`\mat{J}` which maps

        .. math::

            \begin{pmatrix} \dot{\phi} \\ \dot{r} \end{pmatrix} = \mat{J}(\vec{p}, z) \vec{\nu}

        camera spatial velocity :math:`\vec{\nu}` to the image plane velocity
        of the point expressed in polar coordinate form :math:`(\phi, r)`.

        If ``p`` describes multiple points then return a stack of these 
        :math:`2\times 6` matrices, one per point.
        
        Depth is the z-component of the point's coordinate in the camera frame.
        If ``depth`` is a scalar then it is the depth for all points. 

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> from spatialmath import SE3
            >>> camera = CentralCamera.Default()
            >>> camera.visjac_p_polar((200, 300), 2)

        :references:
            - Combining Cartesian and polar coordinates in IBVS. 
              Corke PI, Spindler F, Chaumette F 
              IROS 2009, pp 5962–5967
            - Robotics, Vision & Control for Python, Section 16.2 P. Corke, 
              Springer 2023.
        
        :seealso: :meth:`visjac_p` :meth:`visjac_l` :meth:`visjac_e`
        """

        J = []
        p = smbase.getmatrix(p, (2, None))
        f = self.f[0]

        if smbase.isscalar(Z):
            Z = [Z] * p.shape[1]

        for (phi, r), Zk in zip(p.T, Z):

            # k = (f**2 + r**2) / f
            # k2 = f / (r * Zk)

            c = np.cos(phi)
            s = np.sin(phi)
            
            r = max(r, 0.05)
            Jk = np.array([
                [ -s/r/Zk, c/r/Zk, 0, -c/r, -s/r, 1],
                [c/Zk, s/Zk, -r/Zk, -(1+r**2)*s, (1+r**2)*c, 0]
            ])

            # Jk = np.array([
            #     [     k2 * sth,     -k2 * cth,      0, f / r * cth, f / r * sth , -1],
            #     [-f / Zk * cth, -f / Zk * sth, r / Zk,     k * sth,     -k * cth,  0],
            #     ])
            # Jk = np.array([
            #     [cth/Zk , sth / Zk, -r / Zk,   -(1+r**2) * sth,     -k * cth,  0],
            #     [     k2 * sth,     -k2 * cth,      0, f / r * cth, f / r * sth , -1]])

            J.append(Jk)

        return np.vstack(J)

        # if 0
        # r = rt(1); theta = rt(2);

        # % compute the mapping from uv-dot to r-theta dot 
        # M = 1/r * [r*cos(theta) r*sin(theta); -sin(theta) cos(theta)];

        # % convert r-theta form to uv form
        # u = r * cos(theta); v = r * sin(theta);

        # % compute the Jacobian
        # J = M * cam.visjac_p([u; v], Z);


    def visjac_l(self, lines, plane):
        r"""
        Visual Jacobian for line features

        :param lines: image plane line parameters
        :type p: array_like(2), ndarray(2,N)
        :param plane: plane containing the line
        :type plane: array_like(4)
        :return: visual Jacobian matrix for line feature
        :rtype: ndarray(2,6), ndarray(2N,6)

        Compute the Jacobian which gives the rates of change of the line
        parameters in terms of camera spatial velocity. 

        For image planes lines
        
        .. math:: u \cos \theta + v \sin \theta = \rho
        
        the image Jacobian :math:`\mat{J}` maps

        .. math::

            \begin{pmatrix} \dot{\theta} \\ \dot{\rho} \end{pmatrix} = \mat{J}(\vec{p}, z) \vec{\nu}

        camera spatial velocity :math:`\vec{\nu}` to the image plane velocity
        of the line parameters :math:`(\theta, \rho)`.

        The world plane containing the line is also required, and is provided
        as a vector :math:`(a,b,c,d)` such that

        .. math: aX + bY +cZ + d = 0

        If ``lines`` describes multiple points then return a stack of these 
        :math:`2\times 6` matrices, one per point.
        
        Depth is the z-component of the point's coordinate in the camera frame.
        If ``depth`` is a scalar then it is the depth for all points. 

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> from spatialmath import SE3
            >>> camera = CentralCamera.Default()
            >>> camera.visjac_l((0.2, 500), (0, 0, 1, -3))

        :references:
            - A New Approach to Visual Servoing in Robotics,
              B. Espiau, F. Chaumette, and P. Rives,
              IEEE Transactions on Robotics and Automation, 
              vol. 8, pp. 313-326, June 1992.
            - Visual servo control 2: Advanced approaches
              Chaumette F, Hutchinson S,
              IEEE Robot Autom Mag 14(1):109–118 (2007)
            - Robotics, Vision & Control for Python, Section 15.3.1, P. Corke, 
              Springer 2023.
        
        :seealso: :meth:`visjac_p` :meth:`visjac_p_polar` :meth:`visjac_e`
        """

        a, b, c, d = plane

        lines = smbase.getmatrix(lines, (2, None))
        jac = []
        for theta, rho in lines.T:
            sth = np.sin(theta)
            cth = np.cos(theta)

            lam_th = (a*sth - b*cth ) / d
            lam_rho = (a*rho*cth + b*rho*sth + c) / d

            L = np.array([
                [lam_th*cth, lam_th*sth,  -lam_th*rho, -rho*cth, -rho*sth, -1],
                [lam_rho*cth, lam_rho*sth, -lam_rho*rho, (1 + rho**2)*sth, -(1 + rho**2)*cth, 0]
                ])
            jac.append(L)

        return np.vstack(jac)    


    def visjac_e(self, E, plane):
        r"""
        Visual Jacobian for ellipse features

        :param E: image plane ellipse parameters
        :type E: array_like(5), ndarray(5,N)
        :param plane: plane containing the ellipse
        :type plane: array_like(4)
        :return: visual Jacobian matrix for ellipse feature
        :rtype: ndarray(2,6), ndarray(2N,6)

        Compute the Jacobian gives the rates of change of the ellipse parameters
        in terms of camera spatial velocity. 

        For image plane ellipses
        
        .. math:: u^2 + E_0 v^2 -2 E_1 u v + 2 E_2 u + 2 E_3 v + E_4 = 0
        
        the image Jacobian :math:`\mat{J}` maps

        .. math::

            \begin{pmatrix} \dot{E_0} \\ \vdots \\ \dot{E_4} \end{pmatrix} = \mat{J}(\vec{p}, z) \vec{\nu}

        camera spatial velocity :math:`\vec{\nu}` to the velocity
        of the ellipse parameters :math:`(E_0 \ldots E_4)`.

        The world plane containing the ellipse is also required, and is provided
        as a vector :math:`(a,b,c,d)` such that

        .. math: aX + bY +cZ + d = 0

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> from spatialmath import SE3
            >>> camera = CentralCamera.Default()
            >>> camera.visjac_e(((0.5, 0, -1000, -500, 374900)), (0, 0, 1, -1)

        :references:
            - A New Approach to Visual Servoing in Robotics,
              B. Espiau, F. Chaumette, and P. Rives,
              IEEE Transactions on Robotics and Automation, 
              vol. 8, pp. 313-326, June 1992.
            - Visual servo control 2: Advanced approaches
              Chaumette F, Hutchinson S,
              IEEE Robot Autom Mag 14(1):109–118 (2007)
            - Robotics, Vision & Control for Python, Section 15.3.2, P. Corke, 
              Springer 2023.
        
        :seealso: :meth:`visjac_p` :meth:`visjac_p_polar` :meth:`visjac_l`
        """

        a = -plane[0] / plane[3]
        b = -plane[1] / plane[3]
        c = -plane[2] / plane[3]
        L = np.array([
            [2*b*E[1]-2*a*E[0], 2*E[0]*(b-a*E[1]), 2*b*E[3]-2*a*E[0]*E[2], 2*E[3], 2*E[0]*E[2], -2*E[1]*(E[0]+1)],
            [b-a*E[1], b*E[1]-a*(2*E[1]**2-E[0]), a*(E[3]-2*E[1]*E[2])+b*E[2], -E[2], -(2*E[1]*E[2]-E[3]), E[0]-2*E[1]**2-1],
            [c-a*E[2], a*(E[3]-2*E[1]*E[2])+c*E[1], c*E[2]-a*(2*E[2]**2-E[4]), -E[1], 1+2*E[2]**2-E[4], E[3]-2*E[1]*E[2]],
            [E[2]*b+E[1]*c-2*a*E[3], E[3]*b+E[0]*c-2*a*E[1]*E[3], b*E[4]+c*E[3]-2*a*E[2]*E[3], E[4]-E[0], 2*E[2]*E[3]+E[1], -2*E[1]*E[3]-E[2]],
            [2*c*E[2]-2*a*E[4], 2*c*E[3]-2*a*E[1]*E[4], 2*c*E[4]-2*a*E[2]*E[4], -2*E[3], 2*E[2]*E[4]+2*E[2], -2*E[1]*E[4]]
            ])

        L = L @ np.diag([0.5, 0.5, 0.5, 1, 1, 1])   # not sure why...
        return L

    def flowfield(self, vel, Z=2):
        """
        Display optical flow field

        :param vel: camera spatial velocity
        :type vel: array_like(6)
        :param Z: _description_, defaults to 2
        :type Z: scalar, optional

        Display the optical flow field using Matplotlib, for a grid of points at
        distance ``Z`` for a camera velocity of ``vel``.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import CentralCamera
            >>> camera = CentralCamera.Default()
            >>> camera.flowfield([0, 0, 0, 0, 1, 0])

        :seealso: :meth:`visjac_p`
        """
        vel = base.getvector(vel, 6)

        u = np.arange(0, self.nu, 50)
        v = np.arange(0, self.nv, 50)
        [U,V] = np.meshgrid(u, v, indexing='ij')
        du = np.empty(shape=U.shape)
        dv = np.empty(shape=U.shape)
        for r in range(U.shape[0]):                      
            for c in range(U.shape[1]):
                J = self.visjac_p((U[r,c], V[r,c]), Z )            
                ud, vd =  J @ vel
                du[r,c] = ud
                dv[r,c] = -vd

        self.clf()
        ax = self._init_imageplane()
        ax.quiver(U, V, du, dv, 0.4, zorder=20)

    def derivatives(self, x, P):
        r"""
        Compute projection and derivatives for bundle adjustment

        :param x: camera pose as translation and quaternion vector part
        :type x: array_like(6)
        :param P: 3D world point
        :type P: array_like(3)
        :return: p, A, B
        :rtype: ndarray(2), ndarray(2,6), ndarray(2,3)

        For a world point :math:`\vec{x}` compute the image plane projection and the
        sensitivity to camera and point change

        .. math:: \mat{A} = \frac{\partial \vec{f}(\vec{x})}{\partial \pose}, \mat{B} = \frac{\partial \vec{f}(\vec{x})}{\partial \vec{P}}

        where :math:`\vec{f}(\vec{x})` is the perspective projection function.

        :seealso: :meth:`project_point`
        """
        #compute Jacobians and projection

        from  machinevisiontoolbox.camera_derivatives import cameraModel
        
        Kp = [self.f[0], self.rhou, self.rhov, self.u0, self.v0]

        return cameraModel(*x, *P, *Kp)
        

    def estpose(self, P, p, method='iterative', frame="world"):
        """
        Estimate object pose

        :param P: A set of 3D points defining the object with respect to its own frame
        :type P: ndarray(3, N)
        :param p: Image plane projection of the object points
        :type p: ndarray(2, N)
        :param method: pose estimation algorithm, see OpenCV solvePnP, defaults to 'iterative'
        :type method: str, optional
        :param frame: estimate pose with respect to frame "world" [default] or "camera"
        :type frame: str, optional
        :return: pose of target frame relative to the world frame
        :rtype: :class:`~spatialmath..pose3d.SE3`

        Using a set of points defining some object with respect to its own frame {B}, and
        a set of image-plane projections, estimate the pose of {B} with respect to the world
        or camera frame.  
        
        To estimate the camera's pose with respect to the world frame the camera's pose
        ``self.pose`` is used.

        .. note::
        
            * All of the OpenCV estimation algorithms are supported.
            * Algorithm ``"ippe-square"`` requires exactly four points at the corners of a
              square and in the order: (-x, y), (x, y), (x, -y), (-x, -y).
        
        :seealso: :meth:`project_point`
            `opencv.solvePnP <https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d>`_
        """

        method_dict = {
            'iterative': cv.SOLVEPNP_ITERATIVE,
            'epnp': cv.SOLVEPNP_EPNP,
            'p3p': cv.SOLVEPNP_P3P,
            'ap3p': cv.SOLVEPNP_AP3P,
            'ippe': cv.SOLVEPNP_IPPE,
            'ippe-square': cv.SOLVEPNP_IPPE_SQUARE,
        }

        # as per the Note on solvePnP page
        #  we need to ensure that the image point data is contiguous nx1x2 array
        n = p.shape[1]
        p = np.ascontiguousarray(p[:2, :].T).reshape((n, 1, 2))

        # do the pose estimation
        sol = cv.solvePnP(P.T, p, self.K, self._distortion, flags=method_dict[method])

        if sol[0]:
            # pose of target with respect to camera
            pose_C_T = SE3(sol[2]) * SE3.EulerVec(sol[1])
            # pose with respect to world frame
            if frame == "camera":
                return pose_C_T
            elif frame == "world":
                return self.pose * pose_C_T
            else:
                raise ValueError(f'bad frame value {frame}')
                
        else:
            return None


# ------------------------------------------------------------------------ #

class FishEyeCamera(CameraBase):
    """
    .. inheritance-diagram:: machinevisiontoolbox.Camera.FishEyeCamera
        :top-classes: machinevisiontoolbox.Camera.Camera
        :parts: 1
    """

    def __init__(self, k=None, projection='equiangular', **kwargs):
        r"""
        Create fisheye camera projection model

        :param k: scale factor
        :type k: float, optional
        :param projection: projection model: ``'equiangular'`` [default], ``'sine'``, ``'equisolid'`` or ``'stereographic'``
        :type projection: str, optional
        :param kwargs: arguments passed to :class:`CameraBase` constructor

        A fisheye camera contains a wide angle lens, and the angle of the
        incoming ray is mapped to a radius with respect to the principal point.
        The mapping from elevation angle :math:`\theta` to image plane radius is
        given by:

            =============   =======================================
            Projection      :math:`r(\theta)`
            =============   =======================================
            equiangular     :math:`r = k \theta`
            sine            :math:`r = k \sin \theta`
            equisolid       :math:`r = k \sin \frac{\theta}{2}`
            stereographic   :math:`r = k \tan \frac{\theta}{2}`
            =============   =======================================

        .. note:: 
            - If ``K`` is not specified it is computed such that the circular
              imaging region maximally fills the square image plane.
            - This camera model assumes central projection, that is, the focal point
              is at z=0 and the image plane is at z=f.  The image is not inverted.

        :references: 
            - Robotics, Vision & Control for Python, Section 13.3.1, P. Corke, Springer 2023.

        :seealso: :class:`CameraBase` :class:`CentralCamera` :class:`CatadioptricCamera`
            :class:`SphericalCamera` 
        """

        super().__init__(camtype='fisheye', **kwargs)

        self.projection = projection

        if k is None:
            r = np.min((self.imagesize - self.pp) * self.rho)

        if self.projection == 'equiangular':
            if k is None:
                k = r / (pi/2)
            rfunc = lambda theta: k * theta
        elif self.projection == 'sine':
            if k is None:
                k = r
            rfunc = lambda theta: k * np.sin(theta)
        elif self.projection == 'equisolid':
            if k is None:
                k = r / sin(pi / 4)
            rfunc = lambda theta: k * np.sin(theta / 2)
        elif self.projection == 'stereographic':
            if k is None:
                k = r / tan(pi / 4)
            rfunc = lambda theta: k * np.tan(theta / 2)
        else:
            raise ValueError('unknown projection model')

        self.k = k
        self.rfunc = rfunc
        
    def __str__(self):
        s = super().__str__()
        s += self.fmt.format('model', self.projection, fmt="{}")
        s += self.fmt.format('k', self.k, fmt="{:.4g}")
        return s        
        
    def project_point(self, P, pose=None, objpose=None):
        r"""
        Project 3D points to image plane

        :param P: 3D world point or points
        :type P: array_like(3), array_like(3,n)
        :param pose: camera pose with respect to the world frame, defaults to
            camera's ``pose`` attribute
        :type pose: SE:class:`~spatialmath..pose3d.SE3`3, optional
        :param objpose:  3D point reference frame, defaults to world frame
        :type objpose: :class:`~spatialmath..pose3d.SE3`, optional
        :param visibility: test if points are visible, default False
        :type visibility: bool
        :raises ValueError: [description]
        :return: image plane points
        :rtype: ndarray(2,n)

        Project world points to the fisheye camera image plane.

        The elevation angle range is from :math:`-pi/2` (below the mirror) to
        maxangle above the horizontal plane. The mapping from elevation angle
        :math:`\theta` to image plane radius is given by:

            =============   =======================================
            Projection      :math:`r(\theta)`
            =============   =======================================
            equiangular     :math:`r = k \theta`
            sine            :math:`r = k \sin \theta`
            equisolid       :math:`r = k \sin \frac{\theta}{2}`
            stereographic   :math:`r = k \tan \frac{\theta}{2}`
            =============   =======================================
        
        World points are given as a 1D array or the columns of a 2D array of
        Euclidean coordinates. The computed image plane coordinates are
        Euclidean and given as a 1D array or the corresponding columns of a 2D
        array.

        If ``pose`` is specified it is used for the camera pose instead of the
        attribute ``pose``.  The object's attribute is not updated.

        The points ``P`` are by default with respect to the world frame, but 
        they can be transformed by specifying ``objpose``.

        :seealso: :meth:`plot_point`
        """
        
        P = base.getmatrix(P, (3, None))

        if pose is not None:
            T = self.pose.inv()
        else:
            T = SE3()
        if objpose is not None:
            T *= objpose
        
        R = np.sqrt(np.sum(P ** 2, axis=0))
        phi = np.arctan2(P[1, :], P[0, :])
        theta = np.arccos(P[2, :] / R)
        
        r = self.rfunc(theta)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        
        uv = np.array([x / self.rhou + self.u0, y / self.rhov + self.v0])
        
        return self._add_noise_distortion(uv)

# ------------------------------------------------------------------------ #

class CatadioptricCamera(CameraBase):
    """
    .. inheritance-diagram:: machinevisiontoolbox.Camera.CatadioptricCamera
        :top-classes: machinevisiontoolbox.Camera.Camera
        :parts: 1
    """

    def __init__(self, k=None, projection='equiangular', maxangle=None, **kwargs):
        r"""
        Create catadioptric camera projection model

        :param k: scale factor
        :type k: float, optional
        :param projection: projection model: ``'equiangular'`` [default], ``'sine'``, ``'equisolid'`` or ``'stereographic'``
        :type projection: str, optional
        :param kwargs: arguments passed to :class:`CameraBase` constructor

        A catadioptric camera comprises a perspective camera pointed at a 
        convex mirror, typically paraboloidal or conical.
        
        The elevation angle range is from :math:`-\pi/2` (below the mirror) to
        maxangle above the horizontal plane. The mapping from elevation angle
        :math:`\theta` to image plane radius is given by:

            =============   =======================================
            Projection      :math:`r(\theta)`
            =============   =======================================
            equiangular     :math:`r = k \theta`
            sine            :math:`r = k \sin \theta`
            equisolid       :math:`r = k \sin \frac{\theta}{2}`
            stereographic   :math:`r = k \tan \frac{\theta}{2}`
            =============   =======================================


        .. note::
            - If ``K`` is not specified it is computed such that the circular
              imaging region maximally fills the image plane.
            - This camera model assumes central projection, that is, the focal point
              is at z=0 and the image plane is at z=f.  The image is not inverted.

        :references: 
            - Robotics, Vision & Control for Python, Section 13.3.2, P. Corke, Springer 2023.

        :seealso: :class:`CameraBase` :class:`CentralCamera` :class:`FishEyeCamera`
            :class:`SphericalCamera`
        """

        super().__init__(camtype='catadioptric', **kwargs)

        self.projection = projection
    
        if k is None:
            r = np.min((self.imagesize - self.pp) * self.rho)

        # compute k if not specified, so that hemisphere fits into
        # image plane, requires maxangle being set

        if self.projection == 'equiangular':
            if k is None:
                if maxangle is not None:
                    k = r / (pi / 2 + maxangle)
                    self.maxangle = maxangle
                else:
                    raise ValueError('must specify either k or maxangle')
            rfunc = lambda theta: k * theta
        elif self.projection == 'sine':
            if k is None:
                k = r
            rfunc = lambda theta: k * np.sin(theta)
        elif self.projection == 'equisolid':
            if k is None:
                k = r / sin(pi / 4)
            rfunc = lambda theta: k * np.sin(theta / 2)
        elif self.projection == 'stereographic':
            if k is None:
                k = r / tan(pi/4)
            rfunc = lambda theta: k * np.tan(theta / 2)
        else:
            raise ValueError('unknown projection model')

        self.k = k
        self.rfunc = rfunc

    def __str__(self):
        s = super().__str__()
        s += self.fmt.format('model', self.projection, fmt="{}")
        s += self.fmt.format('k', self.k, fmt="{:.4g}")
        return s

    def project_point(self, P, pose=None, objpose=None):        
        """
        Project 3D points to image plane

        :param P: 3D world point or points
        :type P: array_like(3), array_like(3,n)
        :param pose: camera pose with respect to the world frame, defaults to
            camera's ``pose`` attribute
        :type pose: :class:`~spatialmath..pose3d.SE3`, optional
        :param objpose:  3D point reference frame, defaults to world frame
        :type objpose: :class:`~spatialmath..pose3d.SE3`, optional
        :param visibility: test if points are visible, default False
        :type visibility: bool
        :raises ValueError: [description]
        :return: image plane points
        :rtype: ndarray(2,n)

        Project world points to the catadioptric camera image plane.

        World points are given as a 1D array or the columns of a 2D array of
        Euclidean coordinates. The computed image plane coordinates are
        Euclidean and given as a 1D array or the corresponding columns of a 2D
        array.

        If ``pose`` is specified it is used for the camera pose instead of the
        attribute ``pose``.  The object's attribute is not updated.

        The points ``P`` are by default with respect to the world frame, but 
        they can be transformed by specifying ``objpose``.

        :seealso: :meth:`plot_point`
        """
        P = base.getmatrix(P, (3, None))

        if pose is not None:
            T = self.pose.inv()
        else:
            T = SE3()
        if objpose is not None:
            T *= objpose

        P = T * P  # transform points to camera frame
    
        # project to the image plane
        R = np.sqrt(np.sum(P ** 2, axis=0))
        phi = np.arctan2(P[1, :], P[0, :])
        theta = np.arccos(P[2, :] / R)
        
        r = self.rfunc(theta)  # depends on projection model
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        
        uv = np.array([x / self.rhou + self.u0, y / self.rhov + self.v0])
        
        return self._add_noise_distortion(uv)

# ------------------------------------------------------------------------ #
class SphericalCamera(CameraBase):
    """
    .. inheritance-diagram:: machinevisiontoolbox.Camera.SphericalCamera
        :top-classes: machinevisiontoolbox.Camera.Camera
        :parts: 1
    """    
        
    def __init__(self, **kwargs):
        """
        Create spherical camera projection model

        :param kwargs: arguments passed to :class:`CameraBase` constructor

        The spherical camera is an idealization with a complete field of view
        that can be used to generalize all camera projection models.

        :references: 
            - Robotics, Vision & Control for Python, Section 13.3.3, P. Corke, Springer 2023.

        :seealso: :class:`CameraBase` :class:`CentralCamera` :class:`CatadioptricCamera`
            :class:`FishEyeCamera` 
        """
        # invoke the superclass constructor
        super().__init__(camtype='spherical', 
            limits=[-pi,pi,0,pi],
            labels=['Longitude φ (rad)', 'Colatitude θ (rad)'],
            **kwargs)

    # return field-of-view angle for x and y direction (rad)
    def fov(self):
        """
        Camera field-of-view angles

        :return: field of view angles in radians
        :rtype: ndarray(2)
        
        Computes the field of view angles (2x1) in radians for the camera
        horizontal and vertical directions.
        """
        return [2 * pi, 2 * pi]
    
    def project_point(self, P, pose=None, objpose=None):
        r"""
        Project 3D points to image plane

        :param P: 3D world point or points
        :type P: array_like(3), array_like(3,n)
        :param pose: camera pose with respect to the world frame, defaults to
            camera's ``pose`` attribute
        :type pose: :class:`~spatialmath..pose3d.SE3`, optional
        :param objpose:  3D point reference frame, defaults to world frame
        :type objpose: :class:`~spatialmath..pose3d.SE3`, optional
        :return: image plane points
        :rtype: ndarray(2,n)

        Project world points to the spherical camera image plane.
        
        World points are given as a 1D array or the columns of a 2D array of
        Euclidean coordinates. The computed image plane coordinates are
        in polar form :math:`(\phi, \theta)` (longitude, colatitude),
        and given as a 1D array or the corresponding columns of a 2D
        array.

        If ``pose`` is specified it is used for the camera pose instead of the
        attribute ``pose``.  The object's attribute is not updated.

        The points ``P`` are by default with respect to the world frame, but 
        they can be transformed by specifying ``objpose``.

        :seealso: :meth:`plot_point`
        """
        P = base.getmatrix(P, (3, None))

        if pose is None:
            pose = self.pose

        pose = pose.inv()

        if objpose is not None:
            pose *= objpose

        P = pose * P         # transform points to camera frame
        
        R = np.linalg.norm(P, axis=0)
        x = P[0, :] / R
        y = P[1, :] / R
        z = P[2, :] / R

        phi = np.arctan2(y, x)
        theta = np.arccos(z)
        return np.array([phi, theta])

    def visjac_p(self, p, depth=None):
        r"""
        Visual Jacobian for point features

        :param p: image plane points
        :type p: array_like(2) or ndarray(2,N)
        :param depth: point depth, defaults to None
        :type depth: float or array_like(N), optional
        :return: visual Jacobian
        :rtype: ndarray(2,6) or ndarray(2N,6)

        Compute the image Jacobian :math:`\mat{J}` which maps

        .. math::

            \dvec{p} = \mat{J}(\vec{p}, z) \vec{\nu}

        camera spatial velocity :math:`\vec{\nu}` to the image plane velocity
        :math:`\dvec{p}` of the point where :math:`\vec{p}=(\phi, \theta)`

        If ``p`` describes multiple points then return a stack of these 
        :math:`2\times 6` matrices, one per point.
        
        Depth is the z-component of the point's coordinate in the camera frame.
        If ``depth`` is a scalar then it is the depth for all points. 
        """

        J = []
        if smbase.isscalar(depth):
            depth = [depth] * p.shape[1]

        for (phi, theta), R in zip(p.T, depth):
            sp = np.sin(phi)
            cp = np.cos(phi)
            st = np.sin(theta)
            ct = np.cos(theta)

            Jk = np.array([
                [sp/R/st, -cp/R/st, 0, cp*ct/st, sp*ct/st, -1],
                [-cp*ct/R, -sp*ct/R, st/R, sp, -cp, 0]
            ])
            J.append(Jk)
        return np.vstack(J)

    def plot(self, frame=False, **kwargs):
        smbase.plot_sphere(radius=1, filled=True, color='lightyellow', resolution=20, centre=self.pose.t)
        self.pose.plot(style='arrow', axislabel=True, length=1.4)



# class CentralCamera_polar(Camera):
#     """
#     Central projection camera class
#     """

#     def __init__(self,
#                  f=1,
#                  distortion=None,
#                  **kwargs):
#         """
#         Create central camera projection model in polar coordinates

#         :param f: focal length, defaults to 8*1e-3
#         :type f: float, optional
#         :param distortion: camera distortion parameters, defaults to None
#         :type distortion: array_like(5), optional

#         :seealso: :meth:`distort`
#         """

#         super().__init__(type='perspective', **kwargs)
#         # TODO some of this logic to f and pp setters
#         self.f = f

#         self._distortion = distortion

#     @classmethod
#     def Default(cls, **kwargs):
#         default = {
#             'f': 0.008, 
#             'rho': 10e-6,
#             'imagesize': 1000, 
#             'pp': (500,500),
#             'name': 'default perspective camera'
#         }

#         return CentralCamera_polar(**{**default, **kwargs})
        
#     def __str__(self):
#         s = super().__str__()
#         s += self.fmt.format('principal pt', self.pp)
#         s += self.fmt.format('focal length', self.f)

#         return s


#     def project_point(self, P, pose=None, objpose=None, **kwargs):
#         r"""
#         Project 3D points to image plane

#         :param P: 3D points to project into camera image plane
#         :type P: array_like(3), array_like(3,n)
#         :param pose: camera pose with respect to the world frame, defaults to
#             camera's ``pose`` attribute
#         :type pose: SE3, optional
#         :param objpose:  3D point reference frame, defaults to world frame
#         :type objpose: SE3, optional
#         :param visibility: test if points are visible, default False
#         :type visibility: bool
#         :param retinal: transform to retinal coordinates, default False
#         :type retinal: bool, optional
#         :return: image plane points
#         :rtype: ndarray(2,n)

#         Project a 3D point to the image plane

#         .. math::

#             \hvec{p} = \mat{C} \hvec{P}

#         where :math:`\mat{C}` is the camera calibration matrix and :math:`\hvec{p}` and :math:`\hvec{P}`
#         are the image plane and world frame coordinates respectively.

#         Example:

#         .. runblock:: pycon

#             >>> from machinevisiontoolbox import CentralCamera
#             >>> camera = CentralCamera()
#             >>> camera.project_point((0.3, 0.4, 2))

#         If ``pose`` is specified it is used for the camera frame pose, otherwise
#         the attribute ``pose``.  The object's ``pose`` attribute is not updated
#         if ``pose`` is specified.

#         A single point can be specified as a 3-vector, multiple points as an
#         array with three rows and one column (x, y, z) per point.

#         The points ``P`` are by default with respect to the world frame, but 
#         they can be transformed by specifying ``objpose``.
        
#         If world points are behind the camera, the image plane points are set to
#         NaN.
        
#         if ``visibility`` is True then each projected point is checked to ensure
#         it lies in the bounds of the image plane.  In this case there are two
#         return values: the image plane coordinates and an array of booleans
#         indicating if the corresponding point is visible.
#         """
#         if pose is None:
#             pose = self.pose

#         C = self.C(pose, retinal=retinal)

#         if isinstance(P, np.ndarray):
#             if P.ndim == 1:
#                 P = P.reshape((-1, 1))  # make it a column
#         else:
#             P = base.getvector(P, out='col')

#         # make it homogeneous if not already
#         if P.shape[0] == 3:
#             P = base.e2h(P)

#         # project 3D points

#         if objpose is not None:
#             P = objpose.A @ P

#         x = C @ P

#         if behind:
#             x[2, x[2, :] < 0] = np.nan  # points behind the camera are set to NaN

#         x = base.h2e(x)

#         # add Gaussian noise and distortion
#         x = self.add_noise_distortion(x)

#         #  do visibility check if required
#         if visibility:
#             visible = ~np.isnan(x[0,:]) \
#                 & (x[0, :] >= 0) \
#                 & (x[1, :] >= 0) \
#                 & (x[0, :] < self.nu) \
#                 & (x[1, :] < self.nv)
            
#             return x, visible
#         else:
#             return x

#     def plot_point(self, Pp
#         ax = _newplot(self, fig, ax)

#         if self._image is not None:
#             # if camera has an image, display said image
#             idisp(self._image,
#                       fig=fig,
#                       ax=ax,
#                       title=self._name,
#                       drawonly=True)
#         else:
#             if self.limits is None:
#                 ax.set_xlim(0, self.nu)
#                 ax.set_ylim(0, self.nv)
#             else:
#                 ax.set_xlim(self.limits[0], self.limits[1])
#                 ax.set_ylim(self.limits[2], self.limits[3])
#             ax.autoscale(False)
#             ax.set_aspect('equal')
#             ax.invert_yaxis()
#             ax.grid(True)
#             if self.labels is None:
#                 ax.set_xlabel('u (pixels)')
#                 ax.set_ylabel('v (pixels)')
#             else:
#                 ax.set_xlabel(self.labels[0])
#                 ax.set_ylabel(self.labels[1])
#             ax.set_title(self.name)
#             ax.set_facecolor('lightyellow')
#             ax.figure.canvas.set_window_title('Machine Vision Toolbox for Python')

#         # TODO figure out axes ticks, etc
#         return ax  # likely this return is not necessary

if __name__ == "__main__":
    from spatialmath import UnitQuaternion

    # im1 = Image.Read("eiffel2-1.png", grey=True)
    # camera = CentralCamera();
    # camera.disp(im1);


    # cam = CentralCamera(f=0.08)
    # print(cam)
    # P = [0.1, 0.2, 3]
    # print(cam.project_point(P))

    cam = CentralCamera(f=0.08, imagesize=1000, rho=10e-6)
    print(cam)

    cam.project_point([1,2,3])

    # P = np.array([[0, 10], [0, 10], [10, 10]])
    # p, visible = cam.project_point(P, visibility=True)
    # visible

    # P = [0.1, 0.2, 3]
    # print(cam.project_point(P))


    # T1 = SE3(-0.1, 0, 0) * SE3.Ry(0.4);
    # camera1 = CentralCamera(name="camera 1", f=0.002, imagesize=1000, rho=10e-6, pose=T1)
    # # print(camera1)

    # camera1.decomposeH(np.eye(3,3))


    # L = Line3.TwoPoints([0, 0, 1], [1, 1, 1])
    # camera = CentralCamera.Default();
    # l = camera.project_line(L)
    # camera.plot_line3(L)


    # x = np.r_[cam.pose.t, UnitQuaternion(cam.pose).vec3]
    # print(x)
    # p, JA, JB = cam.derivatives(x, P)
    # print(p)
    # print(cam.project_point(P))
    # print(JA)
    # print(JB)

    # smbase.plotvol3(2)

    # cam.plot_camera(scale=0.5, shape='camera', T=SE3.Ry(np.pi/2))

    # plt.show(block=True)
    # print(cam)
    # # cam.pose = SE3([0.1, 0.2, 0.3])
    # print(cam.pose)
    # # fig, ax = c.plot_camera(frustum=True)
    # # plt.show()
    # np.set_printoptions(linewidth=120, formatter={'float': lambda x: f"{x:8.4g}" if abs(x) > 1e-10 else f"{0:8.4g}"})


    # print(cam.project([1,2,3]))

    # print(cam.visjac_p((300,300), 1))
    # cam.flowfield([0,0,0, 0,0,1])
    # # fundamental matrix
    # # create +8 world points (20 in this case)
    # nx, ny = (4, 5)
    # depth = 3
    # x = np.linspace(-1, 1, nx)
    # y = np.linspace(-1, 1, ny)
    # X, Y = np.meshgrid(x, y)
    # Z = depth * np.ones(X.shape)
    # P = np.dstack((X, Y, Z))
    # PC = np.ravel(P, order='C')
    # PW = np.reshape(PC, (3, nx * ny), order='F')

    # # create projections from pose 1:
    # print(c.T)
    # p1 = c.project(PW)  # p1 wrt c's T
    # print(p1)
    # c.plot(PW)

    # # define pose 2:
    # T2 = SE3([0.4, 0.2, 0.3])  # just pure x-translation
    # p2 = c.project(PW, T2)
    # print(p2)
    # c.plot(p2)

    # # convert p1, p2 into lists of points?
    # p1 = np.float32(np.transpose(p1))
    # p2 = np.float32(np.transpose(p2))
    # F = c.FfromPoints(p1,
    #                   p2,
    #                   method='8p',
    #                   ransacThresh=3,
    #                   confidence=0.99,
    #                   maxiters=10)

    # # to check F:
    # p1h = e2h(p1.T)
    # p2h = e2h(p2.T)
    # pfp = [p2h[:, i].T @ F @ p1h[:, i] for i in range(p1h.shape[1])]
    # # [print(pfpi) for pfpi in pfp]
    # for pfpi in pfp:
    #     print(pfpi)
    # # should be all close to zero, which they are!

    # # essential matrix from points:
    # E = c.EfromPoints(p1, p2, c.C)

    # # TODO verify E

    # import code
    # code.interact(local=dict(globals(), **locals()))
