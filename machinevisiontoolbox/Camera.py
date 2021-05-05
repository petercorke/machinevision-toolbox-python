#!/usr/bin/env python
"""
Camera class
@author: Dorian Tsai
@author: Peter Corke
"""

import numpy as np
import cv2 as cv
from math import pi, sqrt
from spatialmath import base, Plucker
import machinevisiontoolbox as mvt
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import numpy as np
import scipy

from machinevisiontoolbox.Image import Image
# import CameraVisualizer as CamVis

# from mpl_toolkits.mplot3d import Axes3D, art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# from collections import namedtuple
from spatialmath import SE3
# import spatialmath.base as tr

class Camera(ABC):

    # list of attributes
    _name = []      # camera  name (string)
    _camtype = []   # camera type (string)

    _imagesize = None        # number of pixels (horizontal, vertical)
    _pp = None        # principal point (horizontal, vertical)
    _rhou = []      # pixel imagesize (single pixel) horizontal
    _rhov = []      # pixel imagesize (single pixel) vertical
    _image = []     # image (TODO image class?), for now, just numpy array

    _T = []         # camera pose (homogeneous transform, SE3 class)

    _fig = []       # for plotting, figure handle/object reference
    _ax = []        # for plotting, axes handle

    def __init__(self,
                 name=None,
                 camtype='central',
                 rho=10e-6,
                 imagesize=(1024, 1024),
                 pp=None,
                 noise=None,
                 pose=None,
                 limits=None,
                 labels=None):
        """
        Create instance of a Camera class
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

        rho = base.getvector(rho)
        if len(rho) == 1:
            self._rhou = rho[0]
            self._rhov = rho[0]
        elif len(rho) == 2:
            self._rhou = rho[0]
            self._rhov = rho[1]
        else:
            raise ValueError(rho, 'rho must be a 1- or 2-element vector')

        self._pp = pp

        self.imagesize = imagesize


        if pose is None:
            self._pose = SE3()
        else:
            self._pose = SE3(pose)

        self._noise = noise

        self._image = None

        self._fig = None
        self._ax = None

        self._noise = None
        self._distortion = None
        self.labels = labels
        self.limits = limits


    def __str__(self):
        s = ''
        self.fmt = '{:>15s}: {}\n'
        s += self.fmt.format('Name', self.name + ' [' + self.__class__.__name__ + ']')
        s += self.fmt.format('pixel size', ' x '.join([str(x) for x in self.rho]))
        s += self.fmt.format('image size', ' x '.join([str(x) for x in self.imagesize]))
        s += self.fmt.format('pose', self.pose.printline(file=None, fmt="{:.3g}"))
        return s

    def __repr__(self):
        return str(self)
        
    @abstractmethod
    def project(self, P, **kwargs):
        pass

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, newname):
        if isinstance(newname, str):
            self._name = newname
        else:
            raise TypeError(newname, 'name must be a string')

    @property
    def camtype(self):
        return self._camtype

    @camtype.setter
    def camtype(self, newcamtype):
        if isinstance(newcamtype, str):
            self._camtype = newcamtype
        else:
            raise TypeError(newcamtype, 'camtype must be a string')

    @property
    def imagesize(self):
        return self._imagesize

    @imagesize.setter
    def imagesize(self, npix):
        npix = base.getvector(npix)
        if len(npix) == 1:
            self._imagesize = np.r[npix[0], npix[0]]
        elif len(npix) == 2:
            self._imagesize = npix
        else:
            raise ValueError(
                imagesize, 'imagesize must be a 1- or 2-element vector')
        if self._pp is None:
            self._pp = self._imagesize / 2

    @property
    def nu(self):
        return self._imagesize[0]

    @property
    def nv(self):
        return self._imagesize[1]

    @property
    def width(self):
        return self._imagesize[0]

    @property
    def height(self):
        return self._imagesize[1]

    @property
    def pp(self):
        """
        Get principal point coordinate

        :return: principal point
        :rtype: 2-tuple

        :seealso: :func:`u0`, :func:`v0`
        """
        return self._pp

    @pp.setter
    def pp(self, pp):
        """
        Set principal point coordinate

        :param pp: principal point
        :type pp: array_like(2)

        :seealso: :func:`u0`, :func:`v0`
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
        Get principal point: horizontal coordinate

        :return: horizontal component of principal point
        :rtype: float

        :seealso: :func:`v0`, :func:`pp`
        """
        return self._pp[0]

    @property
    def v0(self):
        """
        Get principal point: vertical coordinate

        :return: vertical component of principal point
        :rtype: float

        :seealso: :func:`u0`, :func:`pp`
        """
        return self._pp[1]

    @property
    def rhou(self):
        """
        Get pixel size: horizontal value

        :return: horizontal pixel size
        :rtype: float

        :seealso: :func:`rhov`, :func:`rho`
        """
        return self._rhou

    # this is generally the centre of the image, has special meaning for
    # perspective camera
    
    @property
    def rhov(self):
        """
        Get pixel size: horizontal value

        :return: horizontal pixel size
        :rtype: float

        :seealso: :func:`rhov`, :func:`rho`
        """
        return self._rhov

    @property
    def rho(self):
        """
        Get pixel dimensions

        :return: horizontal pixel size
        :rtype: float

        :seealso: :func:`rhou`, :func:`rhov`
        """

        return np.array([self._rhou, self._rhov])

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, newimage):
        self._image = Image(newimage)

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, newpose):
        self._pose = SE3(newpose)

    @property
    def noise(self):
        return self._noise

    def fov(self):
        """
        Camera field-of-view angles.
        
        A = C.fov() are the field of view angles (2x1) in radians for the camera x and y
        (horizontal and vertical) directions.
        """
        try:
            return 2 * np.arctan(np.r_[self.imagesize] / 2 * np.r_[self.rho] / self.f)
        except:
            raise ValueError('imagesize or rho properties not set');

    def plotcreate(self, fig=None, ax=None):
        """
        Create plot for camera image plane
        """

        if self._ax is not None:
            return

        if (fig is None) and (ax is None):
            # create our own handle for the figure/plot
            print('creating new figure and axes for camera')
            fig, ax = plt.subplots()  # TODO *args, **kwargs?
        # TODO elif ax is a plot handle, else raise ValueError
        # else:

        if self._image is not None:
            # if camera has an image, display said image
            mvt.idisp(self._image,
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
            ax.figure.canvas.set_window_title('Machine Vision Toolbox for Python')

        # TODO figure out axes ticks, etc
        self._fig = fig
        self._ax = ax
        return fig, ax  # likely this return is not necessary

    def clf(self):
        for artist in self._ax.get_children():
            try:
                artist.remove()
            except:
                pass

    def plot(self, p=None, marker='or', markersize=6, **kwargs):
        """
        Plot points on image plane
        If 3D points, then 3D world points
        If 2D points, then assumed image plane points
        TODO plucker coordinates/lines?
        TODO returns 2d points
        """
        self.plotcreate()

        if p.shape[0] == 3:
            # plot world points
            p = self.project(p, **kwargs)

        elif p.shape[0] != 2:
            raise ValueError('p must have be (2,), (3,), (2,n), (3,n)')

        # TODO plot ip on image plane given self._fig and self._ax
        # TODO accept kwargs for the plotting

        self._ax.plot(p[0, :], p[1, :], marker, markersize=markersize)
        plt.show()

        return p

    def mesh(self, X, Y, Z, objpose=None, pose=None, **kwargs):
        """
        Plot points on image plane
        If 3D points, then 3D world points
        If 2D points, then assumed image plane points
        TODO plucker coordinates/lines?
        """
        # self.plotcreate()
        # # TODO plot ip on image plane given self._fig and self._ax
        # # TODO accept kwargs for the plotting

        # self._ax.plot_surface(X, Y, Z)
        # plt.show()

        #Camera.mesh Plot mesh object on image plane
        #
        # C.mesh(X, Y, Z, OPTIONS) projects a 3D shape defined by the matrices X, Y, Z
        # to the image plane and plots them.  The matrices X, Y, Z are of the same size
        # and the corresponding elements of the matrices define 3D points.
        #
        # Options::
        # 'objpose',T   Transform all points by the homogeneous transformation T before
        #               projecting them to the camera image plane.
        # 'pose',T      Set the camera pose to the homogeneous transformation T before
        #               projecting points to the camera image plane.  Temporarily overrides
        #               the current camera pose C.T.
        #
        # Additional arguments are passed to plot as line style parameters.
        #
        # See also MESH, CYLINDER, SPHERE, MKCUBE, Camera.plot, Camera.hold, Camera.clf.
        
        # check that mesh matrices conform
        if X.shape != Y.shape or X.shape != Z.shape:
            raise ValueError('matrices must be the same size')
        
        if pose is None:
            pose = self.pose
        if objpose is not None:
            pose = objpose.inv() * pose
        
        # get handle for this camera image plane
        self.plotcreate()
        plt.autoscale(False)
        
        # draw 3D line segments
        nsteps = 21
        s = np.linspace(0, 1, nsteps)

        # c.clf
        # holdon = c.hold(1);
        
        for i in range(X.shape[0]-1):      #i=1:numrows(X)-1
            for j in range(X.shape[1]-1):  # j=1:numcols(X)-1
                P0 = np.r_[X[i, j], Y[i, j], Z[i, j]]
                P1 = np.r_[X[i+1, j], Y[i+1, j], Z[i+1, j]]
                P2 = np.r_[X[i, j+1], Y[i, j+1], Z[i, j+1]]
                
                if self.camtype == 'perspective':
                    # straight world lines are straight on the image plane
                    uv = self.project(np.c_[P0, P1], pose=pose)
                else:
                    # straight world lines are not straight, plot them piecewise
                    P = (1 - s) * P0[:, np.newaxis] + s * P1[:, np.newaxis]
                    uv = self.project(P, pose=pose)

                self._ax.plot(uv[0, :], uv[1, :], **kwargs)
                
                if self.camtype == 'perspective':
                    # straight world lines are straight on the image plane
                    uv = self.project(np.c_[P0, P2], pose=pose)
                else:
                    # straight world lines are not straight, plot them piecewise
                    P = (1 - s) * P0[:, np.newaxis] + s * P2[:, np.newaxis]
                    uv = self.project(P, pose=pose)

                self._ax.plot(uv[0, :], uv[1, :], **kwargs)

        
        for j in range(X.shape[1]-1):  # j=1:numcols(X)-1
            P0 = [X[-1,j],   Y[-1,j],   Z[-1,j]]
            P1 = [X[-1,j+1], Y[-1,j+1], Z[-1,j+1]]
            
            # if c.perspective
                # straight world lines are straight on the image plane
            uv = self.project(np.c_[P0, P1], pose=pose);
            # else
            #     # straight world lines are not straight, plot them piecewise
            #     P = bsxfun(@times, (1-s), P0) + bsxfun(@times, s, P1);
            #     uv = c.project(P, 'setopt', opt);
            self._ax.plot(uv[0,:], uv[1,:], **kwargs)
        
        # c.hold(holdon); # turn hold off if it was initially off

        plt.draw()

    def plot_camera(self,
                    T=None,
                    scale=1,
                    shape='frustum',
                    label=True,
                    persist=False,
                    fig=None,
                    ax=None):
        """
        Display camera icon in world view
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

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # draw camera-like object:
        if shape == 'cube':
            # for now, just draw a cube
            # TODO change those points based on pose
            # self.T or input T
            pcube = np.array([[-1, -1, -1],
                              [1, -1, -1],
                              [1, 1, -1],
                              [-1, 1, -1],
                              [-1, -1, 1],
                              [1, -1, 1],
                              [1, 1, 1],
                              [-1, 1, 1]])
            ax.scatter3D(pcube[:, 0], pcube[:, 1], pcube[:, 2])

        elif shape == 'frustum':
            # TODO make this kwargs or optional inputs
            camfrustum = CameraVisualizer(self,
                                          length=scale,
                                          widthb=scale/10,
                                          widtht=scale)
            camfrustumpoly = Poly3DCollection(camfrustum.gen_frustrum_poly(),
                                              facecolors=['r', 'g', 'r', 'y'])
            ax.add_collection3d(camfrustumpoly)

        #  https://stackoverflow.com/questions/33540109/plot-surfaces-on-a-cube
        if label:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

        return fig, ax

    @classmethod
    def plotfrustum(cls,
                    f=0.1,
                    fbwidth=0.05,
                    ftwidth=0.1,
                    fig=None,
                    ax=None):
        """
        Plot camera frustum
        """
        if (fig is None) and (ax is None):
            # create our own handle for the figure/plot
            print('creating new figure and axes for camera')
            fig = plt.figure()
            ax = fig.gca(projection='3d')

        return fig, ax


    def printCameraAttributes(self):
        """
        Print (internal) camera class attributes
        TODO should change this to print relevant camera parameters
        """
        attributes = vars(self)
        for key in attributes:
            print(key, ': \t', attributes[key])

    def common(self, uv):
        # distort the pixels
        
        # add Gaussian noise with specified standard deviation
        if self.noise is not None:
            uv += np.random.normal(0.0, self.noise, size=uv.shape)
        return uv 

class CentralCamera(Camera):
    """
    A (central projection) camera class
    """

    # list of attributes
    _fu = []        # focal length horizontal [units]
    _fv = []        # focal length vertical [units]

    def __init__(self,
                 f=8*1e-3,
                 distortion=None,
                 **kwargs):
        """
        Create instance of a Camera class
        """

        super().__init__(camtype='perspective', **kwargs)
        # TODO some of this logic to f and pp setters
        self.f = f

        self._distortion = distortion


    def __str__(self):
        s = super().__str__()
        s += self.fmt.format('principal pt', self.pp)
        s += self.fmt.format('focal length', self.f)

        return s


    def project(self, P, pose=None, objpose=None, visibility=False):
        """
        Project world points to image plane

        :param P: 3D points to project into camera image plane
        :type P: array_like(3), array_like(3,n)
        :param pose: camera pose with respect to the world frame, defaults to
            camera's ``pose`` attribute
        :type pose: SE3, optional
        :param objpose:  3D point reference frame, defaults to world frame
        :type objpose: SE3, optional
        :param visibility: test if points are visible, default False
        :type visibility: bool
        :raises ValueError: [description]
        :return: image plane points
        :rtype: ndarray(2,n)

        If ``pose`` is specified it is used for the camera pose instead of the
        attribute ``pose``.  The objects attribute is not updated.

        The points ``P`` are by default with respect to the world frame, but 
        they can be transformed 
        
        If points are behind the camera, the image plane points are set to
        NaN.
        
        if ``visibility`` is True then check whether the projected point lies in
        the bounds of the image plane.  Return two values: the image plane
        coordinates and an array of booleans indicating if the corresponding
        point is visible.

        If ``P`` is a Plucker object, then each value is projected into a 
        2D line in homogeneous form :math:`p[0] u + p[1] v + p[2] = 0`.
        """

        if pose is None:
            pose = self.pose

        C = self.getC(pose)

        if isinstance(P, Plucker):
            # project Plucker lines

            x = np.empty(shape=(3, 0))
            for p in P:
                l = base.vex( C @ p.skew @ C.T)
                x = np.c_[x, l / np.max(np.abs(l))]  # normalize by largest element
            return x

        else:
            if isinstance(P, np.ndarray):
                if P.ndim == 1:
                    P = P.reshape((-1, 1))  # make it a column
            else:
                P = base.getvector(P, out='col')

            # make it homogeneous if not already
            if P.shape[0] == 3:
                P = base.e2h(P)

            # project 3D points

            if objpose is not None:
                P = objpose.A @ P

            x = C @ P

            # x[2, x[2, :]<0] = np.nan  # points behind the camera are set to NaN

            x = base.h2e(x)

            if self._distortion is not None:
                x = self._distort(x)
            
            if self._noise is not None:
                # add Gaussian noise with specified standard deviation
                x += np.diag(self._noise) * np.random.randn(x.shape)

            #  do visibility check if required
            if visibility:
                visible = ~np.isnan(x[0,:]) \
                    & (x[0,:] >= 0) \
                    & (x[1,:] >= 0) \
                    & (x[0,:] < self.nu) \
                    & (x[1,:] < self.nv)
                
                return x, visibility
            else:
                return x


    def _distort(self, X):
        """
        CentralCamera.distort Compute distorted coordinate
        
        Xd = cam.distort(X) is the projected image plane point X (2x1) after 
        lens distortion has been applied.
        """
        
        # convert to normalized image coordinates
        X = np.linalg.inv(self.K) * X

        # unpack coordinates
        u = X[0, :]
        v = X[1, :]

        # unpack distortion vector
        k = c.distortion[:3]
        p = c.distortion[3:]

        r = np.sqrt(u ** 2 + v ** 2) # distance from principal point
        
        # compute the shift due to distortion
        delta_u = u * (k[0] * r ** 2 + k[1] * r ** 4 + k[2] * r ** 6) + \
            2 * p[0] * u * v + p[1] * (r ** 2 + 2 * u ** 2)
        delta_v = v  * (k[0] * r ** 2 + k[1] * r ** 4 + k[2] * r ** 6) + \
            p[0] * (r ** 2 + 2 * v ** 2) + 2  *p[1] * u * v
        
        # distorted coordinates
        ud = u + delta_u
        vd = v + delta_v
        
        return self.K * e2h( np.r_[ud, vd] ) # convert to pixel coords



    @property
    def fu(self):
        """
        Get focal length in horizontal direction

        :return: focal length in horizontal direction
        :rtype: 2-tuple

        :seealso: :func:`fv`, :func:`f`
        """
        return self._fu

    @property
    def fv(self):
        """
        Get focal length in vertical direction

        :return: focal length in horizontal direction
        :rtype: 2-tuple

        :seealso: :func:`fu`, :func:`f`
        """
        return self._fv

    @property
    def f(self):
        """
        Get focal length

        :return: focal length
        :rtype: 2-tuple

        :seealso: :func:`fu`, :func:`fv`
        """
        return (self._fu, self._fv)

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
    def K(self):
        """
        Intrinsic matrix of camera
        """
        # fmt: off
        K = np.array([[self.fu / self.rhou, 0,                   self.u0],
                      [ 0,                  self.fv / self.rhov, self.v0],
                      [ 0,                  0,                    1]
                      ], dtype=np.float)
        # fmt: on
        return K

    @property
    def C(self):
        """
        Camera matrix, camera calibration or projection matrix
        """
        P0 = np.eye(3, 4)
        return self.K @ P0 @ self.pose.inv().A

    def getC(self, T=None):
        """
        Get Camera matrix, camera calibration or projection matrix
        """

        P0 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]], dtype=np.float)

        if T is None:
            C = self.K @ P0 @ self.pose.inv().A
        else:
            C = self.K @ P0 @ T.inv().A
        return C

    def H(self, T, N, d):
        """
        Homography matrix

        ``H(T, N, d)`` is the (3, 3) homography matrix for the camera observing
        the plane with normal ``N`` and at distance ``d`` from two viewpoints.
        The first view is from the current camera pose (self.T), and the second
        is after a relative motion represented by the homogeneous
        transformation ``T``
        """

        if d < 0:
            raise ValueError(d, 'plane distance d must be > 0')

        N = base.getvector(N)
        if N[2] < 0:
            raise ValueError(N, 'normal must be away from camera (N[2] >= 0)')

        # T transform view 1 to view 2
        T = SE3(T).inv()

        HH = T.R + 1.0 / d * T.t @ N  # need to ensure column then row = 3x3

        # apply camera intrinsics
        HH = self.K @ HH @ np.linalg.inv(self.K)

        return HH / HH[2, 2]  # normalised

    def invH(self, H, K=None, ):
        """
        Decompose homography matrix

        ``self.invH(H)`` decomposes the homography ``H`` (3,3) into the camerea
        motion and the normal to the plane. In practice, there are multiple
        solutions and the return ``S``  is a named tuple with elements
        ``S.T``, the camera motion as a homogeneous transform matrix (4,4), and
        translation not to scale, and ``S.N`` the normal vector to the plawne
        (3,3).  # TODO why is the normal vector a 3x3?
        """

        if K is None:
            K = np.identity(3)
            # also have K = self.K

        H = np.linalg.inv(K) @ H @ K

        # normalise so that the second singular value is one
        U, S, V = np.linalg.svd(H, compute_uv=True)
        H = H / S[1, 1]

        # compute the SVD of the symmetric matrix H'*H = VSV'
        U, S, V = np.linalg.svd(np.transpose(H) @ H)

        # ensure V is right-handed
        if np.linalg.det(V) < 0:
            print('det(V) was < 0')
            V = -V

        # get squared singular values
        s0 = S[0, 0]
        s2 = S[2, 2]

        # v0 = V[0:, 0]
        # v1 = V[0:, 1]
        # v2 = V[0:, 2]

        # pure rotation - where all singular values == 1
        if np.abs(s0 - s2) < (100 * np.spacing(1)):
            print('Warning: Homography due to pure rotation')
            if np.linalg.det(H) < 0:
                H = -H
            # sol = namedtuple('T', T, ''
        # TODO finish from invhomog.m
        print('Unfinished')
        return False


    @classmethod
    def FfromPoints(cls,
                    P1,
                    P2,
                    method,
                    ransacThresh,
                    confidence,
                    maxiters):
        """
        Compute fundamental matrix from two sets of corresponding image points
        see https://docs.opencv.org/master/d9/d0c/
        group__calib3d.html#gae850fad056e407befb9e2db04dd9e509
        """
        # TODO check valid input
        # need at least 7 pairs of points
        # TODO sort options in a user-friendly manner
        fopt = {'7p': cv.FM_7POINT,
                '8p': cv.FM_8POINT,
                'ransac': cv.FM_RANSAC,
                'lmeds': cv.FM_LMEDS}

        F, mask = cv.findFundamentalMat(P1, P2,
                                        method=fopt[method],
                                        ransacReprojThreshold=ransacThresh,
                                        confidence=confidence,
                                        maxIters=maxiters)
        # print('Fund mat = ', F)

        return F

    @classmethod
    def EfromPoints(cls,
                    P1,
                    P2,
                    camMat=None):
        """
        Compute essential matrix from two sets of corresponding image points
        TODO there are many more ways of computing E, but can tackle those
        later
        """
        # TODO check valid input
        # need at least 5 pairs of points
        # TODO sort options
        # if camMat is None:
        #    camMat = cls.C
        # TODO set default options, but user-configurable for method, prob,
        # threshold, etc

        # in the MVT we define C as a 3x4, but opencV just wants 3x3 fx, fy,
        # cx, cy, so simply cut off the 4th column
        if np.all(camMat.shape == (3, 4)):
            camMat = camMat[:, 0:3]

        E, mask = cv.findEssentialMat(P1, P2, cameraMatrix=camMat)
        # method=cv.RANSAC,
        #                               prob=0.999,
        #                               threshold=1.0
        print('Ess mat =', E)
        return E

    @classmethod
    def invcamcal(cls, C):

        def rq(S):
            # [R,Q] = vgg_rq(S)  Just like qr but the other way around.
            # If [R,Q] = vgg_rq(X), then R is upper-triangular, Q is orthogonal, and X==R*Q.
            # Moreover, if S is a real matrix, then det(Q)>0.
            # By awf

            S = S.T
            Q, U = scipy.linalg.qr(S[::-1, ::-1])
            Q = Q.T
            Q = Q[::-1, ::-1]
            U = U.T
            U = U[::-1, ::-1]

            if np.linalg.det(Q) < 0:
                U[:,0] = -U[:,0]
                Q[0,:] = -Q[0,:]

            return U, Q


        if C.shape != (3,4):
            raise ValueError('argument is not a 3x4 matrix')

        u, s, v = scipy.linalg.svd(C)

        t = v[3,:]  # not svd returns v transpose
        t = t / t[3]
        t = t[0:3]

        M = C[0:3,0:3]
        
        # M = K * R
        K, R = rq(M)

        # deal with K having negative elements on the diagonal
        
        # make a matrix to fix this, K*C has positive diagonal
        C = np.diag(np.sign(np.diag(K)));
        
        # now  K*R = (K*C) * (inv(C)*R), so we need to check C is a proper rotation
        # matrix.  If isn't then the situation is unfixable
        
        print(C)
        if np.linalg.det(C) != 1:
            raise ValueError('cannot correct signs in the intrinsic matrix')
        
        # all good, let's fix it
        K = K @ C
        R = C.T @ R
        
        # normalize K so that lower left is 1
        K = K / K[2,2]
        
        # pull out focal length and scale factors
        f = K[0,0]
        s = [1, K[1,1] / K[0,0]]
        T = SE3(t) * SE3.SO3(R.T)

        return cls(f=f, pp=K[0:2,2], rho=s, pose=T)

    @staticmethod
    def camcal(P, p):
        z4 = np.zeros((4,))

        A = np.empty(shape=(0,11))
        b = np.empty(shape=(0,))
        for uv, X in zip(p.T, P.T):
            u, v = uv
            row = np.array([
                    np.r_[ X, 1, z4, -u * X],
                    np.r_[z4, X,  1, -v * X]
                ])
            A = np.vstack((A, row))
            b = np.r_[b, uv]

        # solve Ax = b where c is 11 elements of camera matrix
        c, *_ = scipy.linalg.lstsq(A, b)

        # compute and print the residual
        r = A @ c - b
        print(f"residual is {np.linalg.norm(r):.3g} px")

        c = np.r_[c, 1]   # append a 1
        C = c.reshape((3,4))  # make a 3x4 matrix



        return C

    def visjac_p(self, uv, Z):
        '''
        Image Jacobian for point features (interaction matrix)
        
        Returns a 2Nx6 matrix of stacked Jacobians, one per image-plane point.
        uv is a 2xN matrix of image plane points
        Z  is the depth of the corresponding world points. Can be scalar, same distance to every
        point, or a vector or list of length N.
        
        References:
        * A tutorial on Visual Servo Control", Hutchinson, Hager & Corke, 
            IEEE Trans. R&A, Vol 12(5), Oct, 1996, pp 651-670.
        * Robotics, Vision & Control, Corke, Springer 2017, Chap 15.
        '''
        uv = base.getmatrix(uv, (2, None))

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
            Lp = K[:2,:2] @ np.array(
                [ [-1/z,  0,     x/z, x * y,      -(1 + x**2), y],
                  [ 0,   -1/z,   y/z, (1 + y**2), -x*y,       -x] ])

            # stack them vertically
            L = np.vstack([L, Lp])

        return L

    def flowfield(self, vel, Z=2):
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

        self.plotcreate()
        plt.quiver(U, V, du, dv, 0.4)
        plt.show(block=True)


    @staticmethod
    def camcald(XYZ, uv):
        """
        CAMCALD Camera calibration from data points


        :param XYZ: [description]
        :type XYZ: [type]
        :param uv: [description]
        :type uv: [type]

        C = CAMCALD(D) is the camera matrix (3x4) determined by least squares 
        from corresponding world and image-plane points.  D is a table 
        of points with rows of the form [X Y Z U V] where (X,Y,Z) is the 
        coordinate of a world point and [U,V] is the corresponding image 
        plane coordinate. 

        [C,E] = CAMCALD(D) as above but E is the maximum residual error after 
        back substitution [pixels]. 

        Notes:
        - This method assumes no lense distortion affecting the image plane
        coordinates.

        See also CentralCamera.
        """

        if XYZ.shape[1] != uv.shape[1]:
            raise ValueError('must have same number of world and image-plane points')

        n = XYZ.shape[1]
        if n < 6:
            raise ValueError('At least 6 points required for calibration')

        # build the matrix as per Ballard and Brown p.482

        # the row pair are one row at this point
        A = np.hstack((XYZ.T, np.ones((n,1)), np.zeros((n,4)), -uv[0, :, np.newaxis] * XYZ.T,
            np.zeros((n,4)), XYZ.T, np.ones((n,1)), -uv[1,:, np.newaxis] * XYZ.T))

        # reshape the matrix, so that the rows interleave
        A = A.reshape((n * 2, 11))

        if np.linalg.matrix_rank(A) < 11:
            raise ValueError('Rank deficient, perhaps points are coplanar or collinear');

        b = uv.reshape(-1, 1, order='F')

        x, resid, *_ = np.linalg.lstsq(A, b)  # least squares solution

        resid = sqrt(resid)
        if resid > 1:
            print('Residual greater than 1 pixel');
        print(f"maxm residual {resid:.3g} pixels")
        
        x = np.vstack((x, 1)).reshape((3, 4))

        return x, resid

    @classmethod
    def InvCamcal(cls, C):
        """
        Inverse camera calibration

        :param C: [description]
        :type C: [type]


            c = INVCAMCAL(C)

            Decompose, or invert, a 3x4camera calibration matrix C.
        The result is a camera object with the following parameters set:
            f
            sx, sy  (with sx=1)
            (u0, v0)  principal point
            Tcam is the homog xform of the world origin wrt camera

        Since only f.sx and f.sy can be estimated we set sx = 1.

        REF:	Multiple View Geometry, Hartley&Zisserman, p 163-164

        SEE ALSO: camera
        """

        if not C.shape == (3,4):
            raise ValueError('argument is not a 3x4 matrix')

        u, s, v = np.linalg.svd(C)
        v = v.T

        # determine camera position
        t = v[:, 3]  # last column
        t = t / t[3]
        t = t[:3]

        # determine camera orientation

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

        M = C[:3, :3]
        
        K, R = rq(M)

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
        return cls(name='invcamcal',
            f=f, pp=K[:2, 2], rho=s, pose=SE3.Rt(R.T, t))

    def estpose(self, P, p, method='iterative'):

        method_dict = {
            'iterative': cv.SOLVEPNP_ITERATIVE,
            'epnp': cv.SOLVEPNP_EPNP,
            'p3p': cv.SOLVEPNP_P3P,
            'ap3p': cv.SOLVEPNP_AP3P,
            'ippe': cv.SOLVEPNP_IPPE,
            'ippe-square': cv.SOLVEPNP_IPPE_SQUARE,
        }

        sol = cv.solvePnP(P.T, p.T, self.K, self._distortion, flags=method_dict[method])
        
        if sol[0]:
            return SE3(sol[2]) * SE3.EulerVec(sol[1])
        else:
            return None

# ----------------------------------------------------------------------------#
class CameraVisualizer:
    """
    Class for visualizer of a camera object. Used to generate frustrums in
    Matplotlib

        Constructor:
        `CamVisualizer(parameters)
            camera Camera object being visualized
            f_length  length of the frustrum
            fb_width  width of base of frustrum (camera centre end)
            ft_width  width of top of frustrum (lens end)
        Methods:
          gen_frustrum_poly()  return 4x4x3 matrix of points to create
          Poly3DCollection with Matplotlib
                           Order of sides created [top, right, bottom, left]
    """

    def __init__(self, camera, length=0.1, widthb=0.05, widtht=0.1):
        """
        Create instance of CamVisualizer class

        Required parameters: camera  Camera object being visualized (see
            common.py for Camera class)

        Optional parameters: f_length length of the displayed frustrum (0.1
            default) fb_width width of the base of displayed frustrum (camera
            centre end) (0.05 default) ft_width width of the top of displayed
            frustrum (lens end) (0.1 default)
        """
        self.camera = camera

        # Define corners of polygon in cameras frame (cf) in homogenous
        # coordinates b is base t is top rectangle

        widthb /= 2
        widtht /= 2
        self.b0 = np.array([-widthb, -widthb, 0, 1])
        self.b1 = np.array([-widthb, widthb, 0, 1])
        self.b2 = np.array([widthb, widthb, 0, 1])
        self.b3 = np.array([widthb, -widthb, 0, 1])
        self.t0 = np.array([-widtht, -widtht, length, 1])
        self.t1 = np.array([-widtht, widtht, length, 1])
        self.t2 = np.array([widtht, widtht, length, 1])
        self.t3 = np.array([widtht, -widtht, length, 1])

    def gen_frustrum_poly(self):

        # Transform frustrum points to world coordinate frame using the camera
        # extrinsics
        # T = self.camera.pose.A
        T = self.camera.pose.A

        # bottom/narrow end
        b0 = (T @ self.b0)[:-1]
        b1 = (T @ self.b1)[:-1]
        b2 = (T @ self.b2)[:-1]
        b3 = (T @ self.b3)[:-1]

        # wide/top end
        t0 = (T @ self.t0)[:-1]
        t1 = (T @ self.t1)[:-1]
        t2 = (T @ self.t2)[:-1]
        t3 = (T @ self.t3)[:-1]

        # Each set of four points is a single side of the Frustrum
        # points = np.array([[b0, b1, t1, t0], [b1, b2, t2, t1], [
        #                   b2, b3, t3, t2], [b3, b0, t0, t3]])
        points = [
            np.array([b0, b1, t1, t0]),  # -x face
            np.array([b1, b2, t2, t1]),  # +y face
            np.array([b2, b3, t3, t2]),  # +x face
            np.array([b3, b0, t0, t3])   # -y face
        ]
        return points




# ------------------------------------------------------------------------ #


class FishEyeCamera(Camera):

    """
    Fish eye camera class

    This camera model assumes central projection, that is, the focal point
    is at z=0 and the image plane is at z=f.  The image is not inverted.

    See also Camera.
    """

    def __init__(self, k=None, projection='equiangular', **kwargs):
        """
        Fisheye camera object

        :param k: [description], defaults to None
        :type k: [type], optional
        :param projection: [description], defaults to 'equiangular'
        :type projection: str, optional
        :raises ValueError: [description]

        Notes::
        - If K is not specified it is computed such that the circular imaging region
          maximally fills the square image plane.
        
        See also Camera, CentralCamera, CatadioptricCamera, SphericalCamera.
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
            
        
    def project(self, P, pose=None, objpose=None):
        #FishEyeCamera.project Project world points to image plane
        #
        # UV = C.project(P, OPTIONS) are the image plane coordinates for the world
        # points P.  The columns of P (3xN) are the world points and the columns
        # of UV (2xN) are the corresponding image plane points.
        #
        # Options::
        # 'pose',T         Set the camera pose to the pose T (homogeneous transformation (4x4) or SE3)
        #                  before projecting points to the camera image plane.  Temporarily overrides
        #                  the current camera pose C.T.
        # 'objpose',T      Transform all points by the pose T (homogeneous transformation (4x4) or SE3)
        #                  before projecting them to the camera image plane.
        #
        # See also CatadioprtricCamera.plot, Camera.plot.
        
        
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
        
        return self.common(uv)


# ------------------------------------------------------------------------ #

#CatadioptricCamera  Catadioptric camera class
"""
A concrete class for a catadioptric camera, subclass of Camera.

Methods::

project          project world points to image plane

plot             plot/return world point on image plane
hold             control hold for image plane
ishold           test figure hold for image plane
clf              clear image plane
figure           figure holding the image plane
mesh             draw shape represented as a mesh
point            draw homogeneous points on image plane
line             draw homogeneous lines on image plane
plot_camera      draw camera

rpy              set camera attitude
move             copy of Camera after motion
centre           get world coordinate of camera centre

delete           object destructor
char             convert camera parameters to string
display          display camera parameters

Properties (read/write)::
npix         image dimensions in pixels (2x1)
pp           intrinsic: principal point (2x1)
rho          intrinsic: pixel dimensions (2x1) [metres]
f            intrinsic: focal length [metres]
p            intrinsic: tangential distortion parameters
T            extrinsic: camera pose as homogeneous transformation

Properties (read only)::
nu    number of pixels in u-direction
nv    number of pixels in v-direction
u0    principal point u-coordinate
v0    principal point v-coordinate

Notes::
 - Camera is a reference object.
 - Camera objects can be used in vectors and arrays

See also CentralCamera, Camera.
"""


# TODO:
#   make a parent imaging class and subclass perspective, fisheye, panocam
#   test for points in front of camera and set to NaN if not
#   test for points off the image plane and set to NaN if not
#     make clipping/test flags

class CatadioptricCamera(Camera):
    
    def __init__(self, k=None, projection='equiangular', maxangle=None, **kwargs):
        """
        Fisheye camera object

        :param k: [description], defaults to None
        :type k: [type], optional
        :param projection: [description], defaults to 'equiangular'
        :type projection: str, optional
        :raises ValueError: [description]

        Notes::
        - If K is not specified it is computed such that the circular imaging region
          maximally fills the square image plane.

                  Notes::
        - The elevation angle range is from -pi/2 (below the mirror) to
          maxangle above the horizontal plane.
        
        See also Camera, FisheyeCamera, CatadioptricCamera, SphericalCamera.
        
        See also Camera, CentralCamera, CatadioptricCamera, SphericalCamera.
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
        
    # return field-of-view angle for x and y direction (rad)
    def fov(self):
        return 2 * arctan(self.imagesize / 2 * self.s / self.f)

    def project(self, P, pose=None, objpose=None):
        #Project world points to image plane
        #
        # UV = self.project(P, OPTIONS) are the image plane coordinates for the world
        # points P.  The columns of P (3xN) are the world points and the columns
        # of UV (2xN) are the corresponding image plane points.
        #
        # Options::
        # 'pose',T         Set the camera pose to the pose T (homogeneous transformation (4x4) or SE3)
        #                  before projecting points to the camera image plane.  Temporarily overrides
        #                  the current camera pose self.T.
        # 'objpose',T      Transform all points by the pose T (homogeneous transformation (4x4) or SE3)
        #                  before projecting them to the camera image plane.
        #
        # See also FishEyeCamera.plot, Camera.plot.         
        
        P = base.getmatrix(P, (3, None))

        if pose is not None:
            T = self.pose.inv()
        else:
            T = SE3()
        if objpose is not None:
            T *= objpose

        P = T * P         # transform points to camera frame
    
        R = np.sqrt(np.sum(P ** 2, axis=0))
        phi = np.arctan2(P[1, :], P[0, :])
        theta = np.arccos(P[2, :] / R)
        
        r = self.rfunc(theta)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        
        uv = np.array([x / self.rhou + self.u0, y / self.rhov + self.v0])
        
        return self.common(uv)



# ------------------------------------------------------------------------ #
class SphericalCamera(Camera):
    
    #TODO
    # pixel noise
    # image paint and rotate
    # Tcam and Tobj animation handle this in the superclass project function
    
        
    def __init__(self, **kwargs):
        #SphericalCamera.Spherical Create spherical projection camera object
        #
        # C = SphericalCamera() creates a spherical projection camera with canonic
        # parameters: f=1 and name='canonic'.
        #
        # C = CentralCamera(OPTIONS) as above but with specified parameters.
        #
        # Options::
        # 'name',N                  Name of camera
        # 'pixel',S                 Pixel size: SxS or S(1)xS(2)
        # 'pose',T                  Pose of the camera as a homogeneous
        #                           transformation
        #
        # See also Camera, CentralCamera, FisheyeCamera, CatadioptricCamera.
        
        # invoke the superclass constructor
        super().__init__(camtype='spherical', 
            limits=[-pi,pi,0,pi],
            labels=['Longitude $\phi$ (rad)', 'Colatitude $\theta$ (rad)'],
            **kwargs)

    # return field-of-view angle for x and y direction (rad)
    def fov(self):
        return 2 * pi
    
    def project(self, P, pose=None, objpose=None):
        #SphericalCamera.project Project world points to image plane
        #
        # PT = self.project(P, OPTIONS) are the image plane coordinates for the world
        # points P.  The columns of P (3xN) are the world points and the columns
        # of PT (2xN) are the corresponding spherical projection points, each column
        # is phi (longitude) and theta (colatitude).
        #
        # Options::
        # 'pose',T         Set the camera pose to the pose T (homogeneous transformation (4x4) or SE3)
        #                  before projecting points to the camera image plane.  Temporarily overrides
        #                  the current camera pose self.T.
        # 'objpose',T      Transform all points by the pose T (homogeneous transformation (4x4) or SE3)
        #                  before projecting them to the camera image plane.
        #
        # See also SphericalCamera.plot.
        P = base.getmatrix(P, (3, None))

        if pose is not None:
            T = self.pose.inv()
        else:
            T = SE3()
        if objpose is not None:
            T *= objpose

        P = T * P         # transform points to camera frame
        
        R = np.sqrt( np.sum(P ** 2, axis=0))
        x = P[0, :] / R
        y = P[1, :] / R
        z = P[2, :] / R
        # r = sqrt( x.^2 + y.^2)
        #theta = atan2(r, z)
        theta = np.arccos(P[2, :] / R)
        phi = np.arctan2(y, x)
        return np.array([phi, theta])


if __name__ == "__main__":

    cam = CentralCamera()
    print(cam)
    # cam.pose = SE3([0.1, 0.2, 0.3])
    print(cam.pose)
    # fig, ax = c.plot_camera(frustum=True)
    # plt.show()
    np.set_printoptions(linewidth=120, formatter={'float': lambda x: f"{x:8.4g}" if abs(x) > 1e-10 else f"{0:8.4g}"})


    print(cam.project([1,2,3]))

    print(cam.visjac_p((300,300), 1))
    cam.flowfield([0,0,0, 0,0,1])
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