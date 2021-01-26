#!/usr/bin/env python
"""
Camera class
@author: Dorian Tsai
@author: Peter Corke
"""

import numpy as np
import cv2 as cv
from spatialmath import base
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
    
    def __init__(self,
                 name=None,
                 camtype=None,
                 rho=10e-6,
                 imagesize=(500, 500),
                 pose=None):
        """
        Create instance of a Camera class
        """
        if name is None:
            self._name = 'MVTB camera'
        else:
            if not isinstance(name, str):
                raise TypeError(name, 'name must be a string')
            self._name = name

        if camtype is None:
            self._camtype = 'perspective'
        else:
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

        imagesize = base.getvector(imagesize)
        if len(imagesize) == 1:
            self._nu = imagesize
            self._nv = imagesize
        elif len(imagesize) == 2:
            self._nu = imagesize[0]
            self._nv = imagesize[1]
        else:
            raise ValueError(
                imagesize, 'imagesize must be a 1- or 2-element vector')

        if pose is None:
            self._pose = SE3()
        else:
            self._pose = SE3(pose)

        self._image = None

        self._fig = None
        self._ax = None

        self._noise = None
        self._distortion = None

    def __str__(self):
        s = ''
        self.fmt = '{:>15s}: {}\n'
        s += self.fmt.format('Name', self.name + ' [' + self.__class__.__name__ + ']')
        s += self.fmt.format('pixel size', ' x '.join([str(x) for x in self.rho]))
        s += self.fmt.format('image size', ' x '.join([str(x) for x in self.imagesize]))
        s += self.fmt.format('pose', self.pose.printline(file=None, fmt="{:.3g}"))
        return s

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
    def nu(self):
        return self._nu

    @property
    def nv(self):
        return self._nv

    @property
    def imagesize(self):
        return (self._nu, self._nv)

    @property
    def rhou(self):
        """
        Get pixel size: horizontal value

        :return: horizontal pixel size
        :rtype: float

        :seealso: :func:`rhov`, :func:`rho`
        """
        return self._rhou

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
        Get pixel size: horizontal value

        :return: horizontal pixel size
        :rtype: float

        :seealso: :func:`rhov`, :func:`rho`
        """

        return (self._rhov, self._rhov)

    @property
    def rho(self):
        """
        Get pixel size: horizontal value

        :return: horizontal pixel size
        :rtype: float

        :seealso: :func:`rhov`, :func:`rho`
        """
        return (self._rhov, self._rhov)


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

    def plotcreate(self, fig=None, ax=None):
        """
        Create plot for camera image plane
        """
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
            ax.set_xlim(0, self.nu)
            ax.set_ylim(0, self.nv)
            ax.autoscale(False)
            ax.invert_yaxis()
            ax.grid(True)
            ax.set_xlabel('u (pixels)')
            ax.set_ylabel('v (pixels)')
            ax.set_title(self.name)
            ax.figure.canvas.set_window_title('Machine Vision Toolbox for Python')

        # TODO figure out axes ticks, etc
        self._fig = fig
        self._ax = ax
        return fig, ax  # likely this return is not necessary

    def plot(self, p=None, marker='or', markersize=10, **kwargs):
        """
        Plot points on image plane
        If 3D points, then 3D world points
        If 2D points, then assumed image plane points
        TODO plucker coordinates/lines?
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
        
        # get handle for this camera image plane
        self.plotcreate()
        plt.autoscale(False)
        
        # draw 3D line segments
        nsteps = 21;
        
        # c.clf
        # holdon = c.hold(1);
        # s = linspace(0, 1, nsteps);
        
        for i in range(X.shape[0]-1):      #i=1:numrows(X)-1
            for j in range(X.shape[1]-1):  # j=1:numcols(X)-1
                P0 = [X[i,j], Y[i,j], Z[i,j]]
                P1 = [X[i+1,j], Y[i+1,j], Z[i+1,j]]
                P2 = [X[i,j+1], Y[i,j+1], Z[i,j+1]]
                
                # if c.perspective
                    # straight world lines are straight on the image plane
                uv = self.project(np.c_[P0, P1], pose=pose);
                # else
                #     # straight world lines are not straight, plot them piecewise
                #     P = bsxfun(@times, (1-s), P0) + bsxfun(@times, s, P1);
                #     uv = c.project(P, 'setopt', opt);

                self._ax.plot(uv[0,:], uv[1,:], **kwargs);
                
                # if c.perspective
                    # straight world lines are straight on the image plane
                uv = self.project(np.c_[P0, P2], pose=pose);
                # else
                #     # straight world lines are not straight, plot them piecewise
                #     P = bsxfun(@times, (1-s), P0) + bsxfun(@times, s, P2);
                #     uv = c.project(P, 'setopt', opt);
                self._ax.plot(uv[0,:], uv[1,:], **kwargs);

        
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
                    scale=None,
                    frustum=False,
                    cube=False,
                    label=False,
                    persist=False,
                    fig=None,
                    ax=None):
        """
        Display camera icon in world view
        """

        if (fig is None) and (ax is None):
            # create our own handle for the figure/plot
            print('creating new figure and axes for camera')
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            # ax.set_aspect('equal')

        # draw camera-like object:
        if cube:
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

        if frustum:
            # TODO make this kwargs or optional inputs
            camfrustum = CameraVisualizer(self,
                                          f_length=0.5,
                                          fb_width=0.05,
                                          ft_width=0.5)
            camfrustumpoly = Poly3DCollection(camfrustum.gen_frustrum_poly(),
                                              facecolors=['g', 'r', 'b', 'y'])
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

class CentralCamera(Camera):
    """
    A (central projection) camera class
    """

    # list of attributes
    _name = []      # camera  name (string)
    _camtype = []   # camera type (string)

    _nu = []        # number of pixels horizontal
    _nv = []        # number of pixels vertical
    _u0 = []        # principal point horizontal
    _v0 = []        # principal point vertical
    _rhou = []      # pixel imagesize (single pixel) horizontal
    _rhov = []      # pixel imagesize (single pixel) vertical
    _fu = []        # focal length horizontal [units]
    _fv = []        # focal length vertical [units]
    _image = []     # image (TODO image class?), for now, just numpy array

    _T = []         # camera pose (homogeneous transform, SE3 class)

    _fig = []       # for plotting, figure handle/object reference
    _ax = []        # for plotting, axes handle

    def __init__(self,
                 f=8*1e-3,
                 rho=10e-6,
                 pp=None,
                 **kwargs):
        """
        Create instance of a Camera class
        """

        super().__init__(**kwargs)

        # TODO some of this logic to f and pp setters
        self.f = f

        if pp is None:
            print('principal point not specified, \
                   setting it to centre of image plane')
            self.pp = (self._nu / 2, self._nv / 2)
        else:
            self.pp = pp


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

        P = base.getmatrix(P, (3,None))

        if pose is None:
            pose = self.pose

        C = self.getC(pose)

        if isinstance(P, np.ndarray):
            # project 3D points

            if objpose is not None:
                P = objpose * P

            x = C @ base.e2h(P)

            x[2,x[2,:]<0] = np.nan  # points behind the camera are set to NaN


            x = base.h2e(x)

            # if self._distortion is not None:
            #     x = self.distort(x)
            
            # if self._noise is not None:
            #     # add Gaussian noise with specified standard deviation
            #     x += np.diag(self._noise) * np.random.randn(x.shape)

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

        elif isinstance(P, Plucker):
            # project Plucker lines

            x = np.empty(shape=(3, 0))
            for p in P:
                l = base.vex( C * p.skew * C.T)
                x = np.c_[x, l / np.max(np.abs(l))] # normalize by largest element
            return x

    @property
    def u0(self):
        """
        Get principal point: horizontal coordinate

        :return: horizontal component of principal point
        :rtype: float

        :seealso: :func:`v0`, :func:`pp`
        """
        return self._u0

    @property
    def v0(self):
        """
        Get principal point: vertical coordinate

        :return: vertical component of principal point
        :rtype: float

        :seealso: :func:`u0`, :func:`pp`
        """
        return self._v0

    @property
    def pp(self):
        """
        Get principal point coordinate

        :return: principal point
        :rtype: 2-tuple

        :seealso: :func:`u0`, :func:`v0`
        """
        return (self._u0, self._v0)

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
            self._u0 = pp[0]
            self._v0 = pp[0]
        elif len(pp) == 2:
            self._u0 = pp[0]
            self._v0 = pp[1]
        else:
            raise ValueError(pp, 'pp must be a 1- or 2-element vector')

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
        K = np.array([[self._fu / self._rhou, 0, self._u0],
                      [ 0, self._fv / self._rhov, self._v0],
                      [ 0, 0, 1]], dtype=np.float)
        return K

    @property
    def C(self):
        """
        Camera matrix, camera calibration or projection matrix
        """
        P0 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]], dtype=np.float)

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

        t = v[:,3]
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

    def __init__(self, camera, f_length=0.1, fb_width=0.05, ft_width=0.1):
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
        self.cf_b0 = np.array([-fb_width/2, -fb_width/2, 0, 1]).reshape(4, 1)
        self.cf_b1 = np.array([-fb_width/2, fb_width/2, 0, 1]).reshape(4, 1)
        self.cf_b2 = np.array([fb_width/2, fb_width/2, 0, 1]).reshape(4, 1)
        self.cf_b3 = np.array([fb_width/2, -fb_width/2, 0, 1]).reshape(4, 1)
        self.cf_t0 = np.array(
            [-ft_width/2, -ft_width/2, f_length, 1]).reshape(4, 1)
        self.cf_t1 = np.array(
            [-ft_width/2, ft_width/2, f_length, 1]).reshape(4, 1)
        self.cf_t2 = np.array(
            [ft_width/2, ft_width/2, f_length, 1]).reshape(4, 1)
        self.cf_t3 = np.array(
            [ft_width/2, -ft_width/2, f_length, 1]).reshape(4, 1)

    def gen_frustrum_poly(self):

        # Transform frustrum points to world coordinate frame using the camera
        # extrinsics
        T = self.camera.T.A

        b0 = (T @ self.cf_b1)[:-1].flatten()
        b1 = (T @ self.cf_b2)[:-1].flatten()
        b2 = (T @ self.cf_b3)[:-1].flatten()
        b3 = (T @ self.cf_b0)[:-1].flatten()
        t0 = (T @ self.cf_t1)[:-1].flatten()
        t1 = (T @ self.cf_t2)[:-1].flatten()
        t2 = (T @ self.cf_t3)[:-1].flatten()
        t3 = (T @ self.cf_t0)[:-1].flatten()

        # Each set of four points is a single side of the Frustrum
        points = np.array([[b0, b1, t1, t0], [b1, b2, t2, t1], [
                          b2, b3, t3, t2], [b3, b0, t0, t3]])
        return points


if __name__ == "__main__":

    c = CentralCamera()
    print(c)
    c.T = SE3([0.1, 0.2, 0.3])
    print(c.T)
    # fig, ax = c.plot_camera(frustum=True)
    # plt.show()

    print(c.project([1,2,-3]))
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
