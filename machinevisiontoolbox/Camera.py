#!/usr/bin/env python
"""
Camera class
@author: Dorian Tsai
@author: Peter Corke
"""

import numpy as np
import cv2 as cv
import spatialmath.base.argcheck as argcheck
import machinevisiontoolbox as mvt
import matplotlib.pyplot as plt

# from collections import namedtuple
from spatialmath import SE3
from spatialmath.base import e2h, h2e


class Camera:
    """
    A camera class
    """

    # list of attributes
    _name = []      # camera  name (string)
    _camtype = []   # camera type (string)

    _nu = []        # number of pixels horizontal
    _nv = []        # number of pixels vertical
    _u0 = []        # principal point horizontal
    _v0 = []        # principal point vertical
    _rhou = []      # pixel resolution (single pixel) horizontal
    _rhov = []      # pixel resolution (single pixel) vertical
    _fu = []        # focal length horizontal [units]
    _fv = []        # focal length vertical [units]
    _image = []     # image (TODO image class?), for now, just numpy array

    _T = []         # camera pose (homogeneous transform, SE3 class?)

    _fig = []       # for plotting, figure handle/object reference
    _ax = []        # for plotting, axes handle

    def __init__(self,
                 name=None,
                 camtype=None,
                 f=81e-3,
                 rho=10e-6,
                 resolution=(500, 500),
                 pp=None,
                 T=None):
        """
        Create instance of a Camera class
        """
        if name is None:
            self._name = 'mvtcamera'
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

        f = argcheck.getvector(f)
        if len(f) == 1:
            self._fu = f
            self._fv = f
        elif len(f) == 2:
            self._fu = f[0]
            self._fv = f[1]
        else:
            raise ValueError(f, 'f must be a 1- or 2-element vector')

        rho = argcheck.getvector(rho)
        if len(rho) == 1:
            self._rhou = rho
            self._rhov = rho
        elif len(rho) == 2:
            self._rhou = rho[0]
            self._rhov = rho[1]
        else:
            raise ValueError(rho, 'rho must be a 1- or 2-element vector')

        resolution = argcheck.getvector(resolution)
        if len(resolution) == 1:
            self._nu = resolution
            self._nv = resolution
        elif len(resolution) == 2:
            self._nu = resolution[0]
            self._nv = resolution[1]
        else:
            raise ValueError(resolution, 'resolution must be a 1- or 2-element vector')

        if pp is None:
            print('principal point not specified, \
                   setting it to centre of image plane')
            self._u0 = self._nu / 2
            self._v0 = self._nv / 2
        else:
            pp = argcheck.getvector(pp)
            if len(pp) == 1:
                self._u0 = pp
                self._v0 = pp
            elif len(pp) == 2:
                self._u0 = pp[0]
                self._v0 = pp[1]
            else:
                raise ValueError(pp, 'pp must be a 1- or 2-element vector')

        # TODO how to check T? various input formats? assume this is taken care
        # of by SE3(T)
        if T is None:
            self._T = np.identity(4)
        else:
            if not isinstance(T, SE3):
                self._T = SE3(T)
            else:
                self._T = T

        self._image = None

        self._fig = None
        self._ax = None

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
    def resolution(self):
        return (self._nu, self._nv)

    @property
    def u0(self):
        return self._u0

    @u0.setter
    def u0(self, value):
        self._u0 = float(value)

    @property
    def v0(self):
        return self._v0

    @property
    def pp(self):
        return (self._u0, self._v0)

    @property
    def rhou(self):
        return self._rhou

    @property
    def rhov(self):
        return self._rhov

    @property
    def rho(self):
        return (self._rhov, self._rhov)

    @property
    def fu(self):
        return self._fu

    @property
    def fv(self):
        return self._fv

    @property
    def f(self):
        return (self._fu, self._fv)

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, newimage):
        self._image = mvt.getimage(newimage)

    @property
    def T(self):
        return self._T

    @property
    def t(self):
        return SE3(self._T).t

    @t.setter
    def t(self, x, y=None, z=None):
        """
        Set camera 3D position [m]
        If y, z are None, then x is a 3-vector xyz
        """
        # TODO check all valid inputs
        if (y is None) and (z is None) and (len(x) == 3):
            # x is a 3-vector
            x = argcheck.getvector(x)
            y = x[1]
            z = x[2]
            x = x[0]
        # order matters, TODO check
        self._T = SE3.Tx(x) * SE3.Ty(y) * SE3.Tz(z)

    @property
    def rpy(self, unit='deg', order='zyx'):
        return self._T.rpy(unit, order)

    @rpy.setter
    def rpy(self, roll, pitch=None, yaw=None):
        """
        Set camera attitude/orientation [rad] vs [deg]
        If pitch and yaw are None, then roll is an rpy 3-vector
        """
        # TODO check all valid inputs, eg rad vs deg
        if (pitch is None) and (yaw is None) and (len(roll) == 3):
            # roll is 3-vector rpy
            roll = argcheck.getvector(roll)
            pitch = roll[1]
            yaw = roll[2]
            roll = roll[0]
            self._T = SE3.Ry(yaw) * SE3.Rx(pitch) * SE3.Rz(roll)
        elif argcheck.isscalar(pitch) and \
                argcheck.isscalar(roll) and argcheck.isscalar(yaw):
            self._T = SE3.Ry(yaw) * SE3.Rx(pitch) * SE3.Rz(roll)
        else:
            raise ValueError(roll, 'roll must be a 3-vector, or \
                roll, pitch, yaw must all be scalars')

    @property
    def K(self):
        """
        Intrinsic matrix of camera
        """
        K = np.array([[self._fu/self._rhou, 0, self._u0],
                      [0, self._fv/self._rhov, self._v0],
                      [0, 0, 1]], dtype=np.float)
        return K

    @property
    def C(self, T=None):
        """
        Camera matrix
        """
        P0 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]], dtype=np.float)
        if T is None:
            C = self.K @ P0 @ np.linalg.inv(self.T)
        else:
            C = self.K @ P0 @ np.linalg.inv(T)
        return C

    def printCamera(self):
        """
        Print (internal) camera class attributes
        TODO should change this to print relevant camera parameters
        """
        attributes = vars(self)
        for key in attributes:
            print(key, ': \t', attributes[key])

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

        # TODO figure out axes ticks, etc
        self._fig = fig
        self._ax = ax
        return fig, ax  # likely this return is not necessary

    def plot(self, p=None):
        """
        Plot points on image plane
        If 3D points, then 3D world points
        If 2D points, then assumed image plane points
        TODO plucker coordinates/lines?
        """
        self.plotcreate()
        ip = self.project(p)
        # TODO plot ip on image plane given self._fig and self._ax
        # TODO don't do this in a for loop
        # TODO accept kwargs for the plotting

        #import code
        #code.interact(local=dict(globals(), **locals()))

        for i in range(ip.shape[1]):
            self._ax.plot(ip[0, i], ip[1, i], 'or', markersize=10)
        plt.show()


    def project(self, P, T=None):
        """
        Central projection for now
        P world points or image plane points in column vectors only
        TODO how to tell? based on shape and display warning otherwise/ambiguity?
        """

        # TODO check P.
        # for now, assume column vectors (wide and short)
        if P.shape[0] == 3:
            # for 3D world points
            if T is None:
                C = self.C
            else:
                C = self.C(T)
            ip = h2e(C @ e2h(P))
        elif P.shape[0] == 2:
            # for 2D imageplane points
            ip = P
        return ip

    """
    TODO Methods
    labelaxes (pixels vs m)
    E
    F
    """


if __name__ == "__main__":

    c = Camera()

    print(c.name)
    c.name = 'machine vision toolbox'
    print(c.name)

    print(c.K)

    print('\n')
    c.printCamera()
    print('\n')

    print(c.t)
    c.t = np.r_[1, 2, 3]
    print(c.t)
    print(c.T)

    print(c.rpy)
    c.rpy = np.r_[0.1, 0.2, 0.3]
    print(c.rpy)
    print(c.T)

    npts = 10
    p = np.random.randint(0, 200, (2, npts))

    imfile = 'images/shark1.png'
    im = mvt.iread(imfile)

    c.image = im

    c.plotcreate()
    c.plot(p)

    #import code
    #code.interact(local=dict(globals(), **locals()))



