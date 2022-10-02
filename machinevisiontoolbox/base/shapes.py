from math import pi, sin, cos

import numpy as np
import scipy

from spatialmath import SE3
from spatialmath.base import base

def mkgrid(n=2, side=1, pose=None):
    """
    Create planar grid of points

    :param n: number of points in each dimension, defaults to 2
    :type n: int, array_like(2)
    :param s: side length of the whole grid, defaults to 1
    :type s: float, array_like(2)
    :param pose: pose of the grid, defaults to None
    :type pose: SE3, optional
    :return: 3D coordinates
    :rtype: ndarray(3,n**2), ndarray(3,n[0]*n[1])
 
    Compute a set of coordinates, as column vectors, that define a
    uniform grid of points over a planar region of given size.

    If ``n`` or ``s`` are scalar it is assumed to apply in the x- and y-directions.

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox import mkgrid
        >>> from spatialmath import SE3
        >>> mkgrid()  # 2x2 grid, side length 1m
        >>> mkgrid(side=2)  # 2x2 grid, side length 2m
        >>> mkgrid([2, 3], [4, 6]) # 2x3 grid, side length 4x6m 
        >>> mkgrid(pose=SE3.Trans(1,2,3))

    .. note:: By default the grid lies in the xy-plane, symmetric about the origin.  The
        ``pose`` argument can be used to transform all the points.
 
    """
    side = base.getvector(side)
    if len(side) == 1:
        sx = side[0]
        sy = side[0]
    elif len(side) == 2:
        sx = side[0]
        sy = side[1]
    else:
        raise ValueError('bad s')

    n = base.getvector(n)
    if len(n) == 1:
        nx = n[0]
        ny = n[0]
    elif len(n) == 2:
        nx = n[0]
        ny = n[1]
    else:
        raise ValueError('bad number of points')

    if n[0] == 2:
        # special case, we want the points in specific order
        P = np.array([
            [-sx, -sy, 0],
            [-sx,  sy, 0],
            [ sx,  sy, 0],
            [ sx, -sy, 0],
        ]).T / 2
    else:
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), sparse=False, indexing='ij')
        X = ( X / (nx-1) - 0.5 ) * sx
        Y = ( Y / (ny-1) - 0.5 ) * sy
        Z = np.zeros(X.shape)
        P = np.column_stack((X.flatten(), Y.flatten(), Z.flatten())).T
    
    # optionally transform the points
    if pose is not None:
        P = pose * P

    return P

def mkcube(s=1, facepoint=False, pose=None, centre=None, edge=False, **kwargs):
    """
    Create a cube

    :param s: side length, defaults to 1
    :type s: float
    :param facepoint: add extra point in the centre of each face, defaults to False
    :type facepoint: bool, optional
    :param pose: pose of the cube, defaults to None
    :type pose: SE3, optional
    :param centre: centre of the cube, defaults to None
    :type centre: array_like(3), optional
    :param edge: create edges, defaults to False
    :type edge: bool, optional
    :raises ValueError: ``centre`` and ``pose`` both specified


    Compute vertices or edges of a cube.

    **Vertex mode**

    :return: vertex coordinates
    :rtype: ndarray(3,8), ndarray(3,14)

    Compute the eight vertex coordinates. If facepoint is True then add
    an extra point in the centre of each face.
    
    By default, the cube is drawn centred at the origin but its centre
    can be changed using ``centre`` or it can be arbitrarily positioned and
    oriented by specifying its ``pose``.

    Example:

    .. runblock:: pycon
    
        >>> from machinevisiontoolbox import mkcube
        >>> from spatialmath import SE3
        >>> mkcube()  # cube of side length 1
        >>> mkcube(pose=SE3.Trans(1,2,3))  # cube of side length 1

    **Edge mode**

    :return: edges as X, Y, Z coordinate arrays
    :rtype: ndarray(2,5), ndarray(2,5), ndarray(2,5)

    Compute the edge line segments in the form of three coordinate matrices matrices that
    can be used to create a wireframe plot.

    By default, the cube is drawn centred at the origin but its centre
    can be changed using ``centre`` or it can be arbitrarily positioned and
    oriented by specifying its ``pose``.

    Example:

        >>> from machinevisiontoolbox import mkcube
        >>> import matplotlib.pyplot as plt
        >>> S = mkcube(edge=True)  # cube of side length 1
        >>> fig = plt.figure()
        >>> ax = fig.gca(projection='3d')
        >>> ax.plot_wireframe(*S)

    We can also use MATLAB-like syntax::

        >>> X, Y, Z = mkcube(edge=True)
        >>> ax.plot_wireframe(X, Y, Z)

    :seealso: :func:`mksphere`, :func:`mkcylinder`
    """
    
    if pose is not None and centre is not None:
        raise ValueError('Cannot specify centre and pose options')

    # offset it
    if centre is not None:
        pose = SE3(base.getvector(centre, 3))

    # vertices of a unit cube with one corner at origin
    cube = np.array([
       [ -1,    -1,     1,     1,    -1,    -1,     1,     1],
       [ -1,     1,     1,    -1,    -1,     1,     1,    -1],
       [ -1,    -1,    -1,    -1,     1,     1,     1,     1]
       ])

    if facepoint:
        # append face centre points if required
        faces = np.array([
          [1,    -1,     0,     0,     0,     0],
          [0,     0,     1,    -1,     0,     0],
          [0,     0,     0,     0,     1,    -1]
        ])
        cube = np.hstack((cube, faces))

    # vertices of cube about the origin
    if base.isvector(s, 3):
        s = np.diagonal(getvector(s, 3))
        cube = s @ cube / 2
    else:
        cube = s * cube / 2
    
    # optionally transform the vertices
    if pose is not None:
        cube = pose * cube

    if edge:
        # edge model, return coordinate matrices
        cube = cube[:,[0,1,2,3,0,4,5,6,7,4]]
        o1 = np.reshape(cube[0,:], (2,5))
        o2 = np.reshape(cube[1,:], (2,5))
        o3 = np.reshape(cube[2,:], (2,5))
        return o1, o2, o3

    else:
        return cube


def mksphere(r=1, n=20, centre=[0,0,0]):
    """
    Create a sphere

    :param r: radius, defaults to 1
    :type r: float, optional
    :param n: number of points around the equator, defaults to 20
    :type n: int, optional
    :return: edges as X, Y, Z coordinate arrays
    :rtype: ndarray(n,n), ndarray(n,n), ndarray(n,n)

    Computes a tuple of three coordinate arrays that can be
    passed to matplotlib `plot_wireframe` to draw a sphere.  By default, the
    sphere is drawn about the origin but its position can be changed using
    the ``centre`` option.

    Example::

        >>> from machinevisiontoolbox import mksphere
        >>> import matplotlib.pyplot as plt
        >>> S = mksphere()
        >>> fig = plt.figure()
        >>> ax = fig.gca(projection='3d')
        >>> ax.plot_wireframe(*S)

    We can also use MATLAB-like syntax::

        >>> X, Y, Z = mksphere()
        >>> ax.plot_wireframe(X, Y, Z)

    :seealso: :func:`mkcube`, :func:`mkcylinder`
    """

    theta = np.linspace(-pi, pi, n).reshape((1,n))
    phi = np.linspace(-pi / 2, pi / 2, n).reshape((n,1))
    cosphi = np.cos(phi)
    sintheta = np.sin(theta)

    # fix rounding errors
    cosphi[0,0] = 0
    cosphi[0,n-1] = 0
    sintheta[0,0] = 0
    sintheta[n-1,0] = 0

    X = r * cosphi @ cos(theta) + centre[0]
    Y = r * cosphi @ sintheta + centre[1]
    Z = r * sin(phi) @ np.ones((1,n)) + centre[2]

    return X, Y, Z


def mkcylinder(r=1, h=1, n=20, symmetric=False, pose=None):
    """
    Create a cylinder

    :param r: radius, defaults to 1
    :type r: array_like(m), optional
    :param h: height of the cylinder, defaults to 1
    :type h: float, optional
    :param n: number of points around circumference, defaults to 20
    :type n: int, optional
    :param symmetric: the cylinder's z-extent is [-``h``/2, ``h``/2]
    :param pose: pose of the cylinder, defaults to None
    :type pose: SE3, optional
    :return: three coordinate arrays
    :rtype: three ndarray(2,n)

    Computes a tuple of three coordinate arrays that can be passed to matplotlib
    `plot_wireframe` to draw a cylinder of radius ``r`` about the z-axis from
    z=0 to z=``h``. The cylinder can be repositioned or reoriented using the
    ``pose`` option.

    If radius ``r`` is array_like(2) then it represents the radius at the bottom
    and top of the cylinder and can be used to create a cone or conical frustum.
    If len(r)>2 then it allows the creation of a more complex shape with radius
    as a function of z. 

    Example::

        >>> from machinevisiontoolbox import mkcylinder
        >>> import matplotlib.pyplot as plt
        >>> # draw a horizontal diablo shape
        >>> r = np.linspace(0, 2*pi, 50)
        >>> S = mkcylinder(r=np.cos(r) + 1.5, symmetric=True, pose=SE3.Rx(pi/2))
        >>> fig = plt.figure()
        >>> ax = fig.gca(projection='3d')
        >>> ax.plot_wireframe(*S)
        >>> plt.show()

    .. note:: We can also use MATLAB-like syntax::

        >>> X, Y, Z = mkcylinder()
        >>> ax.plot_wireframe(X, Y, Z)

    :seealso: :func:`mkcube`, :func:`mksphere`
    """

    if isinstance(r, (int, float)):
        r = [r, r]
    r = np.array(r).reshape((-1,1))

    theta = np.linspace(0, 2 * pi, n).reshape((1,n))
    sintheta = np.sin(theta)
    sintheta[0,n-1] = 0

    X = r @ np.cos(theta)
    Y = r @ sintheta
    m = len(r)
    Z = h  * np.linspace(0, 1, m).reshape((m,1)) @ np.ones((1,n))
    if symmetric:
        Z = Z - h / 2

    if pose is not None:
        P = np.row_stack((X.flatten(), Y.flatten(), Z.flatten()))
        P = pose * P
        X = P[0,:].reshape(X.shape)
        Y = P[1,:].reshape(X.shape)
        Z = P[2,:].reshape(X.shape)

    return X, Y, Z

# if __name__ == "__main__":

#     from spatialmath import SE3

#     S = mkcylinder(pose=SE3())

#     r = np.linspace(0, 2*pi, 50)
#     import matplotlib.pyplot as plt
#     S = mkcylinder(r=np.cos(r) + 1.5, symmetric=True, pose=SE3.Rx(pi/2))
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.plot_wireframe(*S)
#     plt.show()