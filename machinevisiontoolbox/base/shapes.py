from math import pi
import numpy as np
import scipy
from spatialmath import SE3
from spatialmath.base import isvector, getvector

def mkgrid(n, s, pose=None):
    """
    Create grid of points

    :param n: number of points
    :type n: int or array_like(2)
    :param s: side length of the whole grid
    :type s: float or array_like(2)
    :param pose: pose of the grid, defaults to None
    :type pose: SE3, optional
    :return: 3D coordinates
    :rtype: ndarray(3,n)
 
    - ``mkgrid(n, s)`` is a set of coordinates (3 x n^2) that define a uniform
      grid over a planar region of size ``s`` x ``s``. The points are the
      columns of P.

    - ``mkgrid(n, [sx, sy])`` as above but the grid spans a region of size
      ``sx`` x ``sy``.

    - ``mkgrid([nx,ny], s)`` as above but the grid is an array of ``nx`` points
      in the x-direction and ``ny`` points in the y-direction.

    - ``mkgrid([nx, ny], [sx, sy])`` as above but specify the grid size and
      number of points in each direction.

    By default the grid lies in the xy-plane, symmetric about the origin.
 
    """
    s = base.getvector(s)
    if len(s) == 1:
        sx = s[0]
        sy = s[0]
    elif len(s) == 2:
        sx = s[0]
        sy = s[1]
    else:
        raise ValueError('bad s')

    N = base.getvector(N)
    if len(N) == 1:
        nx = N[0]
        ny = N[0]
    elif len(N) == 2:
        nx = N[0]
        ny = N[1]
    else:
        raise ValueError('bad N')

    if N == 2:
        # special case, we want the points in specific order
        p = np.array([
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


def mkcube(s=1, facepoint=False, pose=None, centre=None, edge=True):
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
    :param edge: create edges, defaults to True
    :type edge: bool, optional
    :raises ValueError: ``centre`` and ``pose`` both specified
    :return: points, or edges as plaid matrices
    :rtype: ndarray(3,n) or three matrices ndarray(2,5)

    - ``S = mkcube()`` is a tuple of three "plaid" matrices that can be
      passed to matplotlib `plot_wireframe`.

    - ``S = mkcube(edge=False)`` is a 3x8 matrix of vertex coordinates.

    - ``S = mkcube(facepoint=True, edge=False)`` is a 3x14 matrix of vertex and
      face centre coordinates.

    By default, the cube is drawn centred at the origin but it's centre
    can be changed using ``centre`` or it can be arbitrarily positioned and
    oriented by specifying its ``pose``.


    Example:

    .. autorun:: pycon

        >>> from machinevisiontoolbox import mkcube
        >>> import matplotlib.pyplot as plt
        >>> S = mkcube(1)  # cube of side length 1
        >>> fig = plt.figure()
        >>> ax = fig.gca(projection='3d')
        >>> ax.plot_wireframe(*S)

    .. note:: We can also use MATLAB-like syntax::

        X, Y, Z = mkcube(1)
        ax.plot_wireframe(X, Y, Z)


    .. warning:: cannot specify both ``centre`` and ``pose``

    :seealso: :func:`mksphere`, :func:`mkcylinder`
    """
    
    if pose is not None and centre is not None:
        raise ValueError('Cannot specify centre and pose options')

    # offset it
    if centre is not None:
        pose = SE3(getvector(centre, 3))

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
        cube = np.r_[cube, faces]

    # vertices of cube about the origin
    if isvector(s, 3):
        s = np.diagonal(getvector(s, 3))
        cube = s @ cube / 2
    else:
        cube = s * cube / 2
    
    # optionally transform the vertices
    if pose is not None:
        cube = pose * cube

    if edge:
        # edge model, return plaid matrices
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
    :return: mesh arrays
    :rtype: three ndarray(n,n)

    ``S = mksphere()`` is a tuple of three "plaid" matrices that can be
    passed to matplotlib `plot_wireframe`.

    Example:

    .. autorun:: pycon

        >>> from machinevisiontoolbox import mksphere
        >>> import matplotlib.pyplot as plt
        >>> S = mksphere()
        >>> fig = plt.figure()
        >>> ax = fig.gca(projection='3d')
        >>> ax.plot_wireframe(*S)

    .. note:: We can also use MATLAB-like syntax::

        X, Y, Z = mksphere()
        ax.plot_wireframe(X, Y, Z)

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


def mkcylinder(r=[1, 1], h=1, n=20, symmetric=False, pose=None):
    """
    Create a cylinder

    :param r: radius, defaults to [1, 1]
    :type r: array_like(m), optional
    :param h: height of the cylinder, defaults to 1
    :type h: float, optional
    :param n: number of points around circumference, defaults to 20
    :type n: int, optional
    :param symmetric: the cylinder's z-extent is [-``h``/2, ``h``/2]
    :param pose: pose of the cylinder, defaults to None
    :type pose: SE3, optional
    :return: plaid arrays
    :rtype: three ndarray(2,n)

    ``S = mkscylinder()`` creates a cylinder of height ``h`` and radius ``r``
    drawn about the z-axis. If ``r`` is a vector allows drawing a solid of
    revolution about the z-axis.  The cylinder can be repositioned or reoriented
    using the ``pose`` option.

    The cylinder is described by a tuple of three "plaid" matrices that can be
    passed to matplotlib `plot_wireframe`.

    Example:

    .. autorun:: pycon

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

        X, Y, Z = mkcylinder()
        ax.plot_wireframe(X, Y, Z)

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

if __name__ == "__main__":

    from spatialmath import SE3

    S = mkcylinder(pose=SE3())

    r = np.linspace(0, 2*pi, 50)
    import matplotlib.pyplot as plt
    S = mkcylinder(r=np.cos(r) + 1.5, symmetric=True, pose=SE3.Rx(pi/2))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_wireframe(*S)
    plt.show()