import numpy as np

def meshgrid(width, height):
    """
    Coordinate arrays for an image

    :param width: image width in pixels
    :type width: int
    :param height: image height in pixels
    :type height: int
    :return: coordinate arrays
    :rtype: ndarray(H,W), ndarray(H,W)

    Returns arrays ``U`` and ``V`` such that ``U[u,v] = u`` and ``V[u,v] = v``.
    This can be used to define a 2D-function, for example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox import meshgrid
        >>> U, V = meshgrid(3, 4)
        >>> U
        >>> V
        >>> Z = U**2 + V**2 # z=u^2 + v^2
        >>> Z

    :seealso: :func:`Image.warp` :func:`~numpy.meshgrid`
    """
    u = np.arange(width)
    v = np.arange(height)

    return np.meshgrid(u, v)#, indexing='ij')

def sphere_rotate(Phi, Theta, R):
    r"""
    Rotate coordinate matrices for a spherical image

    :param Phi: coordinate array for azimuth
    :type Phi: ndarray(H,W)
    :param Theta: coordinate array for colatitude
    :type Theta: ndarray(H,W)
    :param R: an SO(3) rotation matrix
    :type R: :class:`spatialmath.pose3d.SO3`
    :return: transformed coordinate arrays
    :rtype: ndarray(H,W), ndarray(H,W)

    The coordinates of points in a spherical image can be represented by a pair
    of coordinate matrices that describe azimuth :math:`\phi \in [0, 2\pi]` and
    colatitude :math:`\theta \in [0, \pi]` for each pixel: ``Phi[u,v]``
    :math:`=\phi_{u,v}`, ``Theta[u,v]`` :math:`=\theta_{u,v}`.
    
    This function rotates the spherical image about its centre by
    transforming the coordinate arrays

    .. math:: \begin{pmatrix} \phi^\prime_{u,v} \\ \theta^\prime_{u,v} \end{pmatrix} =
        \mat{R} \begin{pmatrix} \phi_{u,v} \\ \theta_{u,v} \end{pmatrix}, \forall u, v

    :seealso: :class:`spatialmath.pose3d.SO3`
    """

    # convert the spherical coordinates to Cartesian
    x = np.sin(Theta) * np.cos(Phi)
    y = np.sin(Theta) * np.sin(Phi)
    z = np.cos(Theta)

    # convert to 3xN format
    p = np.array([x.ravel(), y.ravel(), z.ravel()])

    # transform the points
    p = R * p

    # convert back to Cartesian coordinate matrices
    x = p[0, :].reshape(x.shape)
    y = p[1, :].reshape(x.shape)
    z = p[2, :].reshape(x.shape)

    nTheta = np.arccos(z)
    nPhi = np.arctan2(y, x)

    return nPhi, nTheta
