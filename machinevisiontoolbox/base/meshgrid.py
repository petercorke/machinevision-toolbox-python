import numpy as np

def meshgrid(width, height):
    u = np.arange(width)
    v = np.arange(height)

    return np.meshgrid(u, v)#, indexing='ij')

def sphere_rotate(Phi, Theta, T):

    # convert the spherical coordinates to Cartesian
    x = np.sin(Theta) * np.cos(Phi)
    y = np.sin(Theta) * np.sin(Phi)
    z = np.cos(Theta)

    # convert to 3xN format
    p = np.array([x.ravel(), y.ravel(), z.ravel()])

    # transform the points
    p = T * p

    # convert back to Cartesian coordinate matrices
    x = p[0, :].reshape(x.shape)
    y = p[1, :].reshape(x.shape)
    z = p[2, :].reshape(x.shape)

    nTheta = np.arccos(z)
    nPhi = np.arctan2(y, x)

    return nPhi, nTheta
