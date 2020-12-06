# Create cube
#
# P = MKCUBE(S, OPTIONS) is a set of points (3x8) that define the 
# vertices of a cube of side length S and centred at the origin.
#
# [X,Y,Z] = MKCUBE(S, OPTIONS) as above but return the rows of P as three 
# vectors.
#
# [X,Y,Z] = MKCUBE(S, 'edge', OPTIONS) is a mesh that defines the edges of
# a cube.
#
# Options::
# 'facepoint'    Add an extra point in the middle of each face, in this case
#                the returned value is 3x14 (8 vertices + 6 face centres).
# 'centre',C     The cube is centred at C (3x1) not the origin
# 'pose',T       The pose of the cube coordinate frame is defined by the homogeneous transform T,
#                allowing all points in the cube to be translated or rotated.
# 'edge'         Return a set of cube edges in MATLAB mesh format rather
#                than points.
#
# See also CYLINDER, SPHERE.

import numpy as np
from spatialmath.base import isvector, getvector
from spatialmath import SE3

def mkcube(s, facepoint=False, pose=None, centre=None, edge=True):
    
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
        return cube[0,:], cube[1,:], cube[2,:]
