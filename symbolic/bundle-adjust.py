from machinevisiontoolbox import Camera
from spatialmath import *
from sympy import *
import numpy as np

# Camera intrinsics
u0, v0, rhox, rhoy, f = base.symbol('u_0 v_0 rho_x rho_y f')
K = np.array([[f / rhox, 0, u0], [0, f / rhoy, v0], [0, 0, 1]])
print(K)
# World point
P = base.symbol('P(x:z)')

## Camera pose
# Camera translation
t = base.symbol('t(x:z)')

# Camera rotation
# - scalar part of quaternion is a function of the vector part
# which we keep as the camera rotation state
qx, qy, qz = base.symbol('qx qy qz')
qs = sqrt(1 - qx ** 2 - qy ** 2 - qz ** 2)
print(qs)
R = base.q2r([qs, qx, qy, qz])
print(R)

## Camera projection model
#  In homogeneous coordinates
uvw = K @ R.T @ (np.r_[P] - np.r_[t])

# In Euclidean coordinates
uv = uvw[:2] / uvw[2]
uv = simplify(uv)
print(uv)

## Jacobians
# Compute the Jacobian (2x6) of pixel coordinate with respect to camera pose
A = Matrix(uv).jacobian([*t, qx, qy, qz])  # wrt camera pose

# and the Jacobian (2x3) of pixel coordinate with respect to landmark position
B = Matrix(uv).jacobian(P)              #  wrt landmark position

with open('camera_derivatives.py', 'w') as f:
    print("from numpy import sqrt, array\n", file=f)
    print("def cameraModel(tx, ty, tz, qx, qy, qz, Px, Py, Pz, f, rho_x, rho_y, u_0, v_0):", file=f)
    print("    p = ", pycode(uv.tolist(), fully_qualified_modules=False), file=f)
    print("    A = ", pycode(A.tolist(), fully_qualified_modules=False), file=f)
    print("    B = ", pycode(B.tolist(), fully_qualified_modules=False), file=f)
    print("    return array(p), array(A), array(B)", file=f)