from machinevisiontoolbox import CentralCamera, mkcube
from spatialmath import SE3
camera = CentralCamera.Default()
X, Y, Z = mkcube(0.2, pose=SE3(0, 0, 1), edge=True)
camera.plot_wireframe(X, Y, Z, 'k--')