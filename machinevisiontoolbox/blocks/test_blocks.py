#!/usr/bin/env python3

try:
    from bdsim.blocks.linalg import *

    _bdsim = True
except ImportError:
    _bdsim = False

import unittest
import numpy.testing as nt

from machinevisiontoolbox import CentralCamera, mkcube
from spatialmath import SE3

from machinevisiontoolbox.blocks import *


@unittest.skipIf(not _bdsim, reason="bdsim is not installed")
class CameraBlockTest(unittest.TestCase):
    def test_camera(self):

        cam = CentralCamera.Default()
        block = Camera(cam)

        P = np.array([1, 2, 5])
        T = SE3()
        p = cam.project_point(P, pose=T)

        nt.assert_array_almost_equal(block._output(P, T)[0], p)

        T = SE3.Trans(0.2, 0.3, 0.4)
        p = cam.project_point(P, pose=T)

        nt.assert_array_almost_equal(block._output(P, T)[0], p)

    def test_visjac(self):

        cam = CentralCamera.Default()
        block = Visjac_p(cam, 5)

        P = np.array([1, 2, 5])
        p = cam.project_point(P)

        J = cam.visjac_p(p, 5)
        nt.assert_array_almost_equal(block._output(p)[0], J)

    def test_estpose(self):

        cam = CentralCamera.Default()

        P = mkcube(0.2)
        T_unknown = SE3.Trans(0.1, 0.2, 1.5) * SE3.RPY(0.1, 0.2, 0.3)
        p = cam.project_point(P, objpose=T_unknown)

        T_est = cam.estpose(P, p)

        block = EstPose_p(cam, P)

        nt.assert_array_almost_equal(block._output(p)[0], T_est)

    def test_imageplane(self):

        cam = CentralCamera.Default()
        block = ImagePlane(cam)

        # block._start()
        # block._step()


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":

    unittest.main()
