import math
import numpy as np
import numpy.testing as nt
import unittest
from machinevisiontoolbox.base.meshgrid import *
from spatialmath import SO3

class TestMeshgrid(unittest.TestCase):

    def test_meshgrid(self):

        U, V = meshgrid(200, 100)
        self.assertEqual(U.shape, (100, 200))
        self.assertEqual(V.shape, (100, 200))

        # for coordinate (20,30)
        self.assertEqual(U[30, 20], 20)
        self.assertEqual(V[30, 20], 30)

    @unittest.skip
    def test_spherical_rotate(self):

        U, V = meshgrid(10, 10)
        Phi = U / 10 * 2 * math.pi
        Theta = V / 10 * math.pi

        P, T = spherical_rotate(Phi, Theta, SO3.Rz(-math.pi))
        nt.assert_array_almost_equal(Phi[:5, :5], P[:5, 100:105])
        nt.assert_array_almost_equal(Theta[:5, :5], T[:5, 100:105])

        P, T = spherical_rotate(Phi, Theta, SO3.Ry(-math.pi/2))
        nt.assert_array_almost_equal(Phi[:5, :5], P[100:105, :5])
        nt.assert_array_almost_equal(Theta[:5, :5], T[100:105, :5])
      
# ------------------------------------------------------------------------ #
if __name__ == '__main__':

    unittest.main()