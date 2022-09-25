#!/usr/bin/env python

import numpy as np
import numpy.testing as nt
import unittest
import numpy.testing as nt
from spatialmath import SE3
import matplotlib.pyplot as plt

from machinevisiontoolbox import CentralCamera

class TestCamera(unittest.TestCase):

    def test_parameters(self):

        c = CentralCamera(f=0.2)
        nt.assert_array_almost_equal(c.f, (0.2, 0.2))
        self.assertEqual(c.camtype, 'perspective')
        self.assertEqual(c.fu, 0.2)
        self.assertEqual(c.fv, 0.2)

        c.f = 0.3
        nt.assert_array_almost_equal(c.f, (0.3, 0.3))

        c = CentralCamera(imagesize=4000)
        nt.assert_array_almost_equal(c.imagesize, (4000, 4000))
        self.assertEqual(c.width, 4000)
        self.assertEqual(c.height, 4000)
        self.assertEqual(c.nu, 4000)
        self.assertEqual(c.nv, 4000)
        nt.assert_array_almost_equal(c.pp, (2000, 2000))

        c = CentralCamera(imagesize=(4000, 3000))
        nt.assert_array_almost_equal(c.imagesize, (4000, 3000))
        self.assertEqual(c.width, 4000)
        self.assertEqual(c.height, 3000)
        self.assertEqual(c.nu, 4000)
        self.assertEqual(c.nv, 3000)
        nt.assert_array_almost_equal(c.pp, (2000, 1500))

        c.imagesize = 5000
        self.assertEqual(c.width, 5000)
        self.assertEqual(c.height, 5000)
        self.assertEqual(c.nu, 5000)
        self.assertEqual(c.nv, 5000)

        c.imagesize = (5000, 4000)
        self.assertEqual(c.width, 5000)
        self.assertEqual(c.height, 4000)
        self.assertEqual(c.nu, 5000)
        self.assertEqual(c.nv, 4000)

        c = CentralCamera(rho=0.1)
        self.assertEqual(c.rhou, 0.1)
        self.assertEqual(c.rhov, 0.1)

        c = CentralCamera(rho=(0.1, 0.2))
        nt.assert_array_almost_equal(c.rho, (0.1, 0.2))
        self.assertEqual(c.rhou, 0.1)
        self.assertEqual(c.rhov, 0.2)

        c = CentralCamera(f=2, rho=(0.1, 0.2))

        c = CentralCamera(pp=(200, 300))
        self.assertEqual(c.u0, 200)
        self.assertEqual(c.v0, 300)
        nt.assert_array_almost_equal(c.pp, (200, 300))

        c = CentralCamera(pp=(200, 300), imagesize=3000)
        self.assertEqual(c.u0, 200)
        self.assertEqual(c.v0, 300)
        nt.assert_array_almost_equal(c.pp, (200, 300))

    def test_pose(self):

        c = CentralCamera()
        self.assertTrue(c.pose == SE3())

        x = SE3(1,2,3)
        y = SE3(4,5,6)
        c = CentralCamera(pose=x)
        self.assertTrue(c.pose == x)

        c.pose = y
        self.assertTrue(c.pose == y)

        c = CentralCamera()
        c2 = c.move(x)
        self.assertIsNot(c, c2)
        self.assertTrue(c2.pose == x)

    def test_str(self):
        c = CentralCamera(f=0.123, imagesize=(2000, 3000), name='fred')
        s = str(c)
        self.assertIsInstance(s, str)
        self.assertTrue('fred' in s)
        self.assertTrue('CentralCamera' in s)
        self.assertTrue('2000' in s)
        self.assertTrue('3000' in s)
        self.assertTrue('0.123' in s)

    def test_project(self):

        c = CentralCamera(f=1, rho=1, pp=(0, 0))
        p = c.project_point([0, 0, 5])
        self.assertTrue(p.shape == (2,1))
        nt.assert_array_almost_equal(p.flatten(), [0, 0])

        p = c.project_point([1, 0, 5])
        nt.assert_array_almost_equal(p.flatten(), [0.2, 0])

        p = c.project_point([-1, 0, 5])
        nt.assert_array_almost_equal(p.flatten(), [-0.2, 0])

        p = c.project_point([0, 1, 5])
        nt.assert_array_almost_equal(p.flatten(), [0, 0.2])

        p = c.project_point([0, -1, 5])
        nt.assert_array_almost_equal(p.flatten(), [0, -0.2])

        p = c.project_point([1, -1, 5])
        nt.assert_array_almost_equal(p.flatten(), [0.2, -0.2])

        p = c.project_point([1, -1, 2.5])
        nt.assert_array_almost_equal(p.flatten(), [0.4, -0.4])

        p = c.project_point([1, -1, 5], pose=SE3(1, -1, 0))
        nt.assert_array_almost_equal(p.flatten(), [0, 0])

        p = c.project_point([1, -1, 5], pose=SE3(0, 0, 2.5))
        nt.assert_array_almost_equal(p.flatten(), [0.4, -0.4])

        p = c.project_point([1, -1, 5], objpose=SE3(-1, 1, 0))
        nt.assert_array_almost_equal(p.flatten(), [0, 0])

        c = CentralCamera(f=1, rho=1, pp=(1,1))

        p = c.project_point([0, -1, 5])
        nt.assert_array_almost_equal(p.flatten(), [1, 0.8])

        c = CentralCamera(f=2, rho=1, pp=(0, 0))
        p = c.project_point([1, -1, 5])
        nt.assert_array_almost_equal(p.flatten(), [0.4, -0.4])

        c = CentralCamera(f=2, rho=2, pp=(0, 0))
        p = c.project_point([1, -1, 5])
        nt.assert_array_almost_equal(p.flatten(), [0.2, -0.2])

    def test_graphics(self):
        c = CentralCamera.Default()

        c.plot_point([1, -1, 5])
        c.clf()
        p = c.project_point([1, -1, 5])
        c.plot_point(p)

        plt.close()

    # test visibility
    # test multiple points

    def test_epiline(self):
        c1 = CentralCamera.Default()
        c2 = c1.move(SE3(0.1, 0, 0))
        P = [1, -1, 5]
        p1 = c1.project_point(P)
        F = c1.F(c2)

        c1.plot_epiline(F, p1)

        plt.close()

    def test_plucker(self):

        c = CentralCamera.Default()
        P = [1, -1, 5]
        p = c.project_point(P)

        line = c.ray(p)
        self.assertTrue(line.contains(P))




# ----------------------------------------------------------------------- #
if __name__ == '__main__':

    unittest.main()