#!/usr/bin/env python
"""
Smoke tests for Visual Servo classes.
"""


import unittest

from machinevisiontoolbox import CentralCamera, IBVS, PBVS, mkgrid
from spatialmath import SE3
import numpy as np


class TestPbvs(unittest.TestCase):

    def test_run(self):

        camera = CentralCamera.Default(
            pose=SE3.Trans(1, 1, -2)
        )  # create camera at initial pose
        P = mkgrid(2, 0.5)  # create marker points in world frame
        T_Cd_G = SE3.Tz(1)  # desired pose of goal frame {G} with respect to camera
        pbvs = PBVS(
            camera,
            P=P,
            pose_g=SE3.Trans(-1, -1, 2),
            pose_d=T_Cd_G,
            plotvol=[-1, 2, -1, 2, -3, 2.5],
        )  # create PBVS object
        pbvs.run(10)  # simulate it for 200 iterations


class TestIbvs(unittest.TestCase):

    def test_run(self):

        camera = CentralCamera.Default(
            pose=SE3.Trans(1, 1, -3) * SE3.Rz(0.6)
        )  # create camera at initial pose
        P = mkgrid(2, side=0.5, pose=SE3.Tz(3))
        # marker points in world frame
        pd = (
            200 * np.array([[-1, -1, 1, 1], [-1, 1, 1, -1]]) + np.c_[camera.pp]
        )  # desired coordinates in image plane

        ibvs = IBVS(camera, P=P, p_d=pd)  # create IBVS object
        ibvs.run(10)  # simulate it for 25 iterations


if __name__ == "__main__":
    unittest.main()
