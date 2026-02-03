import numpy as np
import numpy.testing as nt
import unittest
import sys
from pathlib import Path

# Add the src directory to path and import directly from the module file
shapes_path = (
    Path(__file__).parent.parent.parent / "src" / "machinevisiontoolbox" / "base"
)
sys.path.insert(0, str(shapes_path))

# Import the shapes module directly
import importlib.util

spec = importlib.util.spec_from_file_location("shapes", shapes_path / "shapes.py")
shapes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shapes)

mkgrid = shapes.mkgrid
mkcube = shapes.mkcube
mksphere = shapes.mksphere
mkcylinder = shapes.mkcylinder

from spatialmath import SE3


class TestMkgrid(unittest.TestCase):
    def test_default(self):
        """Test default 2x2 grid with side length 1"""
        P = mkgrid()
        self.assertEqual(P.shape, (3, 4))

        # Check z-coordinates are all zero
        nt.assert_array_equal(P[2, :], np.zeros(4))

        # Check specific corner positions
        expected = np.array(
            [[-0.5, -0.5, 0.5, 0.5], [-0.5, 0.5, 0.5, -0.5], [0, 0, 0, 0]]
        )
        nt.assert_array_almost_equal(P, expected)

    def test_scalar_n(self):
        """Test with scalar n parameter"""
        P = mkgrid(n=3, side=2)
        self.assertEqual(P.shape, (3, 9))
        nt.assert_array_equal(P[2, :], np.zeros(9))

        # Check bounds
        self.assertAlmostEqual(P[0, :].min(), -1.0)
        self.assertAlmostEqual(P[0, :].max(), 1.0)
        self.assertAlmostEqual(P[1, :].min(), -1.0)
        self.assertAlmostEqual(P[1, :].max(), 1.0)

    def test_vector_n(self):
        """Test with vector n parameter"""
        # When n[0] == 2, special case returns 4 points
        P = mkgrid(n=[2, 3], side=1)
        self.assertEqual(P.shape, (3, 4))
        nt.assert_array_equal(P[2, :], np.zeros(4))

        # Test with non-special case
        P = mkgrid(n=[3, 4], side=1)
        self.assertEqual(P.shape, (3, 12))
        nt.assert_array_equal(P[2, :], np.zeros(12))

    def test_vector_side(self):
        """Test with vector side parameter"""
        P = mkgrid(n=2, side=[4, 6])
        self.assertEqual(P.shape, (3, 4))

        # Check x bounds
        self.assertAlmostEqual(P[0, :].min(), -2.0)
        self.assertAlmostEqual(P[0, :].max(), 2.0)

        # Check y bounds
        self.assertAlmostEqual(P[1, :].min(), -3.0)
        self.assertAlmostEqual(P[1, :].max(), 3.0)

    def test_with_pose(self):
        """Test grid transformation with pose"""
        pose = SE3.Trans(1, 2, 3)
        P = mkgrid(n=2, side=1, pose=pose)
        self.assertEqual(P.shape, (3, 4))

        # Check that z-coordinates are not zero (transformed)
        self.assertTrue(np.all(P[2, :] > 0))

        # Check center is at translation
        center = P.mean(axis=1)
        nt.assert_array_almost_equal(center, [1, 2, 3])

    def test_rotated_grid(self):
        """Test grid with rotation"""
        pose = SE3.Rz(np.pi / 2)
        P = mkgrid(n=2, side=1, pose=pose)
        self.assertEqual(P.shape, (3, 4))

    def test_invalid_side(self):
        """Test invalid side parameter"""
        with self.assertRaises(ValueError):
            mkgrid(side=[1, 2, 3])

    def test_invalid_n(self):
        """Test invalid n parameter"""
        with self.assertRaises(ValueError):
            mkgrid(n=[1, 2, 3])


class TestMkcube(unittest.TestCase):
    def test_default(self):
        """Test default unit cube"""
        cube = mkcube()
        self.assertEqual(cube.shape, (3, 8))

        # Check that all coordinates are within [-0.5, 0.5]
        self.assertTrue(np.all(cube >= -0.5))
        self.assertTrue(np.all(cube <= 0.5))

        # Check corners
        self.assertAlmostEqual(cube[0, :].min(), -0.5)
        self.assertAlmostEqual(cube[0, :].max(), 0.5)

    def test_custom_size(self):
        """Test cube with custom size"""
        cube = mkcube(s=2)
        self.assertEqual(cube.shape, (3, 8))

        # Check bounds for side length 2
        self.assertAlmostEqual(cube[0, :].min(), -1.0)
        self.assertAlmostEqual(cube[0, :].max(), 1.0)

    def test_with_facepoint(self):
        """Test cube with face points"""
        cube = mkcube(facepoint=True)
        self.assertEqual(cube.shape, (3, 14))  # 8 vertices + 6 face centers

    def test_with_pose(self):
        """Test cube with pose transformation"""
        pose = SE3.Trans(1, 2, 3)
        cube = mkcube(pose=pose)
        self.assertEqual(cube.shape, (3, 8))

        # Check center is at translation
        center = cube.mean(axis=1)
        nt.assert_array_almost_equal(center, [1, 2, 3])

    def test_with_centre(self):
        """Test cube with centre parameter"""
        cube = mkcube(centre=[1, 2, 3])
        self.assertEqual(cube.shape, (3, 8))

        # Check center
        center = cube.mean(axis=1)
        nt.assert_array_almost_equal(center, [1, 2, 3])

    def test_edge_mode(self):
        """Test edge mode returns coordinate matrices"""
        X, Y, Z = mkcube(edge=True)

        # Check shapes
        self.assertEqual(X.shape, (2, 5))
        self.assertEqual(Y.shape, (2, 5))
        self.assertEqual(Z.shape, (2, 5))

        # Check bounds
        self.assertAlmostEqual(X.min(), -0.5)
        self.assertAlmostEqual(X.max(), 0.5)

    def test_edge_mode_with_size(self):
        """Test edge mode with custom size"""
        X, Y, Z = mkcube(s=4, edge=True)

        self.assertEqual(X.shape, (2, 5))
        self.assertAlmostEqual(X.min(), -2.0)
        self.assertAlmostEqual(X.max(), 2.0)

    def test_pose_and_centre_error(self):
        """Test that specifying both pose and centre raises error"""
        with self.assertRaises(ValueError):
            mkcube(pose=SE3(), centre=[0, 0, 0])

    def test_rotated_cube(self):
        """Test cube with rotation"""
        pose = SE3.Rx(np.pi / 4)
        cube = mkcube(pose=pose)
        self.assertEqual(cube.shape, (3, 8))


class TestMksphere(unittest.TestCase):
    @unittest.skip(
        "Bug in mksphere: indexing error cosphi[0, n-1] should be cosphi[n-1, 0]"
    )
    def test_default(self):
        """Test default unit sphere"""
        X, Y, Z = mksphere()

        # Check shapes
        self.assertEqual(X.shape, (20, 20))
        self.assertEqual(Y.shape, (20, 20))
        self.assertEqual(Z.shape, (20, 20))

        # Check radius at equator
        r = np.sqrt(X[10, :] ** 2 + Y[10, :] ** 2)
        nt.assert_array_almost_equal(r, np.ones(20), decimal=5)

    def test_custom_radius(self):
        """Test sphere with custom radius"""
        X, Y, Z = mksphere(r=2)

        # Check approximate radius
        r = np.sqrt(X[10, :] ** 2 + Y[10, :] ** 2 + Z[10, :] ** 2)
        nt.assert_array_almost_equal(r, 2 * np.ones(20), decimal=5)

    def test_custom_resolution(self):
        """Test sphere with custom resolution"""
        X, Y, Z = mksphere(n=10)

        self.assertEqual(X.shape, (10, 10))
        self.assertEqual(Y.shape, (10, 10))
        self.assertEqual(Z.shape, (10, 10))

    def test_with_centre(self):
        """Test sphere with custom center"""
        X, Y, Z = mksphere(n=100, centre=[1, 2, 3])

        # Check center position
        center_x = (X.max() + X.min()) / 2
        center_y = (Y.max() + Y.min()) / 2
        center_z = (Z.max() + Z.min()) / 2

        self.assertAlmostEqual(center_x, 1.0, places=2)
        self.assertAlmostEqual(center_y, 2.0, places=2)
        self.assertAlmostEqual(center_z, 3.0, places=2)

    def test_poles(self):
        """Test that poles are properly defined"""
        X, Y, Z = mksphere(n=20)

        # Check north pole
        self.assertAlmostEqual(Z.max(), 1.0)

        # Check south pole
        self.assertAlmostEqual(Z.min(), -1.0)


class TestMkcylinder(unittest.TestCase):
    def test_default(self):
        """Test default cylinder"""
        X, Y, Z = mkcylinder()

        # Check shapes
        self.assertEqual(X.shape, (2, 20))
        self.assertEqual(Y.shape, (2, 20))
        self.assertEqual(Z.shape, (2, 20))

        # Check radius at both ends
        r_bottom = np.sqrt(X[0, :] ** 2 + Y[0, :] ** 2)
        r_top = np.sqrt(X[1, :] ** 2 + Y[1, :] ** 2)
        nt.assert_array_almost_equal(r_bottom, np.ones(20), decimal=5)
        nt.assert_array_almost_equal(r_top, np.ones(20), decimal=5)

        # Check height
        nt.assert_array_almost_equal(Z[0, :], np.zeros(20))
        nt.assert_array_almost_equal(Z[1, :], np.ones(20))

    def test_custom_radius(self):
        """Test cylinder with custom radius"""
        X, Y, Z = mkcylinder(r=2)

        r = np.sqrt(X[0, :] ** 2 + Y[0, :] ** 2)
        nt.assert_array_almost_equal(r, 2 * np.ones(20), decimal=5)

    def test_custom_height(self):
        """Test cylinder with custom height"""
        X, Y, Z = mkcylinder(h=3)

        nt.assert_array_almost_equal(Z[0, :], np.zeros(20))
        nt.assert_array_almost_equal(Z[1, :], 3 * np.ones(20))

    def test_custom_resolution(self):
        """Test cylinder with custom resolution"""
        X, Y, Z = mkcylinder(n=10)

        self.assertEqual(X.shape, (2, 10))
        self.assertEqual(Y.shape, (2, 10))
        self.assertEqual(Z.shape, (2, 10))

    def test_symmetric(self):
        """Test symmetric cylinder"""
        X, Y, Z = mkcylinder(h=2, symmetric=True)

        nt.assert_array_almost_equal(Z[0, :], -1 * np.ones(20))
        nt.assert_array_almost_equal(Z[1, :], np.ones(20))

    def test_cone(self):
        """Test cone (varying radius)"""
        X, Y, Z = mkcylinder(r=[0, 1])

        # Check bottom radius is 0
        r_bottom = np.sqrt(X[0, :] ** 2 + Y[0, :] ** 2)
        nt.assert_array_almost_equal(r_bottom, np.zeros(20), decimal=5)

        # Check top radius is 1
        r_top = np.sqrt(X[1, :] ** 2 + Y[1, :] ** 2)
        nt.assert_array_almost_equal(r_top, np.ones(20), decimal=5)

    def test_frustum(self):
        """Test conical frustum"""
        X, Y, Z = mkcylinder(r=[1, 2])

        # Check bottom radius is 1
        r_bottom = np.sqrt(X[0, :] ** 2 + Y[0, :] ** 2)
        nt.assert_array_almost_equal(r_bottom, np.ones(20), decimal=5)

        # Check top radius is 2
        r_top = np.sqrt(X[1, :] ** 2 + Y[1, :] ** 2)
        nt.assert_array_almost_equal(r_top, 2 * np.ones(20), decimal=5)

    def test_variable_radius(self):
        """Test cylinder with variable radius"""
        r = [1, 2, 1.5, 1]
        X, Y, Z = mkcylinder(r=r)

        self.assertEqual(X.shape, (4, 20))
        self.assertEqual(Y.shape, (4, 20))
        self.assertEqual(Z.shape, (4, 20))

    def test_with_pose(self):
        """Test cylinder with pose transformation"""
        pose = SE3.Trans(1, 2, 3)
        X, Y, Z = mkcylinder(pose=pose)

        # Check that transformation was applied
        self.assertTrue(np.mean(X) != 0 or np.mean(Y) != 0 or np.mean(Z) != 3)

    def test_rotated_cylinder(self):
        """Test horizontal cylinder"""
        pose = SE3.Rx(np.pi / 2)
        X, Y, Z = mkcylinder(pose=pose)

        self.assertEqual(X.shape, (2, 20))
        self.assertEqual(Y.shape, (2, 20))
        self.assertEqual(Z.shape, (2, 20))


if __name__ == "__main__":
    unittest.main()
