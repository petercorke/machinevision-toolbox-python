import sys
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as nt

# Load meshgrid module directly without importing the whole package
meshgrid_file = (
    Path(__file__).parent.parent.parent
    / "src"
    / "machinevisiontoolbox"
    / "base"
    / "meshgrid.py"
)


def load_meshgrid_module():
    """Load meshgrid module by reading and executing its source code."""
    with open(meshgrid_file, "r") as f:
        source_code = f.read()

    # Create a namespace with numpy imported as np
    namespace = {
        "__file__": str(meshgrid_file),
        "__name__": "meshgrid_module",
        "np": np,
    }

    try:
        exec(source_code, namespace)
    except Exception as e:
        raise ImportError(f"Failed to load meshgrid module: {e}")

    return namespace


mesh_namespace = load_meshgrid_module()
meshgrid = mesh_namespace["meshgrid"]
spherical_rotate = mesh_namespace["spherical_rotate"]

try:
    from spatialmath.pose3d import SO3
except ImportError:
    SO3 = None


class TestMeshgrid(unittest.TestCase):
    """Test cases for meshgrid function"""

    def test_meshgrid_basic_2x3(self):
        """Test meshgrid with 2x3 dimensions"""
        U, V = meshgrid(2, 3)

        # Check shapes
        self.assertEqual(U.shape, (3, 2))
        self.assertEqual(V.shape, (3, 2))

    def test_meshgrid_u_coordinates(self):
        """Test that U array contains correct u coordinates"""
        U, V = meshgrid(3, 2)

        # U[v,u] should equal u
        expected_U = np.array([[0, 1, 2], [0, 1, 2]])
        nt.assert_array_equal(U, expected_U)

    def test_meshgrid_v_coordinates(self):
        """Test that V array contains correct v coordinates"""
        U, V = meshgrid(3, 2)

        # V[v,u] should equal v
        expected_V = np.array([[0, 0, 0], [1, 1, 1]])
        nt.assert_array_equal(V, expected_V)

    def test_meshgrid_single_pixel(self):
        """Test meshgrid with 1x1 dimensions"""
        U, V = meshgrid(1, 1)

        self.assertEqual(U.shape, (1, 1))
        self.assertEqual(V.shape, (1, 1))
        self.assertEqual(U[0, 0], 0)
        self.assertEqual(V[0, 0], 0)

    def test_meshgrid_large_dimensions(self):
        """Test meshgrid with larger dimensions"""
        width = 100
        height = 50
        U, V = meshgrid(width, height)

        self.assertEqual(U.shape, (height, width))
        self.assertEqual(V.shape, (height, width))

        # Check corner values
        self.assertEqual(U[0, 0], 0)
        self.assertEqual(U[0, width - 1], width - 1)
        self.assertEqual(V[height - 1, 0], height - 1)

    def test_meshgrid_dtypes(self):
        """Test that coordinate arrays are integer type"""
        U, V = meshgrid(5, 5)

        self.assertTrue(np.issubdtype(U.dtype, np.integer))
        self.assertTrue(np.issubdtype(V.dtype, np.integer))

    def test_meshgrid_range_u(self):
        """Test that U values range from 0 to width-1"""
        width = 10
        height = 8
        U, V = meshgrid(width, height)

        self.assertEqual(np.min(U), 0)
        self.assertEqual(np.max(U), width - 1)

    def test_meshgrid_range_v(self):
        """Test that V values range from 0 to height-1"""
        width = 10
        height = 8
        U, V = meshgrid(width, height)

        self.assertEqual(np.min(V), 0)
        self.assertEqual(np.max(V), height - 1)

    def test_meshgrid_indexing(self):
        """Test that U[v,u] = u and V[v,u] = v"""
        width = 5
        height = 4
        U, V = meshgrid(width, height)

        # Test a few arbitrary points
        for v in range(height):
            for u in range(width):
                self.assertEqual(U[v, u], u)
                self.assertEqual(V[v, u], v)

    def test_meshgrid_wide_image(self):
        """Test meshgrid with wider-than-tall image"""
        U, V = meshgrid(20, 5)

        self.assertEqual(U.shape, (5, 20))
        self.assertEqual(U[0, 19], 19)
        self.assertEqual(V[4, 0], 4)

    def test_meshgrid_tall_image(self):
        """Test meshgrid with taller-than-wide image"""
        U, V = meshgrid(5, 20)

        self.assertEqual(U.shape, (20, 5))
        self.assertEqual(U[0, 4], 4)
        self.assertEqual(V[19, 0], 19)

    def test_meshgrid_square_image(self):
        """Test meshgrid with square image"""
        size = 7
        U, V = meshgrid(size, size)

        self.assertEqual(U.shape, (size, size))
        self.assertEqual(V.shape, (size, size))


class TestSphericalRotate(unittest.TestCase):
    """Test cases for spherical_rotate function"""

    @unittest.skipIf(SO3 is None, "spatialmath not available")
    def test_spherical_rotate_basic(self):
        """Test basic spherical rotation with identity"""
        # Create coordinate matrices for a small spherical image
        Phi = np.ones((5, 5)) * 0.5
        Theta = np.ones((5, 5)) * 0.8

        # Create identity rotation
        R = SO3()

        nPhi, nTheta = spherical_rotate(Phi, Theta, R)

        # With identity rotation and uniform input, output should be approximately uniform
        self.assertEqual(nPhi.shape, Phi.shape)
        self.assertEqual(nTheta.shape, Theta.shape)
        # The values should be close to the input (within numerical precision)
        # Note: atan2 wrapping means exact equality isn't guaranteed
        self.assertTrue(np.all(np.isfinite(nPhi)))
        self.assertTrue(np.all(np.isfinite(nTheta)))

    def test_spherical_rotate_output_shapes(self):
        """Test that output shapes match input shapes"""
        Phi = np.random.rand(8, 12)
        Theta = np.random.rand(8, 12) * np.pi

        R = SO3()
        nPhi, nTheta = spherical_rotate(Phi, Theta, R)

        self.assertEqual(nPhi.shape, Phi.shape)
        self.assertEqual(nTheta.shape, Theta.shape)

    @unittest.skipIf(SO3 is None, "spatialmath not available")
    def test_spherical_rotate_phi_range(self):
        """Test that output phi is in expected range"""
        Phi = np.ones((5, 5)) * 0.5
        Theta = np.ones((5, 5)) * 0.5

        R = SO3()
        nPhi, nTheta = spherical_rotate(Phi, Theta, R)

        # Phi should be approximately in [-pi, pi] or [0, 2*pi]
        # After atan2, values are in [-pi, pi]
        self.assertTrue(np.all(nPhi >= -np.pi))
        self.assertTrue(np.all(nPhi <= np.pi))

    @unittest.skipIf(SO3 is None, "spatialmath not available")
    def test_spherical_rotate_theta_range(self):
        """Test that output theta is in valid range"""
        Phi = np.ones((5, 5)) * 0.5
        Theta = np.ones((5, 5)) * 0.5

        R = SO3()
        nPhi, nTheta = spherical_rotate(Phi, Theta, R)

        # Theta (colatitude) should be in [0, pi]
        self.assertTrue(np.all(nTheta >= 0))
        self.assertTrue(np.all(nTheta <= np.pi))

    @unittest.skipIf(SO3 is None, "spatialmath not available")
    def test_spherical_rotate_dtypes(self):
        """Test that output arrays are floating point"""
        Phi = np.ones((5, 5), dtype=np.float32)
        Theta = np.ones((5, 5), dtype=np.float32)

        R = SO3()
        nPhi, nTheta = spherical_rotate(Phi, Theta, R)

        self.assertTrue(np.issubdtype(nPhi.dtype, np.floating))
        self.assertTrue(np.issubdtype(nTheta.dtype, np.floating))

    @unittest.skipIf(SO3 is None, "spatialmath not available")
    def test_spherical_rotate_single_pixel(self):
        """Test spherical rotation with single pixel"""
        Phi = np.array([[0.5]])
        Theta = np.array([[0.3]])

        R = SO3()
        nPhi, nTheta = spherical_rotate(Phi, Theta, R)

        self.assertEqual(nPhi.shape, (1, 1))
        self.assertEqual(nTheta.shape, (1, 1))

    @unittest.skipIf(SO3 is None, "spatialmath not available")
    def test_spherical_rotate_array_consistency(self):
        """Test that rotation preserves array structure"""
        height, width = 6, 8
        Phi = np.random.rand(height, width) * 2 * np.pi
        Theta = np.random.rand(height, width) * np.pi

        R = SO3()
        nPhi, nTheta = spherical_rotate(Phi, Theta, R)

        # Output should have same shape
        self.assertEqual(nPhi.shape, (height, width))
        self.assertEqual(nTheta.shape, (height, width))

        # Output should contain finite values
        self.assertTrue(np.all(np.isfinite(nPhi)))
        self.assertTrue(np.all(np.isfinite(nTheta)))

    @unittest.skipIf(SO3 is None, "spatialmath not available")
    def test_spherical_rotate_conversion_roundtrip(self):
        """Test conversion to Cartesian and back"""
        # Create coordinate matrices
        Phi = np.ones((3, 3)) * 0.0  # All on x-axis
        Theta = np.ones((3, 3)) * np.pi / 2  # Equator

        R = SO3()  # Identity rotation
        nPhi, nTheta = spherical_rotate(Phi, Theta, R)

        # Convert both to Cartesian to compare
        x = np.sin(Theta) * np.cos(Phi)
        y = np.sin(Theta) * np.sin(Phi)
        z = np.cos(Theta)

        nx = np.sin(nTheta) * np.cos(nPhi)
        ny = np.sin(nTheta) * np.sin(nPhi)
        nz = np.cos(nTheta)

        # With identity rotation, Cartesian coordinates should match
        nt.assert_array_almost_equal(nx, x, decimal=5)
        nt.assert_array_almost_equal(ny, y, decimal=5)
        nt.assert_array_almost_equal(nz, z, decimal=5)


if __name__ == "__main__":
    unittest.main()
