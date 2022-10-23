import math
import numpy as np
import numpy.testing as nt
import unittest
from machinevisiontoolbox.base.shapes import *
from spatialmath import SE3

class TestShapes(unittest.TestCase):

    def test_mkgrid(self):

        p = mkgrid(3, side=2)
        self.assertIsInstance(p, np.ndarray)
        self.assertEqual(p.shape, (3, 9))
        nt.assert_array_almost_equal(p.mean(axis=1), [0, 0, 0])
        nt.assert_array_almost_equal(p.min(axis=1), [-1, -1, 0])
        nt.assert_array_almost_equal(p.max(axis=1), [1, 1, 0])

        p = mkgrid(3, side=2, pose=SE3.Trans(1,2,3))
        self.assertIsInstance(p, np.ndarray)
        self.assertEqual(p.shape, (3, 9))
        nt.assert_array_almost_equal(p.mean(axis=1), [1, 2, 3])
        nt.assert_array_almost_equal(p.min(axis=1), [0, 1, 3])
        nt.assert_array_almost_equal(p.max(axis=1), [2, 3, 3])

    def test_mkcube_vertex(self):

        p = mkcube(side=2)
        self.assertIsInstance(p, np.ndarray)
        self.assertEqual(p.shape, (3, 8))
        nt.assert_array_almost_equal(p.mean(axis=1), [0, 0, 0])
        nt.assert_array_almost_equal(p.min(axis=1), [-1, -1, -1])
        nt.assert_array_almost_equal(p.max(axis=1), [1, 1, 1])

        p = mkcube(side=2, facepoint=True)
        self.assertIsInstance(p, np.ndarray)
        self.assertEqual(p.shape, (3, 14))
        nt.assert_array_almost_equal(p.mean(axis=1), [0, 0, 0])
        nt.assert_array_almost_equal(p.min(axis=1), [-1, -1, -1])
        nt.assert_array_almost_equal(p.max(axis=1), [1, 1, 1])

        p = mkcube(side=2, centre=[1,2,3])
        self.assertIsInstance(p, np.ndarray)
        self.assertEqual(p.shape, (3, 8))
        nt.assert_array_almost_equal(p.mean(axis=1), [1, 2, 3])
        nt.assert_array_almost_equal(p.min(axis=1), [0, 1, 2])
        nt.assert_array_almost_equal(p.max(axis=1), [2, 3, 4])

        p = mkcube(side=2, pose=SE3.Trans([1,2,3]))
        self.assertIsInstance(p, np.ndarray)
        self.assertEqual(p.shape, (3, 8))
        nt.assert_array_almost_equal(p.mean(axis=1), [1, 2, 3])
        nt.assert_array_almost_equal(p.min(axis=1), [0, 1, 2])
        nt.assert_array_almost_equal(p.max(axis=1), [2, 3, 4])

    def test_mkcube_edge(self):

        X, Y, Z = mkcube(side=2, edge=True)

        for p in [X, Y, Z]:
            self.assertIsInstance(p, np.ndarray)
            self.assertEqual(p.shape, (2, 5))
    
            nt.assert_array_almost_equal(p.min(), -1)
            nt.assert_array_almost_equal(p.max(), 1)

        X, Y, Z = mkcube(side=2, edge=True, centre=[1, 2, 3])
        for p in [X, Y, Z]:
            self.assertIsInstance(p, np.ndarray)
            self.assertEqual(p.shape, (2, 5))

        nt.assert_array_almost_equal(X.min(), 0)
        nt.assert_array_almost_equal(X.max(), 2)

        nt.assert_array_almost_equal(Y.min(), 1)
        nt.assert_array_almost_equal(Y.max(), 3)

        nt.assert_array_almost_equal(Z.min(), 2)
        nt.assert_array_almost_equal(Z.max(), 4)

        X, Y, Z = mkcube(side=2, edge=True, pose=SE3.Trans([1,2,3]))
        for p in [X, Y, Z]:
            self.assertIsInstance(p, np.ndarray)
            self.assertEqual(p.shape, (2, 5))

        nt.assert_array_almost_equal(X.min(), 0)
        nt.assert_array_almost_equal(X.max(), 2)

        nt.assert_array_almost_equal(Y.min(), 1)
        nt.assert_array_almost_equal(Y.max(), 3)

        nt.assert_array_almost_equal(Z.min(), 2)
        nt.assert_array_almost_equal(Z.max(), 4)

    def mksphere(self):
        X, Y, Z = mksphere(r=2, n=30)
        for p in [X, Y, Z]:
            self.assertIsInstance(p, np.ndarray)
            self.assertEqual(p.shape, (30, 30))
            nt.assert_array_almost_equal(p.mean(), 0)
            nt.assert_array_almost_equal(p.min(), -2)
            nt.assert_array_almost_equal(p.max(), 2)

        X, Y, Z = mksphere(r=2, n=30, centre=[1,2,3])
        for p, c in zip([X, Y, Z], [1,2,3]):
            self.assertIsInstance(p, np.ndarray)
            self.assertEqual(p.shape, (30, 30))
            nt.assert_array_almost_equal(p.mean(), c)
            nt.assert_array_almost_equal(p.min(), -2+c)
            nt.assert_array_almost_equal(p.max(), 2+c)

    def mkcylinder(self):
        X, Y, Z = mkcylinder(r=2, h=6, n=30)
        for p in [X, Y, Z]:
            self.assertIsInstance(p, np.ndarray)
            self.assertEqual(p.shape, (2, 30))
        nt.assert_array_almost_equal(X.mean(), 0)
        nt.assert_array_almost_equal(X.min(), -2)
        nt.assert_array_almost_equal(X.max(), 2)

        nt.assert_array_almost_equal(Y.mean(), 0)
        nt.assert_array_almost_equal(Y.min(), -2)
        nt.assert_array_almost_equal(Y.max(), 2)

        nt.assert_array_almost_equal(Z.mean(), 3)
        nt.assert_array_almost_equal(Z.min(), 0)
        nt.assert_array_almost_equal(Z.max(), 6)

        # symmetric
        X, Y, Z = mkcylinder(r=2, h=6, n=30, symmetric=True)
        for p in [X, Y, Z]:
            self.assertIsInstance(p, np.ndarray)
            self.assertEqual(p.shape, (2, 30))
        nt.assert_array_almost_equal(X.mean(), 0)
        nt.assert_array_almost_equal(X.min(), -2)
        nt.assert_array_almost_equal(X.max(), 2)

        nt.assert_array_almost_equal(Y.mean(), 0)
        nt.assert_array_almost_equal(Y.min(), -2)
        nt.assert_array_almost_equal(Y.max(), 2)

        nt.assert_array_almost_equal(Z.mean(), 0)
        nt.assert_array_almost_equal(Z.min(), -3)
        nt.assert_array_almost_equal(Z.max(), 3)

        # shifted
        X, Y, Z = mkcylinder(r=2, h=6, n=30, pose=SE3.Trans([1,2,3]))
        for p in [X, Y, Z]:
            self.assertIsInstance(p, np.ndarray)
            self.assertEqual(p.shape, (2, 30))
        nt.assert_array_almost_equal(X.mean(), 1)
        nt.assert_array_almost_equal(X.min(), -1)
        nt.assert_array_almost_equal(X.max(), 3)

        nt.assert_array_almost_equal(Y.mean(), 2)
        nt.assert_array_almost_equal(Y.min(), 0)
        nt.assert_array_almost_equal(Y.max(), 4)

        nt.assert_array_almost_equal(Z.mean(), 6)
        nt.assert_array_almost_equal(Z.min(), 3)
        nt.assert_array_almost_equal(Z.max(), 9)

        # cone
        X, Y, Z = mkcylinder(r=[1,2], h=6, n=30, symmetric=True)
        for p in [X, Y, Z]:
            self.assertIsInstance(p, np.ndarray)
            self.assertEqual(p.shape, (2, 30))
        nt.assert_array_almost_equal(X.mean(), 0)
        nt.assert_array_almost_equal(X.min(), -2)
        nt.assert_array_almost_equal(X.max(), 2)

        nt.assert_array_almost_equal(Y.mean(), 0)
        nt.assert_array_almost_equal(Y.min(), -2)
        nt.assert_array_almost_equal(Y.max(), 2)

        nt.assert_array_almost_equal(Z.mean(), 0)
        nt.assert_array_almost_equal(Z.min(), -3)
        nt.assert_array_almost_equal(Z.max(), 3)

# ------------------------------------------------------------------------ #
if __name__ == '__main__':

    unittest.main()