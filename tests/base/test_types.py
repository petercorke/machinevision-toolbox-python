import math
import numpy as np
import numpy.testing as nt
import unittest
from machinevisiontoolbox.base.types import *
from spatialmath import SE3



class TestTypes(unittest.TestCase):

    def compare_int_images(self, im1, max1, im2, max2, tol=5e-3):
        d = im1.astype('float') / max1 - im2.astype('float') / max2
        self.assertTrue(np.all(np.abs(d) < tol))

    def test_int_image(self):

        # float image
        f = np.random.rand(20, 30).astype('float32')
        i = int_image(f)
        self.assertIsInstance(i, np.ndarray)
        self.assertEqual(i.dtype.type, np.uint8)
        self.compare_int_images(i, 0xff, f, 1)

        f = np.random.rand(20, 30).astype('float64')
        i = int_image(f)
        self.assertIsInstance(i, np.ndarray)
        self.assertEqual(i.dtype.type, np.uint8)
        self.compare_int_images(i, 0xff, f, 1)

        f = np.random.rand(20, 30).astype('float16')
        i = int_image(f)
        self.assertIsInstance(i, np.ndarray)
        self.assertEqual(i.dtype.type, np.uint8)
        self.compare_int_images(i, 0xff, f, 1)

        f = np.random.rand(20, 30, 3).astype('float32')
        i = int_image(f)
        self.assertIsInstance(i, np.ndarray)
        self.assertEqual(i.dtype.type, np.uint8)
        self.compare_int_images(i, 0xff, f, 1)

        f = np.random.rand(20, 30).astype('float32')
        i = int_image(f, intclass='int8')
        self.assertIsInstance(i, np.ndarray)
        self.assertEqual(i.dtype.type, np.int8)
        self.compare_int_images(i, 0x7f, f, 1)

        f = np.random.rand(20, 30).astype('float32')
        i = int_image(f, intclass='uint16')
        self.assertIsInstance(i, np.ndarray)
        self.assertEqual(i.dtype.type, np.uint16)
        self.compare_int_images(i, 0xffff, f, 1)

        f = np.random.rand(20, 30).astype('float32')
        i = int_image(f, intclass='int16')
        self.assertIsInstance(i, np.ndarray)
        self.assertEqual(i.dtype.type, np.int16)
        self.compare_int_images(i, 0x7fff, f, 1)

        # int image, little int to big int

        ii = np.random.randint(1, 255, size=(20, 30), dtype='uint8')
        i = int_image(ii, intclass='uint8')
        self.assertIsInstance(i, np.ndarray)
        self.assertEqual(i.dtype.type, np.uint8)
        self.compare_int_images(ii, 0xff, i, 0xff)

        i = int_image(ii, intclass='uint16')
        self.assertIsInstance(i, np.ndarray)
        self.assertEqual(i.dtype.type, np.uint16)
        self.compare_int_images(ii, 0xff, i, 0xffff)

        i = int_image(ii, intclass='int16')
        self.assertIsInstance(i, np.ndarray)
        self.assertEqual(i.dtype.type, np.int16)
        self.compare_int_images(ii, 0xff, i, 0x7fff)

        i = int_image(ii, intclass='int32')
        self.assertIsInstance(i, np.ndarray)
        self.assertEqual(i.dtype.type, np.int32)
        self.compare_int_images(ii, 0xff, i, 0x7fffffff)

        # int image, big int to little int

        ii = np.random.randint(1, 0xffff, size=(20, 30), dtype='uint16')
        i = int_image(ii, intclass='uint8')
        self.assertIsInstance(i, np.ndarray)
        self.assertEqual(i.dtype.type, np.uint8)
        self.compare_int_images(ii, 0xffff, i, 0xff)

        ii = np.random.randint(1, 1000, size=(20, 30), dtype='uint16')
        i = int_image(ii, intclass='uint8', maxintval=1000)
        self.assertIsInstance(i, np.ndarray)
        self.assertEqual(i.dtype.type, np.uint8)
        self.compare_int_images(ii, 1000, i, 0xff)

        # logical image
        li = np.random.rand(20, 30) >= 0.5
        i = int_image(li)
        self.assertIsInstance(i, np.ndarray)
        self.assertEqual(i.dtype.type, np.uint8)
        nt.assert_array_almost_equal(li*255, i)

        i = int_image(li, intclass='int16')
        self.assertIsInstance(i, np.ndarray)
        self.assertEqual(i.dtype.type, np.int16)
        nt.assert_array_almost_equal(li*0x7fff, i)

    def float_image(self):

        ## from float

        # convert to float32 by default
        fi = np.random.rand(20, 30).astype('float32')
        f = float_image(fi)
        self.assertIsInstance(f, np.ndarray)
        self.assertEqual(f.dtype.type, np.float32)
        nt.assert_array_almost_equal(fi, f)

        fi = np.random.rand(20, 30).astype('float64')
        f = float_image(fi)
        self.assertIsInstance(f, np.ndarray)
        self.assertEqual(f.dtype.type, np.float32)
        nt.assert_array_almost_equal(fi, f)

        fi = np.random.rand(20, 30).astype('float16')
        f = float_image(fi)
        self.assertIsInstance(f, np.ndarray)
        self.assertEqual(f.dtype.type, np.float32)
        nt.assert_array_almost_equal(fi, f)

        # convert to different types of float
        fi = np.random.rand(20, 30).astype('float32')
        f = float_image(fi, floatclass='float64')
        self.assertIsInstance(f, np.ndarray)
        self.assertEqual(f.dtype.type, np.float64)
        nt.assert_array_almost_equal(fi, f)

        f = float_image(fi, floatclass='float16')
        self.assertIsInstance(f, np.ndarray)
        self.assertEqual(f.dtype.type, np.float16)
        nt.assert_array_almost_equal(fi, f)

        ## from int

        # to different float types
        ii = np.random.randint(1, 0xff, size=(20, 30), dtype='uint8')
        f = float_image(ii)
        self.assertIsInstance(f, np.ndarray)
        self.assertEqual(f.dtype.type, np.float32)
        self.compare_int_images(ii, 0xff, f, 1)

        f = float_image(ii, floatclass='float64')
        self.assertIsInstance(f, np.ndarray)
        self.assertEqual(f.dtype.type, np.float64)
        self.compare_int_images(ii, 0xff, f, 1)

        f = float_image(ii, floatclass='float16')
        self.assertIsInstance(f, np.ndarray)
        self.assertEqual(f.dtype.type, np.float16)
        self.compare_int_images(ii, 0xff, f, 1)

        f = float_image(ii, floatclass='single')
        self.assertIsInstance(f, np.ndarray)
        self.assertEqual(f.dtype.type, np.float32)
        self.compare_int_images(ii, 0xff, f, 1)

        f = float_image(ii, floatclass='half')
        self.assertIsInstance(f, np.ndarray)
        self.assertEqual(f.dtype.type, np.float16)
        self.compare_int_images(ii, 0xff, f, 1)

        f = float_image(ii, floatclass='double')
        self.assertIsInstance(f, np.ndarray)
        self.assertEqual(f.dtype.type, np.float64)
        self.compare_int_images(ii, 0xff, f, 1)

        # from different int types
        ii = np.random.randint(1, 0xffff, size=(20, 30), dtype='uint16')
        f = float_image(ii)
        self.assertIsInstance(f, np.ndarray)
        self.assertEqual(f.dtype.type, np.float32)
        self.compare_int_images(ii, 0xff, f, 1)

        f = float_image(ii, floatclass='float16')
        self.assertIsInstance(f, np.ndarray)
        self.assertEqual(f.dtype.type, np.float16)
        self.compare_int_images(ii, 0xff, f, 1)

        ii = np.random.randint(1, 0xffff, size=(20, 30), dtype='int8')
        f = float_image(ii)
        self.assertIsInstance(f, np.ndarray)
        self.assertEqual(f.dtype.type, np.float32)
        self.compare_int_images(ii, 0x7f, f, 1)

        ii = np.random.randint(1, 0xffff, size=(20, 30), dtype='int16')
        f = float_image(ii)
        self.assertIsInstance(f, np.ndarray)
        self.assertEqual(f.dtype.type, np.float32)
        self.compare_int_images(ii, 0x7fff, f, 1)

        ii = np.random.randint(1, 1000, size=(20, 30), dtype='uint16')
        f = float_image(ii, maxintval=1000)
        self.assertIsInstance(f, np.ndarray)
        self.assertEqual(f.dtype.type, np.float32)
        self.compare_int_images(ii, 1000, f, 1)

        ## from logical image
        li = np.random.rand(20, 30) >= 0.5
        f = float_image(li)
        self.assertIsInstance(f, np.ndarray)
        self.assertEqual(f.dtype.type, np.float32)
        nt.assert_array_almost_equal(li, f)
# ------------------------------------------------------------------------ #
if __name__ == '__main__':

    unittest.main()