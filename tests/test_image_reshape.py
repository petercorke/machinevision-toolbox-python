import unittest

import numpy as np
import numpy.testing as nt

from machinevisiontoolbox import Image


class TestImageReshape(unittest.TestCase):
    def test_trim(self):
        # Add test cases for the trim method
        im = Image.String("123|456|789")

        x = im.trim()
        self.assertEqual(x, Image.String("123|456|789"))

        x = im.trim(left=1)
        self.assertEqual(x, Image.String("23|56|89"))

        x = im.trim(left=2)
        self.assertEqual(x, Image.String("3|6|9"))

        x = im.trim(right=1)
        self.assertEqual(x, Image.String("12|45|78"))

        x = im.trim(right=2)
        self.assertEqual(x, Image.String("1|4|7"))

        x = im.trim(top=1)
        self.assertEqual(x, Image.String("456|789"))

        x = im.trim(top=2)
        self.assertEqual(x, Image.String("789"))

        x = im.trim(bottom=1)
        self.assertEqual(x, Image.String("123|456"))

        x = im.trim(bottom=2)
        self.assertEqual(x, Image.String("123"))

        x = im.trim(left=1, right=1)
        self.assertEqual(x, Image.String("2|5|8"))

        x = im.trim(top=1, bottom=1)
        self.assertEqual(x, Image.String("456"))

        x = im.trim(left=1, right=1, top=1, bottom=1)
        self.assertEqual(x, Image.String("5"))

        # color image
        im = Image.String("123|456|789", "ABC|DEF|GHI", "JKL|MNO|PQR")

        x = im.trim()
        self.assertEqual(x, Image.String("123|456|789", "ABC|DEF|GHI", "JKL|MNO|PQR"))

        x = im.trim(left=1)
        self.assertEqual(x, Image.String("23|56|89", "BC|EF|HI", "KL|NO|QR"))

        x = im.trim(left=2)
        self.assertEqual(x, Image.String("3|6|9", "C|F|I", "L|O|R"))

        x = im.trim(right=1)
        self.assertEqual(x, Image.String("12|45|78", "AB|DE|GH", "JK|MN|PQ"))

        x = im.trim(right=2)
        self.assertEqual(x, Image.String("1|4|7", "A|D|G", "J|M|P"))

        x = im.trim(top=1)
        self.assertEqual(x, Image.String("456|789", "DEF|GHI", "MNO|PQR"))

        x = im.trim(top=2)
        self.assertEqual(x, Image.String("789", "GHI", "PQR"))

        x = im.trim(bottom=1)
        self.assertEqual(x, Image.String("123|456", "ABC|DEF", "JKL|MNO"))

        x = im.trim(bottom=2)
        self.assertEqual(x, Image.String("123", "ABC", "JKL"))

        x = im.trim(left=1, right=1)
        self.assertEqual(x, Image.String("2|5|8", "B|E|H", "K|N|Q"))

        x = im.trim(top=1, bottom=1)
        self.assertEqual(x, Image.String("456", "DEF", "MNO"))

        x = im.trim(left=1, right=1, top=1, bottom=1)
        self.assertEqual(x, Image.String("5", "E", "N"))

    def test_pad(self):
        # Add test cases for the pad method
        im = Image.String("123|456")

        x = im.pad()
        self.assertEqual(x, Image.String("123|456"))

        x = im.pad(left=1)
        self.assertEqual(x, Image.String("0123|0456"))

        x = im.pad(left=2)
        self.assertEqual(x, Image.String("00123|00456"))

        x = im.pad(left=2, value=9)
        self.assertEqual(x, Image.String("99123|99456"))

        x = im.pad(right=1)
        self.assertEqual(x, Image.String("1230|4560"))

        x = im.pad(right=2)
        self.assertEqual(x, Image.String("12300|45600"))

        x = im.pad(right=2, value=9)
        self.assertEqual(x, Image.String("12399|45699"))

        x = im.pad(top=1)
        self.assertEqual(x, Image.String("000|123|456"))

        x = im.pad(top=2)
        self.assertEqual(x, Image.String("000|000|123|456"))

        x = im.pad(top=2, value=9)
        self.assertEqual(x, Image.String("999|999|123|456"))

        x = im.pad(bottom=1)
        self.assertEqual(x, Image.String("123|456|000"))

        x = im.pad(bottom=2)
        self.assertEqual(x, Image.String("123|456|000|000"))

        x = im.pad(bottom=2, value=9)
        self.assertEqual(x, Image.String("123|456|999|999"))

        x = im.pad(left=1, right=2, value=9)
        self.assertEqual(x, Image.String("912399|945699"))

        x = im.pad(left=1, right=2, top=2, bottom=1, value=9)
        self.assertEqual(x, Image.String("999999|999999|912399|945699|999999"))

        # color image
        im = Image.String("123|456", "789|ABC", "DEF|GHI")

        x = im.pad()
        self.assertEqual(x, Image.String("123|456", "789|ABC", "DEF|GHI"))

        x = im.pad(left=1)
        self.assertEqual(x, Image.String("0123|0456", "0789|0ABC", "0DEF|0GHI"))

        x = im.pad(left=2)
        self.assertEqual(x, Image.String("00123|00456", "00789|00ABC", "00DEF|00GHI"))

        x = im.pad(left=2, value=(7, 8, 9))
        self.assertEqual(x, Image.String("77123|77456", "88789|88ABC", "99DEF|99GHI"))

        x = im.pad(right=1)
        self.assertEqual(x, Image.String("1230|4560", "7890|ABC0", "DEF0|GHI0"))

        x = im.pad(right=2)
        self.assertEqual(x, Image.String("12300|45600", "78900|ABC00", "DEF00|GHI00"))

        x = im.pad(right=2, value=(7, 8, 9))
        self.assertEqual(x, Image.String("12377|45677", "78988|ABC88", "DEF99|GHI99"))

        x = im.pad(top=1)
        self.assertEqual(x, Image.String("000|123|456", "000|789|ABC", "000|DEF|GHI"))

        x = im.pad(top=2)
        self.assertEqual(
            x, Image.String("000|000|123|456", "000|000|789|ABC", "000|000|DEF|GHI")
        )

        x = im.pad(top=2, value=(7, 8, 9))
        self.assertEqual(
            x, Image.String("777|777|123|456", "888|888|789|ABC", "999|999|DEF|GHI")
        )

        x = im.pad(bottom=1)
        self.assertEqual(x, Image.String("123|456|000", "789|ABC|000", "DEF|GHI|000"))

        x = im.pad(bottom=2)
        self.assertEqual(
            x, Image.String("123|456|000|000", "789|ABC|000|000", "DEF|GHI|000|000")
        )

        x = im.pad(bottom=2, value=(7, 8, 9))
        self.assertEqual(
            x, Image.String("123|456|777|777", "789|ABC|888|888", "DEF|GHI|999|999")
        )

        x = im.pad(left=1, right=2, value=(7, 8, 9))
        self.assertEqual(
            x, Image.String("712377|745677", "878988|8ABC88", "9DEF99|9GHI99")
        )

        x = im.pad(left=1, right=2, top=2, bottom=1, value=(7, 8, 9))
        self.assertEqual(
            x,
            Image.String(
                "777777|777777|712377|745677|777777",
                "888888|888888|878988|8ABC88|888888",
                "999999|999999|9DEF99|9GHI99|999999",
            ),
        )

        # float image
        im = Image.Random(size=(3, 4), dtype="float")
        x = im.pad(left=1, right=2, top=2, bottom=1, value=0.7)
        nt.assert_almost_equal(x.array[2:6, 1:4], im.array)
        nt.assert_almost_equal(x.array[:, 0], 0.7)  # left
        nt.assert_almost_equal(x.array[:, 4:], 0.7)  # right
        nt.assert_almost_equal(x.array[0:2, :], 0.7)  # top
        nt.assert_almost_equal(x.array[6, :], 0.7)  # bottom

        # color float image
        im = Image.Random(size=(3, 4), dtype="float", colororder="RGB")
        x = im.pad(left=1, right=2, top=2, bottom=1, value=0.7)
        for p in range(3):
            nt.assert_almost_equal(x.array[2:6, 1:4, p], im.array[:, :, p])
        nt.assert_almost_equal(x.array[:, 0, :], 0.7)  # left
        nt.assert_almost_equal(x.array[:, 4:, :], 0.7)  # right
        nt.assert_almost_equal(x.array[0:2, :, :], 0.7)  # top
        nt.assert_almost_equal(x.array[6, :, :], 0.7)  # bottom

    def test_dice(self):
        # Add test cases for the dice method
        im = Image.Random(size=(20, 30))

        sims = im.dice(shape=10)
        self.assertEqual(len(sims), 6)
        for sim in sims:
            self.assertEqual(sim.shape, (10, 10))
        nt.assert_array_equal(sims[0].array[:10, :10], im.array[:10, :10])
        nt.assert_array_equal(sims[1].array[:10, :10], im.array[:10, 10:20])
        nt.assert_array_equal(sims[2].array[:10, :10], im.array[10:20, :10])
        nt.assert_array_equal(sims[3].array[:10, :10], im.array[10:20, 10:20])
        nt.assert_array_equal(sims[4].array[:10, :10], im.array[20:30, :10])
        nt.assert_array_equal(sims[5].array[:10, :10], im.array[20:30, 10:200])

    def test_decimate(self):
        # Add test cases for the decimate method
        im = Image.String("1234|5678|9ABC|DEFG")

        x = im.decimate(1)
        self.assertEqual(x, im)

        x = im.decimate(2)
        self.assertEqual(x, Image.String("13|9C"))

        x = im.decimate(4)
        self.assertEqual(x, Image.String("1"))

        im = Image.String(
            "1234|5678|9ABC|DEFG", "2345|6789|ABCD|EFGH", "3456|789A|BCDE|FGHI"
        )
        x = im.decimate(1)
        self.assertEqual(x, im)

        x = im.decimate(2)
        self.assertEqual(x, Image.String("13|9B", "24|AC", "35|BD"))

        x = im.decimate(4)
        self.assertEqual(x, Image.String("1", "2", "3"))

        # TODO test for smoothing option

    def test_replicate(self):
        im = Image.String("123|456|789")

        x = im.replicate(1)
        self.assertEqual(x, im)

        x = im.replicate(2)
        self.assertEqual(x, Image.String("112233|112233|445566|445566|778899|778899"))

        # color image
        im = Image.String("123|456|789", "ABC|DEF|GHI", "JKL|MNO|PQR")
        # x = im.replicate(1)
        # self.assertEqual(x, im)

        x = im.replicate(2)
        self.assertEqual(
            x,
            Image.String(
                "112233|112233|445566|445566|778899|778899",
                "AABBCC|AABBCC|DDEEFF|DDEEFF|GGHHII|GGHHII",
                "JJKKLL|JJKKLL|MMNNOO|MMNNOO|PPQQRR|PPQQRR",
            ),
        )

    def test_roi(self):
        # Add test cases for the roi method
        im = Image.Random(size=(20, 30))
        roi = im.roi((5, 14, 10, 19))
        self.assertEqual(roi.shape, (10, 10))
        nt.assert_array_equal(roi.array, im.array[10:20, 5:15])

        im = Image.Random(size=(20, 30), colororder="RGB")
        roi = im.roi((5, 14, 10, 19))
        self.assertEqual(roi.shape, (10, 10, 3))
        self.assertEqual(roi.colororder_str, "R:G:B")
        nt.assert_array_equal(roi.array, im.array[10:20, 5:15, :])

    def test_samesize(self):
        im1 = Image.Random(size=(100, 120))
        im2 = Image.Random(size=(90, 105))

        x = im1.samesize(im2)
        self.assertEqual(x.size, im2.size)

        im1 = Image.Random(size=(100, 120), colororder="RGB")
        im2 = Image.Random(size=(90, 105))

        x = im1.samesize(im2)
        self.assertEqual(x.size, im2.size)
        self.assertEqual(x.colororder_str, "R:G:B")

        # float image
        im1 = Image.Random(size=(100, 120), dtype="float")
        im2 = Image.Random(size=(90, 105), dtype="float")

        x = im1.samesize(im2)
        self.assertEqual(x.size, im2.size)

        im1 = Image.Random(size=(100, 120), colororder="RGB")
        im2 = Image.Random(size=(90, 105))

        x = im1.samesize(im2)
        self.assertEqual(x.size, im2.size)
        self.assertEqual(x.colororder_str, "R:G:B")

    def test_scale(self):
        # Add test cases for the scale method
        im = Image.Squares(1, size=256)
        x = im.scale(1)
        self.assertEqual(x.size, (256, 256))
        nt.assert_array_equal(x.array, im.array)

        im = Image.Squares(1, size=256)
        x = im.scale(0.5)
        self.assertEqual(x.size, (128, 128))

        # very crude check that the image is still a square
        nt.assert_approx_equal(im.mpq(0, 0) / x.mpq(0, 0), 4, significant=2)
        y = im.decimate(2)
        self.assertTrue((x ^ y).mpq(0, 0) < 130)

        # now expand the image and compare to original
        x = x.scale(2)
        self.assertTrue((x ^ im).mpq(0, 0) < 260)

        # color image
        im = Image.Squares(1, size=256).colorize(colororder="RGB")
        x = im.scale(0.5)
        self.assertEqual(x.size, (128, 128))
        self.assertEqual(x.colororder_str, "R:G:B")

        # very crude check that the image is still a square
        y = im.decimate(2)

        for p in range(3):
            nt.assert_approx_equal(im[p].mpq(0, 0) / x[p].mpq(0, 0), 4, significant=2)
            self.assertTrue((x[p] ^ y[p]).mpq(0, 0) < 130)

    def test_rotate(self):
        def centroid(im):
            y, x = np.where(im.array)
            return np.mean(x), np.mean(y)

        im = Image.Squares(1, size=256)
        x = im.rotate(0)
        self.assertEqual(x.size, (256, 256))
        nt.assert_array_equal(x.array, im.array)

        im = Image.Squares(1, size=256)
        x = im.rotate(np.pi / 4)
        self.assertEqual(x.size, (256, 256))

        # very crude check that the image is still a square
        nt.assert_approx_equal(im.mpq(0, 0) / x.mpq(0, 0), 1, significant=2)
        xc, yc = centroid(im)
        nt.assert_approx_equal(xc, 128, significant=2)
        nt.assert_approx_equal(yc, 128, significant=2)

        # rotate about top left corner
        im = Image.Squares(1, size=256)
        x = im.rotate(0.3, centre=(0, 0))
        self.assertEqual(x.size, (256, 256))

        # very crude check that the image is still a square
        nt.assert_approx_equal(im.mpq(0, 0) / x.mpq(0, 0), 1, significant=2)
        xc, yc = centroid(x)
        self.assertTrue(xc > 128)
        self.assertTrue(yc < 128)

        # color

        im = Image.Squares(1, size=256).colorize(colororder="RGB")
        x = im.rotate(np.pi / 4)
        self.assertEqual(x.size, (256, 256))

        # very crude check that the image is still a square
        for p in range(3):
            nt.assert_approx_equal(im[p].mpq(0, 0) / x[p].mpq(0, 0), 1, significant=2)
            xc, yc = centroid(im[p])
            nt.assert_approx_equal(xc, 128, significant=2)
            nt.assert_approx_equal(yc, 128, significant=2)

    def test_rotate_spherical(self):
        # Add test cases for the rotate_spherical method
        pass

    def test_fliplr(self):
        """Test horizontal flip"""
        img = Image(np.arange(20).reshape(4, 5))
        flipped = img.fliplr()
        self.assertEqual(flipped.array[0, 0], img.array[0, 4])

    def test_flipud(self):
        """Test vertical flip"""
        img = Image(np.arange(20).reshape(4, 5))
        flipped = img.flipud()
        self.assertEqual(flipped.array[0, 0], img.array[3, 0])

    def test_meshgrid(self):
        # instance method call
        im = Image.Random(size=(3, 4))
        U, V = im.meshgrid()

        for u in range(im.width):
            for v in range(im.height):
                self.assertEqual(U[v, u], u)
                self.assertEqual(V[v, u], v)

        # class method call for book compatibility
        Uc, Vc = Image.meshgrid(im.width, im.height)
        nt.assert_array_equal(Uc, U)
        nt.assert_array_equal(Vc, V)

    def test_warp(self):
        """Sample an analytic ramp at half-pixel coordinates."""
        v, u = np.mgrid[:8, :10]
        im = Image((2 * u + 3 * v).astype(np.float32))
        U, V = np.meshgrid(np.arange(7) + 0.5, np.arange(5) + 0.5)
        result = im.warp(U, V)
        self.assertEqual(result.shape, U.shape)
        nt.assert_allclose(result.array, 2 * U + 3 * V, atol=0.1, rtol=0)

    def test_warp_affine(self):
        """Translate a ramp and check the inverse mapping and border."""
        data = np.arange(48, dtype=np.float32).reshape(6, 8)
        im = Image(data)
        M = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
        result = im.warp_affine(M, bgcolor=-1)
        nt.assert_allclose(result.array[1:, 2:], data[:-1, :-2], atol=0.1, rtol=0)
        nt.assert_allclose(result.array[0, :], -1, atol=0.1, rtol=0)
        nt.assert_allclose(result.array[:, :2], -1, atol=0.1, rtol=0)
        inverse = im.warp_affine(M, inverse=True)
        nt.assert_allclose(inverse.array[:-1, :-2], data[1:, 2:], atol=0.1, rtol=0)

    def test_warp_perspective(self):
        """Check a non-affine homography against an analytic ramp."""
        v, u = np.mgrid[:20, :24]
        im = Image((u + 2 * v).astype(np.float32))
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.01, 0.02, 1.0]])
        result = im.warp_perspective(H, size=(10, 8))
        vo, uo = np.mgrid[:8, :10]
        denominator = 1 - 0.01 * uo - 0.02 * vo
        expected = (uo + 2 * vo) / denominator
        self.assertEqual(result.shape, expected.shape)
        nt.assert_allclose(result.array, expected, atol=0.1, rtol=0)
        inverse = im.warp_perspective(np.linalg.inv(H), inverse=True, size=(10, 8))
        nt.assert_allclose(inverse.array, expected, atol=0.1, rtol=0)

    def test_undistort(self):
        """Check nonzero radial distortion on a synthetic intensity ramp."""
        v, u = np.mgrid[:41, :41]
        im = Image((u + 2 * v).astype(np.float32))
        K = np.array([[30.0, 0.0, 20.0], [0.0, 30.0, 20.0], [0.0, 0.0, 1.0]])
        result = im.undistort(K, np.array([0.2, 0.0, 0.0, 0.0, 0.0]))
        x, y = (u - 20) / 30, (v - 20) / 30
        radial = 1 + 0.2 * (x**2 + y**2)
        expected = 20 + 30 * x * radial + 2 * (20 + 30 * y * radial)
        nt.assert_allclose(
            result.array[8:33, 8:33], expected[8:33, 8:33], atol=0.1, rtol=0
        )
        identity = im.undistort(K, np.zeros(5))
        nt.assert_allclose(identity.array, im.array, atol=0.1, rtol=0)

    def test_interp2d(self):
        """Interpolate a plane in pixel coordinates and an explicit domain."""
        v, u = np.mgrid[:6, :8]
        im = Image((2 * u + 3 * v).astype(np.float32))
        U, V = np.meshgrid(np.arange(6) + 0.25, np.arange(4) + 0.5)
        nt.assert_allclose(im.interp2d(U, V).array, 2 * U + 3 * V, atol=1e-6, rtol=0)
        Ud, Vd = 10 + 2 * u, -5 + 3 * v
        result = im.interp2d(10 + 2 * U, -5 + 3 * V, Ud=Ud, Vd=Vd)
        nt.assert_allclose(result.array, 2 * U + 3 * V, atol=1e-6, rtol=0)

    def test_tile(self):
        ims = []
        N = 16
        Nw = 4
        Nh = 4
        W = 10
        H = 12
        for i in range(N):
            ims.append(Image.Random(size=(W, H)))

        sep = 2  # default
        x = Image.Tile(ims)
        self.assertEqual(x.size, (W * Nw + sep * (Nw - 1), H * Nh + sep * (Nh - 1)))
        for c in range(Nw):
            for r in range(Nh):
                nt.assert_array_equal(
                    x.array[
                        r * (H + sep) : r * (H + sep) + H,
                        c * (W + sep) : c * (W + sep) + W,
                    ],
                    ims[r * Nw + c].array,
                )

        # test explicit sep
        sep = 2
        x = Image.Tile(ims, sep=sep)
        self.assertEqual(x.size, (W * Nw + sep * (Nw - 1), H * Nh + sep * (Nh - 1)))
        for c in range(Nw):
            for r in range(Nh):
                nt.assert_array_equal(
                    x.array[
                        r * (H + sep) : r * (H + sep) + H,
                        c * (W + sep) : c * (W + sep) + W,
                    ],
                    ims[r * Nw + c].array,
                )

        sep = 3
        x = Image.Tile(ims, sep=sep)
        self.assertEqual(x.size, (W * Nw + sep * (Nw - 1), H * Nh + sep * (Nh - 1)))
        for c in range(Nw):
            for r in range(Nh):
                nt.assert_array_equal(
                    x.array[
                        r * (H + sep) : r * (H + sep) + H,
                        c * (W + sep) : c * (W + sep) + W,
                    ],
                    ims[r * Nw + c].array,
                )

        # tests bgcolor
        sep = 3
        x = Image.Tile(ims, sep=sep, bgcolor=42)
        self.assertEqual(x.size, (W * Nw + sep * (Nw - 1), H * Nh + sep * (Nh - 1)))
        for c in range(Nw):
            for r in range(Nh):
                nt.assert_array_equal(
                    x.array[
                        r * (H + sep) : r * (H + sep) + H,
                        c * (W + sep) : c * (W + sep) + W,
                    ],
                    ims[r * Nw + c].array,
                )
        nt.assert_array_equal(x.array[:, W : W + sep], 42)
        nt.assert_array_equal(x.array[H : H + sep, :], 42)

        # test for partial grid
        sep = 3
        x = Image.Tile(ims[:-1], sep=sep, bgcolor=42, columns=4)
        self.assertEqual(x.size, (W * Nw + sep * (Nw - 1), H * Nh + sep * (Nh - 1)))
        for c in range(Nw):
            for r in range(Nh):
                if c * Nw + r < N - 1:
                    nt.assert_array_equal(
                        x.array[
                            r * (H + sep) : r * (H + sep) + H,
                            c * (W + sep) : c * (W + sep) + W,
                        ],
                        ims[r * Nw + c].array,
                    )
                else:
                    # test the empty square
                    nt.assert_array_equal(
                        x.array[
                            r * (H + sep) : r * (H + sep) + H,
                            c * (W + sep) : c * (W + sep) + W,
                        ],
                        42,
                    )

        # test 2x8 grid
        Nw = 2
        Nh = 8
        sep = 3
        x = Image.Tile(ims, sep=sep, columns=2)
        self.assertEqual(x.size, (W * Nw + sep * (Nw - 1), H * Nh + sep * (Nh - 1)))
        for c in range(Nw):
            for r in range(Nh):
                nt.assert_array_equal(
                    x.array[
                        r * (H + sep) : r * (H + sep) + H,
                        c * (W + sep) : c * (W + sep) + W,
                    ],
                    ims[r * Nw + c].array,
                )

        # test for color images
        ims = []
        Nw = 4
        Nh = 4

        for i in range(N):
            ims.append(Image.Random(size=(W, H), colororder="RGB"))

        sep = 2
        x = Image.Tile(ims, sep=sep)
        self.assertEqual(x.size, (W * Nw + sep * (Nw - 1), H * Nh + sep * (Nh - 1)))
        self.assertEqual(x.colororder_str, "R:G:B")
        for c in range(Nw):
            for r in range(Nh):
                nt.assert_array_equal(
                    x.array[
                        r * (H + sep) : r * (H + sep) + H,
                        c * (W + sep) : c * (W + sep) + W,
                        ...,
                    ],
                    ims[r * Nw + c].array,
                )

        # tests bgcolor
        sep = 3
        x = Image.Tile(ims, sep=sep, bgcolor=(42, 43, 44))
        self.assertEqual(x.size, (W * Nw + sep * (Nw - 1), H * Nh + sep * (Nh - 1)))
        self.assertEqual(x.colororder_str, "R:G:B")
        for c in range(Nw):
            for r in range(Nh):
                nt.assert_array_equal(
                    x.array[
                        r * (H + sep) : r * (H + sep) + H,
                        c * (W + sep) : c * (W + sep) + W,
                        ...,
                    ],
                    ims[r * Nw + c].array,
                )

        for p, v in enumerate((42, 43, 44)):
            nt.assert_array_equal(x.array[:, W : W + sep, p], v)
            nt.assert_array_equal(x.array[H : H + sep, :, p], v)

    def test_hstack(self):
        # Add test cases for the HStack method
        ims = []
        N = 20
        W = 10
        H = 12

        for i in range(N):
            ims.append(Image.Random(size=(W, H)))

        sep = 1
        x = Image.Hstack(ims)
        self.assertEqual(x.size, (W * N + sep * (N - 1), H))
        for i in range(N):
            nt.assert_array_equal(
                x.array[:, i * (W + sep) : i * (W + sep) + W], ims[i].array
            )

        # test explicit sep
        sep = 1
        x = Image.Hstack(ims, sep=sep)
        self.assertEqual(x.size, (W * N + sep * (N - 1), H))
        for i in range(N):
            nt.assert_array_equal(
                x.array[:, i * (W + sep) : i * (W + sep) + W], ims[i].array
            )

        sep = 3
        x = Image.Hstack(ims, sep=sep)
        self.assertEqual(x.size, (W * N + sep * (N - 1), H))
        for i in range(N):
            nt.assert_array_equal(
                x.array[:, i * (W + sep) : i * (W + sep) + W], ims[i].array
            )

        # tests bgcolor
        sep = 3
        x = Image.Hstack(ims, sep=sep, bgcolor=42)
        self.assertEqual(x.size, (W * N + sep * (N - 1), H))
        for i in range(N):
            nt.assert_array_equal(
                x.array[:, i * (W + sep) : i * (W + sep) + W], ims[i].array
            )
        nt.assert_array_equal(x.array[:, W : W + sep], 42)

        # test for color images
        ims = []
        for i in range(N):
            ims.append(Image.Random(size=(W, H), colororder="RGB"))

        sep = 1
        x = Image.Hstack(ims, sep=sep)
        self.assertEqual(x.size, (W * N + sep * (N - 1), H))
        for i in range(N):
            nt.assert_array_equal(
                x.array[:, i * (W + sep) : i * (W + sep) + W, ...], ims[i].array
            )

        sep = 3
        x = Image.Hstack(ims, sep=sep)
        self.assertEqual(x.size, (W * N + sep * (N - 1), H))
        for i in range(N):
            nt.assert_array_equal(
                x.array[:, i * (W + sep) : i * (W + sep) + W, ...], ims[i].array
            )

        # tests bgcolor
        sep = 3
        x = Image.Hstack(ims, sep=sep, bgcolor=(42, 43, 44))
        self.assertEqual(x.size, (W * N + sep * (N - 1), H))
        for i in range(N):
            nt.assert_array_equal(
                x.array[:, i * (W + sep) : i * (W + sep) + W, ...], ims[i].array
            )
        for p, v in enumerate((42, 43, 44)):
            nt.assert_array_equal(x.array[:, W : W + sep, p], v)

    def test_vstack(self):
        ims = []
        N = 20
        W = 10
        H = 12

        for i in range(N):
            ims.append(Image.Random(size=(W, H)))

        sep = 1
        x = Image.Vstack(ims)
        self.assertEqual(x.size, (W, H * N + sep * (N - 1)))
        for i in range(N):
            nt.assert_array_equal(
                x.array[i * (H + sep) : i * (H + sep) + H, :], ims[i].array
            )

        # test explicit sep
        sep = 1
        x = Image.Vstack(ims, sep=sep)
        self.assertEqual(x.size, (W, H * N + sep * (N - 1)))
        for i in range(N):
            nt.assert_array_equal(
                x.array[i * (H + sep) : i * (H + sep) + H, :], ims[i].array
            )

        sep = 3
        x = Image.Vstack(ims, sep=sep)
        self.assertEqual(x.size, (W, H * N + sep * (N - 1)))
        for i in range(N):
            nt.assert_array_equal(
                x.array[i * (H + sep) : i * (H + sep) + H, :], ims[i].array
            )

        # tests bgcolor
        sep = 3
        x = Image.Vstack(ims, sep=sep, bgcolor=42)
        self.assertEqual(x.size, (W, H * N + sep * (N - 1)))
        for i in range(N):
            nt.assert_array_equal(
                x.array[i * (H + sep) : i * (H + sep) + H, :], ims[i].array
            )
        nt.assert_array_equal(x.array[H : H + sep, :], 42)

        # test for color images
        ims = []
        for i in range(N):
            ims.append(Image.Random(size=(W, H), colororder="RGB"))

        sep = 1
        x = Image.Vstack(ims, sep=sep)
        self.assertEqual(x.size, (W, H * N + sep * (N - 1)))
        for i in range(N):
            nt.assert_array_equal(
                x.array[i * (H + sep) : i * (H + sep) + H, :, ...], ims[i].array
            )

        sep = 3
        x = Image.Vstack(ims, sep=sep)
        self.assertEqual(x.size, (W, H * N + sep * (N - 1)))
        for i in range(N):
            nt.assert_array_equal(
                x.array[i * (H + sep) : i * (H + sep) + H, :, ...], ims[i].array
            )

        # tests bgcolor
        sep = 3
        x = Image.Vstack(ims, sep=sep, bgcolor=(42, 43, 44))
        self.assertEqual(x.size, (W, H * N + sep * (N - 1)))
        for i in range(N):
            nt.assert_array_equal(
                x.array[i * (H + sep) : i * (H + sep) + H, :, ...], ims[i].array
            )

        for p, v in enumerate((42, 43, 44)):
            nt.assert_array_equal(x.array[H : H + sep, :, p], v)

    def test_view1d(self):
        im = Image.Random(size=(20, 30))
        x = im.view1d()
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.shape, (600,))
        nt.assert_array_equal(x, im.array.flatten())

        im = Image.Random(size=(20, 30), colororder="RGB")
        x = im.view1d()
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.shape, (600, 3))
        for p in range(3):
            nt.assert_array_equal(x[:, p], im.array[:, :, p].flatten())


if __name__ == "__main__":
    unittest.main()
