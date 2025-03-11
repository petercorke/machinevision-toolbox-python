import unittest
from machinevisiontoolbox import Image
import numpy.testing as nt
import numpy as np


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
        nt.assert_almost_equal(x.A[2:6, 1:4], im.A)
        nt.assert_almost_equal(x.A[:, 0], 0.7)  # left
        nt.assert_almost_equal(x.A[:, 4:], 0.7)  # right
        nt.assert_almost_equal(x.A[0:2, :], 0.7)  # top
        nt.assert_almost_equal(x.A[6, :], 0.7)  # bottom

        # color float image
        im = Image.Random(size=(3, 4), dtype="float", colororder="RGB")
        x = im.pad(left=1, right=2, top=2, bottom=1, value=0.7)
        for p in range(3):
            nt.assert_almost_equal(x.A[2:6, 1:4, p], im.A[:, :, p])
        nt.assert_almost_equal(x.A[:, 0, :], 0.7)  # left
        nt.assert_almost_equal(x.A[:, 4:, :], 0.7)  # right
        nt.assert_almost_equal(x.A[0:2, :, :], 0.7)  # top
        nt.assert_almost_equal(x.A[6, :, :], 0.7)  # bottom

    def test_dice(self):
        # Add test cases for the dice method
        im = Image.Random(size=(20, 30))

        sims = im.dice(shape=10)
        self.assertEqual(len(sims), 6)
        for sim in sims:
            self.assertEqual(sim.shape, (10, 10))
        nt.assert_array_equal(sims[0].A[:10, :10], im.A[:10, :10])
        nt.assert_array_equal(sims[1].A[:10, :10], im.A[:10, 10:20])
        nt.assert_array_equal(sims[2].A[:10, :10], im.A[10:20, :10])
        nt.assert_array_equal(sims[3].A[:10, :10], im.A[10:20, 10:20])
        nt.assert_array_equal(sims[4].A[:10, :10], im.A[20:30, :10])
        nt.assert_array_equal(sims[5].A[:10, :10], im.A[20:30, 10:200])

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
        nt.assert_array_equal(roi.A, im.A[10:20, 5:15])

        im = Image.Random(size=(20, 30), colororder="RGB")
        roi = im.roi((5, 14, 10, 19))
        self.assertEqual(roi.shape, (10, 10, 3))
        self.assertEqual(roi.colororder_str, "R:G:B")
        nt.assert_array_equal(roi.A, im.A[10:20, 5:15, :])

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
        nt.assert_array_equal(x.A, im.A)

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
            y, x = np.where(im.A)
            return np.mean(x), np.mean(y)

        im = Image.Squares(1, size=256)
        x = im.rotate(0)
        self.assertEqual(x.size, (256, 256))
        nt.assert_array_equal(x.A, im.A)

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

    def test_meshgrid(self):
        # Add test cases for the meshgrid method
        im = Image.Random(size=(3, 4))
        U, V = im.meshgrid()

        for u in range(im.width):
            for v in range(im.height):
                self.assertEqual(U[v, u], u)
                self.assertEqual(V[v, u], v)

    def test_warp(self):
        # Add test cases for the decimate method
        pass

    def test_warp_affine(self):
        # Add test cases for the decimate method
        pass

    def test_warp_perspective(self):
        # Add test cases for the decimate method
        pass

    def test_undistort(self):
        # Add test cases for the decimate method
        pass

    def test_interp2d(self):
        # Add test cases for the decimate method
        pass

    def test_Tile(self):
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
                    x.A[
                        r * (H + sep) : r * (H + sep) + H,
                        c * (W + sep) : c * (W + sep) + W,
                    ],
                    ims[r * Nw + c].A,
                )

        # test explicit sep
        sep = 2
        x = Image.Tile(ims, sep=sep)
        self.assertEqual(x.size, (W * Nw + sep * (Nw - 1), H * Nh + sep * (Nh - 1)))
        for c in range(Nw):
            for r in range(Nh):
                nt.assert_array_equal(
                    x.A[
                        r * (H + sep) : r * (H + sep) + H,
                        c * (W + sep) : c * (W + sep) + W,
                    ],
                    ims[r * Nw + c].A,
                )

        sep = 3
        x = Image.Tile(ims, sep=sep)
        self.assertEqual(x.size, (W * Nw + sep * (Nw - 1), H * Nh + sep * (Nh - 1)))
        for c in range(Nw):
            for r in range(Nh):
                nt.assert_array_equal(
                    x.A[
                        r * (H + sep) : r * (H + sep) + H,
                        c * (W + sep) : c * (W + sep) + W,
                    ],
                    ims[r * Nw + c].A,
                )

        # tests bgcolor
        sep = 3
        x = Image.Tile(ims, sep=sep, bgcolor=42)
        self.assertEqual(x.size, (W * Nw + sep * (Nw - 1), H * Nh + sep * (Nh - 1)))
        for c in range(Nw):
            for r in range(Nh):
                nt.assert_array_equal(
                    x.A[
                        r * (H + sep) : r * (H + sep) + H,
                        c * (W + sep) : c * (W + sep) + W,
                    ],
                    ims[r * Nw + c].A,
                )
        nt.assert_array_equal(x.A[:, W : W + sep], 42)
        nt.assert_array_equal(x.A[H : H + sep, :], 42)

        # test for partial grid
        sep = 3
        x = Image.Tile(ims[:-1], sep=sep, bgcolor=42, columns=4)
        self.assertEqual(x.size, (W * Nw + sep * (Nw - 1), H * Nh + sep * (Nh - 1)))
        for c in range(Nw):
            for r in range(Nh):
                if c * Nw + r < N - 1:
                    nt.assert_array_equal(
                        x.A[
                            r * (H + sep) : r * (H + sep) + H,
                            c * (W + sep) : c * (W + sep) + W,
                        ],
                        ims[r * Nw + c].A,
                    )
                else:
                    # test the empty square
                    nt.assert_array_equal(
                        x.A[
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
                    x.A[
                        r * (H + sep) : r * (H + sep) + H,
                        c * (W + sep) : c * (W + sep) + W,
                    ],
                    ims[r * Nw + c].A,
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
                    x.A[
                        r * (H + sep) : r * (H + sep) + H,
                        c * (W + sep) : c * (W + sep) + W,
                        ...,
                    ],
                    ims[r * Nw + c].A,
                )

        # tests bgcolor
        sep = 3
        x = Image.Tile(ims, sep=sep, bgcolor=(42, 43, 44))
        self.assertEqual(x.size, (W * Nw + sep * (Nw - 1), H * Nh + sep * (Nh - 1)))
        self.assertEqual(x.colororder_str, "R:G:B")
        for c in range(Nw):
            for r in range(Nh):
                nt.assert_array_equal(
                    x.A[
                        r * (H + sep) : r * (H + sep) + H,
                        c * (W + sep) : c * (W + sep) + W,
                        ...,
                    ],
                    ims[r * Nw + c].A,
                )

        for p, v in enumerate((42, 43, 44)):
            nt.assert_array_equal(x.A[:, W : W + sep, p], v)
            nt.assert_array_equal(x.A[H : H + sep, :, p], v)

    def test_Hstack(self):
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
            nt.assert_array_equal(x.A[:, i * (W + sep) : i * (W + sep) + W], ims[i].A)

        # test explicit sep
        sep = 1
        x = Image.Hstack(ims, sep=sep)
        self.assertEqual(x.size, (W * N + sep * (N - 1), H))
        for i in range(N):
            nt.assert_array_equal(x.A[:, i * (W + sep) : i * (W + sep) + W], ims[i].A)

        sep = 3
        x = Image.Hstack(ims, sep=sep)
        self.assertEqual(x.size, (W * N + sep * (N - 1), H))
        for i in range(N):
            nt.assert_array_equal(x.A[:, i * (W + sep) : i * (W + sep) + W], ims[i].A)

        # tests bgcolor
        sep = 3
        x = Image.Hstack(ims, sep=sep, bgcolor=42)
        self.assertEqual(x.size, (W * N + sep * (N - 1), H))
        for i in range(N):
            nt.assert_array_equal(x.A[:, i * (W + sep) : i * (W + sep) + W], ims[i].A)
        nt.assert_array_equal(x.A[:, W : W + sep], 42)

        # test for color images
        ims = []
        for i in range(N):
            ims.append(Image.Random(size=(W, H), colororder="RGB"))

        sep = 1
        x = Image.Hstack(ims, sep=sep)
        self.assertEqual(x.size, (W * N + sep * (N - 1), H))
        for i in range(N):
            nt.assert_array_equal(
                x.A[:, i * (W + sep) : i * (W + sep) + W, ...], ims[i].A
            )

        sep = 3
        x = Image.Hstack(ims, sep=sep)
        self.assertEqual(x.size, (W * N + sep * (N - 1), H))
        for i in range(N):
            nt.assert_array_equal(
                x.A[:, i * (W + sep) : i * (W + sep) + W, ...], ims[i].A
            )

        # tests bgcolor
        sep = 3
        x = Image.Hstack(ims, sep=sep, bgcolor=(42, 43, 44))
        self.assertEqual(x.size, (W * N + sep * (N - 1), H))
        for i in range(N):
            nt.assert_array_equal(
                x.A[:, i * (W + sep) : i * (W + sep) + W, ...], ims[i].A
            )
        for p, v in enumerate((42, 43, 44)):
            nt.assert_array_equal(x.A[:, W : W + sep, p], v)

    def test_Vstack(self):
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
            nt.assert_array_equal(x.A[i * (H + sep) : i * (H + sep) + H, :], ims[i].A)

        # test explicit sep
        sep = 1
        x = Image.Vstack(ims, sep=sep)
        self.assertEqual(x.size, (W, H * N + sep * (N - 1)))
        for i in range(N):
            nt.assert_array_equal(x.A[i * (H + sep) : i * (H + sep) + H, :], ims[i].A)

        sep = 3
        x = Image.Vstack(ims, sep=sep)
        self.assertEqual(x.size, (W, H * N + sep * (N - 1)))
        for i in range(N):
            nt.assert_array_equal(x.A[i * (H + sep) : i * (H + sep) + H, :], ims[i].A)

        # tests bgcolor
        sep = 3
        x = Image.Vstack(ims, sep=sep, bgcolor=42)
        self.assertEqual(x.size, (W, H * N + sep * (N - 1)))
        for i in range(N):
            nt.assert_array_equal(x.A[i * (H + sep) : i * (H + sep) + H, :], ims[i].A)
        nt.assert_array_equal(x.A[H : H + sep, :], 42)

        # test for color images
        ims = []
        for i in range(N):
            ims.append(Image.Random(size=(W, H), colororder="RGB"))

        sep = 1
        x = Image.Vstack(ims, sep=sep)
        self.assertEqual(x.size, (W, H * N + sep * (N - 1)))
        for i in range(N):
            nt.assert_array_equal(
                x.A[i * (H + sep) : i * (H + sep) + H, :, ...], ims[i].A
            )

        sep = 3
        x = Image.Vstack(ims, sep=sep)
        self.assertEqual(x.size, (W, H * N + sep * (N - 1)))
        for i in range(N):
            nt.assert_array_equal(
                x.A[i * (H + sep) : i * (H + sep) + H, :, ...], ims[i].A
            )

        # tests bgcolor
        sep = 3
        x = Image.Vstack(ims, sep=sep, bgcolor=(42, 43, 44))
        self.assertEqual(x.size, (W, H * N + sep * (N - 1)))
        for i in range(N):
            nt.assert_array_equal(
                x.A[i * (H + sep) : i * (H + sep) + H, :, ...], ims[i].A
            )

        for p, v in enumerate((42, 43, 44)):
            nt.assert_array_equal(x.A[H : H + sep, :, p], v)

    def test_view1d(self):
        im = Image.Random(size=(20, 30))
        x = im.view1d()
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.shape, (600,))
        nt.assert_array_equal(x, im.A.flatten())

        im = Image.Random(size=(20, 30), colororder="RGB")
        x = im.view1d()
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.shape, (600, 3))
        for p in range(3):
            nt.assert_array_equal(x[:, p], im.A[:, :, p].flatten())


if __name__ == "__main__":
    unittest.main()
