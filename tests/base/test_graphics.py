import unittest
from unittest.case import skip

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt

from machinevisiontoolbox.base import *


class TestBlobs(unittest.TestCase):
    def test_draw_line(self):
        # test outline box draws properly
        img = np.zeros((1000, 1000), dtype="uint8")
        # horizontal line x=100-600, y=200
        draw_line(img, (100, 200), (600, 200), color=100, thickness=1)
        v, u = np.argwhere(img == 100).T
        self.assertEqual(len(u), 501)
        self.assertEqual(u.min(), 100)
        self.assertEqual(u.max(), 600)
        self.assertEqual(v.min(), 200)
        self.assertEqual(v.max(), 200)
        nt.assert_array_equal(img[200, 100:601], 100)

        img = np.zeros((1000, 1000), dtype="uint8")
        # horizontal line x=100-600, y=200
        draw_line(img, (100, 200), (600, 200), color=100, thickness=3)
        v, u = np.argwhere(img == 100).T
        self.assertEqual(len(u), 2513)
        self.assertEqual(u.min(), 98)
        self.assertEqual(u.max(), 602)
        self.assertEqual(v.min(), 198)
        self.assertEqual(v.max(), 202)
        nt.assert_array_equal(img[200, 100:601], 100)

        img = np.zeros((1000, 1000, 3), dtype="uint8")
        # horizontal line x=100-600, y=200
        draw_line(img, (100, 200), (600, 200), color=100, thickness=1)
        v, u = np.argwhere(img[:, :, 0] > 0).T
        self.assertEqual(len(u), 501)
        self.assertEqual(u.min(), 100)
        self.assertEqual(u.max(), 600)
        self.assertEqual(v.min(), 200)
        self.assertEqual(v.max(), 200)
        nt.assert_array_equal(img[200, 100:601, 0], 100)
        nt.assert_array_equal(img[200, 100:601, 1], 100)
        nt.assert_array_equal(img[200, 100:601, 2], 100)

        img = np.zeros((1000, 1000, 3), dtype="uint8")
        # horizontal line x=100-600, y=200
        draw_line(img, (100, 200), (600, 200), color="red", thickness=1)
        v, u = np.argwhere(img[:, :, 0] > 0).T
        self.assertEqual(len(u), 501)
        self.assertEqual(u.min(), 100)
        self.assertEqual(u.max(), 600)
        self.assertEqual(v.min(), 200)
        self.assertEqual(v.max(), 200)
        nt.assert_array_equal(img[200, 100:601, 0], 255)
        nt.assert_array_equal(img[200, 100:601, 1], 0)
        nt.assert_array_equal(img[200, 100:601, 2], 0)

        img = np.zeros((1000, 1000, 3), dtype="float32")
        # horizontal line x=100-600, y=200
        draw_line(img, (100, 200), (600, 200), color="red", thickness=1)
        v, u = np.argwhere(img[:, :, 0] > 0).T
        self.assertEqual(len(u), 501)
        self.assertEqual(u.min(), 100)
        self.assertEqual(u.max(), 600)
        self.assertEqual(v.min(), 200)
        self.assertEqual(v.max(), 200)
        nt.assert_array_equal(img[200, 100:601, 0], 1.0)
        nt.assert_array_equal(img[200, 100:601, 1], 0)
        nt.assert_array_equal(img[200, 100:601, 2], 0)

    def test_draw_box(self):
        # test outline box draws properly
        img = np.zeros((1000, 1000), dtype="uint8")
        draw_box(img, lrbt=[100, 400, 200, 500], color=100, thickness=1)
        v, u = np.argwhere(img == 100).T
        self.assertEqual(len(u), 1200)
        self.assertEqual(u.min(), 100)
        self.assertEqual(u.max(), 400)
        self.assertEqual(v.min(), 200)
        self.assertEqual(v.max(), 500)
        self.assertEqual(img[200, 100], 100)  # top left
        self.assertEqual(img[500, 400], 100)  # bottom right
        nt.assert_array_equal(img[201:500, 101:40], 0)  # middle is black

        # test filled box draws properly
        img = np.zeros((1000, 1000), dtype="uint8")
        draw_box(img, lrbt=[100, 400, 200, 500], color=100, thickness=-1)
        v, u = np.argwhere(img == 100).T
        self.assertEqual(len(u), 301**2)
        self.assertEqual(u.min(), 100)
        self.assertEqual(u.max(), 400)
        self.assertEqual(v.min(), 200)
        self.assertEqual(v.max(), 500)
        nt.assert_array_equal(img[200:501, 100:401], 100)

        # test filled color box draws properly
        img = np.zeros((1000, 1000, 3), dtype="uint8")
        draw_box(img, lrbt=[100, 400, 200, 500], color="red", thickness=-1)
        v, u = np.argwhere(img[:, :, 0] > 0).T
        self.assertEqual(u.min(), 100)
        self.assertEqual(u.max(), 400)
        self.assertEqual(v.min(), 200)
        self.assertEqual(v.max(), 500)
        nt.assert_array_equal(img[200:501, 100:401, 0], 255)
        nt.assert_array_equal(img[200:501, 100:401, 1], 0)
        nt.assert_array_equal(img[200:501, 100:401, 2], 0)

        # test filled float color box draws properly
        img = np.zeros((1000, 1000, 3), dtype="float32")
        draw_box(img, lrbt=[100, 400, 200, 500], color="red", thickness=-1)
        v, u = np.argwhere(img[:, :, 0] > 0).T
        self.assertEqual(u.min(), 100)
        self.assertEqual(u.max(), 400)
        self.assertEqual(v.min(), 200)
        self.assertEqual(v.max(), 500)
        nt.assert_array_equal(img[200:501, 100:401, 0], 1.0)
        nt.assert_array_equal(img[200:501, 100:401, 1], 0.0)
        nt.assert_array_equal(img[200:501, 100:401, 2], 0.0)

        img = np.zeros((1000, 1000), dtype="uint8")
        draw_box(img, lrbt=[100, 400, 200, 500], color=100, thickness=-1)
        draw_box(img, lrbt=[500, 800, 200, 500], color=150, thickness=-1)
        draw_box(img, lrbt=[100, 400, 600, 900], color=200, thickness=-1)
        draw_box(img, lrbt=[500, 800, 600, 900], color=250, thickness=-1)

        self.assertEqual(img[200, 100], 100)
        self.assertEqual(img[200, 500], 150)
        self.assertEqual(img[600, 100], 200)
        self.assertEqual(img[600, 500], 250)

        img = np.zeros((1000, 1000, 3), dtype="uint8")
        draw_box(img, lrbt=[100, 400, 100, 400], color="red", thickness=-1)
        draw_box(img, lrbt=[500, 800, 100, 400], color="green", thickness=-1)
        draw_box(img, lrbt=[100, 400, 500, 800], color="blue", thickness=-1)
        draw_box(img, lrbt=[500, 800, 500, 800], color="white", thickness=-1)

        nt.assert_array_equal(img[200, 100], [255, 0, 0])
        nt.assert_array_equal(img[200, 500], [0, 128, 0])
        nt.assert_array_equal(img[600, 100], [0, 0, 255])
        nt.assert_array_equal(img[600, 500], [255, 255, 255])

        img = np.zeros((1000, 1000, 3), dtype="float32")
        draw_box(img, lrbt=[100, 400, 100, 400], color="red", thickness=-1)
        draw_box(img, lrbt=[500, 800, 100, 400], color="green", thickness=-1)
        draw_box(img, lrbt=[100, 400, 500, 800], color="blue", thickness=-1)
        draw_box(img, lrbt=[500, 800, 500, 800], color="white", thickness=-1)

        nt.assert_array_equal(img[200, 100], [1.0, 0, 0])
        nt.assert_array_almost_equal(img[200, 500], [0, 0.5, 0], decimal=2)
        nt.assert_array_equal(img[600, 100], [0, 0, 1.0])
        nt.assert_array_equal(img[600, 500], [1.0, 1.0, 1.0])

    def test_draw_circle(self):
        # test outline circle
        img = np.zeros((1000, 1000), dtype="uint8")
        draw_circle(img, centre=(200, 300), radius=100, color=200, thickness=1)
        v, u = np.argwhere(img == 200).T
        self.assertEqual(len(u), 564)
        self.assertEqual(u.min(), 100)
        self.assertEqual(u.max(), 300)
        self.assertEqual(v.min(), 200)
        self.assertEqual(v.max(), 400)
        self.assertEqual(img[300, 100], 200)  # left
        self.assertEqual(img[300, 300], 200)  # right
        self.assertEqual(img[200, 200], 200)  # top
        self.assertEqual(img[400, 200], 200)  # bottom

        # test filled circle draws properly
        img = np.zeros((1000, 1000), dtype="uint8")
        draw_circle(img, centre=(200, 300), radius=100, color=200, thickness=-1)
        v, u = np.argwhere(img == 200).T
        self.assertEqual(len(u), 31417)
        self.assertEqual(u.min(), 100)
        self.assertEqual(u.max(), 300)
        self.assertEqual(v.min(), 200)
        self.assertEqual(v.max(), 400)
        self.assertEqual(img[300, 100], 200)  # left
        self.assertEqual(img[300, 300], 200)  # right
        self.assertEqual(img[200, 200], 200)  # top
        self.assertEqual(img[400, 200], 200)  # bottom
        self.assertEqual(img[300, 200], 200)  # centre

        img = np.zeros((1000, 1000, 3), dtype="uint8")
        draw_circle(img, centre=(200, 300), radius=100, color="red", thickness=-1)
        v, u = np.argwhere(img[:, :, 0] > 0).T
        self.assertEqual(u.min(), 100)
        self.assertEqual(u.max(), 300)
        self.assertEqual(v.min(), 200)
        self.assertEqual(v.max(), 400)
        nt.assert_array_equal(img[300, 100], [255, 0, 0])  # left
        nt.assert_array_equal(img[300, 300], [255, 0, 0])  # right
        nt.assert_array_equal(img[200, 200], [255, 0, 0])  # top
        nt.assert_array_equal(img[400, 200], [255, 0, 0])  # bottom
        nt.assert_array_equal(img[300, 200], [255, 0, 0])  # centre

        img = np.zeros((1000, 1000, 3), dtype="float32")
        draw_circle(img, centre=(200, 300), radius=100, color="red", thickness=-1)
        v, u = np.argwhere(img[:, :, 0] > 0).T
        self.assertEqual(u.min(), 100)
        self.assertEqual(u.max(), 300)
        self.assertEqual(v.min(), 200)
        self.assertEqual(v.max(), 400)
        nt.assert_array_equal(img[300, 100], [1.0, 0, 0])  # left
        nt.assert_array_equal(img[300, 300], [1.0, 0, 0])  # right
        nt.assert_array_equal(img[200, 200], [1.0, 0, 0])  # top
        nt.assert_array_equal(img[400, 200], [1.0, 0, 0])  # bottom
        nt.assert_array_equal(img[300, 200], [1.0, 0, 0])  # centre

    def test_draw_labelbox(self):
        img = np.zeros((1000, 1000), dtype="uint8")
        draw_labelbox(
            img,
            "Topleft",
            lrbt=[100, 400, 100, 400],
            textcolor=0,
            labelcolor=100,
            color=200,
            thickness=2,
            fontsize=1,
            position="topleft",
        )
        draw_labelbox(
            img,
            "Topright",
            lrbt=[500, 800, 100, 400],
            textcolor=0,
            labelcolor=100,
            color=200,
            thickness=2,
            fontsize=1,
            position="topright",
        )
        draw_labelbox(
            img,
            "Bottomleft",
            lrbt=[100, 400, 500, 800],
            textcolor=0,
            labelcolor=100,
            color=200,
            thickness=2,
            fontsize=1,
            position="bottomleft",
        )
        draw_labelbox(
            img,
            "Bottomright",
            lrbt=[500, 800, 500, 800],
            textcolor=0,
            labelcolor=100,
            color=200,
            thickness=2,
            fontsize=1,
            position="bottomright",
        )

    def test_plot_labelbox(self):
        img = np.zeros((1000, 1000), dtype="uint8")
        idisp(img)
        plot_labelbox(
            "Topleft",
            lrbt=[100, 400, 100, 400],
            textcolor="red",
            labelcolor="blue",
            color="yellow",
            linewidth=2,
            position="topleft",
        )
        plot_labelbox(
            "Topright",
            lrbt=[500, 800, 100, 400],
            textcolor="red",
            labelcolor="blue",
            color="yellow",
            linewidth=2,
            position="topright",
        )
        plot_labelbox(
            "Bottomleft",
            lrbt=[100, 400, 500, 800],
            textcolor="red",
            labelcolor="blue",
            color="yellow",
            linewidth=2,
            position="bottomleft",
        )
        plot_labelbox(
            "Bottomright",
            lrbt=[500, 800, 500, 800],
            textcolor="red",
            labelcolor="blue",
            color="yellow",
            linewidth=2,
            position="bottomright",
        )

    def test_draw_point(self):
        img = np.zeros((1000, 1000), dtype="uint8")
        draw_point(img, (100, 200), color=100, fontheight=20)
        v, u = np.argwhere(img == 100).T
        self.assertEqual(len(u), 94)
        self.assertTrue(abs(u.mean() - 100) < 2)
        self.assertTrue(abs(v.mean() - 200) < 2)
        self.assertEqual(img[200, 100], 100)

        img = np.zeros((1000, 1000), dtype="uint8")
        draw_point(img, (100, 200), text="Hello", color=100, fontheight=20)
        v, u = np.argwhere(img == 100).T
        self.assertEqual(len(u), 593)
        self.assertEqual(img[200, 100], 100)

        img = np.zeros((1000, 1000), dtype="uint8")
        draw_point(
            img,
            [
                (100, 200),
                (200, 300),
                (300, 400),
                (400, 500),
                (500, 600),
            ],
            text="#{0}",
            color=100,
            fontheight=20,
        )

        img = np.zeros((1000, 1000), dtype="uint8")
        draw_point(
            img,
            [
                (100, 200),
                (200, 300),
                (300, 400),
                (400, 500),
                (500, 600),
            ],
            text=["dasher", "dancer", "prancer", "vixen", "comet"],
            color=100,
            fontheight=20,
        )
        pass

    def test_draw_text(self):
        # test outline box draws properly
        img = np.zeros((1000, 1000), dtype="uint8")
        # horizontal line x=100-600, y=200
        draw_line(img, (100, 200), (600, 200), color=100, thickness=1)
        v, u = np.argwhere(img == 100).T
        self.assertEqual(len(u), 501)
        self.assertEqual(u.min(), 100)
        self.assertEqual(u.max(), 600)
        self.assertEqual(v.min(), 200)
        self.assertEqual(v.max(), 200)
        nt.assert_array_equal(img[200, 100:601], 100)


# ------------------------------------------------------------------------ #
if __name__ == "__main__":
    unittest.main()
