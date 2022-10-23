import math
import numpy as np
import numpy.testing as nt
import unittest
from machinevisiontoolbox.base.graphics import *
from machinevisiontoolbox.base.moments import mpq

class TestGraphics(unittest.TestCase):

    def test_draw_box(self):

        ## square
        im = np.zeros((100, 100))
        draw_box(im, centre=(40, 60), wh=51, color=123)

        # check grey values
        self.assertEqual(im.min(), 0)
        self.assertEqual(im.max(), 123)

        # check centroid
        x0 = mpq(im, 1, 0) / mpq(im, 0, 0)
        y0 = mpq(im, 0, 1) / mpq(im, 0, 0)
        self.assertAlmostEqual(x0, 40)
        self.assertAlmostEqual(y0, 60)

        # check bounds
        v, u = im.nonzero()
        self.assertEqual(u.min(), 40-25)
        self.assertEqual(u.max(), 40+25)

        self.assertEqual(v.min(), 60-25)
        self.assertEqual(v.max(), 60+25)

        # check number of set pixels
        self.assertEqual(im.sum()/123, 4*(51-1))

        ## rectangle by centre and width-height
        im = np.zeros((150, 150))
        draw_box(im, centre=(40, 60), wh=[51, 80], color=123)

        # check grey values
        self.assertEqual(im.min(), 0)
        self.assertEqual(im.max(), 123)

        # check centroid
        x0 = mpq(im, 1, 0) / mpq(im, 0, 0)
        y0 = mpq(im, 0, 1) / mpq(im, 0, 0)
        self.assertAlmostEqual(x0, 40)
        self.assertAlmostEqual(y0, 59.5)

        # check bounds
        v, u = im.nonzero()
        self.assertEqual(u.min(), 40-25)
        self.assertEqual(u.max(), 40+25)

        self.assertEqual(v.min(), 60-40)
        self.assertEqual(v.max(), 60+39)

        # check number of set pixels
        self.assertEqual(im.sum()/123, 2*(51-1) + 2*(80-1))

        ## rectangle by corners
        im = np.zeros((150, 150))
        draw_box(im, lt=(15,20), rb=(65,99), color=123)

        # check grey values
        self.assertEqual(im.min(), 0)
        self.assertEqual(im.max(), 123)

        # check centroid
        x0 = mpq(im, 1, 0) / mpq(im, 0, 0)
        y0 = mpq(im, 0, 1) / mpq(im, 0, 0)
        self.assertAlmostEqual(x0, 40)
        self.assertAlmostEqual(y0, 59.5)

        # check bounds
        v, u = im.nonzero()
        self.assertEqual(u.min(), 40-25)
        self.assertEqual(u.max(), 40+25)

        self.assertEqual(v.min(), 60-40)
        self.assertEqual(v.max(), 60+39)

        # check number of set pixels
        self.assertEqual(im.sum()/123, 2*(51-1) + 2*(80-1))

        ## rectangle by corners 2
        im = np.zeros((150, 150))
        draw_box(im, lb=(15,99), rt=(65,20), color=123)

        # check grey values
        self.assertEqual(im.min(), 0)
        self.assertEqual(im.max(), 123)

        # check centroid
        x0 = mpq(im, 1, 0) / mpq(im, 0, 0)
        y0 = mpq(im, 0, 1) / mpq(im, 0, 0)
        self.assertAlmostEqual(x0, 40)
        self.assertAlmostEqual(y0, 59.5)

        # check bounds
        v, u = im.nonzero()
        self.assertEqual(u.min(), 40-25)
        self.assertEqual(u.max(), 40+25)

        self.assertEqual(v.min(), 60-40)
        self.assertEqual(v.max(), 60+39)

        # check number of set pixels
        self.assertEqual(im.sum()/123, 2*(51-1) + 2*(80-1))

        ## rectangle by corners 3
        im = np.zeros((150, 150))
        draw_box(im, l=15, t=20, r=65, b=99, color=123)

        # check grey values
        self.assertEqual(im.min(), 0)
        self.assertEqual(im.max(), 123)

        # check centroid
        x0 = mpq(im, 1, 0) / mpq(im, 0, 0)
        y0 = mpq(im, 0, 1) / mpq(im, 0, 0)
        self.assertAlmostEqual(x0, 40)
        self.assertAlmostEqual(y0, 59.5)

        # check bounds
        v, u = im.nonzero()
        self.assertEqual(u.min(), 40-25)
        self.assertEqual(u.max(), 40+25)

        self.assertEqual(v.min(), 60-40)
        self.assertEqual(v.max(), 60+39)

        # check number of set pixels
        self.assertEqual(im.sum()/123, 2*(51-1) + 2*(80-1))

        ## rectangle by corners 4
        im = np.zeros((150, 150))
        draw_box(im, ltrb=[15, 20, 65, 99], color=123)

        # check grey values
        self.assertEqual(im.min(), 0)
        self.assertEqual(im.max(), 123)

        # check centroid
        x0 = mpq(im, 1, 0) / mpq(im, 0, 0)
        y0 = mpq(im, 0, 1) / mpq(im, 0, 0)
        self.assertAlmostEqual(x0, 40)
        self.assertAlmostEqual(y0, 59.5)

        # check bounds
        v, u = im.nonzero()
        self.assertEqual(u.min(), 40-25)
        self.assertEqual(u.max(), 40+25)

        self.assertEqual(v.min(), 60-40)
        self.assertEqual(v.max(), 60+39)

        # check number of set pixels
        self.assertEqual(im.sum()/123, 2*(51-1) + 2*(80-1))

        ## rectangle by corners 5
        im = np.zeros((150, 150))
        draw_box(im, bbox=[15, 65, 20, 99], color=123)

        # check grey values
        self.assertEqual(im.min(), 0)
        self.assertEqual(im.max(), 123)

        # check centroid
        x0 = mpq(im, 1, 0) / mpq(im, 0, 0)
        y0 = mpq(im, 0, 1) / mpq(im, 0, 0)
        self.assertAlmostEqual(x0, 40)
        self.assertAlmostEqual(y0, 59.5)

        # check bounds
        v, u = im.nonzero()
        self.assertEqual(u.min(), 40-25)
        self.assertEqual(u.max(), 40+25)

        self.assertEqual(v.min(), 60-40)
        self.assertEqual(v.max(), 60+39)

        # check number of set pixels
        self.assertEqual(im.sum()/123, 2*(51-1) + 2*(80-1))

        ## thick square
        #    square sides are 7 pixels thick and external corners are rounded
        im = np.zeros((100, 100))
        draw_box(im, centre=(40, 60), wh=51, color=123, thickness=5)

        # check grey values
        self.assertEqual(im.min(), 0)
        self.assertEqual(im.max(), 123)

        # check centroid
        x0 = mpq(im, 1, 0) / mpq(im, 0, 0)
        y0 = mpq(im, 0, 1) / mpq(im, 0, 0)
        self.assertAlmostEqual(x0, 40)
        self.assertAlmostEqual(y0, 60)

        # check bounds
        v, u = im.nonzero()
        self.assertEqual(u.min(), 40-25-3)
        self.assertEqual(u.max(), 40+25+3)

        self.assertEqual(v.min(), 60-25-3)
        self.assertEqual(v.max(), 60+25+3)

        # check number of set pixels
        self.assertEqual(im.sum()/123, 1380)

        ## filled square
        im = np.zeros((100, 100))
        draw_box(im, centre=(40, 60), wh=51, color=123, thickness=-1)

        # check grey values
        self.assertEqual(im.min(), 0)
        self.assertEqual(im.max(), 123)

        # check centroid
        x0 = mpq(im, 1, 0) / mpq(im, 0, 0)
        y0 = mpq(im, 0, 1) / mpq(im, 0, 0)
        self.assertAlmostEqual(x0, 40)
        self.assertAlmostEqual(y0, 60)

        # check bounds
        v, u = im.nonzero()
        self.assertEqual(u.min(), 40-25)
        self.assertEqual(u.max(), 40+25)

        self.assertEqual(v.min(), 60-25)
        self.assertEqual(v.max(), 60+25)

        # check number of set pixels
        self.assertEqual(im.sum()/123, 51**2)
        
        ## square with color
        im = np.zeros((100, 100, 3))
        colors = (1,2,3)
        draw_box(im, centre=(40, 60), wh=51, color=colors)

        # check grey values
        for i in range(3):
            plane = im[:,:,i]
            self.assertEqual(plane.min(), 0)
            self.assertEqual(plane.max(), colors[i])

            # check centroid
            x0 = mpq(plane, 1, 0) / mpq(plane, 0, 0)
            y0 = mpq(plane, 0, 1) / mpq(plane, 0, 0)
            self.assertAlmostEqual(x0, 40)
            self.assertAlmostEqual(y0, 60)

            # check bounds
            v, u = plane.nonzero()
            self.assertEqual(u.min(), 40-25)
            self.assertEqual(u.max(), 40+25)

            self.assertEqual(v.min(), 60-25)
            self.assertEqual(v.max(), 60+25)

            # check number of set pixels
            self.assertEqual(plane.sum()/colors[i], 4*(51-1))

    def test_draw_circle(self):
        ## circle
        im = np.zeros((100, 100))
        draw_circle(im, centre=(40, 60), radius=30, color=123)

        # check grey values
        self.assertEqual(im.min(), 0)
        self.assertEqual(im.max(), 123)

        # check centroid
        x0 = mpq(im, 1, 0) / mpq(im, 0, 0)
        y0 = mpq(im, 0, 1) / mpq(im, 0, 0)
        self.assertAlmostEqual(x0, 40)
        self.assertAlmostEqual(y0, 60)

        # check bounds
        v, u = im.nonzero()
        self.assertEqual(u.min(), 40-30)
        self.assertEqual(u.max(), 40+30)

        self.assertEqual(v.min(), 60-30)
        self.assertEqual(v.max(), 60+30)

        # check number of set pixels
        perimeter = im.sum()/123
        self.assertAlmostEqual(perimeter, 168)

        ## filled circle
        im = np.zeros((100, 100))
        draw_circle(im, centre=(40, 60), radius=30, thickness=-1, color=123)

        # check grey values
        self.assertEqual(im.min(), 0)
        self.assertEqual(im.max(), 123)

        # check centroid
        x0 = mpq(im, 1, 0) / mpq(im, 0, 0)
        y0 = mpq(im, 0, 1) / mpq(im, 0, 0)
        self.assertAlmostEqual(x0, 40)
        self.assertAlmostEqual(y0, 60)

        # check bounds
        v, u = im.nonzero()
        self.assertEqual(u.min(), 40-30)
        self.assertEqual(u.max(), 40+30)

        self.assertEqual(v.min(), 60-30)
        self.assertEqual(v.max(), 60+30)

        # check number of set pixels
        self.assertAlmostEqual(im.sum()/123, math.pi*30**2, delta=10)

        circularity = 4 * math.pi * im.sum() / 123 / perimeter**2
        self.assertTrue(circularity > 0.9)
   
        ## circle with color
        im = np.zeros((100, 100, 3))
        colors = (1,2,3)
        draw_circle(im, centre=(40, 60), radius=30, color=colors)

        # check grey values
        for i in range(3):
            plane = im[:,:,i]
            self.assertEqual(plane.min(), 0)
            self.assertEqual(plane.max(), colors[i])

            # check centroid
            x0 = mpq(plane, 1, 0) / mpq(plane, 0, 0)
            y0 = mpq(plane, 0, 1) / mpq(plane, 0, 0)
            self.assertAlmostEqual(x0, 40)
            self.assertAlmostEqual(y0, 60)

            # check bounds
            v, u = plane.nonzero()
            self.assertEqual(u.min(), 40-30)
            self.assertEqual(u.max(), 40+30)

            self.assertEqual(v.min(), 60-30)
            self.assertEqual(v.max(), 60+30)

            # check number of set pixels
            self.assertEqual(plane.sum()/colors[i], 168)

    def test_draw_line(self):
        ## vertical line
        im = np.zeros((100, 100))
        draw_line(im, start=(20,30), end=(20,70), color=123)

        # check grey values
        self.assertEqual(im.min(), 0)
        self.assertEqual(im.max(), 123)

        # check centroid
        x0 = mpq(im, 1, 0) / mpq(im, 0, 0)
        y0 = mpq(im, 0, 1) / mpq(im, 0, 0)
        self.assertAlmostEqual(x0, 20)
        self.assertAlmostEqual(y0, 50)

        # check bounds
        v, u = im.nonzero()
        self.assertEqual(u.min(), 20)
        self.assertEqual(u.max(), 20)

        self.assertEqual(v.min(), 30)
        self.assertEqual(v.max(), 70)

        # check number of set pixels
        self.assertAlmostEqual(im.sum()/123, 41)

        ## horizontal line
        im = np.zeros((100, 100))
        draw_line(im, start=(20,30), end=(70,30), color=123)

        # check grey values
        self.assertEqual(im.min(), 0)
        self.assertEqual(im.max(), 123)

        # check centroid
        x0 = mpq(im, 1, 0) / mpq(im, 0, 0)
        y0 = mpq(im, 0, 1) / mpq(im, 0, 0)
        self.assertAlmostEqual(x0, 45)
        self.assertAlmostEqual(y0, 30)

        # check bounds
        v, u = im.nonzero()
        self.assertEqual(u.min(), 20)
        self.assertEqual(u.max(), 70)

        self.assertEqual(v.min(), 30)
        self.assertEqual(v.max(), 30)

        # check number of set pixels
        self.assertAlmostEqual(im.sum()/123, 51)

        ## thick vertical line
        im = np.zeros((100, 100))
        draw_line(im, start=(20,30), end=(20,70), color=123, thickness=5)

        # check grey values
        self.assertEqual(im.min(), 0)
        self.assertEqual(im.max(), 123)

        # check centroid
        x0 = mpq(im, 1, 0) / mpq(im, 0, 0)
        y0 = mpq(im, 0, 1) / mpq(im, 0, 0)
        self.assertAlmostEqual(x0, 20)
        self.assertAlmostEqual(y0, 50)

        # check bounds
        v, u = im.nonzero()
        self.assertEqual(u.min(), 20-3)
        self.assertEqual(u.max(), 20+3)

        self.assertEqual(v.min(), 30-3)
        self.assertEqual(v.max(), 70+3)

        # check number of set pixels
        self.assertAlmostEqual(im.sum()/123, 309)

        ## oblique line
        im = np.zeros((100, 100))
        draw_line(im, start=(20,30), end=(70,90), color=123)

        # check grey values
        self.assertEqual(im.min(), 0)
        self.assertEqual(im.max(), 123)

        # check centroid
        x0 = mpq(im, 1, 0) / mpq(im, 0, 0)
        y0 = mpq(im, 0, 1) / mpq(im, 0, 0)
        self.assertAlmostEqual(x0, 45, delta=1)
        self.assertAlmostEqual(y0, 60, delta=1)

        # check bounds
        v, u = im.nonzero()
        self.assertEqual(u.min(), 20)
        self.assertEqual(u.max(), 70)

        self.assertEqual(v.min(), 30)
        self.assertEqual(v.max(), 90)

        # check number of set pixels
        self.assertAlmostEqual(im.sum()/123, 61)

        ## colored vertical line
        im = np.zeros((100, 100, 3))
        colors = (1,2,3)
        draw_line(im, start=(20,30), end=(20,70), color=colors)

        # check grey values
        for i in range(3):
            plane = im[:,:,i]
            self.assertEqual(plane.min(), 0)
            self.assertEqual(plane.max(), colors[i])

            # check centroid
            x0 = mpq(plane, 1, 0) / mpq(plane, 0, 0)
            y0 = mpq(plane, 0, 1) / mpq(plane, 0, 0)
            self.assertAlmostEqual(x0, 20)
            self.assertAlmostEqual(y0, 50)

            # check bounds
            v, u = plane.nonzero()
            self.assertEqual(u.min(), 20)
            self.assertEqual(u.max(), 20)

            self.assertEqual(v.min(), 30)
            self.assertEqual(v.max(), 70)

            # check number of set pixels
            self.assertEqual(plane.sum()/colors[i], 41)

    def test_draw_point(self):
        ## pure marker
        im = np.zeros((100, 100))
        draw_point(im, pos=(20, 30), color=123)
        # check grey values
        self.assertEqual(im.min(), 0)
        self.assertEqual(im.max(), 123)

        # check centroid
        x0 = mpq(im, 1, 0) / mpq(im, 0, 0)
        y0 = mpq(im, 0, 1) / mpq(im, 0, 0)
        self.assertAlmostEqual(x0, 20, delta=1)
        self.assertAlmostEqual(y0, 30, delta=1)

        # check bounds
        v, u = im.nonzero()
        self.assertEqual(u.min(), 15)
        self.assertEqual(u.max(), 23)

        self.assertEqual(v.min(), 27)
        self.assertEqual(v.max(), 34)

        # check number of set pixels
        self.assertAlmostEqual(im.sum()/123, 34)

        ## marker with text
        im = np.zeros((100, 100))
        draw_point(im, text='#1', pos=(20, 30), color=123)
        # check grey values
        self.assertEqual(im.min(), 0)
        self.assertEqual(im.max(), 123)

        # check centroid
        x0 = mpq(im, 1, 0) / mpq(im, 0, 0)
        y0 = mpq(im, 0, 1) / mpq(im, 0, 0)
        self.assertAlmostEqual(x0, 29, delta=1)
        self.assertAlmostEqual(y0, 30, delta=1)

        # check bounds
        v, u = im.nonzero()
        self.assertEqual(u.min(), 15)
        self.assertEqual(u.max(), 39)

        self.assertEqual(v.min(), 26)
        self.assertEqual(v.max(), 36)

        ## color marker
        im = np.zeros((100, 100, 3))
        colors = (1,2,3)
        draw_point(im, pos=(20, 30), color=colors)
        for i in range(3):
            plane = im[:,:,i]
            self.assertEqual(plane.min(), 0)
            self.assertEqual(plane.max(), colors[i])

            # check centroid
            x0 = mpq(plane, 1, 0) / mpq(plane, 0, 0)
            y0 = mpq(plane, 0, 1) / mpq(plane, 0, 0)
            self.assertAlmostEqual(x0, 20, delta=1)
            self.assertAlmostEqual(y0, 30, delta=1)

            # check bounds
            v, u = plane.nonzero()
            self.assertEqual(u.min(), 15)
            self.assertEqual(u.max(), 23)

            self.assertEqual(v.min(), 27)
            self.assertEqual(v.max(), 34)

            # check number of set pixels
            self.assertAlmostEqual(plane.sum()/colors[i], 34)

    def test_draw_text(self):
        im = np.zeros((100, 100))
        draw_text(im, pos=(20, 30), text='+', color=123)
        # check grey values
        self.assertEqual(im.min(), 0)
        self.assertEqual(im.max(), 123)

        # check centroid
        x0 = mpq(im, 1, 0) / mpq(im, 0, 0)
        y0 = mpq(im, 0, 1) / mpq(im, 0, 0)
        self.assertAlmostEqual(x0, 24, delta=1)
        self.assertAlmostEqual(y0, 30, delta=1)

        # check bounds
        v, u = im.nonzero()
        self.assertEqual(u.min(), 20)
        self.assertEqual(u.max(), 28)

        self.assertEqual(v.min(), 27)
        self.assertEqual(v.max(), 34)

        # check number of set pixels
        self.assertAlmostEqual(im.sum()/123, 34)

        ## test align options

        im = np.zeros((100, 100))
        draw_text(im, pos=(40, 60), text='XXXX', color=123, align=('right', 'bottom'))
        v, u = im.nonzero()
        self.assertAlmostEqual(u.max(), 40, delta=1)
        self.assertAlmostEqual(v.max(), 60, delta=1)

        im = np.zeros((100, 100))
        draw_text(im, pos=(40, 60), text='XXXX', color=123, align=('left', 'top'))
        v, u = im.nonzero()
        self.assertAlmostEqual(u.min(), 40, delta=1)
        self.assertAlmostEqual(v.min(), 60, delta=1)

        im = np.zeros((100, 100))
        draw_text(im, pos=(40, 60), text='XXXX', color=123, align=('centre', 'centre'))
        v, u = im.nonzero()
        self.assertAlmostEqual(u.mean(), 40, delta=1)
        self.assertAlmostEqual(v.mean(), 60, delta=1)

    def test_draw_labelbox(self):
        im = np.zeros((100, 100))
        draw_labelbox(im, "hello", textcolor=10, labelcolor=5, centre=(20, 30), wh=(40, 20))

    def test_plot_labelbox(self):
        plot_labelbox("hello", textcolor='r', labelcolor='b', centre=(20, 30), wh=(40, 20))
        plt.close('all')
# ------------------------------------------------------------------------ #
if __name__ == '__main__':

    unittest.main()