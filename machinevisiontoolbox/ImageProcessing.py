#!/usr/bin/env python

import numpy as np
import spatialmath.base.argcheck as argcheck
import cv2 as cv


import scipy as sp

from scipy import signal  # TODO figure out sp.signal.convolve2d()?
from scipy import interpolate

from collections import namedtuple
from pathlib import Path

from collections.abc import Iterable


class ImageProcessingMixin:
    """
    Image processing class
    """


# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':

    # testing idisp:
    # im_name = 'longquechen-moon.png'
    im_name = 'multiblobs.png'
    im = mvtb.iread((Path('images') / im_name).as_posix())
    im = Image(im)

    # se = np.ones((3, 3))
    ip = mvt.ImageProcessing()
    # immi = ip.morph(im, se, oper='min', n=25)

    #p = ip.pyramid(ip.mono(im))
    # p[0].disp()

    # p[2].disp()
    # mvt.idisp(immi.image)
    # immi.disp

    # im = iread((Path('images') / 'test' / im_name).as_posix())
    # imo = mono(im)

    # m = mpq(imo, 1, 2)
    # print(m)
    # for debugging interactively
    # import code
    # code.interact(local=dict(globals(), **locals()))

    # show original image
    # idisp(im, title='space rover 2020')

    # do canny:
    # imcan = canny(im, sigma=3, th0=50, th1=100)

    # idisp(imcan, title='canny')

    # K = kgauss(sigma=1)
    # ic = convolve(im, K, optmode='same', optboundary='wrap')

    # import code
    # code.interact(local=dict(globals(), **locals()))

    # idisp(ic,title='convolved')

    # do mono
    # im2 = mono(im1)
    # idisp(im2, title='mono')

    # test color # RGB
    # im3 = color(im2, c=[1, 0, 0])
    # idisp(im3, title='color(red)')

    # test stretch
    # im4 = stretch(im3)
    # idisp(im4, title='stretch')

    # test erode
    # im = np.array([[1, 1, 1, 0],
    #               [1, 1, 1, 0],
    #               [0, 0, 0, 0]])
    # im5 = erode(im, se=np.ones((3, 3)), opt='wrap')
    # print(im5)
    # idisp(im5, title='eroded')

    # im = [[0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 1, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0]]
    # im6 = dilate(im, se=np.ones((3, 3)))
    # print(im6)
    # idisp(im6, title='dilated')

    print('done')
