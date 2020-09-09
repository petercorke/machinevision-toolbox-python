#!/usr/bin/env python

import io as io
import numpy as np
import spatialmath.base.argcheck as argcheck
import cv2 as cv
import matplotlib.path as mpath
import sys as sys

from collections import namedtuple
from pathlib import Path


def idisp(im, **kwargs):
    """
    Interactive image display tool

    :param im: image
    :type im: numpy array, shape (N,M,3)
    :param *args: arguments - options for idisp
    :type *args: see dictionary below TODO
    :param **kwargs: key word arguments - options for idisp
    :type **kwargs: see dictionary below TODO
    :return: :rtype:

    ``idisp(im, **kwargs)`` displays an image and allows interactive
    investigation of pixel values, linear profiles, histograms and zooming. The
    image is displayed in a figure with a toolbar across the top.

    Options::
    'nogui'          don't display the GUI
    'noaxes'         don't display axes on the image
    'noframe'        don't display axes or frame on the image
    'plain'          don't display axes, frame or GUI
    'axis',A         TODO display the image in the axes given by handle A, the
                    'nogui' option is enforced.
    'here'           display the image in the current axes
    'title',T        put the text T in the title bar of the window
    'clickfunc',F    invoke the function handle F(x,y) on a down-click in
                    the window
    'ncolors',N      number of colors in the color map (default 256)
    'bar'            add a color bar to the image
    'print',F        write the image to file F in EPS format
    'square'         display aspect ratio so that pixels are square
    'wide'           make figure full screen width, useful for displaying stereo pair
    'flatten'        display image planes (colors or sequence) as horizontally
                    adjacent images
    'black',B        change black to grey level B (range 0 to 1)
    'ynormal'        y-axis interpolated spectral data and corresponding wavelengthincreases upward, image is inverted
    'histeq'         apply histogram equalization
    'cscale',C       C is a 2-vector that specifies the grey value range that
                    spans the colormap.
    'xydata',XY      XY is a cell array whose elements are vectors that span
                    the x- and y-axes respectively.
    'colormap',C     set the colormap to C (Nx3)
    'grey'           color map: greyscale unsigned, zero is black, maximum
                    value is white
    'invert'         color map: greyscale unsigned, zero is white, maximum
                    value is black
    'signed'         color map: greyscale signed, positive is blue, negative
                    is red, zero is black
    'invsigned'      color map: greyscale signed, positive is blue, negative
                    is red, zero is white
    'random'         color map: random values, highlights fine structure
    'dark'           color map: greyscale unsigned, darker than 'grey',
                    good for superimposed graphics
    'new'            create a new figure

    Example::

        #TODO

    :notes:

    - Is a wrapper around the MATLAB builtin function IMAGE. See the MATLAB help
      on "Display Bit-Mapped Images" for details of color mapping.
    - Color images are displayed in MATLAB true color mode: pixel triples map to
      display RGB values.  (0,0,0) is black, (1,1,1) is white.
    - Greyscale images are displayed in indexed mode: the image pixel value is
      mapped through the color map to determine the display pixel value.
    - For grey scale images the minimum and maximum image values are mapped to
      the first and last element of the color map, which by default
      ('greyscale') is the range black to white. To set your own scaling
      between displayed grey level and pixel value use the 'cscale' option.
    - The title of the figure window by default is the name of the variable
      passed in as the image, this can't work if the first argument is an
      expression.


    References:

        - Robotics, Vision & Control, Section 10.1, P. Corke, Springer 2011.
    """

    # TODO options via *args, **kwargs

    # check if im is valid input

    # set default values for options
    opt = {'nogui': False,
           'noaxes': False,
           'noframe': False,
           'plain': False,
           'axis': False,
           'here': False,
           'title': 'display window',
           'clickfunc': None,
           'ncolors': 256,
           'bar': False,
           'print': None,
           'square': True,
           'wide': False,
           'flatten': False,
           'black': None,
           'ynormal': None,
           'histeq': None,
           'cscale': None,
           'xydata': None,
           'colormap': None,
           'grey': False,
           'invert': False,
           'signed': False,
           'invsigned': False,
           'random': False,
           'dark': False,
           'new': True
           }

    # find the common keys in kwargs
    common_keys = list(kwargs.keys() & opt.keys())

    # apply kwargs to opt
    for k, v in kwargs.items():
        if k in opt:
            opt[k] = v

    # cv.namedWindow(displayWindowTitle, cv.WINDOW_AUTOSIZE)
    cv.imshow(opt['title'], im)
    k = cv.waitKey(0)
    cv.destroyAllWindows()




def iread(file, *args, **kwargs):
    """
    Read image from file

    :param file: file name of image
    :type file: string
    :param *args: arguments
    :type *args: ?
    :param **kwargs: key word arguments - options for idisp
    :type **kwargs: see dictionary below TODO
    :return: :rtype:

    ``iread(file, *args, **kwargs)`` reads the specified image file and returns
    a matrix. The image can by greyscale or color in any of the wide range of
    formats supported by the OpenCV imread function.

    Options::
    'uint8'         return an image with 8-bit unsigned integer pixels in
                    the range 0 to 255
    'single'        return an image with single precision floating point pixels
                    in the range 0 to 1.
    'double'        return an image with double precision floating point pixels
                    in the range 0 to 1.
    'grey'          convert image to greyscale, if it's color, using ITU rec 601
    'grey_709'      convert image to greyscale, if it's color, using ITU rec 709
    'gamma',G       apply this gamma correction, either numeric or 'sRGB'
    'reduce',R      decimate image by R in both dimensions
    'roi',R         apply the region of interest R to each image,
                    where R=[umin umax; vmin vmax].

    Example::

        #TODO

    :notes:

    - A greyscale image is returned as an HxW matrix
    - A color image is returned as an HxWx3 matrix
    - A greyscale image sequence is returned as an HxWxN matrix where N is the
      sequence length
    - A color image sequence is returned as an HxWx3xN matrix where N is the
      sequence length

    References:

        - Robotics, Vision & Control, Section 10.1, P. Corke, Springer 2011.
    """

    # determine if file is valid:
    assert isinstance(file, str), 'file must be a string'

    opt = {
        'uint8': False,
        'single': False,
        'double': False,
        'grey': False,
        'grey_709': False,
        'gamma': 'sRGB',
        'reduce': 1.0,
        'roi': None
    }

    # TODO
    # parse options
    # if empty, display list of images to automatically read

    # check if file is a valid pathname:
    img = cv.imread(file)
    if img is None:
        sys.exit('Could not read the image specified by "file".')


    # TODO check for wild cards
    # TODO search paths automatically for specified file?
    # TODO fetch from server

    return img


def iscolor(im):
    """
    Test for color image

    :param im: file name of image
    :type im: string
    :return s: true if color image (if third dimension of im == 3)
    :rtype: boolean

    ``iscolor(im)`` is true if ``im`` is a color image, that is, its third
    dimension is equal to three.

    Example::

        #TODO

    References:

        - Robotics, Vision & Control, Section 10.1, P. Corke, Springer 2011.
    """

    # W,H is mono
    # W,H,3 is color
    # W,H,N is mono sequence (ambiguous for N=3 mono image sequence)
    # W,H,3,N is color sequence

    # TODO check if image is valid image
    if (im.ndim == 3) and (im.shape[0] > 1) and (im.shape[1] > 1) and (im.shape[2] == 3):
        s = True
    else:
        s = False
    return s


def imono(im, opt='r601'):
    """
    Convert color image to monochrome

    :param im: image
    :type im: numpy array (N,H,3)
    :param opt: greyscale conversion option
    :type opt: string
    :return out: greyscale image
    :rtype: numpy array (N,H)

    ``imono(im)`` is a greyscale equivalent of the color image ``im``

    Example::

        #TODO

    References:

        - Robotics, Vision & Control, Section 10.1, P. Corke, Springer 2011.
    """
    # grayscale conversion option names:
    grey_601 = {'r601', 'grey', 'gray', 'mono', 'grey_601', 'gray_601'}
    grey_709 = {'r709', 'grey_709', 'gray_709'}

    # W,H is mono
    # W,H,3 is color
    # W,H,N is mono sequence (ambiguous for N=3 mono image sequence)
    # W,H,3,N is color sequence

    # check if image is valid input
    # check if opt is valid input
    assert isinstance(opt, str), '"opt" must be a string'

    if not iscolor(im):
        return im

    # determine how many images there
    if (im.ndim == 4):
        nimg = im.shape[3]  # for the W,H,3,N case
    elif im.shape[2] == 3:
        nimg = im.shape[2]  # for the W,H,N case
    else:
        nimg = 1

    # for each image
    for i in range(0, nimg):
        if im.ndim == 4:
            rgb = np.squeeze(im[:, :, :, i])
        else:
            rgb = im[:, :, i]

        if opt in grey_601:
            # rec 601 luma
            outi = 0.229*rgb[:, :, 0] + 0.587*rgb[:, :, 1] + 0.114*rgb[:, :, 2]
        elif opt in grey_709:
            # rec 709 luma
            outi = 0.2126*rgb[:, :, 0] + 0.7152*rgb[:, :, 1] + 0.0722*rgb[:, :, 2]
        elif opt == 'value':
            # 'value' refers to the V in HSV space, not the CIE L*
            # the mean of the max and min of RGB values at each pixel
            mn = rgb[:, :, 2].min(axis=2)
            mx = rgb[:, :, 2].max(axis=2)

            if isinstance(rgb, float):
                outi = 0.5 * (mn + mx)
            else:
                z = (np.int32(mx) + np.int32(mn)) / 2
                outi = z.astype(im.dtype)
        else:
            # TODO there was some raise error/assert thing - check color.py
            print('Error: unknown option for opt')
        out = outi  # TODO append outi to out for each i
        return out


def idouble(im, opt=None):
    """
    Convert integer image to double

    :param im: image
    :type im: numpy array (N,H,3)
    :param opt: either 'single', 'double'  TODO should it be float32 vs float64?
    :type opt: string
    :return out: image with double precision elements ranging from 0 to 1
    :rtype: numpy array (N,H,3)

    ``idouble(im)`` is an image with double precision elements in the range 0 to
    1 corresponding to the elewments of ``im``. The integer pixels ``im`` are
    assumed to span the range 0 to the maximum value of their integer class.

    Options::
    'single'        return an array of single precision floats instead of
                    doubles
    'float'         as above


    Example::

        #TODO

    References:

        - Robotics, Vision & Control, Section 10.1, P. Corke, Springer 2011.
    """

    # make sure image is valid
    # make sure opt is either None or a string
    if (opt == 'float') or (opt == 'single'):
        # convert to float pixel values
        if isinstance(im, int):
            out = im.astype(float) / np.float32(im.max())
        else:
            out = im.astype(float)
    else:
        # convert to double pixel values (default)
        if isinstance(im, int):
            out = np.float64(im) / np.float64(im.max())
        else:
            out = np.float64(im)

    return out
