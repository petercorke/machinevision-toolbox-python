#!/usr/bin/env python

import io as io
import numpy as np
# np.linalg.eig()
import spatialmath.base.argcheck as argcheck
import cv2 as cv
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import sys as sys
import machinevisiontoolbox.color
import time
import scipy as sp
# sp.signal.convolve2d()

from scipy import signal  # TODO figure out sp.signal.convolve2d()?

# code.interact(local=dict(globals(), **locals()))

from collections import namedtuple
from pathlib import Path


def idisp(im, **kwargs):
    """
    Interactive image display tool

    :param im: image
    :type im: numpy array, shape (N,M,3) or (N, M)
    :param args: arguments - options for idisp
    :type args: see dictionary below TODO
    :param kwargs: key word arguments - options for idisp
    :type kwargs: see dictionary below TODO
    :return: :rtype:

    ``idisp(im, **kwargs)`` displays an image and allows interactive
    investigation of pixel values, linear profiles, histograms and zooming. The
    image is displayed in a figure with a toolbar across the top.

    :options:

        - 'nogui'          don't display the GUI
        - 'noaxes'         don't display axes on the image
        - 'noframe'        don't display axes or frame on the image
        - 'plain'          don't display axes, frame or GUI
        - 'axis',A         TODO display the image in the axes given by handle A, the
          'nogui' option is enforced.
        - 'here'           display the image in the current axes
        - 'title',T        put the text T in the title bar of the window
        - 'clickfunc',F    invoke the function handle F(x,y) on a down-click in
          the window
        - 'ncolors',N      number of colors in the color map (default 256)
        - 'bar'            add a color bar to the image
        - 'print',F        write the image to file F in EPS format
        - 'square'         display aspect ratio so that pixels are square
        - 'wide'           make figure full screen width, useful for displaying stereo pair
        - 'flatten'        display image planes (colors or sequence) as horizontally
          adjacent images
        - 'black',B        change black to grey level B (range 0 to 1)
        - 'ynormal'        y-axis interpolated spectral data and corresponding wavelengthincreases upward, image is inverted
        - 'histeq'         apply histogram equalization
        - 'cscale',C       C is a 2-vector that specifies the grey value range that
          spans the colormap.
        - 'xydata',XY      XY is a cell array whose elements are vectors that span
          the x- and y-axes respectively.
        - 'colormap',C     set the colormap to C (Nx3)
        - 'grey'           color map: greyscale unsigned, zero is black, maximum
          value is white
        - 'invert'         color map: greyscale unsigned, zero is white, maximum
          value is black
        - 'signed'         color map: greyscale signed, positive is blue, negative
          is red, zero is black
        - 'invsigned'      color map: greyscale signed, positive is blue, negative
          is red, zero is white
        - 'random'         color map: random values, highlights fine structure
        - 'dark'           color map: greyscale unsigned, darker than 'grey',
          good for superimposed graphics
        - 'new'            create a new figure

    Example::

        # TODO

    .. note::

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


    :references:

        - Robotics, Vision & Control, Section 10.1, P. Corke, Springer 2011.
    """

    # TODO options via *args, **kwargs?

    # check if im is valid input
    im = getimage(im)

    # set default values for options
    opt = {'nogui': False,
           'noaxes': False,
           'noframe': False,
           'plain': False,
           'axis': False,
           'here': False,
           'title': 'Machine Vision Toolbox for Python',
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

    # apply kwargs to opt
    # TODO can be written in one line "a comprehension"
    for k, v in kwargs.items():
        if k in opt:
            opt[k] = v

    # if we are running in a Jupyter notebook, print to matplotlib,
    # otherwise print to opencv imshow/new window. This is done because
    # cv.imshow does not play nicely with .ipynb
    if _isnotebook():
        # recall that matplotlib uses RGB, while opencv uses BGR, so we need to
        # switch the image channels here
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        plt.imshow(im)
        plt.title(opt['title'])
        plt.show()

    else:
        cv.namedWindow(opt['title'], cv.WINDOW_AUTOSIZE)
        cv.imshow(opt['title'], im)  # make sure BGR format image
        k = cv.waitKey(delay=0)  # non blocking, by default False
        # cv.destroyAllWindows()

        # TODO would like to find if there's a more graceful way of
        # exiting/destroying the window, or keeping it running in the background
        # (eg, start a new python process for each figure)
        # if ESC pressed, close the window, otherwise it persists until program
        # exits
        if k == 27:
            # only destroy the specific window
            cv.destroyWindow(opt['title'])


def _isnotebook():
    """
    Determine if code is being run from a Jupyter notebook

    ``_isnotebook`` is True if running Jupyter notebook, else False

    :references:

        - https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-
          is-executed-in-the-ipython-notebook/39662359#39662359
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def iread(file, *args, **kwargs):
    """
    Read image from file

    :param file: file name of image
    :type file: string
    :param args: arguments
    :type args: args
    :param kwargs: key word arguments - options for idisp
    :type kwargs: see dictionary below TODO
    :return: image
    :rtype: numpy array

    ``iread(file, *args, **kwargs)`` reads the specified image file and returns
    a matrix. The image can by greyscale or color in any of the wide range of
    formats supported by the OpenCV imread function.

    :options:

        - 'uint8'         return an image with 8-bit unsigned integer pixels in
          the range 0 to 255
        - 'single'        return an image with single precision floating point pixels
          in the range 0 to 1.
        - 'double'        return an image with double precision floating point pixels
          in the range 0 to 1.
        - 'grey'          convert image to greyscale, if it's color, using ITU rec 601
        - 'grey_709'      convert image to greyscale, if it's color, using ITU rec 709
        - 'gamma',G       apply this gamma correction, either numeric or 'sRGB'
        - 'reduce',R      decimate image by R in both dimensions
        - 'roi',R         apply the region of interest R to each image,
          where R=[umin umax; vmin vmax].

    Example::

        # TODO

    .. note::

        - A greyscale image is returned as an HxW matrix
        - A color image is returned as an HxWx3 matrix
        - A greyscale image sequence is returned as an HxWxN matrix where N is the
          sequence length
        - A color image sequence is returned as an HxWx3xN matrix where N is the
          sequence length

    :references:

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
    im = cv.imread(file)
    if im is None:
        # TODO check ValueError
        raise ValueError('Could not read the image specified by ``file``.')

    # TODO check for wild cards
    # TODO search paths automatically for specified file?
    # TODO fetch from server

    return im


def isimage(im):
    """
    Test if input is an image

    :param im: input image
    :type im: numpy array, shape (N,M), (N,M,3) or (N, M,3, P)
    :return: out
    :rtype: boolean True or False

    .. note::

        - ``isimage(im)`` returns False if.
        - ('im is not of type int or float').
        - ('im has ndims < 2').
        - ('im is (H,W), but does not have enough columns or rows to be an image').
        - ('im (H,W,N), but does not have enough N to be either a color image (N=3).
        - or a sequence of monochrome images (N > 1)').
        - ('im (H,W,M,N) should be a sequence of color images, but M is not equal to
          3').
    """
    # return a consistent data type/format?
    # Can we convert any format to BGR 32f? How would we know format in is RGB vs
    # BGR?

    # convert im to nd.array
    im = np.array(im)

    # TODO consider complex floats?
    # check if image is int or floats
    # TODO shouldn't np.integer and np.float be the catch-all types?
    if not (np.issubdtype(im.dtype, np.integer) or
            np.issubdtype(im.dtype, np.float) or
            np.issubdtype(im.dtype, np.bool_) or
            np.issubdtype(im.dtype, np.uint8) or
            np.issubdtype(im.dtype, np.uint16) or
            np.issubdtype(im.dtype, np.uint32) or
            np.issubdtype(im.dtype, np.uint64) or
            np.issubdtype(im.dtype, np.int8) or
            np.issubdtype(im.dtype, np.int16) or
            np.issubdtype(im.dtype, np.int32) or
            np.issubdtype(im.dtype, np.int64) or
            np.issubdtype(im.dtype, np.float32) or
            np.issubdtype(im.dtype, np.float64)):
        return False

    # check im.ndims > 1
    if im.ndim < 2:
        return False

    # check if im.ndims == 2, then im.shape (W,H), W >= 1, H >= 1
    if (im.ndim == 2) and ((im.shape[0] >= 1) and im.shape[1] >= 1):
        return True

    # check if im.ndims == 3, then im.shape(W,H,N), N >= 1
    if (im.ndim == 3) and (im.shape[2] >= 1):
        return True

    # check if im.ndims == 4, then im.shape(W,H,N,M), then N == 3
    if (im.ndim == 4) and (im.shape[2] == 3):
        return True

    # return consistent image format
    return False


def iint(im, intclass='uint8'):
    """
    Convert image to integer class

    :param im: image
    :type im: numpy array (N,H,3) or (N,H)
    :param intclass: either 'uint8', or any intclass supported by numpy
    :type intclass: string
    :return out: image (uint8)
    :rtype: numpy array (N,H,3)

    ``iint(im)`` is an image with unsigned 8-bit integer elements in the range 0
    to 255 corresponding to the elements of the image ``im``.

    ``int(im, intclass) as above but the output pixels belong to the integer
    class ``intclass``.

    :options:

        # TODO

    .. note::

        - Works for an image with arbitrary number of dimensions, eg. a color
          image or image sequence.
        - If the input image is floating point (single or double) the pixel values
          are scaled from an input range of [0,1] to a range spanning zero to the
          maximum positive value of the output integer class.
        - If the input image is an integer class then the pixels are cast to
          change type but not their value.

    Example::

        # TODO

    :references:

        - Robotics, Vision & Control, Section 12.1, P. Corke, Springer 2011.
    """

    if not isimage(im):
        raise TypeError(im, 'im is not a valid image')

    if np.issubdtype(im.dtype, np.float):
        # rescale to integer
        return (np.rint(im *
                np.float64(np.iinfo(intclass).max))).astype(intclass)
    else:
        return im.astype(intclass)


def idouble(im, opt='float32'):
    """
    Convert integer image to double

    :param im: image
    :type im: numpy array (N,H,3)
    :param opt: either 'single', 'double', or 'float32', or 'float64'
    :type opt: string
    :return out: image with double precision elements ranging from 0 to 1
    :rtype: numpy array (N,H,3)

    ``idouble(im)`` is an image with double precision elements in the range 0 to
    1 corresponding to the elewments of ``im``. The integer pixels ``im`` are
    assumed to span the range 0 to the maximum value of their integer class.

    :options:

        - 'single'        return an array of single precision floats instead of
          doubles
        - 'float'         as above

    Example::

        # TODO

    :references:

        - Robotics, Vision & Control, Section 12.1, P. Corke, Springer 2011.
    """

    # make sure image is valid
    if not isimage(im):
        raise TypeError(im, 'im is not a valid image')

    # make sure opt is either None or a string
    if (opt == 'float') or (opt == 'single') or (opt == 'float32'):
        # convert to float pixel values
        if np.issubdtype(im.dtype, np.integer):
            return im.astype(np.float32) / np.float32(np.iinfo(im.dtype).max)
        else:
            return im.astype(np.float32)
    else:
        # convert to double pixel values (default)
        if np.issubdtype(im.dtype, np.integer):
            return im.astype(np.float64) / np.float64(np.iinfo(im.dtype).max)
        else:
            # the preferred method, compared to np.float64(im)
            return im.astype(np.float64)


def getimage(im):
    """
    converts ``im`` to image compatible with OpenCV

    :param im: image
    :type im: numpy array (N,H,3) or (N,H)
    :return out: image of type np.(uint8, uint16, int16, float32, float64)
    :rtype: numpy array of the size of im

    ``getimage(im)`` converts ``im`` into a compatible datatype with OpenCV:
    CV_8U, CV_16U, CV_16S, CV_32F or CV_64F. By default, if int then CV_8U, and
    if float then CV_64F. Boolean images are converted to 0's and 1's int
    """

    if not isimage(im):
        raise TypeError(im, 'im is not a valid image')

    im = np.array(im)

    validTypes = (np.uint8, np.uint16, np.int16, np.float32, np.float64)
    # if im.dtype is not one of the valid image types,
    # convert to the nearest type or default to float64
    # TODO: what about image scaling?
    if im.dtype not in validTypes:
        # if float, just convert to CV_64F
        if np.issubdtype(im.dtype, np.float):
            im = idouble(im)
        elif np.issubdtype(im.dtype, np.integer):
            if im.min() < 0:
                # use iint (which has scaling), or np.astype()?
                # in this case, since we are converting int to int, or float to
                # float, it should not matter
                im = iint(im, np.int16)
            elif im.max() < np.iinfo(np.uint8).max:
                im = iint(im, np.uint8)
            elif im.max() < np.iinfo(np.uint16).max:
                im = iint(im, np.uint16)
            else:
                raise ValueError(im, 'max value of im exceeds np.uint16')
        elif np.issubdtype(im.dtype, np.bool_):
            im = iint(im)

    return im


def iscolor(im):
    """
    Test for color image

    :param im: image
    :type im: numpy array (N,H) or (N,H,3)
    :return s: true if color image (if third dimension of im == 3)
    :rtype: boolean

    ``iscolor(im)`` is true if ``im`` is a color image, that is, its third
    dimension is equal to three.

    Example::

        # TODO

    :references:

        - Robotics, Vision & Control, Section 10.1, P. Corke, Springer 2011.
    """

    # W,H is mono
    # W,H,3 is color
    # W,H,N is mono sequence (ambiguous for N=3 mono image sequence)
    # W,H,3,N is color sequence

    # TODO check if image is valid image
    if (im.ndim == 3) and (im.shape[0] > 1) and (im.shape[1] > 1) and \
       (im.shape[2] == 3):
        s = True
    else:
        s = False
    return s


def mono(im, opt='r601'):
    """
    Convert color image to monochrome

    :param im: image
    :type im: numpy array (N,H,3), (N,H)
    :param opt: greyscale conversion option
    :type opt: string
    :return out: greyscale image
    :rtype: numpy array (N,H)

    ``mono(im)`` is a greyscale equivalent of the color image ``im``

    Example::

        # TODO

    :references:

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
    # assert isinstance(opt, str), '"opt" must be a string'
    if not isinstance(opt, str):
        raise ValueError(opt, 'opt must be a string')

    if not iscolor(im):
        return im

    # determine how many images there
    if (im.ndim == 4):
        nimg = im.shape[3]  # for the W,H,3,N case
    elif im.shape[2] == 3:
        nimg = 1   # for the W,H,N case
    else:
        nimg = im.shape[2]
    # TODO for W,H,N sequence, should just return sequence of im

    # for each image
    for i in range(0, nimg):
        if im.ndim == 4:
            bgr = np.squeeze(im[:, :, :, i])
        else:
            bgr = im

        if opt in grey_601:
            # print('in grey_601')
            # rec 601 luma
            # NOTE: OpenCV uses BGR
            # for RGB: outi = 0.229 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + \
            #     0.114 * rgb[:, :, 2]
            outi = 0.229 * bgr[:, :, 2] + 0.587 * bgr[:, :, 1] + \
                0.114 * bgr[:, :, 0]
            outi = outi.astype(im.dtype)
        elif opt in grey_709:
            # rec 709 luma
            # TODO fix for BGR (OpenCV)!
            outi = 0.2126 * bgr[:, :, 0] + 0.7152 * bgr[:, :, 1] + \
                0.0722 * bgr[:, :, 2]
            outi = outi.astype(im.dtype)
        elif opt == 'value':
            # 'value' refers to the V in HSV space, not the CIE L*
            # the mean of the max and min of RGB values at each pixel
            mn = bgr[:, :, 2].min(axis=2)
            mx = bgr[:, :, 2].max(axis=2)

            if np.issubdtype(bgr.dtype, np.float):
                outi = 0.5 * (mn + mx)
                outi = outi.astype(im.dtype)
            else:
                z = (np.int32(mx) + np.int32(mn)) / 2
                outi = z.astype(im.dtype)
        else:
            raise TypeError('unknown type for opt')
        out = outi  # TODO append outi to out for each i
        return out


def color(im, c=[1, 1, 1]):
    """
    Colorise a greyscale image

    :param im: image
    :type im: numpy array (N,H)
    :param c: color to color image
    :type c: string or rgb-tuple
    :return out: image with float64 precision elements ranging from 0 to 1
    :rtype: numpy array (N,H,3)

    ``color(im)`` is a color image out ``c`` (N,H,3), where each color
    plane is equal to im.

    ``imcolor(im, c)`` as above but each output pixel is ``c``(3,1) times
    the corresponding element of ``im``.

    Example::

        # TODO

    .. note::

        - Can convert a monochrome sequence (h,W,N) to a color image sequence
          (H,W,3,N).

    :references:

        - Robotics, Vision & Control, Section 10.1, P. Corke, Springer 2011.
    """

    # make sure input image is an image
    im = getimage(im)
    c = argcheck.getvector(c).astype(im.dtype)

    # TODO for im (N,H)
    if im.ndim == 2:
        # only one plan to convert
        # recall opencv uses BGR
        out = np.stack((c[2] * im, c[1] * im, c[0] * im), axis=2)
    else:
        for i in range(im.shape[2]):  # (W,H,3 or N, N (if [2] == 3))
            cplane = None
            for k in range(len(c)):  # should be 0,1,2
                cplane = np.stack((cplane, im[:, :, i] * c[k]), axis=2)
            out[:, :, :, i] = cplane
    return out


def stretch(im, max=1, range=None):
    """
    Image normalisation

    :param im: image
    :type im: numpy array (N,H,3)
    :param max: M   TODO  pixels are mapped to the range 0 to M
    :type max: scalar integer or float
    :param range: range R(1) is mapped to 0, R(2) is mapped to 1 (or max value)
    :type range: 2-tuple or numpy array (2,1)
    :return out: image
    :rtype: numpy array (N,H,3), type double

    ``stretch(im)`` is a normalised image in which all pixel values lie in the
    range of 0 to 1. That is, a linear mapping where the minimum value of ``im``
    is mapped to 0 and the maximum value of ``im`` is mapped to 1.

    .. note::

        - For an integer image the result is a double image in the range 0 to max value

    Example::

        # TODO

    :references:

        - Robotics, Vision & Control, Section 12.1, P. Corke, Springer 2011.
    """

    im = getimage(im)

    # TODO make all infinity values = None?

    if range is None:
        mn = im.min()
        mx = im.max()
    else:
        r = argcheck.getvector(range)
        mn = r[0]
        mx = r[1]

    zs = (im - mn) / (mx - mn) * max

    if r is not None:
        zs = np.maximum(0, np.minimum(max, zs))
    return zs


def getse(se):
    """
    Get structuring element

    :param se: structuring element
    :type se: array (N,H)
    :return se: structuring element
    :rtype: numpy array (N,H) as uint8

    ``getse(se)`` converts matrix ``se`` into a uint8 numpy array for opencv,
    which only accepts kernels of type CV_8U
    """

    # TODO isse test?
    se = np.array(se)

    return se.astype(np.uint8)


def erode(im, se, n=1, opt='replicate', **kwargs):
    """
    Morphological erosion

    :param im: image
    :type im: numpy array (N,H,3) or (N,H)
    :param se: structuring element
    :type se: numpy array (S,T), where S < N and T < H
    :param n: number of times to apply the erosion
    :type n: integer
    :param opt: option specifying the type of erosion
    :type opt: string
    :return out: image
    :rtype: numpy array (N,H,3) or (N,H)

    ``erode(im, se, opt)`` is the image ``im`` after morphological erosion with
    structuring element ``se``.

    ``erode(im, se, n, opt)`` as above, but the structruring element ``se`` is
    applied ``n`` times, that is ``n`` erosions.

    :options:
        -  'replicate'     the border value is replicated (default)
        -  'none'          pixels beyond the border are not included in the window
        -  'trim'          output is not computed for pixels where the structuring
           element crosses the image border, hence output image has
           reduced dimensions TODO

    .. note::

        - Cheaper to apply a smaller structuring element multiple times than
          one large one, the effective structuing element is the Minkowski sum
          of the structuring element with itself N times.

    Example::

        # TODO

    :references:

        - Robotics, Vision & Control, Section 12.5, P. Corke, Springer 2011.
    """

    # check if valid input:
    im = getimage(im)
    se = getse(se)

    # if not isimage(se):
    #     raise TypeError(se, 'se is not a valid image')
    # TODO check to see if se is a valid structuring element
    # TODO check if se is valid (odd number and less than im.shape)
    # consider cv.getStructuringElement?

    if not isinstance(n, int):
        n = int(n)
    if n <= 0:
        raise ValueError(n, 'n must be greater than 0')

    if not isinstance(opt, str):
        raise TypeError(opt, 'opt must be a string')

    # convert options TODO trim?
    cvopt = {
        'replicate': cv.BORDER_REPLICATE,
        'none': cv.BORDER_ISOLATED,
        # 'wrap': cv.BORDER_WRAP # BORDER_WRAP is not supported in OpenCV
    }

    if opt not in cvopt.keys():
        raise ValueError(opt, 'opt is not a valid option')

    # TODO use **kwargs to accept getStructuringElements settings
    # will have to split up getStructuringElement keywords with erode keywords

    # se = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    return cv.erode(im, se, iterations=n, borderType=cvopt[opt], **kwargs)


def dilate(im, se, n=1, opt='replicate', **kwargs):
    """
    Morphological dilation

    :param im: image
    :type im: numpy array (N,H,3) or (N,H)
    :param se: structuring element
    :type se: numpy array (S,T), where S < N and T < H
    :param n: number of times to apply the dilation
    :type n: integer
    :param opt: option specifying the type of dilation
    :type opt: string
    :return out: image
    :rtype: numpy array (N,H,3) or (N,H)

    ``dilate(im, se, opt)`` is the image ``im`` after morphological dilation with
    structuring element ``se``.

    ``dilate(im, se, n, opt)`` as above, but the structruring element ``se`` is
    applied ``n`` times, that is ``n`` dilations.

    :options::

        - 'replicate'     the border value is replicated (default)
        - 'none'          pixels beyond the border are not included in the window
        - 'trim'          output is not computed for pixels where the structuring
          element crosses the image border, hence output image has
          reduced dimensions TODO

    .. note::

        - Cheaper to apply a smaller structuring element multiple times than
          one large one, the effective structuing element is the Minkowski sum
          of the structuring element with itself N times.

    Example::

        # TODO

    :references:

        - Robotics, Vision & Control, Section 12.5, P. Corke, Springer 2011.
    """

    # check if valid input:
    im = getimage(im)
    se = getse(se)

    # TODO check if se is valid (odd number and less than im.shape)
    # if not isimage(se):
    #    raise TypeError(se, 'se is not a valid image')

    if not isinstance(n, int):
        n = int(n)
    if n <= 0:
        raise ValueError(n, 'n must be greater than 0')

    if not isinstance(opt, str):
        raise TypeError(opt, 'opt must be a string')

    # convert options TODO trim?
    cvopt = {
        'replicate': cv.BORDER_REPLICATE,
        'none': cv.BORDER_ISOLATED,
        # 'wrap': cv.BORDER_WRAP # TODO wrap not supported in 4.40
    }

    if opt not in cvopt.keys():
        raise ValueError(opt, 'opt is not a valid option')

    # se = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    return cv.dilate(im, se, iterations=n, borderType=cvopt[opt], **kwargs)


def morph(im, se, oper, n=1, opt='replicate', **kwargs):
    """
    Morphological neighbourhood processing

    :param im: image
    :type im: numpy array (N,H,3) or (N,H)
    :param se: structuring element
    :type se: numpy array (S,T), where S < N and T < H
    :param oper: option specifying the type of morphological operation
    :type oper: string
    :param n: number of times to apply the operation
    :type n: integer
    :param opt: option specifying the border options
    :type opt: string
    :return out: image
    :rtype: numpy array (N,H,3) or (N,H)

    ``morph(im, se, opt)`` is the image ``im`` after morphological operation
    with structuring element ``se``.

    ``morph(im, se, n, opt)`` as above, but the structruring element ``se`` is
    applied ``n`` times, that is ``n`` morphological operations.

    :operation options:

        - 'min'       minimum value over the structuring element
        - 'max'       maximum value over the structuring element
        - 'diff'      maximum - minimum value over the structuring element
        - 'plusmin'   the minimum of the pixel value and the pixelwise sum of the ()
          structuring element and source neighbourhood. :TODO:

    :border options:

        - 'replicate'    the border value is replicated (default)
        - 'none'      pixels beyond the border are not included in the window
        - 'trim'      output is not computed for pixels where the structuring element
          crosses the image border, hence output image has reduced
          dimensions TODO

    .. note::

        - Cheaper to apply a smaller structuring element multiple times than
          one large one, the effective structuing element is the Minkowski sum
          of the structuring element with itself N times.
        - Performs greyscale morphology
        - The structuring element shoul dhave an odd side length.
        - For binary image, min = erosion, max = dilation.
        - The ``plusmin`` operation can be used to compute the distance transform.
        - The input can be logical, uint8, uint16, float or double.
        - The output is always double

    Example::

        # TODO

    :references:

        - Robotics, Vision & Control, Section 12.5, P. Corke, Springer 2011.
    """

    # check if valid input:
    im = getimage(im)
    se = getse(se)

    # TODO check if se is valid (odd number and less than im.shape), can also
    # be a scalar
    # if not isimage(se):
    #    raise TypeError(se, 'se is not a valid image')

    if not isinstance(oper, str):
        raise TypeError(oper, 'oper must be a string')

    if not isinstance(n, int):
        n = int(n)
    if n <= 0:
        raise ValueError(n, 'n must be greater than 0')

    if not isinstance(opt, str):
        raise TypeError(opt, 'opt must be a string')

    # convert options TODO trim?
    cvopt = {
        'replicate': cv.BORDER_REPLICATE,
        'none': cv.BORDER_ISOLATED
    }

    if opt not in cvopt.keys():
        raise ValueError(opt, 'opt is not a valid option')
    # note: since we are calling erode/dilate, we stick with opt. we use
    # cvopt[opt] only when calling the cv.erode/cv.dilate functions

    if oper == 'min':
        #import code
        #code.interact(local=dict(globals(), **locals()))
        out = erode(im, se, n=n, opt=opt)
        # erode(im, se, n=1, opt='border', **kwargs)
    elif oper == 'max':
        out = dilate(im, se, n=n, opt=opt)
    elif oper == 'diff':
        se = getse(se)
        out = cv.morphologyEx(im, cv.MORPH_GRADIENT, se, iterations=n,
                              bordertype=cvopt[opt])
    elif oper == 'plusmin':
        out = None  # TODO
    else:
        raise ValueError(oper, 'morph does not support oper')

    # se = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    return out


def hitormiss(im, s1, s2=None):
    """
    Hit or miss transform

    :param im: image
    :type im: numpy array (N,H,3) or (N,H)
    :param s1: structuring element 1
    :type s1: numpy array (S,T), where S < N and T < H
    :param s2: structuring element 2
    :type s2: numpy array (S,T), where S < N and T < H
    :return out: image
    :rtype: numpy array (N,H,3) or (N,H)

    ``hitormiss(im, s1, s2)`` is the hit-or-miss transform of the binary image
    ``im`` with the structuring element ``s1``. Unlike standard morphological
    operations, ``s1`` has three possible values: 0, 1 and don't care
    (represented by nans).

    Example::

        # TODO

    :references:

        - Robotics, Vision & Control, Section 12.5, P. Corke, Springer 2011.
    """
    # check valid input
    im = getimage(im)
    # TODO also check if binary image?

    if s2 is None:
        s2 = np.float32(s1 == 0)
        s1 = np.float32(s1 == 1)

    return morph(im, s1, 'min') * morph((1 - im), s2, 'min')


def endpoint(im):
    """
    Find end points on a binary skeleton image

    :param im: image
    :type im: numpy array (N,H,3) or (N,H)
    :return out: image
    :rtype: numpy array (N,H,3) or (N,H)

    ``endpoint(im)`` is a binary image where pixels are set if the
    corresponding pixel in the binary image ``im`` is the end point of a
    single-pixel wide line such as found in an image skeleton.  Computed
    using the hit-or-miss morphological operator.

    :references:

        - Robotics, Vision & Control, Section 12.5.3, P. Corke, Springer 2011.
    """

    im = getimage(im)

    se = np.zeros((3, 3, 8))
    se[:, :, 0] = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    se[:, :, 1] = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]])
    se[:, :, 2] = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
    se[:, :, 3] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
    se[:, :, 4] = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
    se[:, :, 5] = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    se[:, :, 6] = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
    se[:, :, 7] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    o = np.zeros(im.shape)
    for i in range(se.shape[2]):
        o = np.logical_or(o, hitormiss(im, se[:, :, i]))
    return o


def triplepoint(im):
    """
    Find triple points

    :param im: image
    :type im: numpy array (N,H,3) or (N,H)
    :return out: image
    :rtype: numpy array (N,H,3) or (N,H)

    ``triplepoint(im)`` is a binary image where pixels are set if the
    corresponding pixel in the binary image ``im`` is a triple point, that is
    where three single-pixel wide line intersect. These are the Voronoi points
    in an image skeleton.  Computed using the hit-or-miss morphological
    operator.

    :references:

     - Robotics, Vision & Control, Section 12.5.3, P. Corke, Springer 2011.
    """

    im = getimage(im)
    se = np.zeros((3, 3, 16))
    se[:, :, 0] = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]])
    se[:, :, 1] = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    se[:, :, 2] = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 0]])
    se[:, :, 3] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 1]])
    se[:, :, 4] = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]])
    se[:, :, 5] = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]])
    se[:, :, 6] = np.array([[0, 1, 0], [1, 1, 0], [0, 1, 0]])
    se[:, :, 7] = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]])
    se[:, :, 8] = np.array([[0, 1, 0], [0, 1, 1], [1, 0, 0]])
    se[:, :, 9] = np.array([[0, 0, 1], [1, 1, 0], [0, 0, 1]])
    se[:, :, 10] = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0]])
    se[:, :, 11] = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 1]])
    se[:, :, 12] = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0]])
    se[:, :, 13] = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 0]])
    se[:, :, 14] = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
    se[:, :, 15] = np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]])

    o = np.zeros(im.shape)
    for i in range(se.shape[2]):
        o = np.logical_or(o, hitormiss(im, se[:, :, i]))
    return o


def iopen(im, se, **kwargs):
    """
    Morphological opening

    :param im: image
    :type im: numpy array (N,H,3) or (N,H)
    :param se: structuring element
    :type se: numpy array (S,T), where S < N and T < H
    :param n: number of times to apply the dilation
    :type n: integer
    :param opt: option specifying the type of dilation
    :type opt: string
    :return out: image
    :rtype: numpy array (N,H,3) or (N,H)

    ``iopen(im, se, opt)`` is the image ``im`` after morphological opening with
    structuring element ``se``. This is a morphological erosion followed by
    dilation.

    ``iopen(im, se, n, opt)`` as above, but the structruring element ``se`` is
    applied ``n`` times, that is ``n`` erosions followed by ``n`` dilations.

    :options:

        - 'border'    the border value is replicated (default)
        - 'none'      pixels beyond the border are not included in the window
        - 'trim'      output is not computed for pixels where the structuring element
          crosses the image border, hence output image has reduced
          dimensions TODO

    .. note::

        - For binary image an opening operation can be used to eliminate small
          white noise regions.
        - Cheaper to apply a smaller structuring element multiple times than
          one large one, the effective structuing element is the Minkowski sum
          of the structuring element with itself N times.

    Example::

        # TODO

    :references:

        - Robotics, Vision & Control, Section 12.5, P. Corke, Springer 2011.
    """

    a = erode(im, se, **kwargs)
    return dilate(a, se, **kwargs)


def iclose(im, se, **kwargs):
    """
    Morphological closing

    :param im: image
    :type im: numpy array (N,H,3) or (N,H)
    :param se: structuring element
    :type se: numpy array (S,T), where S < N and T < H
    :param n: number of times to apply the operation
    :type n: integer
    :param opt: option specifying the type of border behaviour
    :type opt: string
    :return out: image
    :rtype: numpy array (N,H,3) or (N,H)

    ``iclose(im, se, opt)`` is the image ``im`` after morphological closing with
    structuring element ``se``. This is a morphological dilation followed by
    erosion.

    ``iclose(im, se, n, opt)`` as above, but the structuring element ``se`` is
    applied ``n`` times, that is ``n`` dilations followed by ``n`` erosions.

    :options:

        - 'border'    the border value is replicated (default)
        - 'none'      pixels beyond the border are not included in the window
        - 'trim'      output is not computed for pixels where the structuring element
          crosses the image border, hence output image has reduced
          dimensions TODO

    .. note::

        - For binary image an opening operation can be used to eliminate small
          white noise regions.
        - Cheaper to apply a smaller structuring element multiple times than
          one large one, the effective structuing element is the Minkowski sum
          of the structuring element with itself N times.

    Example::

        # TODO

    :references:

        - Robotics, Vision & Control, Section 12.5, P. Corke, Springer 2011.
    """

    a = dilate(im, se, **kwargs)
    return dilate(a, se, **kwargs)


def thin(im, delay=0.0):
    """
    Morphological skeletonization

    :param im: image
    :type im: numpy array (N,H,3) or (N,H)
    :param delay: seconds between each iteration of display
    :type delay: float
    :return out: image
    :rtype: numpy array (N,H,3) or (N,H)

    ``thin(im, delay)`` as above but graphically displays each iteration
    of the skeletonization algorithm with a pause of ``delay`` seconds between
    each iteration.

    Example::

        # TODO

    :references:

        - Robotics, Vision & Control, Section 12.5, P. Corke, Springer 2011.
    """

    # ensure valid input
    im = getimage(im)

    # TODO make sure delay is a float > 0

    # create a binary image (True/False)
    # im = im > 0

    # create structuring elements
    sa = np.array([[0, 0, 0],
                   [np.nan, 1, np.nan],
                   [1, 1, 1]])
    sb = np.array([[np.nan, 0, 0],
                   [1, 1, 0],
                   [np.nan, 1, np.nan]])

    out = im
    while True:
        for i in range(4):
            r = hitormiss(im, sa)
            im = np.logical_xor(im, r)  # might also use the bitwise operator ^
            r = hitormiss(im, sb)
            im = np.logical_xor(im, r)
            sa = np.rot90(sa)
            sb = np.rot90(sb)
        if delay > 0.0:
            idisp(im)
            time.sleep(5)  # TODO work delay into waitKey as optional input!
        if np.all(out == im):
            break
        out = im

    return out


def smooth(im, sigma, hw=None, opt='full'):
    """
    Smooth image

    :param im: image
    :type im: numpy array (N,H,3) or (N,H)
    :param sigma: standard deviation of the Gaussian kernel
    :type sigma: float
    :param hw: half-width of the kernel
    :type hw: float
    :param opt: convolution options np.convolve (see below)
    :type opt: string
    :return out: image
    :rtype: numpy array (N,H,3) or (N,H)

    ``smooth(im, sigma)`` is the image ``im`` after convolution with a
     Gaussian kernel of standard deviation ``sigma``

    ``smooth(im, sigma, hw)`` as above with kernel half-width ``hw``.

    ``smooth(im, sigma, opt)`` as above with options passed to np.convolve

    :options:

        - 'full'    returns the full 2-D convolution (default)
        - 'same'    returns OUT the same size as IM
        - 'valid'   returns  the valid pixels only, those where the kernel does not
          exceed the bounds of the image.

    .. note::

        - By default (option 'full') the returned image is larger than the
          passed image.
        - Smooths all planes of the input image.
        - The Gaussian kernel has a unit volume.
        - If input image is integer it is converted to float, convolved, then
          converted back to integer.
    """

    im = getimage(im)
    if not argcheck.isscalar(sigma):
        raise ValueError(sigma, 'sigma must be a scalar')

    is_int = False
    if np.issubdtype(im.dtype, np.integer):
        is_int = True
        im = idouble(im)

    m = kgauss(sigma, hw)

    # convolution options from smooth.m, which relate to Matlab's conv2.m
    convOpt = {'full', 'same', 'valid'}
    if opt not in convOpt:
        raise ValueError(
            opt, 'opt must be a string of either ''full'', ''same'', or ''valid''')

    if im.ndims == 2:
        # greyscale image
        ims = np.convolve(im, m, opt)
    elif im.ndims == 3:
        # colour image, need to convolve for each image channel
        for i in range(im.shape[2]):
            ims[:, :, i] = np.convolve(im[:, :, i], m, opt)
    elif im.ndims == 4:
        # sequence of colour images
        for j in range(im.shape[3]):
            for i in range(im.shape[2]):
                ims[:, :, i, j] = np.convolve(im[:, :, i, j], m, opt)
    else:
        raise ValueError(im, 'number of dimensions of im is invalid')

    if is_int:
        ims = iint(ims)

    return ims


def kgauss(sigma, hw=None):
    """
    Gaussian kernel

    :param sigma: standard deviation of Gaussian kernel
    :type sigma: float
    :param hw: width of the kernel
    :type hw: integer
    :return k: kernel
    :rtype: numpy array (N,H)

    ``kgauss(sigma)`` is a 2-dimensional Gaussian kernel of standard deviation
    ``sigma``, and centred within the matrix ``k`` whose half-width is
    ``hw=2*sigma`` and ``w=2*hw+1``.

    ``kgauss(sigma, hw)`` as above but the half-width ``hw`` is specified.

    .. note::

        - The volume under the Gaussian kernel is one.
    """

    # make sure sigma, w are valid input
    if hw is None:
        hw = np.ceil(3 * sigma)

    wi = np.arange(-hw, hw + 1)
    x, y = np.meshgrid(wi, wi)

    m = 1.0/(2.0 * np.pi * sigma ** 2) * \
        np.exp(-(np.power(x, 2) + np.power(y, 2)) / 2.0 / sigma ** 2)
    # area under the curve should be 1, but the discrete case is only
    # an approximation
    return m / np.sum(m)


def klaplace():
    r"""
    Laplacian kernel

    :return k: kernel
    :rtype: numpy array (3,3)

    https://stackoverflow.com/questions/31861792/how-to-show-matrix-in-sphinx-docs

    ``klaplace()`` is the Laplacian kernel:

    .. math::

        K = \begin{bmatrix}
            0 & 1 & 0 \\
            1 & -4 & 1 \\
            0 & 1 & 0
            \end{bmatrix}

    .. note::

        - This kernel has an isotropic response to image gradient.
    """
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])


def ksobel():
    r"""
    Sobel edge detector

    :return k: kernel
    :rtype: numpy array (3,3)

    ``ksobel()`` is the Sobel x-derivative kernel:

    .. math::

        K = \frac{1}{8} \begin{bmatrix}
            1 & 0 & -1 \\
            2 & 0 & -2 \\
            1 & 0 & -1
            \end{bmatrix}

    .. note::

        - This kernel is an effective vertical-edge detector
        - The y-derivative (horizontal-edge) kernel is K'
    """
    return np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])/8.0


def kdog(sigma1, sigma2=None, hw=None):
    """
    Difference of Gaussians kernel

    :param sigma1: standard deviation of first Gaussian kernel
    :type sigma1: float
    :param sigma2: standard deviation of second Gaussian kernel
    :type sigma2: float
    :param hw: half-width of Gaussian kernel
    :type hw: integer
    :return k: kernel
    :rtype: numpy array

    ``kdog(sigma1)`` is a 2-dimensional difference of Gaussian kernel equal
    to ``kgauss(sigma1) - kgauss(sigma2)``, where ``sigma1`` > ``sigma2.
    By default, ``sigma2 = 1.6 * sigma1``.  The kernel is centred within
    the matrix ``k`` whose half-width ``hw = 3xsigma1`` and full width of the
    kernel is ``2xhw+1``.

    ``kdog(sigma1, sigma2)`` as above but sigma2 is specified directly.

    ``kdog(sigma1, sigma2, hw)`` as above but the kernel half-width is specified

    .. note::

        - This kernel is similar to the Laplacian of Gaussian and is often used
          as an efficient approximation.
    """

    # sigma1 > sigma2
    if sigma2 is None:
        sigma2 = 1.6 * sigma1
    else:
        if sigma2 > sigma1:
            t = sigma1
            sigma1 = sigma2
            sigma2 = t

    # thus, sigma2 > sigma1
    if hw is None:
        hw = np.ceil(3.0 * sigma1)

    m1 = kgauss(sigma1, hw)  # thin kernel
    m2 = kgauss(sigma2, hw)  # wide kernel

    return m2 - m1


def klog(sigma, hw=None):
    """
    Laplacian of Gaussian kernel

    :param sigma1: standard deviation of first Gaussian kernel
    :type sigma1: float
    :param hw: half-width of kernel
    :type hw: integer
    :return k: kernel
    :rtype: numpy array (2 * 3 * sigma + 1, 2 * 3 * sigma + 1)

    ``klog(sigma)`` is a 2-dimensional Laplacian of Gaussian kernel of
    width (standard deviation) sigma and centred within the matrix ``k`` whose
    half-width is ``hw=3xsigma``, and ``w=2xhw+1``.

    ``klog(sigma, hw)`` as above but the half-width ``w`` is specified.
    """

    # TODO ensure valid input
    if hw is None:
        hw = np.ceil(3.0 * sigma)
    wi = np.arange(-hw, hw + 1)
    x, y = np.meshgrid(wi, wi)
    return 1.0 / (np.pi * sigma ** 4.0) * \
        ((np.power(x, 2) + np.power(y, 2)) / (2.0 * sigma ** 2) - 1) * \
        np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2.0 * sigma ** 2))


def kdgauss(sigma, hw=None):
    """
    Derivative of Gaussian kernel

    :param sigma1: standard deviation of first Gaussian kernel
    :type sigma1: float
    :param hw: half-width of kernel
    :type hw: integer
    :return k: kernel
    :rtype: numpy array (2 * 3 * sigma + 1, 2 * 3 * sigma + 1)

    ``kdgauss(sigma)`` is a 2-dimensional derivative of Gaussian kernel
    ``(w,w)`` of width (standard deviation) sigma and centred within the
    matrix ``k`` whose half-width ``hw = 3xsigma`` and ``w=2xhw+1``.

    ``kdgauss(sigma, hw)`` as above but the half-width is explictly specified.

     .. note::

        - This kernel is the horizontal derivative of the Gaussian, dG/dx.
        - The vertical derivative, dG/dy, is k'.
        - This kernel is an effective edge detector.
    """
    if hw is None:
        hw = np.ceil(3.0 * sigma)

    wi = np.arange(-hw, hw + 1)
    x, y = np.meshgrid(wi, wi)

    return -x / sigma ** 2 / (2.0 * np.pi) * np.exp(-np.power(x, 2) +
                np.power(y, 2) / 2.0 / sigma ** 2)


def kcircle(r, hw=None):
    """
    Circular structuring element

    :param r: radius of circle structuring element, or 2-vector (see below)
    :type r: float, 2-tuple or 2-element vector of floats
    :param hw: half-width of kernel
    :type hw: integer
    :return k: kernel
    :rtype: numpy array (2 * 3 * sigma + 1, 2 * 3 * sigma + 1)

    ``kcircle(r)`` is a square matrix ``(w,w)`` where ``w=2r+1`` of zeros with
    a maximal centred circular region of radius ``r`` pixels set to one.

    ``kcircle(r,w)`` as above but the dimension of the kernel is explicitly
     specified.

    .. note::

        - If ``r`` is a 2-element vector the result is an annulus of ones, and
          the two numbers are interpretted as inner and outer radii.
    """

    # check valid input:
    r = argcheck.getvector(r)
    if r.shape[1] > 1:  # TODO check if this is a good scalar check? I think argcheck has iscalar?
        rmax = r.max()
        rmin = r.min()
    else:
        rmax = r

    if hw is not None:
        w = hw * 2.0 + 1.0
    elif hw is None:
        w = 2.0 * rmax + 1.0

    k = np.zeros(w, w)
    c = np.ceil(w / 2.0)

    if r.shape[1] > 1:
        k = kcircle(rmax, w) - kcircle(rmin, w)
    else:
        x, y = imeshgrid(s)
        x = x - c
        y = y - c
        ll = np.where(np.round(np.power(x, 2) +
                      np.power(y, 2) - np.power(r, 2) <= 0))
        k[ll] = 1
    return k


def imeshgrid(a1, a2=None):
    """
    Domain matrices for image

    :param a1: array input 1
    :type a1: numpy array
    :param a2: array input 2
    :type a2: numpy array
    :return u: domain of image, horizontal
    :rtype u: numpy array
    :return v: domain of image, vertical
    :rtype v: numpy array

    ``imeshgrid(im)`` are matrices that describe the domain of image
    ``im (h,w)`` and are each ``(h,w)``. These matrices are used for the
    evaluation of functions over the image. The element ``u(r,c) = c``
    and ``v(r,c) = r``.

    ``imeshgrid(w, h)`` as above but the domain is ``(w,h)``.

    ``imeshgrid(s)`` as above but the domain is described by ``s`` which can
    be a scalar ``(s,s)`` or a 2-vector ``s=[w,h]``.

    """

    # TODO check valid input, though tricky if a1 is a 2D array
    # (argcheck.ismatrix()?)
    if a2 is None:
        if np.minimum(a1.shape) <= 1:
            # we specify a size for a square output image
            ai = np.arange(0, a1)
            u, v = np.meshgrid(ai, ai)
        elif len(a1) == 2:
            # we specify a size for a rectangular output image (w, h)
            a10 = np.arange(0, a1[0])
            a11 = np.arange(0, a1[1])
            u, v = np.meshgrid(a10, a11)
        elif (a1.ndims >= 2) and (a1.length(a1) > 2):
            u, v = np.meshgrid(
                np.arange(0, a1.shape[0]), np.arange(0, a1.shape[1]))
        else:
            raise ValueError(a1, 'incorrect argument')
    else:
        u, v = np.meshgrid(np.arange(0, a1), np.arange(0, a2))
    return u, v


def pyramid(im, sigma=1, N=None):
    """
    Pyramidal image decomposition

    :param im: image
    :type im: numpy array (N,H,3) or (N,H)
    :param sigma: standard deviation of Gaussian kernel
    :type sigma: float
    :return N: number of levels of pyramid computed
    :rtype N: integer

    ``pyramid(im)`` is a pyramid decomposition of input image ``im`` using
    Gaussian smoothing with standard deviation of 1. The return is a list
    array of images each one having dimensions half that of the previous image.
    The pyramid is computed down to a non-halvable image size.

    ``pyramid(im, sigma)`` as above but the Gaussian standard deviation
    is ``sigma``.

    ``pyramid(im, sigma, N)`` as above but only ``N`` levels of the pyramid are
    computed.

    .. note::

        - Works for greyscale images only.
    """

    # check inputs
    im = getimage(im)
    if not argcheck.isscalar(sigma):
        raise ValueError(sigma, 'sigma must be a scalar')

    if (not argcheck.isscalar(N)) and (N >= 0) and (N <= max(im.shape)):
        raise ValueError(N, 'N must be a scalar and 0 <= N <= max(im.shape)')

    if N is None:
        N = max(im.shape)

    # TODO options to accept different border types,
    # note that the Matlab implementation is hard-coded to 'same'

    # return cv.buildPyramid(im, N, borderType=cv.BORDER_REPLICATE)
    # Python version does not seem to be implemented

    # list comprehension approach
    # TODO pyr = [cv.pyrdown(inputs(i)) for i in range(blah)]

    p = [im]
    for i in range(N):
        if any(im.shape == 1):
            break
        im = cv.pyrDown(im, borderType=cv.BORDER_REPLICATE)
        p.append(im)

    return p


def sad(im1, im2):
    """
    Sum of absolute differences

    :param im1: image 1
    :type im1: numpy array
    :param im2: image 2
    :type im2: numpy array
    :return out: sad
    :rtype out: scalar

    ``sad(im1, im2)`` is the sum of absolute differences between the
    two equally sized image patches ``im1`` and ``im2``.
    The result is a scalar that indicates image similarity, a value of 0
    indicates identical pixel patterns and is increasingly positive as image
    dissimilarity increases.
    """

    im1 = getimage(im1)
    im2 = getimage(im2)
    if not np.all(im1.shape == im2.shape):
        raise ValueError(im2, 'im2 shape is not equal to im1')

    m = np.abs(im1 - im2)
    return np.sum(m)


def ssd(im1, im2):
    """
    Sum of squared differences

    :param im1: image 1
    :type im1: numpy array
    :param im2: image 2
    :type im2: numpy array
    :return out: ssd
    :rtype out: scalar

    ``ssd(im1, im2)`` is the sum of squared differences between the
    two equally sized image patches ``im1`` and ``im2``.  The result M is a
    scalar that indicates image similarity, a value of 0 indicates identical
    pixel patterns and is increasingly positive as image
    dissimilarity increases.
    """

    im1 = getimage(im1)
    im2 = getimage(im2)
    if not np.all(im1.shape == im2.shape):
        raise ValueError(im2, 'im2 shape is not equal to im1')
    m = np.power((im1 - im2), 2)
    return np.sum(m)


def ncc(im1, im2):
    """
    Normalised cross correlation

    :param im1: image 1
    :type im1: numpy array
    :param im2: image 2
    :type im2: numpy array
    :return out: ncc
    :rtype out: scalar

    ``ncc(im1, im2)`` is the normalized cross-correlation between the
    two equally sized image patches ``im1`` and ``im2``.
    The result is a scalar in the interval -1 (non match) to
    1 (perfect match) that indicates similarity.

    .. note::

        - A value of 1 indicates identical pixel patterns.
        - The ``ncc`` similarity measure is invariant to scale changes in image
          intensity.
    """

    im1 = getimage(im1)
    im2 = getimage(im2)
    if not np.all(im1.shape == im2.shape):
        raise ValueError(im2, 'im2 shape is not equal to im1')
    denom = np.sqrt(np.sum(np.power(im1, 2) * np.power(im2, 2)))

    if denom < 1e-10:
        return 0
    else:
        return np.sum(im1 * im2) / denom


def zsad(im1, im2):
    """
    Zero-mean sum of absolute differences

    :param im1: image 1
    :type im1: numpy array
    :param im2: image 2
    :type im2: numpy array
    :return out: zsad
    :rtype out: scalar

    ``zsad(im1, im2)`` is the zero-mean sum of absolute differences between the
    two equally sized image patches `im1`` and ``im2``.
    The result is a scalar that indicates image similarity, a value of 0
    indicates identical pixel patterns and is increasingly positive as image
    dissimilarity increases.

    .. note::

        - The ``zsad`` similarity measure is invariant to changes in image
          brightness offset.
    """

    im1 = getimage(im1)
    im2 = getimage(im2)
    if not np.all(im1.shape == im2.shape):
        raise ValueError(im2, 'im2 shape is not equal to im1')
    im1 = im1 - np.mean(im1)
    im2 = im2 - np.mean(im2)
    m = np.abs(im1 - im2)
    return np.sum(m)


def zssd(im1, im2):
    """
    Zero-mean sum of squared differences

    :param im1: image 1
    :type im1: numpy array
    :param im2: image 2
    :type im2: numpy array
    :return out: zssd
    :rtype out: scalar

    ``zssd(im1, im2)`` is the zero-mean sum of squared differences between the
    two equally sized image patches ``im1`` and ``im2``.  The result
    is a scalar that indicates image similarity, a value of 0 indicates
    identical pixel patterns and is increasingly positive as image
    dissimilarity increases.

    .. note::

        - The ``zssd`` similarity measure is invariant to changes in image
          brightness offset.
    """

    im1 = getimage(im1)
    im2 = getimage(im2)
    if not np.all(im1.shape == im2.shape):
        raise ValueError(im2, 'im2 shape is not equal to im1')
    im1 = im1 - np.mean(im1)
    im2 = im2 - np.mean(im2)
    m = np.power(im1 - im2, 2)
    return np.sum(m)


def zncc(im1, im2):
    """
    Zero-mean normalized cross correlation

    :param im1: image 1
    :type im1: numpy array
    :param im2: image 2
    :type im2: numpy array
    :return out: zncc
    :rtype out: scalar

    ``zncc(im1, im2)`` is the zero-mean normalized cross-correlation between the
    two equally sized image patches ``im1`` and ``im2``.  The result is a
    scalar in the interval -1 to 1 that indicates similarity.  A value of 1
    indicates identical pixel patterns.

    .. note::

        - The ``zncc`` similarity measure is invariant to affine changes in image
          intensity (brightness offset and scale).
    """

    im1 = getimage(im1)
    im2 = getimage(im2)
    if not np.all(im1.shape == im2.shape):
        raise ValueError(im2, 'im2 shape is not equal to im1')
    im1 = im1 - np.mean(im1)
    im2 = im2 - np.mean(im2)
    denom = np.sqrt(np.sum(np.power(im1, 2) * np.sum(np.power(im2, 2))))

    if denom < 1e-10:
        return 0
    else:
        return np.sum(im1 * im2) / denom


def thresh(im, t=None, opt='binary'):
    """
    Image threshold

    :param im: image
    :type im: numpy array
    :param t: threshold
    :type t: scalar
    :param opt: threshold option (see below)
    :type opt: string
    :return imt: thresholded binary image
    :rtype imt: numpy array same size as im
    :return threshvalue: threshold if opt is otsu or triangle
    :rtype threshvalue: scalar

    See opencv threshold types for threshold options
    https://docs.opencv.org/4.2.0/d7/d1b/group__imgproc__
    misc.html#gaa9e58d2860d4afa658ef70a9b1115576

    :options:
        - 'binary' # TODO consider the LaTeX formatting of equations
        - 'binary_inv'
        - 'trunc'
        - 'tozero'
        - 'tozero_inv'
        - 'otsu'
        - 'triangle'

    .. note::

        - greyscale only
        - For a uint8 class image the slider range is 0 to 255.
        - For a floating point class image the slider range is 0 to 1.0
    """

    # dictionary of threshold options from OpenCV
    threshopt = {
        'binary': cv.THRESH_BINARY,
        'binary_inv': cv.THRESH_BINARY_INV,
        'trunc': cv.THRESH_TRUNC,
        'tozero': cv.THRESH_TOZERO,
        'tozero_inv': cv.THRESH_TOZERO_INV,
        'otsu': cv.THRESH_OTSU,
        'triangle': cv.THRESH_TRIANGLE
    }

    im = getimage(im)

    if t is not None:
        if not argcheck.isscalar(t):
            raise ValueError(t, 't must be a scalar')
    else:
        # if no threshold is specified, we assume to use Otsu's method
        opt = 'otsu'

    # for image int class, maxval = max of int class
    # for image float class, maxval = 1
    if np.issubdtype(im.dtype, np.integer):
        maxval = np.iinfo(im.dtype).max
    else:
        # float image, [0, 1] range
        maxval = 1.0
    threshvalue, imt = cv.threshold(im, t, maxval, threshopt[opt])

    if opt == 'otsu' or opt == 'triangle':
        return imt, threshvalue
    else:
        return imt


def window(im, se, func, opt='border', **kwargs):
    """
    Generalized spatial operator

    :param im: image
    :type im: numpy array
    :param se: structuring element
    :type se: numpy array
    :param func: function to operate
    :type funct: reference to a callable function
    :param opt: border option
    :type opt: string
    :return out: image after function has operated on every pixel
    :rtype out: numpy array

    ``window(im, se, func)`` is an image where each pixel is the result
    of applying the function ``func`` to a neighbourhood centred on the
    corresponding pixel in ``im``. The neighbourhood is defined by the size
    of the structuring element ``se`` which should have odd side lengths.
    The elements in the neighbourhood corresponding to non-zero elements in
    ``se`` are packed into a vector (in column order from top left) and passed
    to the specified callable function ``func``.
    The return value of ``func`` becomes the corresponding pixel value.

    ``window(im, se, func, opt)`` as above but performance of edge
    pixels can be controlled.

    :options:
        -  'replicate'     the border value is replicated (default)
        -  'none'          pixels beyond the border are not included in the window
        -  'trim'          output is not computed for pixels where the structuring
           element crosses the image border, hence output image has
           reduced dimensions TODO

    Example::

        Compute the maximum value over a 5x5 window:
        window(im, ones(5,5), @max);

        Compute the standard deviation over a 3x3 window:
        window(im, ones(3,3), @std);

    .. note::

        - The structuring element should have an odd side length.
        - Is slow since the function ``func`` must be invoked once for every
          output pixel.
        - The input can be logical, uint8, uint16, float or double, the output is
          always double
    """

    # window's mex function with scipy's ndimage.generic_filter

    # check valid input
    im = getimage(im)
    # se = getimage(se)

    # border options:
    edgeopt = {
        'border': 'nearest',
        'none': 'constant',  # TODO does not seem to be a 'not included' option
        'wrap': 'wrap'
    }
    if opt not in edgeopt:
        raise ValueError(opt, 'opt is not a valid edge option')

    # TODO check valid input for func? callable(func) == True

    return sp.ndimage.generic_filter(im, func, footprint=se, mode=edgeopt[opt])


def rank(im, se, rank=-1, opt='replicate'):
    """
    Rank filter

    :param im: image
    :type im: numpy array
    :param se: structuring element
    :type se: numpy array
    :param rank: rank of filter
    :type rank: integer
    :param opt: border option
    :type opt: string
    :return out: image after rank filter applied to every pixel
    :rtype out: numpy array

    ``rank(im, se, rank)`` is a rank filtered version of ``im``.  Only
    pixels corresponding to non-zero elements of the structuring element ``se``
    are ranked and the ``rank``'ed value in rank becomes the corresponding
    output pixel value.  The highest rank, the maximum, is ``rank=-1``.

    ``rank(im, se, rank, opt)`` as above but the processing of edge
    pixels can be controlled.

    :options:
        -  'replicate'     the border value is replicated (default)
        -  'none'          pixels beyond the border are not included in the window
        -  'trim'          output is not computed for pixels where the structuring
           element crosses the image border, hence output image has
           reduced dimensions TODO

    Example::

        5x5 median filter, 25 elements in the window, the median is the 12th in rank
        rank(im, 12, ones(5,5));

        3x3 non-local maximum, find where a pixel is greater than its 8 neighbours
        se = ones(3,3); se(2,2) = 0;
        im > rank(im, 1, se);

    .. note::

        - The structuring element should have an odd side length.
        - The input can be logical, uint8, uint16, float or double, the output is
          always double
    """

    # TODO replace rank.m mex function with scipy.ndimage.rank_filter

    # check valid input
    im = getimage(im)
    # se = getimage(se)

    if not isinstance(rank, int):
        raise TypeError(rank, 'rank is not an int')

    # border options for rank_filter that are compatible with rank.m
    borderopt = {
        'replicate': 'nearest',
        'wrap': 'wrap'
    }

    if opt not in borderopt:
        raise ValueError(opt, 'opt is not a valid option')

    return sp.ndimage.rank_filter(im, rank, footprint=se, mode=borderopt[opt])


def hist(im, nbins=256, opt=None):
    """
    Image histogram

    :param im: image
    :type im: numpy array
    :param nbins: number of bins for histogram
    :type nbins: integer
    :param opt: histogram option
    :type opt: string
    :return hist: histogram h as a column vector, and corresponding bins x
    :rtype hist: collections.namedtuple


    ``hist(im)`` is the histogram of intensities for image ``im`` as a vector.
    For an image with  multiple planes, the histogram of each plane is given
    in a separate column.

    ``hist(im, nbins)`` as above with the number of bins specified

    ``hist(im, opt)`` as above with histogram options specified

    :options:

        - 'cdf'       compute a cumulative histogram
        - 'normcdf'   compute a normalized cumulative histogram, whose maximum value
          is one
        - 'sorted'    histogram but with occurrence sorted in descending magnitude
          order.  Bin coordinates X reflect this sorting.

    Example::

       [h,x] = hist(im);
       bar(x,h);

       [h,x] = hist(im, 'normcdf');
       plot(x,h);

    .. note::

        - The bins spans the greylevel range 0-255.
        - For a floating point image the histogram spans the greylevel range 0-1.
        - For floating point images all NaN and Inf values are first removed.
    """

    # check inputs
    im = getimage(im)

    optHist = ['cdf', 'normcdf', 'sorted']
    if opt is not None and opt not in optHist:
        raise ValueError(opt, 'opt is not a valid option')

    if np.issubdtype(im.dtype, np.integer):
        maxrange = np.iinfo(im.dtype).max
    else:
        # float image
        maxrange = 1.0

    # if greyscale image, then iterate once,
    # otherwise, iterate for each color channel
    if im.ndim == 2:
        numchannel = 1
    else:
        numchannel = im.shape[2]

    # normal histogram case
    h = np.zeros((nbins, numchannel))
    x = np.linspace(0, maxrange, nbins, endpoint=True)  # bin coordinate
    for i in range(numchannel):
        h[:, i] = cv.calcHist([im], [i], None, [nbins], [0, maxrange])

    if opt == 'cdf':
        h = np.cumsum(h)
    elif opt == 'normcdf':
        h = np.cumsum(h)
        h = h / h[-1]
    elif opt == 'sorted':
        h = np.sort(h, axis=0)
        x = x[np.argsort(h, axis=0)]

    # what should we return? hist, x? named tuple perhaps?
    # TODO cdf and normcdf add to named tuple
    return namedtuple('hist', 'h x')(h, x)


def normhist(im):
    """
    Histogram normalisaton

    :param im: image
    :type im: numpy array
    :return nim: normalised image
    :rtype nim: numpy array

    ``normhist(im)`` is a histogram normalized version of the image ``im``.

    .. note::

        - Highlights image detail in dark areas of an image.
        - The histogram of the normalized image is approximately uniform, that is,
          all grey levels ae equally likely to occur.
    """

    im = getimage(im)
    if im.ndims > 2:
        raise ValueError(im, 'normhist does not support color images')

    # TODO could alternatively just call cv.equalizeHist()?
    # TODO note that cv.equalizeHist might only accept 8-bit images, while normhist can accept float images as well?    # return cv.equalizeHist(im)
    cdf = hist(im, 'cdf')
    cdf.h = cdf.h / np.max(cdf.h)

    if np.issubdtype(im.dtype, np.float):
        nim = np.interp(im.flatten(), cdf.x, cdf.h)
    else:
        nim = np.interp(idouble(im).flatten(), cdf.x, cdf.h)

    # reshape nim to image:
    return nim.reshape(im.shape[0], im.shape[1])


# TODO figure out how to do function handles because of the arguments!


def similarity(T, im, metric=None):
    """
    Locate template in image

    :param T: template image
    :type T: numpy array
    :param im: image
    :type im: numpy array
    :param metric: similarity metric function
    :type metric: callable function reference
    :return S: similarity image
    :rtype S: numpy array

    ``similarity(T, im)`` is an image where each pixel is the ``zncc``
    similarity of the template ``T`` (M,M) to the (M,M) neighbourhood
    surrounding the corresonding input pixel in ``im``.  ``S`` is same
    size as ``im``.

    ``similarity(T, im, metric)`` as above but the similarity metric is
    specified by the function ``metric`` which can be any of
    @sad, @ssd, @ncc, @zsad, @zssd.

    :Example:

        #TODO (see isimilarity.m for example)

    .. note::

        - For NCC and ZNCC the maximum in S corresponds to the most likely template
          location.  For SAD, SSD, ZSAD and ZSSD the minimum value corresponds
          to the most likely location.
        - Similarity is not computed for those pixels where the template crosses
          the image boundary, and these output pixels are set to NaN.
        - The ZNCC function is a MEX file and therefore the fastest
        - User provided similarity metrics can be used, the function accepts
          two regions and returns a scalar similarity score.

    :references:

     - Robotics, Vision & Control, Section 12.4, P. Corke, Springer 2011.
    """

    # check inputs
    im = getimage(im)
    T = getimage(T)
    if ((T.shape[0] % 2) == 0) or ((T.shape[1] % 2) == 0):
        raise ValueError(T, 'template T must have odd dimensions')

    if metric is None:
        metric = zncc

    hc = np.floor(T.shape[0]/2)
    hr = np.floor(T.shape[1]/2)

    S = np.empty(im.shape)

    # TODO can probably replace these for loops with list comprehensions
    for c in range(start=hc+1, stop=im.shape[0]-hc):
        for r in range(start=hr+1, stop=im.shape[1]-hr):
            S[r, c] = metric(T, im[r-hr:r+hr, c-hc:c+hc])  # TODO check indexes
    return S


def convolve(im, K, optmode='same', optboundary='wrap'):
    """
    Image convolution

    :param im: image
    :type im: numpy array
    :param K: kernel
    :type K: numpy array
    :param optmode: option for convolution
    :type optmode: string
    :param optboundary: option for boundary handling
    :type optboundary: string
    :return C: convolved image
    :rtype C: numpy array

    ``convolve(im, K)`` is the convolution of image ``im`` with the kernel ``K``

    ``convolve(im, K, optmode)`` as above but specifies the convolution mode.
    See scipy.signal.convolve2d for details, mode options below

    ``convolve(im, K, optboundary)`` as above but specifies the boundary
    handling options

    :options:

        - 'same'    output image is same size as input image (default)
        - 'full'    output image is larger than the input image
        - 'valid'   output image is smaller than the input image, and contains only
          valid pixels TODO

    .. note::

         - If the image is color (has multiple planes) the kernel is applied to
           each plane, resulting in an output image with the same number of planes.
         - If the kernel has multiple planes, the image is convolved with each
           plane of the kernel, resulting in an output image with the same number of
           planes.
         - This function is a convenience wrapper for the MATLAB function CONV2.
         - Works for double, uint8 or uint16 images.  Image and kernel must be of
           the same type and the result is of the same type.
         - This function replaces iconv().

    :references:

     - Robotics, Vision & Control, Section 12.4, P. Corke, Springer 2011.
    """

    im = getimage(im)
    # if not isinstance(K, np.float):  # TODO check K, kernel, can be numpy
    #  array K = np.float64(K)

    # TODO check opt is valid string based on conv2 options
    modeopt = {
        'full': 'full',
        'valid': 'valid',
        'same': 'same'
    }

    if optmode not in modeopt:
        raise ValueError(optmode, 'opt is not a valid option')

    boundaryopt = {
        'fill': 'fill',
        'wrap': 'wrap',
        'reflect': 'symm'
    }
    if optboundary not in boundaryopt:
        raise ValueError(optboundary, 'opt is not a valid option')

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.
    # convolve2d.html
    if im.ndim == 2 and K.ndim == 2:
        # simple case, convolve image with kernel, both are 2D
        C = signal.convolve2d(
            im, K, mode=modeopt[optmode], boundary=boundaryopt[optboundary])
    elif im.ndim == 3 and K.ndim == 2:
        # image has multiple planes:
        C = [None]*im.shape[2]
        for i in range(im.shape[2]):
            C[i] = signal.convolve2d(im[:, :, i], K, mode=modeopt[optmode],
                                     boundary=boundaryopt[optboundary])
        C = np.dstack(C)
    elif im.ndim == 2 and K.ndim == 3:
        # kernel has multiple planes:
        C = [None]*K.shape[2]
        for i in range(K.shape[2]):
            C[i] = signal.convolve2d(im, K[:, :, i], mode=modeopt[optmode],
                                     boundary=boundaryopt[optboundary])
        C = np.dstack(C)

    else:
        # TODO how to make image and kernel raise error?
        raise ValueError(
            im, 'image and kernel cannot both have muliple planes')

    return C


def canny(im, sigma=1, th0=None, th1=None):
    """
    Canny edge detection

    :param im: image
    :type im: numpy array
    :param sigma: standard deviation for Gaussian kernel smoothing
    :type sigma: float
    :param th0: lower threshold
    :type th0: float
    :param th1: upper threshold
    :type th1: float
    :return E: edge image
    :rtype E: numpy array

    ``canny(im)`` is an edge image obtained using the Canny edge
    detector algorithm.  Hysteresis filtering is applied to the gradient
    image: edge pixels > ``th1`` are connected to adjacent pixels > ``th0``,
    those below ``th0`` are set to zero.

    ``canny(im, sigma, th0, th1)`` as above, but the standard deviation of the
    Gaussian smoothing, ``sigma``, lower and upper thresholds ``th0``, ``th1``
    can be specified

    .. note::

        - Produces a zero image with single pixel wide edges having non-zero values.
        - Larger values correspond to stronger edges.
        - If th1 is zero then no hysteresis filtering is performed.
        - A color image is automatically converted to greyscale first.

    :references:

        - "A Computational Approach To Edge Detection", J. Canny,
          IEEE Trans. Pattern Analysis and Machine Intelligence, 8(6):679–698, 1986.

    """

    # check valid input
    im = getimage(im)

    # set defaults (eg thresholds, eg one as a function of the other)
    if th0 is None:
        if np.issubdtype(np.float):
            th0 = 0.1
        else:
            # isint
            th0 = np.round(0.1*np.iinfo(im.dtype).max)
    if th1 is None:
        th1 = 1.5*th0

    # compute gradients Ix, Iy using guassian kernel
    dg = kdgauss(sigma)
    Ix = np.abs(convolve(im, dg, 'same'))
    Iy = np.abs(convolve(im, np.transpose(dg), 'same'))

    # Ix, Iy must be 16-bit input image
    Ix = np.array(Ix, dtype=np.int16)
    Iy = np.array(Iy, dtype=np.int16)

    return cv.Canny(Ix, Iy, th0, th1, L2gradient=True)


def replicate(im, M=1):
    """
    Expand image

    :param im: image to expand
    :type im: numpy array
    :param M: number of times to replicate image
    :type M: integer
    :return E: expanded image
    :rtype E: numpy array

    ``replicate(im, M)`` is an expanded version of the image (H,W) where
    each pixel is replicated into a (M,M) tile. If ``im`` is (H,W) the
    result is ((M*H),(M*W)) numpy array.
    """

    im = getimage(im)
    if im.ndims > 2:
        # dealing with multiplane image
        ir2 = []
        for i in range(im.shape[2]):
            ir2 = np.append(replicate(im[:, :, i], M))
        return ir2

    nr = im.shape[0]
    nc = im.shape[1]

    # replicate columns
    ir = np.zeros((M * nr, nc), dtype=im.dtype)
    for r in range(M):
        ir[r:-1:M, :] = im

    # replicate rows
    ir2 = np.zeros((M*nr, M*nc), dtype=im.dtype)
    for c in range(M):
        ir2[:, c:-1:M] = ir

    return ir2


def decimate(im, m=2, sigma=None):
    """
    Decimate an image

    :param im: image
    :type im: numpy array
    :param m: decimation factor TODO probably not the correct term
    :type m: integer
    :param sigma: standard deviation for Gaussian kernel smoothing
    :type sigma: float
    :return out: decimated image
    :rtype out: numpy array

    ``idecimate(im, m)`` is a decimated version of the image IM whose
    size is reduced by m (an integer) in both dimensions.  The image is
    smoothed with a Gaussian kernel with standard deviation m/2
    then subsampled.

    ``idecimate(im, m, sigma)`` as above but the standard deviation of the
    smoothing kernel is set to ``sigma``.

    .. note::

        - If the image has multiple planes, each plane is decimated.
        - Smoothing is used to eliminate aliasing artifacts and the standard
          deviation should be chosen as a function of the maximum spatial frequency
          in the image.
    """

    im = getimage(im)
    if (m - np.ceil(m)) != 0:
        raise ValueError(m, 'decimation factor m must be an integer')

    if sigma is None:
        sigma = m/2

    # smooth image
    im = smooth(im, sigma)

    # decimate image
    return im[0:-1:m, 0:-1:m, :]


def testpattern(t, w, *args, **kwargs):
    """
    Create test images

    :param t: pattern type
    :type t: string
    :param w: image size of output pattern image
    :type w: integer or 2-element vector
    :param args: arguments for test patterns
    :type args: float (varies)
    :param kwargs: keyword arguments for test patterns? Not currently used
    :type kwargs: dictionary
    :return z: test pattern image
    :rtype z: numpy array

    ``testpattern(type, w, args)`` creates a test pattern image.  If ``w`` is a
    scalar the output image has shape ``(w,w)`` else if ``w=(w,h)`` the output
    image shape is ``(w,h)``.  The image is specified by the string ``t`` and
    one or two (type specific) arguments:

    :t options:

    - 'rampx'     intensity ramp from 0 to 1 in the x-direction. ARGS is the number
      of cycles.
    - 'rampy'     intensity ramp from 0 to 1 in the y-direction. ARGS is the number
      of cycles.
    - 'sinx'      sinusoidal intensity pattern (from -1 to 1) in the x-direction.
      ARGS is the number of cycles.
    - 'siny'      sinusoidal intensity pattern (from -1 to 1) in the y-direction.
      ARGS is the number of cycles.
    - 'dots'      binary dot pattern.  ARGS are dot pitch (distance between
      centres); dot diameter.
    - 'squares'   binary square pattern.  ARGS are pitch (distance between
      centres); square side length.
    - 'line'      a line.  ARGS are theta (rad), intercept.

    Example::

        A 256x256 image with 2 cycles of a horizontal sawtooth intensity ramp:
        testpattern('rampx', 256, 2);

        A 256x256 image with a grid of dots on 50 pixel centres and 20 pixels in
        diameter:
        testpattern('dots', 256, 50, 25);

    """

    # check valid input
    topt = ['sinx', 'siny', 'rampx', 'rampy', 'line', 'squares', 'dots']
    if t not in topt:
        raise ValueError(t, 't is an unknown pattern type')

    w = argcheck.getvector(w)
    if np.length(w) == 1:
        z = np.zeros((w, w))
    elif np.length(w) == 2:
        z = np.zeros((w[0], w[1]))
    else:
        raise ValueError(w, 'w has more than two values')

    if t == 'sinx':
        if len(args) > 0:
            ncycles = args[0]
        else:
            ncycles = 1
        x = np.arange(0, z.shape[1]-1)
        c = z.shape[1] / ncycles
        z = np.matlib.repmat(
            np.sin(x / c * ncycles * 2 * np.pi), z.shape[0], 1)

    elif t == 'siny':
        if len(args) > 0:
            ncycles = args[0]
        else:
            ncycles = 1
        c = z.shape[0] / ncycles
        y = np.arange(0, z.shape[0]-1)
        z = np.matlib.repmat(np.sin(y/c * ncycles*2*np.pi), 1, z.shape[0])

    elif t == 'rampx':
        if len(args) > 0:
            ncycles = args[0]
        else:
            ncycles = 1
        c = z.shape[1] / ncycles
        x = np.arange(0, z.shape[1]-1)
        z = np.matlib.repmat(np.mod(x, c) / (c-1), z.shape[0], 1)

    elif t == 'rampy':
        if len(args) > 0:
            ncycles = args[0]
        else:
            ncycles = 1
        c = z.shape[0] / ncycles
        y = np.arange(0, z.shape[0]-1)
        z = np.matlib.repmat(np.mod(y, c) / (c-1), 1, z.shape[1])

    elif t == 'line':
        nr = z.shape[0]
        nc = z.shape[1]
        theta = args[0]
        c = args[1]

        if np.abs(np.tan(theta)) < 1:
            x = np.arange(0, nc-1)
            y = np.round(x * np.tan(theta) + c)
            # TODO warning: np.where might return a tuple
            s = np.where((y >= 1) and (y <= nr))

        else:
            y = np.arange(0, nr-1)
            x = np.round((y-c) / np.tan(theta))
            # note: be careful about 1 vs 0, python vs matlab indexing
            s = np.where((x >= 1) and (x <= nc))

        for k in s:
            z[y[k], x[k]] = 1

    elif t == 'squares':
        nr = z.shape[0]
        nc = z.shape[1]
        pitch = args[0]
        d = args[1]
        if d > (pitch/2):
            print('warning: squares will overlap')
        rad = np.floor(d/2)
        d = 2.0 * rad
        for r in range(pitch/2.0, (nr - pitch/2.0), pitch):
            for c in range(pitch/2, (nc-pitch/2), pitch):
                z[r-rad:r+rad, c-rad:c+rad] = np.ones(d+1)

    elif t == 'dots':
        nr = z.shape[0]
        nc = z.shape[1]
        pitch = args[0]
        d = args[1]
        if d > (pitch/2.0):
            print('warning: dots will overlap')

        rad = np.floor(d/2.0)
        d = 2.0*rad
        s = kcircle(d/2.0)
        for r in range(pitch/2, (nr-pitch/2), pitch):
            for c in range(pitch/2, (nc - pitch/2), pitch):
                z[r-rad:r+rad, c-rad:c+rad] = s

    else:
        raise ValueError(t, 'unknown pattern type')
        z = []

    return z


def scale(im, sfactor, outsize=None, sigma=None):
    """
    Scale an image

    :param im: image
    :type im: numpy array
    :param sfactor: scale factor
    :type sfactor: scalar
    :param outsize: output image size (w, h)
    :type outsize: 2-element vector, integers
    :param sigma: standard deviation of kernel for image smoothing
    :type sigma: float
    :return out: smoothed image
    :rtype out: numpy array

    ``iscale(im, sfactor)`` is a version of ``im`` scaled in both directions
    by ``sfactor`` which is a real scalar. ``sfactor> 1`` makes the image
    larger, ``sfactor < 1`` makes it smaller.

    :options:

        - 'outsize',S     set size of OUT to HxW where S=[W,H]
        - 'smooth',S      initially smooth image with Gaussian of standard deviation
          S (default 1).  S=[] for no smoothing.
    """
    # check inputs
    im = getimage(im)
    if not argcheck.isscalar(sfactor):
        raise TypeError(sfactor, 'factor is not a scalar')

    if np.issubdtype(im.dtype, np.float):
        is_int = False
    else:
        is_int = True
        im = idouble(im)

    # smooth image to prevent aliasing  - TODO should depend on scale factor
    if sigma is not None:
        im = smooth(im, sigma)

    nr = im.shape[0]
    nc = im.shape[1]

    # output image size is determined by input size and scale factor
    # else from specified size
    if outsize is not None:
        nrs = np.floor(nr * sfactor)
        ncs = np.floor(nc * sfactor)
    else:
        nrs = outsize[0]
        ncs = outsize[1]

    # create the coordinate matrices for warping
    U, V = imeshgrid(im)
    U0, V0 = imeshgrid([ncs, nrs])

    U0 = U0 / sfactor
    V0 = V0 / sfactor

    if im.ndims > 2:
        out = np.zeros((ncs, nrs, im.shape[2]))
        for k in range(im.shape[2]):
            out[:, :, k] = sp.interpolate.interp2d(
                U, V, im[:, :, k], U0, V0, kind='linear')
    else:
        out = sp.interpolate.interp2d(U, V, im, U0, V0, kind='linear')

    if is_int:
        out = iint(out)

    return out


def rotate(im, angle, crop=False, sc=1.0, extrapval=0, sm=None, outsize=None):
    """
    Rotate an image

    :param im: image
    :type im: numpy array
    :param angle: rotatation angle [radians]
    :type angle: scalar
    :param crop: output image size (w, h)
    :type crop: 2-element vector, integers
    :param sc: scale factor
    :type sc: float
    :param extrapval: background value of pixels
    :type extrapval: float
    :param sm: smooth (standard deviation of Gaussian kernel, typically sigma)
    :type sm: float
    :param outsize: output image size (w, h)
    :type outsize: 2-element vector, integers
    :return out: rotated image
    :rtype out: numpy array

    ``rotate(im, angle)`` is a version of the image ``im`` that has been
    rotated about its centre by angle ``angle``.

    :options:

        - 'outsize',S     set size of output image to HxW where S=[W,H]
        - 'crop'          return central part of image, same size as IM
        - 'scale',S       scale the image size by S (default 1)
        - 'extrapval',V   set background pixels to V (default 0)
        - 'smooth',S      initially smooth the image with a Gaussian of standard
          deviation S

    .. note::

        - Rotation is defined with respect to a z-axis which is into the image.
        - Counter-clockwise is a positive angle.
        - The pixels in the corners of the resulting image will be undefined and
          set to the 'extrapval'.

    """
    # TODO note that there is cv.getRotationMatrix2D and cv.warpAffine
    # https://appdividend.com/2020/09/24/how-to-rotate-an-image-in-python-
    # using-opencv/

    im = getimage(im)
    if not argcheck.isscalar(angle):
        raise ValueError(angle, 'angle is not a valid scalar')

    # TODO check optional inputs

    if np.issubdtype(im.dtype, np.float):
        is_int = False
    else:
        is_int = True
        im = idouble(im)

    if sm is not None:
        im = smooth(im, sm)

    if outsize is not None:
        # output image is determined by input size
        U0, V0 = np.meshgrid(
            np.arange(0, outsize[0]), np.arange(0, outsize[1]))
        # U0, V0 = meshgrid(0:outsize[0],0:outsize[1])
    else:
        outsize = np.array([im.shape[0], im.shape[1]])
        U0, V0 = imeshgrid(im)

    nr = im.shape[0]
    nc = im.shape[1]

    # creqate coordinate matrices for warping
    Ui, Vi = imeshgrid(im)

    # rotation and scale
    R = cv.getRotationMatrix2D(center=(0, 0), angle=angle, scale=sc)
    uc = nc/2.0
    vc = nr / 2.0
    U02 = 1.0/sc * (R[0, 0] * (U0 - uc) + R[1, 0] * (V0 - vc)) + uc
    V02 = 1.0/sc * (R[0, 1] * (U0-uc) + R[1, 1] * (V0-vc)) + vc

    if crop:
        trimx = np.abs(nr/2.0*np.sin(angle))
        trimy = np.abs(nc/2.0*np.sin(angle))
        if sc < 1:
            trimx = trimx + nc/2.0*(1.0-sc)
            trimy = trimy + nr/2.0*(1.0-sc)
        trimx = np.ceil(trimx)  # +1
        trimy = np.ceil(trimy)  # +1
        U0 = U02[trimy:U02.shape[1]-trimy,
                 trimx:U02.shape[0]-trimx]  # TODO check indices
        V0 = V02[trimy:V02.shape[1]-trimy, trimx:V02.shape[0]-trimx]

    if im.ndims > 2:
        out = np.zeros((outsize[0], outsize[1], im.shape[2]))
        for k in range(im.shape[2]):
            out[:, :, k] = sp.interpolate.interp2(
                Ui, Vi, im[:, :, k], U02, V02, kind='linear')  # TODO extrapval?
    else:
        out = sp.interpolate.interp2(Ui, Vi, im, U02, V02, kind='linear')

    if is_int:
        out = iint(out)

    return out


def samesize(im1, im2, bias=0.5):
    """
    Automatic image trimming

    :param im: image
    :type im: numpy array
    :param im1: image 1
    :type im1: numpy array
    :param bias: bias that controls what part of the image is cropped
    :type bias: float
    :return out: trimmed image
    :rtype out: numpy array

    ``samesize(im1, im2)`` is an image derived from ``im1`` that has
    the same dimensions as ``im2``.  This is achieved by cropping and scaling.

    ``samesize(im1, im2, bias)`` as above but ``bias`` controls which part
    of the image is cropped.  ``bias`` = 0.5 is symmetric cropping,
    ``bias`` < 0.5 moves the crop window up or to the left,
    while ``bias``>0.5 moves the crop window down or to the right.

    """
    # check inputs
    if bias < 0 or bias > 1:
        raise ValueError(bias, 'bias must be in range [0, 1]')

    im1 = getimage(im1)
    im2 = getimage(im2)

    sc = im2.shape / im1.shape
    out = scale(im1, sc.max())

    if out.shape[0] > im2.shape[1]:  # rows then columns
        # scaled image is too high, so trim rows
        d = out.shape[0] - im2.shape[0]
        d1 = np.max(1, np.floor(d * bias))
        d2 = d - d1
        # [1 d d1 d2]
        im2 = out[d1:-1-d2-1, :, :]  # TODO check indexing
    if out.shape[1] > im2.shape[1]:
        # scaled image is too wide, so trim columns
        d = out.shape[1] - im2.shape[1]
        d1 = np.max(1, np.floor(d*bias))
        d2 = d - d1
        # [2 d d1 d2]
        out = out[:, d1:-1-d2-1, :]  # TODO check indexing
    return out


def paste(canvas, pattern, pt, opt='centre', centre=False, zero=True,
          mode='set'):
    """
    Paste an image into an image

    :param canvas: base image
    :type canvas: numpy array
    :param pattern: sub-image super-imposed onto onto canvas
    :type pattern: numpy array
    :param pt: coordinates where pattern is pasted
    :type pt: 2-element vector of integer coordinates
    :param opt: options for paste settings
    :type opt: string
    :param centre: True if pattern is centered at pt, else topleft of pattern
    :type centre: boolean
    :param zero: zero-based coordinates (True) or 1-based coordinates
    :type zero: boolean
    :return out: pasted image
    :rtype out: numpy array

    ``paste(canvas, pattern, pt)`` is the image ``canvas`` with the subimage
    ``pattern`` pasted in at the position ``pt=[U, V]``.

    :options:

        - 'centre'    The pasted image is centred at ``pt``, otherwise ``pt`` is
          the top-left corner of the subimage in ``canvas`` (default)
        - 'set'       ``pattern`` overwrites the pixels in ``canvas`` (default)
        - 'add'       ``pattern`` is added to the pixels in ``canvas``
        - 'mean'      ``pattern`` is set to the mean of pixel values in ``canvas``
          and ``pattern``
        - 'zero'      the coordinates of ``pt`` start at zero, by default 1
          is assumed

    .. note::

        - Pixels outside the pasted in region are unaffected.
    """

    # check inputs
    canvas = getimage(canvas)
    pattern = getimage(pattern)
    pt = argcheck.getvector(pt)

    # TODO check optional inputs valid

    cw = canvas.shape[0]
    ch = canvas.shape[1]
    pw = pattern.shape[0]
    ph = pattern.shape[1]

    pasteOpt = ['set', 'add', 'mean']
    if opt not in pasteOpt:
        raise ValueError(opt, 'opt is not a valid option for paste()')

    if centre:
        left = pt[0] - np.floor(pw/2)
        top = pt[1] - np.floor(ph/2)
    else:
        left = pt[0]  # x
        top = pt[1]  # y

    if not zero:
        left += 1
        top += 1

    if (top+ph-1) > ch:
        raise ValueError(ph, 'pattern falls off bottom edge')
    if (left+pw-1) > cw:
        raise ValueError(pw, 'pattern falls off right edge')

    if pattern.ndims > 2:
        np = pattern.shape[2]
    else:
        np = 1

    if canvas.ndims > 2:
        nc = canvas.shape[2]
    else:
        nc = 1

    if np > nc:
        # pattern has multiple planes, replicate the canvas
        out = np.matlib.repmat(canvas, [1, 1, np])
    else:
        out = canvas

    if np < nc:
        pattern = np.matlib.repmat(pattern, [1, 1, nc])

    if opt == 'set':
        out[top:top+ph-1, left:left+pw-1, :] = pattern
    elif opt == 'add':
        out[top:top+ph-1, left:left+pw-1, :] = out[top:top +
                                                   ph-1, left:left+pw-1, :] + pattern

    elif opt == 'mean':
        old = out[top:top+ph-1, left:left+pw-1, :]
        # TODO check no nans in pattern
        k = ~np.isnan(pattern)  # TODO not sure if this works as intended
        old[k] = 0.5 * (old[k] + pattern[k])
        out[top:top+ph-1, left:left+pw-1, :] = old
    else:
        raise ValueError(opt, 'opt is not valid')

    return out


def peak2(z, npeaks=2, sc=1, interp=False):
    """
    Find peaks in a matrix

    :param z: 2-dimensional signal
    :type z: numpy array
    :param npeaks: number of peaks to return (default all)
    :type npeaks: scalar
    :param sc: scale of peaks to consider
    :type sc: float
    :param interp:  interpolation done on peaks
    :type interp: boolean
    :return: peaks, xy locations, ap? TODO
    :rtype: collections.namedtuple

    ``peak2(z)`` are the peak values in the 2-dimensional signal ``z``

    ``peak2(z, options)`` as above but also returns the indices of the
    maxima in the matrix Z.  Use SUB2IND to convert these to row and column
    coordinates

    :options:

         - 'npeaks',N    Number of peaks to return (default all)
         - 'scale',S     Only consider as peaks the largest value in the horizontal
           and vertical range +/- S points.
         - 'interp'      Interpolate peak (default no interpolation)
         - 'plot'        Display the interpolation polynomial overlaid on the point data

    .. note::

        - A maxima is defined as an element that larger than its eight neighbours.
          Edges elements will never be returned as maxima.
        - To find minima, use PEAK2(-V).
        - The interp options fits points in the neighbourhood about the peak with
          a paraboloid and its peak position is returned.  In this case IJ will
          be non-integer.

    """

    # TODO check valid input

    # create a neighbourhood mask for non-local maxima suppression
    h = sc
    w = 2*h
    M = np.ones((w, w))
    M[h, h] = 0

    # compute the neighbourhood maximum
    znh = window(idouble(z), M, 'max', 'wrap')  # TODO make sure this works

    # find all pixels greater than their neighbourhood
    k = np.where(z > znh)

    # sort these local maxima into descending order
    # [zpk,ks] = sort(z(k), 'descend');
    # k = k(ks);
    ks = [np.argsort(z, axis=0)][:, :, -1]  # TODO check this
    k = k[ks]

    npks = np.min(np.length(k), npeaks)
    k = k[0:npks]

    # TODO use unravel_index and/or ravel_multi_index function to replace ind2sub/sub2ind
    # note that Matlab is column major, while Python/numpy is row major?
    y, x = np.unravel_index(k, z.shape)
    xy = np.stack((y, x), axis=2)

    # interpolate peaks if required
    if interp:
        # TODO see peak2.m, line 87-131
        print('TODO')
    else:
        xyp = xy
        zp = z(k)
        ap = []

    return namedtuple('peaks', 'xy' 'z' 'a')(xyp, zp, ap)


def roi(im, reg=None, wh=None):
    """
    Extract region of interest

    :param im: image
    :type im: numpy array
    :param reg: region
    :type reg: numpy array
    :param wh: width and/or height
    :type wh: 2-element vector of integers, or single integer
    :return: roi
    :rtype: numpy array

    ``iroi(im, rect)`` is a subimage of the image ``im`` described by the
    rectangle ``rect=[umin, umax; vmin, vmax]``. The function returns the
    top, left, bottom and top coordinates of the selected region of
    interest, as vectors.

    ``iroi(im, reg, wh)`` as above but the region is centered at ``reg=(U,V)``
    and has a size ``wh``.  If ``wh`` is scalar then ``W=H=S``
    otherwise ``S=(W,H)``.
    """

    im = getimage(im)
    if reg is not None and wh is not None:
        reg = getimage(reg)  # 2x2?
        wh = argcheck.getvector(wh)

        xc = reg[0]
        yc = reg[1]
        if len(wh) == 1:
            w = np.round(wh/2)
            h = w
        else:
            w = np.round(wh[0]/2)
            h = np.round(wh[1]/2)
        left = xc - w
        right = xc + w
        top = yc - h
        bot = hc + h

    elif reg is not None and wh is None:
        reg = getimage(reg)

        left = reg[0, 0]
        right = reg[0, 1]
        top = reg[1, 0]
        bot = reg[1, 1]

    else:
        raise ValueError(reg, 'reg cannot be None')

    # TODO check row/column ordering, and ndim check
    roi = im[top:bot, left:right, :]

    return namedtuple('roi', 'roi' 'left' 'right' 'top' 'bot') (roi, left,
                right, top, bot)


def pixelswitch(mask, im1, im2):
    """
    Pixel-wise image merge

    :param mask: image mask
    :type mask: numpy array
    :param im1: image 1
    :type im1: numpy array
    :param im2: image 2
    :type im2: numpy array
    :return: out
    :rtype: numpy array

    ``pixelswitch(mask, im1, im2)`` is an image where each pixel is
    selected from the corresponding pixel in ``im1`` or ``im2`` according to the
    corresponding pixel values in ``mask``.  If the element of ``mask``
    is zero ``im1`` is selected, otherwise ``im2`` is selected.

    ``im1`` or ``im2`` can contain a color descriptor which is one of:
    - A scalar value corresponding to a greyscale
    - A 3-vector corresponding to a color value
    - A string containing the name of a color which is found using COLORNAME.

    Example::

     Read a uint8 image
            im = iread('lena.pgm');
     and set high valued pixels to red
            a = ipixswitch(im>120, im, uint8([255 0 0]));
     The result is a uint8 image since both arguments are uint8 images.

            a = ipixswitch(im>120, im, [1 0 0]);
     The result is a double precision image since the color specification
     is a double.

            a = ipixswitch(im>120, im, 'red');
     The result is a double precision image since the result of colorname
     is a double precision 3-vector.

    .. note::

        - ``im1``, ``im2`` and ``mask`` must all have the same number of
          rows and columns (unless ``im1`` or ``im2`` are specifying a color)
        - If ``im1`` and ``im2`` are both greyscale then ``out`` is greyscale.
        - If either of ``im1`` or ``im2`` are color then ``out`` is color.
        - If either one image is double and one is integer then the integer
          image is first converted to a double image.
    """

    # TODO add possibility for alpha layering?
    # TODO might be able to replace all this with np.where()

    im1 = _checkimage(im1, mask)
    im2 = _checkimage(im2, mask)

    if np.issubdtype(im1, np.float) and np.issubdtype(im2, np.integer):
        im2 = idouble(im2)
    elif np.issubdtype(im1, np.integer) and np.issubdtype(im2, np.float):
        im1 = idouble(im1)

    if im1.ndims > 2:
        np1 = im1.shape[2]
    else:
        np1 = 1
    if im2.ndims > 2:
        np2 = im2.shape[2]
    else:
        np2 = 1

    nplanes = np.max(np1, np2)

    if nplanes == 3:
        if np1 == 1:
            im1 = np.matlib.repmat(im1, [1, 1, 3])  # TODO check if this works
        if np2 == 1:
            im2 = np.matlib.repmat(im2, [1, 1, 3])

    # in case one of the images contains NaNs, we can't blend the images using arithmetic
    # out = mask * im1 + (1 - mask) * im2

    out = im2
    mask = np.bool(mask)
    mask = np.matlib.repmat(mask, [1, 1, nplanes])
    out[mask] = im1[mask]

    return out


def _checkimage(im, mask):
    """
    Check image and mask for pixelswitch

    :param im: image, possibly a color vector or identifier
    :type im: numpy array or scalar or string
    :param mask: mask
    :type mask: numpy array
    :return: out
    :rtype: numpy array

    ``_checkimage(im, mask)`` is an image the same shape as ``mask``, and might
    be an image of all one color, depending on the value of ``im``

    """
    if isinstance(im, str):
        # image is a string color name
        col = mvt.colorname(im)
        if col is []:
            raise ValueError(im, 'unknown color')
        out = mvt.color(np.ones(mask.shape), col)
    elif argcheck.isscalar(im):
        # image is a  scalar, create a greyscale image the same size as mask
        # TODO not certain if im.dtype works if im is scalar
        out = np.ones(mask.shape, dtype=im.dtype) * im
    elif im.ndims == 2 and (im.shape == (1, 3) or im.shape == (3, 1) or
             im.shape == (3,)):
        # image is a (1,3), create a color image the same size as mask
        out = mvt.color(np.ones(mask.shape, dtype=im.dtype), im)
    else:
        # actual image, check the dimensions
        if not np.any(im.shape == mask.shape):
            raise ValueError(
                im, 'input image sizes (im or mask) do not conform')

    return out


def label(im, conn=8, ltype='int32', ccalgtype=cv.CCL_DEFAULT):
    """
    Label an image

    :param im: binary image
    :type im: numpy array
    :param conn: connectivity, 4 or 8
    :type conn: integer
    :param ltype: output image type
    :type ltype: string
    :param ccalgtype: specified for connected component algorithm (from OpenCV)
    :type ccalgtype: int
    :return: n_components, labels
    :rtype: int, numpy array

    ``label(im)`` is a label image that indicates connected components within
    the image ``im`` (H,W).  Each pixel in ``labels`` (H,W) is an integer
    label that indicates which connected region the corresponding pixel in
    ``im`` belongs to.  Region labels are in the range 1 to ``n_components``.

    .. note::

        - This algorithm is variously known as region labelling, connectivity
          analysis, connected component analysis, blob labelling.
        - All pixels within a region have the same value (or class).
        - This is a "low level" function, IBLOBS is a higher level interface.
        - The image can be binary or greyscale.
        - Connectivity is only performed in 2 dimensions.
        - Connectivity is performed using 8 nearest neighbours by default.
        - To use 8-way connectivity pass a second argument of 8, eg. ILABEL(IM, 8).
        - 8-way connectivity introduces ambiguities, a chequerboard is two blobs.
    """

    # check valid input:
    im = getimage(im)
    im = mono(im)  # image must be uint8

    # TODO input image must be 8-bit single-channel image
    if im.ndim > 2:
        raise ValueError(im, 'image must be single channel')

    if not (conn in [4, 8]):
        raise ValueError(conn, 'connectivity must be 4 or 8')

    # make labels uint32s, unique and never recycled?
    # set ltype to default to cv.CV_32S
    if ltype == 'int32':
        ltype = cv.CV_32S
        dtype = np.int32
    elif ltype == 'uint16':
        ltype = cv.CV_16U
        dtpye = np.uint16
    else:
        raise ValueError(ltype, 'ltype must be either int32 or uint16')

    labels = np.zeros((im.shape[0], im.shape[1]), dtype=dtype)

    n_components, labels = cv.connectedComponents(im,
                                            labels,
                                            connectivity=conn,
                                            ltype=ltype)

    # print(retval)  # retval not documented, but likely number of labels
    # TODO cv.connectedComponents sees 0 background as one component, unfortunately
    # this differs from ilabel, which sees different blobs as diferent components

    # TODO possible solution: edge detection (eg Canny/findCOntours) on the binary imaage
    # then invert (bitwise negation) the edge image
    # (or do find contours and invert the countour image)
    # limited to connectivity of 4, since Canny is 8-connected though!
    # Could dilate edge image to accommodate 8-connectivity, but feels like a hack

    # TODO or, could just follow ilabel.m

    # consider scipy.ndimage.label
    #from scipy.ndimage import label, generate_binary_structure
    # s = generate_binary_structure(2,2) # 8-way connectivity
    #labels, n_components = label(im, structure=s)
    # TODO consider cv.connectedComponentsWithStats()
    # TODO consider cv.connectedComponentsWithAlgorithms()

    return n_components, labels


# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':

    # testing idisp:
    # im_name = 'longquechen-moon.png'
    im_name = 'multiblobs.png'
    im = iread((Path('images') / im_name).as_posix())
    # im = iread((Path('images') / 'test' / im_name).as_posix())
    imo = mono(im)

    retval, labels = label(imo, ccalgtype=cv.CCL_GRANA)

    plt.imshow(labels)
    plt.show()

    # for debugging interactively
    #import code
    #code.interact(local=dict(globals(), **locals()))

    # show original image
    # idisp(im, title='space rover 2020')

    # do canny:
    #imcan = canny(im, sigma=3, th0=50, th1=100)

    # idisp(imcan, title='canny')

    #K = kgauss(sigma=1)
    #ic = convolve(im, K, optmode='same', optboundary='wrap')

    #import code
    #code.interact(local=dict(globals(), **locals()))

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
    #im5 = erode(im, se=np.ones((3, 3)), opt='wrap')
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
