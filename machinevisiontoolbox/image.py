#!/usr/bin/env python

import io as io
import numpy as np
import spatialmath.base.argcheck as argcheck
import cv2 as cv
import matplotlib.path as mpath
import sys as sys
import machinevisiontoolbox.color
import time
import scipy as sp

# code.interact(local=dict(globals(), **locals()))

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

    cv.namedWindow(opt['title'], cv.WINDOW_AUTOSIZE)
    cv.imshow(opt['title'], im)  # make sure BGR format image
    k = cv.waitKey(delay=0)  # non blocking, by default False
    # cv.destroyAllWindows()

    # if ESC pressed, close the window, otherwise it persists until program exits
    if k == 27:
        # only destroy the specific window
        cv.destroyWindow(opt['title'])


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
        # TODO check ValueError
        raise ValueError('Could not read the image specified by file".')

    # TODO check for wild cards
    # TODO search paths automatically for specified file?
    # TODO fetch from server

    return img


def isimage(im):
    """
    Test if input is an image, return a consistent data type/format
    Can we convert any format to BGR 32f? How would we know format in is RGB vs
    BGR?
    ('im is not of type int or float')
    ('im has ndims < 2')
    ('im is (H,W), but does not have enough columns or rows to be an image')
    ('im (H,W,N), but does not have enough N to be either a color image (N=3), or a sequence of monochrome images (N > 1)')
    ('im (H,W,M,N) should be a sequence of color images, but M is not equal to 3')
    """
    # convert im to nd.array
    im = np.array(im)

    # TODO consider complex floats?
    # check if image is int or floats
    # TODO shouldn't np.integer and np.float be the catch-all types?
    if not (np.issubdtype(im.dtype, np.integer) or
            np.issubdtype(im.dtype, np.float) or
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
    :type im: numpy array (N,H,3)
    :param intclass: either 'uint8', ... TODO
    :type intclass: string
    :return out: image with unsigned 8-bit integer elements ranging 0 to 255
    corresponding to the elements of the image ``im``
    :rtype: numpy array (N,H,3)

    ``iint(im)`` is an image with unsigned 8-bit integer elements in the range 0
    to 255 corresponding to the elements of the image ``im``.

    ``int(im, intclass) as above but the output pixels belong to the integer
    class ``intclass``.

    Options::

    # TODO

    :notes:
    - Works for an image with arbitrary number of dimensions, eg. a color
      image or image sequence.
    - If the input image is floating point (single or double) the pixel values
      are scaled from an input range of [0,1] to a range spanning zero to the
      maximum positive value of the output integer class.
    - If the input image is an integer class then the pixels are cast to
      change type but not their value.

    Example::

        #TODO

    References:

        - Robotics, Vision & Control, Section 12.1, P. Corke, Springer 2011.
    """

    if not isimage(im):
        raise TypeError(im, 'im is not a valid image')

    if np.issubdtype(im.dtype, np.float):
        # rescale to integer
        return (np.rint(im * np.float64(np.iinfo(intclass).max))).astype(intclass)
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

    Options::
    'single'        return an array of single precision floats instead of
                    doubles
    'float'         as above


    Example::

        #TODO

    References:

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
    input an image, converts image into types compatible opencv:
    CV_8U, CV_16U, CV_16S, CV_32F or CV_64F
    default: if int then CV_8U, if float then CV_64F
    TODO check if there is a different type for opencv binary images (probably
    just CV_8U)
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

    return im


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
            print('in grey_601')
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


def icolor(im, c=[1, 1, 1]):
    """
    Colorise a greyscale image

    :param im: image
    :type im: numpy array (N,H)
    :param c: color to color image
    :type c: string or rgb-tuple
    :return out: image with float64 precision elements ranging from 0 to 1
    :rtype: numpy array (N,H,3)

    ``icolor(im)`` is a color image out ``c`` (N,H,3), where each color
    plane is equal to im.

    ``imcolor(im, c)`` as above but each output pixel is ``c``(3,1) times
    the corresponding element of ``im``.


    Example::

        #TODO

    :notes:

    - Can convert a monochrome sequence (h,W,N) to a color image sequence
      (H,W,3,N)

    References:

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


def istretch(im, max=1, range=None):
    """
    Image normalisation

    :param im: image
    :type im: numpy array (N,H,3)
    :param max: M   TODO  pixels are mapped to the range 0 to M
    :type max: scalar integer or float
    :param range: range R(1) is mapped to zero, R(2) is mapped to 1 (or max
    value)
    :type range: 2-tuple or numpy array (2,1)
    :return out: image
    :rtype: numpy array (N,H,3), type double

    ``istretch(im)`` is a normalised image in which all pixel values lie in the
    range of 0 to 1. That is, a linear mapping where the minimum value of ``im``
    is mapped to 0 and the maximum value of ``im`` is mapped to 1.

    :notes:
    - For an integer image the result is a double image in the range 0 to max value

    Example::

        #TODO

    References:

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


def ierode(im, se, n=1, opt='border', **kwargs):
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

    ``ierode(im, se, opt)`` is the image ``im`` after morphological erosion with
    structuring element ``se``.

    ``ierode(im, se, n, opt)`` as above, but the structruring element ``se`` is
    applied ``n`` times, that is ``n`` erosions.

    Options::
    'border'    the border value is replicated (default)
    'none'      pixels beyond the border are not included in the window
    'trim'      output is not computed for pixels where the structuring element
                crosses the image border, hence output image has reduced
                dimensions TODO
    'wrap'      the image is assumed to wrap around, left to right, top to
                bottom
    TODO other border options from opencv


    :notes:
    - Cheaper to apply a smaller structuring element multiple times than
      one large one, the effective structuing element is the Minkowski sum
      of the structuring element with itself N times.

    Example::

        #TODO

    References:

        - Robotics, Vision & Control, Section 12.5, P. Corke, Springer 2011.
    """

    # check if valid input:
    im = getimage(im)

    if not isimage(se):
        raise TypeError(se, 'se is not a valid image')
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
        'border': cv.BORDER_REPLICATE,
        'none': cv.BORDER_ISOLATED,
        'wrap': cv.BORDER_WRAP
    }

    if opt not in cvopt.keys():
        raise ValueError(opt, 'opt is not a valid option')

    # se = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    return cv.erode(im, se, iterations=n, borderType=cvopt[opt])


def idilate(im, se, n=1, opt='border', **kwargs):
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

    ``idilate(im, se, opt)`` is the image ``im`` after morphological dilation with
    structuring element ``se``.

    ``idilate(im, se, n, opt)`` as above, but the structruring element ``se`` is
    applied ``n`` times, that is ``n`` dilations.

    Options::
    'border'    the border value is replicated (default)
    'none'      pixels beyond the border are not included in the window
    'trim'      output is not computed for pixels where the structuring element
                crosses the image border, hence output image has reduced
                dimensions TODO
    'wrap'      the image is assumed to wrap around, left to right, top to
                bottom
    TODO other border options from opencv


    :notes:
    - Cheaper to apply a smaller structuring element multiple times than
      one large one, the effective structuing element is the Minkowski sum
      of the structuring element with itself N times.

    Example::

        #TODO

    References:

        - Robotics, Vision & Control, Section 12.5, P. Corke, Springer 2011.
    """

    # check if valid input:
    im = getimage(im)

    # TODO check if se is valid (odd number and less than im.shape)
    if not isimage(se):
        raise TypeError(se, 'se is not a valid image')

    if not isinstance(n, int):
        n = int(n)
    if n <= 0:
        raise ValueError(n, 'n must be greater than 0')

    if not isinstance(opt, str):
        raise TypeError(opt, 'opt must be a string')

    # convert options TODO trim?
    cvopt = {
        'border': cv.BORDER_REPLICATE,
        'none': cv.BORDER_ISOLATED,
        'wrap': cv.BORDER_WRAP
    }

    if opt not in cvopt.keys():
        raise ValueError(opt, 'opt is not a valid option')

    # se = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    return cv.dilate(im, se, iterations=n, borderType=cvopt[opt])


def imorph(im, se, oper, n=1, opt='border', **kwargs):
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

    ``imorph(im, se, opt)`` is the image ``im`` after morphological operation with
    structuring element ``se``.

    ``imorph(im, se, n, opt)`` as above, but the structruring element ``se`` is
    applied ``n`` times, that is ``n`` morphological operations.

    The operation ``oper`` is:
    'min'       minimum value over the structuring element
    'max'       maximum value over the structuring element
    'diff'      maximum - minimum value over the structuring element (this is morph_gradient)
    'plusmin'   the minimum of the pixel value and the pixelwise sum of the ()
                structuring element and source neighbourhood. :TODO:

    TODO can we call this border options?
    Options::
    'border'    the border value is replicated (default)
    'none'      pixels beyond the border are not included in the window
    'trim'      output is not computed for pixels where the structuring element
                crosses the image border, hence output image has reduced
                dimensions TODO
    'wrap'      the image is assumed to wrap around, left to right, top to
                bottom
    TODO other border options from opencv


    :notes:
    - Cheaper to apply a smaller structuring element multiple times than
      one large one, the effective structuing element is the Minkowski sum
      of the structuring element with itself N times.
    - Performs greyscale morphology
    - The structuring element shoul dhave an odd side length
    - For binary image, ``min`` = erosion, ``max``= dilation
    - The ``plusmin`` operation can be used to compute the distance transform.
    - The input can be logical, uint8, uint16, float or double.
    - The output is always double

    Example::

        #TODO

    References:

        - Robotics, Vision & Control, Section 12.5, P. Corke, Springer 2011.
    """

    # check if valid input:
    im = getimage(im)

    # TODO check if se is valid (odd number and less than im.shape)
    if not isimage(se):
        raise TypeError(se, 'se is not a valid image')

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
        'border': cv.BORDER_REPLICATE,
        'none': cv.BORDER_ISOLATED,
        'wrap': cv.BORDER_WRAP
    }

    if opt not in cvopt.keys():
        raise ValueError(opt, 'opt is not a valid option')

    if oper == 'min':
        out = ierode(im, se, n, cvopt[opt])
    elif oper == 'max':
        out = idilate(im, se, n, cvopt[opt])
    elif oper == 'diff':
        out = cv.morphologyEx(im, cv.MORPH_GRADIENT, se, iterations=n,
                              bordertype=cvopt[opt])
    elif oper == 'plusmin':
        out = None  # TODO
    else:
        raise ValueError(oper, 'imorph does not support oper')

    # se = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    return out


def hitormiss(im, s1=0.0, s2=0.0):
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
    (represenbtedy nans).

    Example::

        #TODO

    References:

        - Robotics, Vision & Control, Section 12.5, P. Corke, Springer 2011.
    """
    # check valid input
    im = getimage(im)

    # TODO also check if binary image?
    return imorph(im, s1, 'min') and imorph((1 - im), s2, 'min')


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

    Options::
    'border'    the border value is replicated (default)
    'none'      pixels beyond the border are not included in the window
    'trim'      output is not computed for pixels where the structuring element
                crosses the image border, hence output image has reduced
                dimensions TODO
    'wrap'      the image is assumed to wrap around, left to right, top to
                bottom
    TODO other border options from opencv


    :notes:
    - For binary image an opening operation can be used to eliminate small
      white noise regions.
    - Cheaper to apply a smaller structuring element multiple times than
      one large one, the effective structuing element is the Minkowski sum
      of the structuring element with itself N times.

    Example::

        #TODO

    References:

        - Robotics, Vision & Control, Section 12.5, P. Corke, Springer 2011.
    """

    a = ierode(im, se, **kwargs)
    return idilate(a, se, **kwargs)


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

    Options::
    'border'    the border value is replicated (default)
    'none'      pixels beyond the border are not included in the window
    'trim'      output is not computed for pixels where the structuring element
                crosses the image border, hence output image has reduced
                dimensions TODO
    'wrap'      the image is assumed to wrap around, left to right, top to
                bottom
    TODO other border options from opencv


    :notes:
    - For binary image an opening operation can be used to eliminate small
      white noise regions.
    - Cheaper to apply a smaller structuring element multiple times than
      one large one, the effective structuing element is the Minkowski sum
      of the structuring element with itself N times.

    Example::

        #TODO

    References:

        - Robotics, Vision & Control, Section 12.5, P. Corke, Springer 2011.
    """

    a = idilate(im, se, **kwargs)
    return idilate(a, se, **kwargs)


def ithin(im, delay=0.0):
    """
    Morphological skeletonization

    :param im: image
    :type im: numpy array (N,H,3) or (N,H)
    :param delay: seconds between each iteration of display
    :type delay: float
    :return out: image
    :rtype: numpy array (N,H,3) or (N,H)

    ``ithin(im, delay)`` as above but graphically displays each iteration
    of the skeletonization algorithm with a pause of ``delay`` seconds between
    each iteration.

    Example::

        #TODO

    References:

        - Robotics, Vision & Control, Section 12.5, P. Corke, Springer 2011.
    """

    # ensure valid input
    im = getimage(im)
    # TODO make sure delay is a float > 0

    # create a binary image (True/False)
    im = im > 0

    # create structuring elements
    sa = np.array([[0, 0, 0],
                   [np.nan, 1, np.nan],
                   [1, 1, 1]])
    sb = np.array([np.nan, 0, 0],
                  [1, 1, 0],
                  [np.nan, 1, np.nan])

    # loop
    out = im
    while True:
        for i in range(4):
            r = hitormiss(im, sa)
            im = im - r
            r = hitormiss(im, sb)
            im = im - r
            sa = np.rot90(sa)
            sb = np.rot90(sb)
        if delay > 0.0:
            idisp(im)
            time.sleep(5)  # TODO work delay into waitKey as optional input!
        if all(out == im):
            break
        out = im

    return out


def ismooth(im, sigma, w=None, opt='full'):
    """
     OUT = ISMOOTH(IM, SIGMA) is the image IM after convolution with a
     Gaussian kernel of standard deviation SIGMA.

     OUT = ISMOOTH(IM, SIGMA, OPTIONS) as above but the OPTIONS are passed
     to CONV2.

     Options::
     'full'    returns the full 2-D convolution (default)
     'same'    returns OUT the same size as IM
     'valid'   returns  the valid pixels only, those where the kernel does not
               exceed the bounds of the image.

     :notes:
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

    m = kgauss(sigma, w)

    # convolution options from ismooth.m, which relate to Matlab's conv2.m
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


def kgauss(sigma, w=None):
    """
    Gaussian kernel

    ``kgauss(sigma)`` is a 2-dimensional Gaussian kernel of standard deviation
    SIGMA, and  centred within the matrix K whose half-width is H=2xSIGMA and
    W=2xH+1.

    K = KGAUSS(SIGMA, H) as above but the half-width H is specified.

    :notes:
    - The volume under the Gaussian kernel is one.
    """

    # make sure sigma, w are valid input
    if w is None:
        w = np.ceil(3*sigma)

    wi = np.arange(-w, w+1)
    x, y = np.meshgrid(wi, wi)

    m = 1.0/(2.0 * np.pi * sigma**2) * \
        np.exp(-(np.power(x, 2) + np.power(y, 2))/2.0/sigma**2)
    # area under the curve should be 1, but the discrete case is only
    # an approximation
    return m / np.sum(m)


def klaplace():
    """
    Laplacian kernel

     K = KLAPLACE() is the Laplacian kernel:
            |0   1  0|
            |1  -4  1|
            |0   1  0| TODO bmatrix/mathmode
    :math:`\alpha`

     :notes:
     - This kernel has an isotropic response to image gradient.
    """
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])


def ksobel():
    """
    Sobel edge detector

     K = KSOBEL() is the Sobel x-derivative kernel:
             1/8  |1  0  -1|
                  |2  0  -2|
                  |1  0  -1|

     :notes:
     - This kernel is an effective vertical-edge detector
     - The y-derivative (horizontal-edge) kernel is K'
    """
    return np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])/8.0


def kdog(sigma1, sigma2=None, w=None):
    """
    Difference of Gaussians kernel

    K = KDOG(SIGMA1) is a 2-dimensional difference of Gaussian kernel equal
    to KGAUSS(SIGMA1) - KGAUSS(SIGMA2), where SIGMA1 > SIGMA2.  By default
    SIGMA2 = 1.6*SIGMA1.  The kernel is centred within the matrix K whose
    half-width H = 3xSIGMA and W=2xH+1.

    K = KDOG(SIGMA1, SIGMA2) as above but SIGMA2 is specified directly.

    K = KDOG(SIGMA1, SIGMA2, H) as above but the kernel half-width is specified.

    :notes:
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
    if w is None:
        w = np.ceil(3.0 * sigma1)

    m1 = kgauss(sigma1, w)  # thin kernel
    m2 = kgauss(sigma2, w)  # wide kernel

    return m2 - m1


def klog(sigma, w=None):
    """
    Laplacian of Gaussian kernel

     K = KLOG(SIGMA) is a 2-dimensional Laplacian of Gaussian kernel of
     width (standard deviation) SIGMA and centred within the matrix K whose
     half-width is H=3xSIGMA, and W=2xH+1.

     K = KLOG(SIGMA, H) as above but the half-width H is specified.
    """
    # TODO ensure valid input
    if w is None:
        w = np.ceil(3.0*sigma)
    wi = np.arange(-w, w+1)
    x, y = np.meshgrid(wi, wi)
    return 1.0/(np.pi*sigma**4.0) * \
        ((np.power(x, 2) + np.power(y, 2))/(2.0*sigma**2) - 1) * \
        np.exp(-(np.power(x, 2) + np.power(y, 2))/(2.0*sigma**2))


def kdgauss(sigma, w=None):
    """
    Derivative of Gaussian kernel

     K = KDGAUSS(SIGMA) is a 2-dimensional derivative of Gaussian kernel (WxW)
     of width (standard deviation) SIGMA and centred within the matrix K whose
     half-width H = 3xSIGMA and W=2xH+1.

     K = KDGAUSS(SIGMA, H) as above but the half-width is explictly specified.

     :notes:
     - This kernel is the horizontal derivative of the Gaussian, dG/dx.
     - The vertical derivative, dG/dy, is K'.
     - This kernel is an effective edge detector.
    """
    if w is None:
        w = np.ceil(3.0*sigma)

    wi = np.arange(-w, w+1)
    x, y = np.meshgrid(wi, wi)

    return -x/sigma**2/(2.0*np.pi) * np.exp(-np.power(x, 2) + np.power(y, 2) / 2.0 / sigma**2)


def kcircle(r, w=None):
    """
    Circular structuring element

     K = KCIRCLE(R) is a square matrix (WxW) where W=2R+1 of zeros with a maximal
     centred circular region of radius R pixels set to one.

     K = KCIRCLE(R,W) as above but the dimension of the kernel is explicitly
     specified.

     :notes:
     - If R is a 2-element vector the result is an annulus of ones, and
       the two numbers are interpretted as inner and outer radii.\
    """

    # check valid input:
    r = argcheck.getvector(r)
    if r.shape[1] > 1:  # TODO check if this is a good scalar check?
        rmax = r.max()
        rmin = r.min()
    else:
        rmax = r

    if w is not None:
        w = w*2.0 + 1.0
    elif w is None:
        w = 2.0*rmax + 1.0

    s = np.zeros(w, w)
    c = np.ceil(w/2.0)

    if r.shape[1] > 1:
        s = kcircle(rmax, w) - kcircle(rmin, w)
    else:
        x, y = imeshgrid(s)
        x = x - c
        y = y - c
        l = np.where(np.round(np.power(x, 2) +
                              np.power(y, 2) - np.power(r, 2) <= 0))
        s[l] = 1
    return s


def imeshgrid(a1, a2=None):
    """
    Domain matrices for image

    [U,V] = IMESHGRID(IM) are matrices that describe the domain of image IM (HxW)
    and are each HxW.  These matrices are used for the evaluation of functions
    over the image. The element U(R,C) = C and V(R,C) = R.

    [U,V] = IMESHGRID(W, H) as above but the domain is WxH.

    [U,V] = IMESHGRID(S) as above but the domain is described by S which can
    be a scalar SxS or a 2-vector S=[W, H].

    """
    # TODO check valid input, though tricky if a1 is a 2D array, I think calling
    # argcheck.getvector will flatten it...
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


def ipyramid(im, sigma=1, N=None):
    """
    Pyramidal image decomposition

    ``ipyramid(im)`` is a pyramid decomposition of input image ``im`` using
    Gaussian smoothing with standard deviation of 1.  The return is a list array of
    images each one having dimensions half that of the previous image. The
    pyramid is computed down to a non-halvable image size.

    ``ipyramid(im, sigma)`` as above but the Gaussian standard deviation
    is ``sigma``.

    ``ipyramid(im, sigma, N)`` as above but only ``N`` levels of the pyramid are
    computed.

    :notes:
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

    # TODO options to accept different border types, note that the Matlab implementation is hard-coded to 'same'

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


def sad(w1, w2):
    """
    Sum of absolute differences

    M = SAD(I1, I2) is the sum of absolute differences between the
    two equally sized image patches I1 and I2.  The result M is a scalar that
    indicates image similarity, a value of 0 indicates identical pixel patterns
    and is increasingly positive as image dissimilarity increases.

    """
    w1 = getimage(w1)
    w2 = getimage(w2)
    m = np.abs(w1 - w2)
    return np.sum(m)


def ssd(w1, w2):
    """
    Sum of squared differences

    M = SSD(I1, I2) is the sum of squared differences between the
    two equally sized image patches I1 and I2.  The result M is a scalar that
    indicates image similarity, a value of 0 indicates identical pixel patterns
    and is increasingly positive as image dissimilarity increases.
    """
    w1 = getimage(w1)
    w2 = getimage(w2)
    m = np.power((w1 - w2), 2)
    return np.sum(m)


def ncc(w1, w2):
    """
    Normalised cross correlation

    % M = NCC(I1, I2) is the normalized cross-correlation between the
    % two equally sized image patches I1 and I2.  The result M is a scalar in
    % the interval -1 (non match) to 1 (perfect match) that indicates similarity.
    %
    % Notes::
    % - A value of 1 indicates identical pixel patterns.
    % - The NCC similarity measure is invariant to scale changes in image
    %   intensity.
    """
    w1 = getimage(w1)
    w2 = getimage(w2)

    denom = np.sqrt(np.sum(np.power(w1, 2) * np.power(w2, 2)))

    if denom < 1e-10:
        return 0
    else:
        return np.sum(w1 * w2) / denom


def zsad(w1, w2):
    """
    Zero-mean sum of absolute differences

    M = ZSAD(I1, I2) is the zero-mean sum of absolute differences between the
    two equally sized image patches I1 and I2.  The result M is a scalar that
    indicates image similarity, a value of 0 indicates identical pixel patterns
    and is increasingly positive as image dissimilarity increases.

    :notes:
    - The ZSAD similarity measure is invariant to changes in image brightness
    offset.
    """

    w1 = getimage(w1)
    w2 = getimage(w2)
    w1 = w1 - np.mean(w1)
    w2 = w2 - np.mean(w2)
    m = np.abs(w1 - w2)
    return np.sum(m)


def zssd(w1, w2):
    """
    Zero-mean sum of squared differences

    M = ZSSD(I1, I2) is the zero-mean sum of squared differences between the
    two equally sized image patches I1 and I2.  The result M is a scalar that
    indicates image similarity, a value of 0 indicates identical pixel patterns
    and is increasingly positive as image dissimilarity increases.

    :notes:
    - The ZSSD similarity measure is invariant to changes in image brightness
    offset.
    """

    w1 = getimage(w1)
    w2 = getimage(w2)
    w1 = w1 - np.mean(w1)
    w2 = w2 - np.mean(w2)
    m = np.power(w1 - w2, 2)
    return np.sum(m)


def zncc(w1, w2):
    """
    Zero-mean normalized cross correlation

    M = ZNCC(I1, I2) is the zero-mean normalized cross-correlation between the
    two equally sized image patches I1 and I2.  The result M is a scalar in
    the interval -1 to 1 that indicates similarity.  A value of 1 indicates
    identical pixel patterns.

    :notes:
    - The ZNCC similarity measure is invariant to affine changes in image
    intensity (brightness offset and scale).
    """

    w1 = getimage(w1)
    w2 = getimage(w2)
    w1 = w1 - np.mean(w1)
    w2 = w2 - np.mean(w2)
    denom = np.sqrt(np.sum(np.power(w1, 2) * np.sum(np.power(w2, 2))))

    if denom < 1e-10:
        return 0
    else:
        return np.sum(w1 * w2) / denom


def ithresh(im, t=None, opt='binary'):
    """
    Image threshold

    See opencv threshold types for threshold options
    https://docs.opencv.org/4.2.0/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576

    :notes:
    - greyscale only
    - For a uint8 class image the slider range is 0 to 255.
    - For a floating point class image the slider range is 0 to 1.0
    """

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
    ret, imt = cv.threshold(im, t, maxval, threshopt[opt])

    if opt == 'otsu' or opt == 'triangle':
        return imt, ret
    else:
        return imt


def iwindow(im, se, func, opt='border', **kwargs):
    """
    Generalized spatial operator

    % OUT = IWINDOW(IM, SE, FUNC) is an image where each pixel is the result
    % of applying the function FUNC to a neighbourhood centred on the corresponding
    % pixel in IM.  The neighbourhood is defined by the size of the structuring
    % element SE which should have odd side lengths.  The elements in the
    % neighbourhood corresponding to non-zero elements in SE are packed into
    % a vector (in column order from top left) and passed to the specified
    % function handle FUNC.  The return value  becomes the corresponding pixel
    % value in OUT.
    %
    % OUT = IWINDOW(IMAGE, SE, FUNC, EDGE) as above but performance of edge
    % pixels can be controlled.  The value of EDGE is:
    % 'border'   the border value is replicated (default)
    % 'none'     pixels beyond the border are not included in the window TODO
    % 'trim'     output is not computed for pixels whose window crosses
    %            the border, hence output image had reduced dimensions. TODO
    % 'wrap'     the image is assumed to wrap around
    %
    % Example::
    % Compute the maximum value over a 5x5 window:
    %      iwindow(im, ones(5,5), @max);
    %
    % Compute the standard deviation over a 3x3 window:
    %      iwindow(im, ones(3,3), @std);
    %
    % Notes::
    % - Is a MEX file.
    % - The structuring element should have an odd side length.
    % - Is slow since the function FUNC must be invoked once for every
    %   output pixel.
    % - The input can be logical, uint8, uint16, float or double, the output is
    %   always double
    """

    # TODO replace iwindow's mex function with scipy's ndimage.generic_filter

    # check valid input
    im = getimage(im)
    se = getimage(se)

    # border options:
    edgeopt = {
        'border': 'nearest',
        'none': 'constant',  # TODO does not seem to be a 'not included' option
        'wrap': 'wrap'
    }
    if opt not in edgeopt:
        raise ValueError(opt, 'opt is not a valid edge option')

    # TODO check valid input for func?

    return sp.ndimage.generic_filter(im, func, footprint=se, mode=edgeopt[opt])


def irank(im, se, rank=-1, opt='border'):
    """
    Rank filter

    TODO replace order with rank
    TODO no more nbins

    % OUT = IRANK(IM, ORDER, SE) is a rank filtered version of IM.  Only
    % pixels corresponding to non-zero elements of the structuring element SE
    % are ranked and the ORDER'th value in rank becomes the corresponding output
    % pixel value.  The highest rank, the maximum, is ORDER=-1.
    %
    % OUT = IRANK(IMAGE, SE, OP, NBINS) as above but the number of histogram
    % bins can be specified.
    %
    % OUT = IRANK(IMAGE, SE, OP, NBINS, EDGE) as above but the processing of edge
    % pixels can be controlled.  The value of EDGE is:
    % 'border'   the border value is replicated (default)
    % 'none'     pixels beyond the border are not included in the window TODO
    % 'trim'     output is not computed for pixels whose window crosses TODO
    %            the border, hence output image had reduced dimensions.
    % 'wrap'     the image is assumed to wrap around left-right, top-bottom.
    %
    % Examples::
    %
    % 5x5 median filter, 25 elements in the window, the median is the 12thn in rank
    %    irank(im, 12, ones(5,5));
    %
    % 3x3 non-local maximum, find where a pixel is greater than its eight neighbours
    %    se = ones(3,3); se(2,2) = 0;
    %    im > irank(im, 1, se);
    %
    % Notes::
    % - The structuring element should have an odd side length.
    % - Is a MEX file.
    % - The median is estimated from a histogram with NBINS (default 256).
    % - The input can be logical, uint8, uint16, float or double, the output is
    %   always double
    """

    # TODO replace irank.m mex function with scipy.ndimage.rank_filter

    # check valid input
    im = getimage(im)
    se = getimage(se)

    if not isinstance(rank, int):
        raise TypeError(rank, 'rank is not an int')

    # border options for rank_filter that are compatible with irank.m
    borderopt = {
        'border': 'nearest',
        'wrap': 'wrap'
    }

    if opt not in borderopt:
        raise ValueError(opt, 'opt is not a valid option')

    return sp.ndimage.rank_filter(im, rank, footprint=se, mode=borderopt[opt])


def ihist(im, nbins=256, opt=None):
    """
    Image histogram

    % IHIST(IM, OPTIONS) displays the image histogram.  For an image with  multiple
    % planes the histogram of each plane is given in a separate subplot.
    %
    % H = IHIST(IM, OPTIONS) is the image histogram as a column vector.  For
    % an image with multiple planes H is a matrix with one column per image plane.
    %
    % [H,X] = IHIST(IM, OPTIONS) as above but also returns the bin coordinates as
    % a column vector X.
    %
    % Options::
    % 'nbins'     number of histogram bins (default 256)
    % 'cdf'       compute a cumulative histogram
    % 'normcdf'   compute a normalized cumulative histogram, whose maximum value
    %             is one
    % 'sorted'    histogram but with occurrence sorted in descending magnitude
    %             order.  Bin coordinates X reflect this sorting.
    %
    % Example::
    %
    %    [h,x] = ihist(im);
    %    bar(x,h);
    %
    %    [h,x] = ihist(im, 'normcdf');
    %    plot(x,h);
    %
    % Notes::
    % - For a uint8 image the MEX function FHIST is used (if available)
    %   - The histogram always contains 256 bins
    %   - The bins spans the greylevel range 0-255.
    % - For a floating point image the histogram spans the greylevel range 0-1.
    % - For floating point images all NaN and Inf values are first removed.
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
    return namedtuple('hist', 'h x')(h, x)


def inormhist(im):
    """
    Histogram normalisaton

    % OUT = INORMHIST(IM) is a histogram normalized version of the image IM.
    %
    % Notes::
    % - Highlights image detail in dark areas of an image.
    % - The histogram of the normalized image is approximately uniform, that is,
    %   all grey levels ae equally likely to occur.
    """"

    im = getimage(im)
    if im.ndims > 2:
        raise ValueError(im, 'inormhist does not support color images')

    # TODO could alternatively just call cv.equalizeHist()?
    # TODO note that cv.equalizeHist might only accept 8-bit images, while inormhist can accept float images as well?    # return cv.equalizeHist(im)
    cdf = ihist(im, 'cdf')
    cdf.h = cdf.h / np.max(cdf.h)

    if np.issubdtype(im.dtype, np.float):
        nim = np.interp(im.flatten(), cdf.x, cdf.h)
    else:
        nim = np.interp(idouble(im).flatten(), cdf.x, cdf.h)

    # reshape nim to image:
    return nim = nim.reshape(im.shape[0], im.shape[1])


# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':

    # testing idisp:
    im_name = 'longquechen-moon.png'
    im = iread((Path('images') / 'test' / im_name).as_posix())

    # for debugging interactively
    import code
    code.interact(local=dict(globals(), **locals()))

    # show original image
    idisp(im, title='space rover 2020')

    # do mono
    #im2 = imono(im1)
    #idisp(im2, title='mono')

    # test icolor # RGB
    #im3 = icolor(im2, c=[1, 0, 0])
    #idisp(im3, title='icolor(red)')

    # test istretch
    #im4 = istretch(im3)
    #idisp(im4, title='istretch')

    # test ierode
    # im = np.array([[1, 1, 1, 0],
    #               [1, 1, 1, 0],
    #               [0, 0, 0, 0]])
    # im5 = ierode(im, se=np.ones((3, 3)))
    # print(im5)
    # idisp(im5, title='eroded')

    # im = [[0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 1, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0]]
    # im6 = idilate(im, se=np.ones((3, 3)))
    # print(im6)
    # idisp(im6, title='dilated')

    print('done')
