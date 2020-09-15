#!/usr/bin/env python

import io as io
import numpy as np
import spatialmath.base.argcheck as argcheck
import cv2 as cv
import matplotlib.path as mpath
import sys as sys
import machinevisiontoolbox.color

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
            return im.astype(np.float64)  # the preferred method, compared to np.float64(im)


def getimage(im):
    """
    input an image, converts image into types compatible opencv:
    CV_8U, CV_16U, CV_16S, CV_32F or CV_64F
    default: if int then CV_8U, if float then CV_64F
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


# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':

    # testing idisp:
    im_name = 'longquechen-moon.png'
    im = iread((Path('images') / 'test' / im_name).as_posix())

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

    #im = [[0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 1, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0]]
    im6 = idilate(im, se=np.ones((3, 3)))
    # print(im6)
    idisp(im6, title='dilated')

    print('done')
