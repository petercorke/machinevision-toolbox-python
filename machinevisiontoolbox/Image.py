#!/usr/bin/env python
"""
Images class
@author: Dorian Tsai
@author: Peter Corke
"""

import numpy as np
import cv2 as cv
# import spatialmath.base.argcheck as argcheck
import matplotlib.pyplot as plt
import machinevisiontoolbox as mvt


class Image():  # or inherit from np.ndarray?
    """
    An image class for MVT
    """

    def __init__(self,
                 rawimage=None,
                 colorspace='BGR',
                 iscolor=None):

        # super().__init__()  # not sure about this

        if rawimage is None:
            # instance-wide attributes
            # init empty image
            self._width = None
            self._height = None
            self._numimagechannels = None
            self._numimages = None
            self._dtype = None
            self._colorspace = None
            self._imlist = None
            self._iscolor = None
            self._filenamelist = None

        else:

            # whatever rawimage input type is, try to convert it to a list of
            # numpy array images
            # TODO input string with glob

            if isinstance(rawimage, Image):
                # TODO list of Image objects?
                self._imlist = rawimage._imlist
                self._filenamelist = rawimage._filenamelist

            elif isinstance(rawimage, str):
                # string = name of an image file to read in
                imlist = [iread(rawimage)]
                # TODO once iread, then filter through imlist and arrange into
                # proper numimages and numchannels, based on user inputs, though
                # default to single list
                if (iscolor is False) and (imlist[0].ndim == 3):
                    # check 3rd dimension of imlist[0], and make into list of
                    # images
                    imlist = [imlist[0][0:, 0:, i]
                              for i in range(imlist[0].shape[2])]

                self._imlist = imlist
                self._filenamelist = [rawimage]

            elif isinstance(rawimage, list) and isinstance(rawimage[0], str):
                # list of image file names
                print('list of image strings')
                imlist = [iread(rawimage[i]) for i in range(len(rawimage))]

                if (iscolor is False) and (imlist[0].ndim == 3):
                    imlistc = []
                    for i in range(len(imlist)):  # for each image in list
                        for j in range(imlist[i].shape[2]):  # for each channel
                            imlistc.append(imlist[i][0:, 0:, j])
                    imlist = imlistc
                self._imlist = imlist
                self._filenamelist = rawimage

            elif isinstance(rawimage, list) and \
                    isinstance(np.asarray(rawimage[0]), np.ndarray):
                # list of images, with each item being a numpy array
                # imlist = TODO deal with iscolor=False case

                if (iscolor is False) and (rawimage[0].ndim == 3):
                    imlist = []
                    for i in range(len(rawimage)):
                        for j in range(rawimage[i].shape[2]):
                            imlist.append(rawimage[i][0:, 0:, j])
                    self._imlist = imlist
                else:
                    self._imlist = rawimage

                self._filenamelist = [None]*len(self._imlist)

            elif Image.isimage(rawimage):
                # is an actual image or sequence of images compounded into
                # single ndarray
                # make this into a list of images
                # if color:
                rawimage = Image.getimage(rawimage)
                if rawimage.ndim == 4:
                    # assume (W,H,3,N)
                    self._imlist = [Image.getimage(rawimage[0:, 0:, 0:, i])
                                    for i in range(rawimage.shape[3])]
                elif rawimage.ndim == 3:
                    # could be single (W,H,3) -> 1 colour image
                    # or (W,H,N) -> N grayscale images
                    if not rawimage.shape[2] == 3:
                        self._imlist = [Image.getimage(rawimage[0:, 0:, i])
                                        for i in range(rawimage.shape[2])]
                    elif (rawimage.shape[2] == 3) and iscolor:
                        # manually specified iscolor is True
                        # single colour image
                        self._imlist = [Image.getimage(rawimage)]
                    elif (rawimage.shape[2] == 3) and (iscolor is None):
                        # by default, we will assume that a (W,H,3) with
                        # unspecified iscolor is a color image, as the
                        # 3-sequence greyscale case is much less common
                        self._imlist = [Image.getimage(rawimage)]
                    else:
                        self._imlist = [Image.getimage(rawimage[0:, 0:, i])
                                        for i in range(rawimage.shape[2])]

                elif rawimage.ndim == 2:
                    # single (W,H)
                    self._imlist = [Image.getimage(rawimage)]

                else:
                    raise ValueError(rawimage, 'unknown rawimage.shape')

                self._filenamelist = [None]*len(self._imlist)

            else:
                raise TypeError(rawimage, 'raw image is not valid image type')
                print('Valid image types: filename string of an image, \
                       list of filename strings, \
                       list of numpy arrays, or a numpy array')

            # check list of images for size consistency

            # VERY IMPORTANT!
            # We assume that the image stack has the same size image for the
            # entire list. TODO maybe in the future, we remove this assumption,
            # which can cause errors if not adhered to,
            # but for now we simply check the shape of each image in the list
            shape = [self._imlist[i].shape for i in range(len(self._imlist))]
            # shape = [img.shape for img in self._imlist[]]
            # if any(shape[i] != list):
            #   raise

            if np.any([shape[i] != shape[0] for i in range(len(self._imlist))]):
                raise ValueError(rawimage, 'inconsistent input image shape')

            self._height = self._imlist[0].shape[0]
            self._width = self._imlist[0].shape[1]

            # ability for user to specify iscolor manually to remove ambiguity
            if iscolor is None:
                self._iscolor = None
                self._iscolor = self.iscolor  # our best guess
            else:
                self._iscolor = iscolor

            self._numimages = len(self._imlist)

            if self._imlist[0].ndim == 3:
                self._numimagechannels = self._imlist[0].shape[2]
            elif self._imlist[0].ndim == 2:
                self._numimagechannels = 1
            else:
                raise ValueError(self._numimagechannels, 'unknown number of \
                                 image channels')

            self._dtype = self._imlist[0].dtype

            validcolorspaces = ('RGB', 'BGR')
            # TODO add more valid colorspaces
            # assume some default: BGR because we import with mvt with
            # opencv's imread(), which imports as BGR by default
            if colorspace in validcolorspaces:
                self._colorspace = colorspace
            else:
                raise ValueError(colorspace, 'unknown colorspace input')

    def __len__(self):
        return len(self._imlist)

    def __getitem__(self, ind):
        # try to return the ind'th image in an image sequence if it exists
        new = Image()
        new._width = self._width
        new._height = self._height
        new._numimagechannels = self._numimagechannels
        new._dtype = self._dtype
        new._colorspace = self._colorspace
        new._iscolor = self._iscolor

        new._imlist = self.listimages(ind)
        new._filenamelist = self.listimagefilenames(ind)
        new._numimages = len(new._imlist)

        return new

    # ------------------------- properties ------------------------------ #

    # properties
    @property
    def size(self):
        return (self._width, self._height)

    @property
    def nimages(self):
        return self._numimages

    @property
    def nchannels(self):
        return self._numimagechannels

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def dtype(self):
        return self._dtype

    @property
    def colorspace(self):
        return self._colorspace

    @property
    def shape(self):
        return self._imlist[0].shape

    @property
    def ndim(self):
        return self._imlist[0].ndim

    # for convenience (90% of the time), just return the first image in list
    @property
    def image(self):
        return self._imlist[0]

    @property
    def imlist(self):
        return self._imlist

    @property
    def filename(self):
        return self._filenamelist[0]

    # TODO make setters for filename

    @property
    def bgr(self):
        # if ind is None:
        #    ind = np.arange(0, len(self._imlist))
        #imlist = self.listimages(ind)

        if self._colorspace == 'BGR':
            return self._imlist[0]
        else:
            # convert to proper colorspace:
            # TODO mvt.colorspace(self._imlist, '(ctype)->BGR')  # TODO
            # for now, assume we are RGB and simply switch the channels:
            if not self._iscolor:
                return self._imlist[0]
            else:
                # bgr = np.zeros(self._imlist.shape)
                # or i in range(self._numimages):
                #    bgr[0:, 0:, 0:, i] = self._imlist[0:, 0:, ::-1, i]
                # (H,W,3,N) for RGB -> (H,W,3,N) for BGR
                if self._imlist[0].ndim > 3:
                    return self._imlist[0][:, :, ::-1, :]
                    # return [self._imlist[i][:, :, ::-1, :]
                    #        for i in range(len(self._imlist))]
                else:
                    return self._imlist[0][:, :, ::-1]
                    # return [self._imlist[i][0:, 0:, ::-1]
                    #        for i in range(len(self._imlist))]

    @property
    def rgb(self):
        if self._colorspace == 'RGB':
            return self._imlist[0]
        else:
            if not self._iscolor:
                return self._imlist[0]
            else:
                if self._imlist[0].ndim > 3:
                    # (H,W,3,N) for BGR -> (H,W,3,N) for RGB
                    # return [self._imlist[i][0:, 0:, ::-1, 0:]
                    #        for i in range(len(self._imlist))]
                    return self._imlist[0][:, :, ::-1, :]
                else:
                    return self._imlist[0][:, :, ::-1]
                    # return [self._imlist[i][0:, 0:, ::-1]
                    #        for i in range(len(self._imlist))]
    """
    def rgb(self, ind=None):
        if ind is None:
            ind = np.arange(0, len(self._imlist))
        imlist = self.listimages(ind)

        if self._colorspace == 'RGB':
            return imlist
        else:
            # convert to proper colorspace first:
            # return mvt.colorspace(self._imlist, '(ctype)->RGB')
            # TODO for now, we just assume RGB or BGR
            if not self._iscolor:
                return imlist
            else:
                if imlist[0].ndim > 3:
                    # (H,W,3,N) for BGR -> (H,W,3,N) for RGB
                    return [imlist[i][0:, 0:, ::-1, 0:]
                            for i in range(len(imlist))]
                else:
                    return [imlist[i][0:, 0:, ::-1]
                            for i in range(len(imlist))]
    """

    @property
    def iscolor(self):
        """
        ``iscolor(im)`` is true if ``im`` is a color image, that is, its third
        dimension is equal to three.
        """
        # W,H is mono
        # W,H,3 is color
        # W,H,N is mono sequence (ambiguous for N=3 mono image sequence)
        # W,H,3,N is color sequence
        if self._iscolor is not None:
            return self._iscolor
        else:
            im = self._imlist[0]
            if (im.ndim == 4) and (im.shape[0] > 1) and \
               (im.shape[1] > 1) and (im.shape[2] == 3):
                # color sequence
                return True
            elif (im.ndim == 3) and (im.shape[0] > 1) and \
                 (im.shape[1] > 1) and (im.shape[2] == 3):
                # could be (W,H,3) or (W,H,(N=3)), but more often than not,
                # likely to be a color image
                return True
            else:
                return False

        # return self._iscolor or Image.iscolor(self._imlist[0])

    # ------------------------- class functions? ---------------------------- #

    def disp(self, **kwargs):
        """
        Display first image in imlist
        """
        idisp(self._imlist[0], title=self._filenamelist[0], **kwargs)

    def write(self, filename):
        """
        Write first image in imlist to filename
        """
        iwrite(self._imlist[0], filename)

    def listimages(self, ind):
        if isinstance(ind, int) and (ind >= -1) and (ind <= len(self._imlist)):
            return [self._imlist[ind]]
        elif isinstance(ind, slice):
            islice = np.arange(ind.start, ind.stop, ind.step)
            return [self._imlist[i] for i in islice]
        # elif isinstance(ind, tuple) and (len(ind) == 3):
        # slice object from numpy as a 3-tuple -> but how can we
        # differentiate between a normal 3-tuple eg (0,1,2) vs a numpy slice
        # (0, 2, 1)? TODO ruminate for later
        #     islice = np.arange()
        elif (len(ind) > 1) and (np.min(ind) >= -1) and (np.max(ind) <= len(self._imlist)):
            return [self._imlist[i] for i in ind]

    def listimagefilenames(self, ind):
        if isinstance(ind, int) and (ind >= -1) and (ind <= len(self._filenamelist)):
            return [self._filenamelist[ind]]
        elif isinstance(ind, slice):
            islice = np.arange(ind.start, ind.stop, ind.step)
            return [self._filenamelist[i] for i in islice]
        # elif isinstance(ind, tuple) and (len(ind) == 3):
        # slice object from numpy as a 3-tuple -> but how can we
        # differentiate between a normal 3-tuple eg (0,1,2) vs a numpy slice
        # (0, 2, 1)? TODO ruminate for later
        #     islice = np.arange()
        elif (len(ind) > 1) and (np.min(ind) >= -1) and (np.max(ind) <= len(self._filenamelist)):
            return [self._filenamelist[i] for i in ind]

    # def min(self):
    # ------------------------- class methods ------------------------------ #

    @classmethod
    def isimage(cls, imarray):
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
        imarray = np.array(imarray)

        # TODO consider complex floats?
        # check if image is int or floats
        # TODO shouldn't np.integer and np.float be the catch-all types?
        # TODO make list
        imtypes = [np.integer,
                   np.float,
                   np.bool_,
                   np.uint8,
                   np.uint16,
                   np.uint32,
                   np.uint64,
                   np.int8,
                   np.int16,
                   np.int32,
                   np.int64,
                   np.float32,
                   np.float64]
        if imarray.dtype not in imtypes:
            return False

        # check im.ndims > 1
        if imarray.ndim < 2:
            return False

        # check if im.ndims == 2, then im.shape (W,H), W >= 1, H >= 1
        if (imarray.ndim == 2) and ((imarray.shape[0] >= 1) and
                                    (imarray.shape[1] >= 1)):
            return True

        # check if im.ndims == 3, then im.shape(W,H,N), N >= 1
        if (imarray.ndim == 3) and (imarray.shape[2] >= 1):
            return True

        # check if im.ndims == 4, then im.shape(W,H,N,M), then N == 3
        if (imarray.ndim == 4) and (imarray.shape[2] == 3):
            return True

        # return consistent image format
        return False

    @classmethod
    def getimage(cls, imarray):
        """
        converts ``im`` to image compatible with OpenCV

        :param im: image
        :type im: numpy array (N,H,3) or (N,H) or TODO Image object
        :return out: image of type np.(uint8, uint16, int16, float32, float64)
        :rtype: numpy array of the size of im

        ``getimage(im)`` converts ``im`` into a compatible datatype with OpenCV:
        CV_8U, CV_16U, CV_16S, CV_32F or CV_64F. By default, if int then CV_8U, and
        if float then CV_64F. Boolean images are converted to 0's and 1's int
        """
        # if isinstance(im, Image):
        #    imlist = im.imlist
        if not Image.isimage(imarray):
            raise TypeError(imarray, 'im is not a valid image')

        imarray = np.array(imarray)

        validTypes = (np.uint8, np.uint16, np.int16, np.float32, np.float64)
        # if im.dtype is not one of the valid image types,
        # convert to the nearest type or default to float64
        # TODO: what about image scaling?
        if imarray.dtype not in validTypes:
            # if float, just convert to CV_64F
            if np.issubdtype(imarray.dtype, np.float):
                imarray = np.float64(imarray)
            elif np.issubdtype(imarray.dtype, np.integer):
                if imarray.min() < 0:
                    imarray = np.int16(imarray)
                elif imarray.max() < np.iinfo(np.uint8).max:
                    imarray = np.uint8(imarray)
                elif imarray.max() < np.iinfo(np.uint16).max:
                    imarray = np.uint16(imarray)
                else:
                    raise ValueError(
                        imarray, 'max value of im exceeds np.uint16')
            elif np.issubdtype(imarray.dtype, np.bool_):
                imarray = np.uint8(imarray)

        return imarray


# ------------------------------ functions  ---------------------------------- #


def idisp(im,
          fig=None,
          ax=None,
          **kwargs):
    """
    Interactive image display tool

    :param im: image
    :type im: numpy array, shape (N,M,3) or (N, M)
    :param fig: matplotlib figure handle to display image on
    :type fig: tuple
    :param ax: matplotlib axes object to plot on
    :type ax: axes object
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

    # check if im is valid input
    if isinstance(im, Image):
        # if Image object, then take the image out
        image = im.image
    else:
        image = Image.getimage(im)

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
           'new': True,
           'matplotlib': True,  # default to matplotlib plotting
           'drawonly': False
           }

    # apply kwargs to opt
    # TODO can be written in one line "a comprehension"
    for k, v in kwargs.items():
        if k in opt:
            opt[k] = v

    # if we are running in a Jupyter notebook, print to matplotlib,
    # otherwise print to opencv imshow/new window. This is done because
    # cv.imshow does not play nicely with .ipynb
    if _isnotebook() or opt['matplotlib']:

        # if (im.ndim == 3) and (im.shape[2] == 3):
        #    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        if fig is None and ax is None:
            fig, ax = plt.subplots()  # fig creates a new window

        if isinstance(im, Image):
            image = im.rgb
        ax.imshow(image)
        # versus fig.suptitle(opt['title'])
        ax.set_title(opt['title'])

        if opt['drawonly']:
            plt.draw()
        else:
            plt.show()

    else:
        cv.namedWindow(opt['title'], cv.WINDOW_AUTOSIZE)
        cv.imshow(opt['title'], image)  # make sure BGR format image
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

        # TODO fig, ax equivalent for OpenCV? how to print/plot to the same
        # window/set of axes?
        fig = None
        ax = None

    return fig, ax


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

    # TODO read options for image
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

    # if empty, display list of images to automatically read

    # check if file is a valid pathname:
    im = cv.imread(file, **kwargs)  # default read-in should be BGR
    if im is None:
        # TODO check ValueError
        raise ValueError('Could not read the image specified by ``file``.')

    # TODO check for wild cards
    # TODO search paths automatically for specified file?
    # TODO fetch from server

    return im


def iwrite(im, filename, **kwargs):
    """
    Write image to file

    :param im: image to write
    :type im: numpy array
    :param filename: filename to write to
    :type filename: string
    """

    # check valid input

    # cv.imwrite can only save 8-bit single channel or 3-channel BGR images
    # with several specific exceptions
    # https://docs.opencv.org/4.4.0/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce
    # TODO imwrite has many quality/type flags
    ret = cv.imwrite(filename, im, **kwargs)

    if ret is False:
        print('Warning: image failed to write to filename')
        print('Image =', im)
        print('Filename =', filename)

    return ret


if __name__ == "__main__":

    # read im image:

    # test for single colour image
    imfile = 'images/test/longquechen-mars.png'
    rawimage = iread(imfile)

    # test for image string
    rawimage = imfile

    # test for multiple images, stack them first:
    flowers = [str(('flowers' + str(i+1) + '.png')) for i in range(8)]
    print(flowers)

    # list of images
    imlist = [iread(('images/' + i)) for i in flowers]

    # plt.show()

    im = Image(imlist)

    print('im.image dtype =', im.image.dtype)
    print('im.shape =', im.shape)
    print('im.iscolor =', im.iscolor)
    print('im.numimages =', im.nimages)
    print('im.numchannels =', im.nchannels)

    import code
    code.interact(local=dict(globals(), **locals()))

    # idisp(im.bgr)

    #import code
    #code.interact(local=dict(globals(), **locals()))
