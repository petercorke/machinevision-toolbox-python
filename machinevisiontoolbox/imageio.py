import numpy as np
import urllib.request
from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib as mpl
# for getting screen resolution
import pyautogui  # requires pip install pyautogui
from spatialmath.base import islistof

def idisp(im,
          title='Machine Vision Toolbox for Python',
          title_window='Machine Vision Toolbox for Python',
          fig=None,
          ax=None,
          block=False,
          grey=False,
          invert=False,
          invsigned=False,
          colormap=None,
          ncolors=256,
          cbar=False,
          noaxes=False,
          nogui=False,
          noframe=False,
          plain=False,
          savefigname=None,
          notsquare=False,
          fwidth=None,
          fheight=None,
          wide=False,
          flatten=False,
          histeq=False,
          vrange=None,
          **kwargs):
    """
    Interactive image display tool
    :param im: image
    :type im: numpy array, shape (N,M,3) or (N, M)
    :param fig: matplotlib figure handle to display image on
    :type fig: tuple
    :param ax: matplotlib axes object to plot on
    :type ax: axes object
    :param block: matplotlib figure blocks python kernel until window closed
    :type block: bool
    :param colormap: colormap
    :type colormap: string? 3-tuple? see plt.colormaps, matplotlib.cm.get_cmap
    :param ncolors: number of colors in colormap
    :type ncolors: int
    :param noaxes: don't display axes on the image
    :type noaxes: bool
    :param cbar: add colorbar to image
    :type cbar: bool
    :type noaxes: bool
    :param nogui: don't display GUI/interactive buttons
    :type nogui: bool
    :param noframe: don't display axes or frame on the image
    :type noframe: bool
    :param plain: don't display axes, frame or GUI
    :type plain: bool
    :param title: title of figure in figure window
    :type title: str
    :param title: title of figure window
    :type title: str
    :param grey: color map: greyscale unsigned, zero is black, maximum value is
    white
    :type grey: bool
    :param invert: color map: greyscale unsigned, zero is white, max is black
    :type invert: bool
    :param invsigned: color map: greyscale signed, positive is blue, negative
    is red, zero is white
    :type invsigned: bool
    :param savefigname: if not None, save figure as savefigname (default eps)
    :type savefigname: str
    :param notsquare: display aspect ratio so that pixels are not square
    :type notsquare: bool
    :param fwidth: figure width in inches (need dpi for relative screen size?)
    :type fwidth: float
    :param fheight: figure height in inches
    :type fheight: float
    :param wide: set to full screen width, useful for displaying stereo pair
    :type wide: bool
    :param flatten: display image planes horizontally as adjacent images
    :type flatten: bool
    :param histeq: apply histogram equalization
    :param histeq: bool
    :return fig: Matplotlib figure handle
    :rtype fig: figure handle
    :return ax: Matplotlib axes handle
    :rtype ax: axes handle

    - ``idisp(im)`` displays an image. TODO how to document all the options?

    :options:
        - 'clickfunc',F    invoke the function handle F(x,y) on a down-click in
          the window
        - 'black',B        change black to grey level B (range 0 to 1)
        - 'ynormal'        y-axis interpolated spectral data and corresponding
          wavelengthincreases upward, image is inverted
        - 'cscale',C       C is a 2-vector that specifies the grey value range
          that spans the colormap.
        - 'xydata',XY      XY is a cell array whose elements are vectors that
          span the x- and y-axes respectively.
        - 'colormap',C     set the colormap to C (Nx3)
        - 'signed'         color map: greyscale signed, positive is blue,
          negative is red, zero is black
        - 'random'         color map: random values, highli`ghts fine structure
        - 'dark'           color map: greyscale unsigned, darker than 'grey',
          good for superimposed graphics
        - 'new'            create a new figure

    Example:

    .. autorun:: pycon

    .. note::

        - Greyscale images are displayed in indexed mode: the image pixel
          value is mapped through the color map to determine the display pixel
          value.
        - For grey scale images the minimum and maximum image values are
          mapped to the first and last element of the color map, which by
          default ('greyscale') is the range black to white. To set your own
          scaling between displayed grey level and pixel value use the 'cscale'
          option.
        - The title of the figure window by default is the name of the variable
          passed in as the image, this can't work if the first argument is an
          expression.

    :references:

        - Robotics, Vision & Control, Section 10.1, P. Corke, Springer 2011.
    """

    # plain: hide GUI, frame and axes:
    if plain:
        nogui = True
        noaxes = True
        noframe = True

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
           'drawonly': False,
           'vrange': None,
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

        # aspect ratio:
        if notsquare:
            mpl.rcParams["image.aspect"] = 'auto'

        # hide interactive toolbar buttons (must be before figure creation)
        if nogui:
            mpl.rcParams['toolbar'] = 'None'

        if flatten:
            # either make new subplots for each channel
            # or concatenate all into one large image and display
            # TODO can we make axes as a list?

            # for now, just concatenate:
            # first check how many channels:
            if im.ndim > 2:
                # create list of image channels
                imcl = [im[:, :, i] for i in range(im.shape[2])]
                # stack horizontally
                im = np.hstack(imcl)
            # else just plot the regular image - only one channel

        # histogram equalisation
        if histeq:
            imobj = Image(im)
            im = imobj.normhist().image

        if fig is None and ax is None:
            fig, ax = plt.subplots()  # fig creates a new window

        # get screen resolution:
        swidth, sheight = pyautogui.size()  # pixels
        dpi = None  # can make this an input option
        if dpi is None:
            dpi = mpl.rcParams['figure.dpi']  # default is 100

        if wide:
            # want full screen width NOTE (/2 for dual-monitor setup)
            fwidth = swidth/dpi/2

        if fwidth is not None:
            fig.set_figwidth(fwidth)  # inches

        if fheight is not None:
            fig.set_figheight(fheight)  # inches

        # colormaps:
        # cmapflags = [grey, invert, invsigned]  # list of booleans
        if colormap is None:
            if grey:
                cmap = 'gray'
            elif invert:
                cmap = 'Greys'
            elif invsigned:
                cmap = 'seismic'
            else:
                cmap = None
        else:
            cmap = colormap

        print('Colormap is ', cmap)
        if vrange is not None:
            cmapobj = ax.imshow(im, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
        else:
            cmapobj = ax.imshow(im, cmap=cmap)

        if cbar:
            fig.colorbar(cmapobj, ax=ax)

        # set title of figure window
        fig.canvas.set_window_title(title_window)

        # set title in figure plot:
        # fig.suptitle(title)  # slightly different positioning
        ax.set_title(title)

        # hide image axes - by default also removes frame
        # back with ax.spines['top'].set_visible(True) ?
        if noaxes:
            ax.axis('off')

        # no frame:
        if noframe:
            # NOTE: for frame tweaking, see matplotlib.spines
            # https://matplotlib.org/3.3.2/api/spines_api.html
            # note: can set spines linewidth:
            # ax.spines['top'].set_linewidth(2.0)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

        if savefigname is not None:
            # TODO check valid savefigname
            # set default save file format
            mpl.rcParams["savefig.format"] = 'eps'
            plt.draw()

            # savefig must be called before plt.show
            # after plt.show(), a new fig is automatically created
            plt.savefig(savefigname)

        # if opt['drawonly']:
        #     plt.draw()
        # else:
        #     plt.show()
        plt.show(block=block)
        return ax
    else:
        cv.namedWindow(opt['title'], cv.WINDOW_AUTOSIZE)
        cv.imshow(opt['title'], im)  # make sure BGR format image
        k = cv.waitKey(delay=0)  # non blocking, by default False
        # cv.destroyAllWindows()

        # TODO would like to find if there's a more graceful way of
        # exiting/destroying the window, or keeping it running in the
        # background (eg, start a new python process for each figure)
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


def iread(filename, *args, verbose=True, **kwargs):
    """
    Read image from file

    :param file: file name or URL
    :type file: string
    :param args: arguments
    :type args: args
    :param dtype: a NumPy dtype string such as "uint8", "int16", "float32"
    :type dtype: str
    :param grey: convert to grey scale
    :type grey: bool
    :param greymethod: ITU recommendation, either 601 [default] or 709
    :type greymethod: int
    :param reduce: subsample image by this amount in u- and v-dimensions
    :type reduce: int
    :param gamma: gamma decoding, either the exponent of "sRGB"
    :type gamma: float or str
    :param roi: extract region of interest [umin, umax, vmin vmax]
    :type roi: array_like(4)
    :param kwargs: key word arguments - options for idisp
    :type kwargs: see dictionary below TODO
    :return: image
    :rtype: 2D or 3D NumPy array

    - ``image, path = iread(file)`` reads the specified image file and returns the
      image as a NumPy matrix, as well as the absolute path name.  The
      image can by greyscale or color in any of the wide range of formats
      supported by the OpenCV ``imread`` function.

    - ``image, url = iread(url)`` as above but reads the image from the given
      URL.

    If ``file`` is a list or contains a wildcard, the result will be a list of
    ``(image, path)`` tuples.  They will be sorted by path.

    - ``iread(filename, dtype="uint8", grey=None, greymethod=601, reduce=1,
      gamma=None, roi=None)``



    :options:

        - 'uint8'         return an image with 8-bit unsigned integer pixels in
          the range 0 to 255
        - 'single'        return an image with single precision floating point
          pixels in the range 0 to 1.
        - 'double'        return an image with double precision floating point
          pixels in the range 0 to 1.
        - 'grey'          convert image to greyscale, if it's color, using
          ITU rec 601
        - 'grey_709'      convert image to greyscale, if it's color, using
          ITU rec 709
        - 'gamma',G       apply this gamma correction, either numeric or 'sRGB'
        - 'reduce',R      decimate image by R in both dimensions
        - 'roi',R         apply the region of interest R to each image,
          where R=[umin umax; vmin vmax].

    Example:

    .. autorun:: pycon

    .. note::

        - A greyscale image is returned as an HxW matrix
        - A color image is returned as an HxWx3 matrix
        - A greyscale image sequence is returned as an HxWxN matrix where N is
          the sequence length
        - A color image sequence is returned as an HxWx3xN matrix where N is
          the sequence length

    :references:

        - Robotics, Vision & Control, Section 10.1, P. Corke, Springer 2011.
    """

    # determine if file is valid:
    # assert isinstance(filename, str),  'filename must be a string'


    # TODO read options for image
    # opt = {
    #     'uint8': False,
    #     'single': False,
    #     'double': False,
    #     'grey': False,
    #     'grey_709': False,
    #     'gamma': 'sRGB',
    #     'reduce': 1.0,
    #     'roi': None
    # }

    if isinstance(filename, str) and (filename.startswith("http://") or filename.startswith("https://")):
        # reading from a URL

        resp = urllib.request.urlopen(filename)
        array = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv.imdecode(array, -1)
        print(image.shape)
        return (image, filename)

    elif isinstance(filename, (str, Path)):
        # reading from a file

        path = Path(filename).expanduser()

        if any([c in "?*" for c in str(path)]):
            # contains wildcard characters, glob it
            # recurse and return a list
            # https://stackoverflow.com/questions/51108256/how-to-take-a-pathname-string-with-wildcards-and-resolve-the-glob-with-pathlib
    
            parts = path.parts[1:] if path.is_absolute() else path.parts
            p = Path(path.root).glob(str(Path("").joinpath(*parts)))
            pathlist = list(p)

            if len(pathlist) == 0 and not path.is_absolute():
                # look in the toolbox image folder
                path = Path(__file__).parent / "images" / path
                parts = path.parts[1:] if path.is_absolute() else path.parts
                p = Path(path.root).glob(str(Path("").joinpath(*parts)))
                pathlist = list(p)
            
            if len(pathlist) == 0:
                raise ValueError("can't expand wildcard")

            imlist = []
            pathlist.sort()
            for p in pathlist:
                imlist.append(iread(p, **kwargs))
            return imlist

        else:
            # read single file

            if not path.exists():
                if path.is_absolute():
                    raise ValueError(f"file {filename} does not exist")
                # file doesn't exist
                # see if it matches the supplied images
                path = Path(__file__).parent / "images" / path

                if not path.exists():
                    raise ValueError(f"file {filename} does not exist, and not found in supplied images")

            # read the image
            # TODO not sure the following will work on Windows
            im = cv.imread(path.as_posix(), **kwargs)  # default read-in as BGR

            if im is None:
                # TODO check ValueError
                raise ValueError(f"Could not read {filename}")

            return (im, str(path))

    elif islistof(filename, (str, Path)):
        # list of filenames or URLs
        # assume none of these are wildcards, TODO should check
        out = []
        for file in filename:
            out.append(iread(file, *args))
        return out
    else:
        raise ValueError(filename, 'invalid filename')

def int_image(image, intclass='uint8'):
    """
    Convert image to integer type

    :param image: input image
    :type image: ndarray(h,w,nc) or ndarray(h,w,nc)
    :param intclass: either 'uint8', or any integer class supported by np
    :type intclass: str
    :return: image with integer pixel types
    :rtype: ndarray(h,w,nc) or ndarray(h,w,nc)

    - ``int_image(image)`` is a copy of image with pixels converted to unsigned
        8-bit integer (uint8) elements in the range 0 to 255.

    - ``int_image(intclass)`` as above but the output pixels are converted to
        the integer class ``intclass``.

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox import iread, idisp, int_image
        >>> im, file = iread('flowers1.png')
        >>> idisp(int_image(im, 'uint16'))

    .. note::

        - Works for greyscale or color (arbitrary number of planes) image
        - If the input image is floating point (single or double) the
            pixel values are scaled from an input range of [0,1] to a range
            spanning zero to the maximum positive value of the output integer
            class.
        - If the input image is an integer class then the pixels are cast
            to change type but not their value.

    :references:

        - Robotics, Vision & Control, Section 12.1, P. Corke,
            Springer 2011.
    """

    if np.issubdtype(image.dtype, np.bool):
        return image.astype(intclass)

    if np.issubdtype(image.dtype, np.floating):
        # rescale to integer
        scaled = im * np.float64(np.iinfo(intclass).max)
        return np.rint(scaled).astype(intclass)
    elif np.issubdtype(image.dtype, np.integer):
        # cast to different integer type
        return image.astype(intclass)
 

def float_image(image, floatclass='float32'):
    """
    Convert image to float type

    :param image: input image
    :type image: ndarray(h,w,nc) or ndarray(h,w,nc)
    :param floatclass: 'single', 'double', 'float32' [default], 'float64'
    :type floatclass: str
    :return: image with floating point pixel types
    :rtype: ndarray(h,w,nc) or ndarray(h,w,nc)

    - ``float_image()`` is a copy of image with pixels converted to
        ``float32`` floating point values spanning the range 0 to 1. The
        input integer pixels are assumed to span the range 0 to the maximum
        value of their integer class.

    - ``float_image(im, floatclass)`` as above but with floating-point pixel
        values belonging to the class ``floatclass``.

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox import iread, idisp, float_image
        >>> im, file = iread('flowers1.png')
        >>> idisp(float_image(im))

    .. note::

        - Works for greyscale or color (arbitrary number of planes) image
        - If the input image is integer the
          pixel values are scaled from an input range
          spanning zero to the maximum positive value of the output integer
          class to [0,1]
        - If the input image is a floating class then the pixels are cast
            to change type but not their value.

    :references:

        - Robotics, Vision & Control, Section 12.1, P. Corke,
            Springer 2011.
    """

    if floatclass in ('float', 'single', 'float32'):
        # convert to float pixel values
        if np.issubdtype(image.dtype, np.integer):
            # rescale the pixel values
            return image.astype(floatclass) / np.iinfo(image.dtype).max
        elif np.issubdtype(image.dtype, np.floating):
            # cast to different float type
            return image.astype(floatclass)
    else:
        raise ValueError('bad float type')


def iwrite(im, filename, **kwargs):
    """
    Write image (numpy array) to filename

    :param filename: filename to write to
    :type filename: string

    - ``iwrite(im, filename, **kwargs)`` writes ``im`` to ``filename`` with
      **kwargs currently for cv.imwrite() options.

    Example:

    .. autorun:: pycon

    """

    # TODO check valid input

    ret = cv.imwrite(filename, im, **kwargs)

    if ret is False:
        print('Warning: image failed to write to filename')
        print('Image =', im)
        print('Filename =', filename)

    return ret

if __name__ == "__main__":


    filename = "~/code/machinevision-toolbox-python/machinevisiontoolbox/images/campus/*.png"

    im, p = iread(filename)
    print(p)