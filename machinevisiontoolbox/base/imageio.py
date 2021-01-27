import numpy as np
import urllib.request
from pathlib import Path
import cv2 as cv

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.backend_tools import ToolBase, ToolToggleBase

# for getting screen resolution
#import pyautogui  # requires pip install pyautogui
from spatialmath.base import islistof
from machinevisiontoolbox.base import colorconvert

def idisp(im,
          title='Machine Vision Toolbox for Python',
          title_window='Machine Vision Toolbox for Python',
          fig=None,
          ax=None,
          block=False,
          colormap=None,
          black=0,
          matplotlib=True,
          ncolors=256,
          cbar=False,
          axes=True,
          gui=True,
          frame=True,
          plain=False,
          savefigname=None,
          square=True,
          fwidth=None,
          fheight=None,
          wide=False,
          darken=None,
          flatten=False,
          histeq=False,
          vrange=None,
          ynormal=False,
          xydata=None,
          **kwargs):

    """
    Interactive image display tool

    :param im: image
    :type im: numpy array, shape (N,M,3) or (N, M)
    :param fig: matplotlib figure handle to display image on, defaults to new figure
    :type fig: tuple
    :param ax: matplotlib axis object to plot on, defaults to new axis
    :type ax: axis object
    :param block: matplotlib figure blocks python kernel until window closed, default False
    :type block: bool

    :param histeq: apply histogram equalization before display, default False
    :param histeq: bool
    :param black: set black (zero) pixels to this value, default 0
    :type black: int or float

    :param colormap: colormap
    :type colormap: str or matplotlib.colors.Colormap
    :param ncolors: number of colors in colormap
    :type ncolors: int
    :param darken: darken the image by factor, default 1
    :type darken: float

    :param axes: display axes on the image, default True
    :type axes: bool
    :param gui: display GUI/interactive buttons, default True
    :type gui: bool
    :param frame: display axes or frame on the image, default True
    :type frame: bool
    :param plain: don't display axes, frame or GUI
    :type plain: bool
    :param cbar: add colorbar to image, default False
    :type cbar: bool
    :param title: title of figure in figure window
    :type title: str
    :param title: title of figure window
    :type title: str

    :param savefigname: if not None, save figure as savefigname (default eps)
    :type savefigname: str
    :param square: set aspect ratio so that pixels are square, default True
    :type square: bool
    :param fwidth: figure width in inches (need dpi for relative screen size?)
    :type fwidth: float
    :param fheight: figure height in inches
    :type fheight: float
    :param wide: set to full screen width, useful for displaying stereo pair
    :type wide: bool
    :param flatten: display image planes horizontally as adjacent images
    :type flatten: bool
    :param xydata: extent of the image in user units
    :type xydata: array_like(4), [xmin, xmax, ymin, ymax]

    :return fig: Matplotlib figure handle
    :rtype fig: figure handle
    :return ax: Matplotlib axes handle
    :rtype ax: axes handle

    - ``idisp(im)`` displays an image.

    Colormap is a string or else a ``Colormap`` subclass object.  Valid strings
    are any valid `matplotlib colormap names <https://matplotlib.org/tutorials/colors/colormaps.html>`_
    or

    =========  ===============================================
    Color      Meaning
    =========  ===============================================
    grey       zero is black, maximum value is white
    inverted   zero is white, maximum value is black
    signed     negative is red, 0 is white, positive is blue
    invsigned  negative is red, 0 is black, positive is blue
    random     random values
    =========  ===============================================

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox import iread, idisp
        >>> im, file = iread('monalisa.png')
        >>> idisp(im)

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

    # options yet to implement
    #    'axis': False,
    #    'here': False,
    #    'clickfunc': None,
    #    'print': None,
    #    'wide': False,
    #    'cscale': None,
    # handle list of images, or image with multiple frames

    # plain: hide GUI, frame and axes:
    if plain:
        gui = False
        axes = False
        frame = False

    # if we are running in a Jupyter notebook, print to matplotlib,
    # otherwise print to opencv imshow/new window. This is done because
    # cv.imshow does not play nicely with .ipynb
    if matplotlib:  # _isnotebook() and 
        ## display using matplotlib

        # aspect ratio:
        if not square:
            mpl.rcParams["image.aspect"] = 'auto'

        # hide interactive toolbar buttons (must be before figure creation)
        if not gui:
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

        # # experiment with addign buttons to the navigation bar
        # matplotlib.rcParams["toolbar"] = "toolmanager"
        # class LineTool(ToolToggleBase):

        #     def trigger(self, *args, **kwargs):
        #         print('hello from trigger')

        # tm = fig.canvas.manager.toolmanager
        # tm.add_tool('newtool', LineTool)
        # fig.canvas.manager.toolbar.add_tool(tm.get_tool("newtool"), "toolgroup")

        # get screen resolution:
        #swidth, sheight = pyautogui.size()  # pixels  TODO REPLACE THIS WITH STUFF FROM BDSIM
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

        # colormaps
        norm = None
        cmap = None
        if colormap == 'invert':
            cmap = 'Greys'
        elif colormap == 'signed':
            # signed color map, red is negative, blue is positive, zero is white

            cmap = 'RdBu'
            min = np.min(im)
            max = np.max(im)
            if abs(min) > abs(max):
                norm = mpl.colors.Normalize(min, abs(min / max) * max)
            else:
                norm = mpl.colors.Normalize(abs(max / min) * min, max)
        elif colormap == 'invsigned':
            # inverse signed color map, red is negative, blue is positive, zero is black
            cdict = {
                'red': [
                            (0, 1, 1),
                            (0.5, 0, 0),
                            (1, 0, 0)
                        ],
                'green': [
                            (0, 0, 0),
                            (1, 0, 0)
                        ],
                'blue': [
                            (0, 0, 0),
                            (0.5, 0, 0),
                            (1, 1, 1)
                        ]
            }
            cmap = mpl.colors.LinearSegmentedColormap('invsigned', cdict, ncolors)
            min = np.min(im)
            max = np.max(im)
            if abs(min) > abs(max):
                norm = mpl.colors.Normalize(min, abs(min / max) * max)
            else:
                norm = mpl.colors.Normalize(abs(max / min) * min, max)
        elif colormap == 'grey':
            cmap = 'gray'
            if darken is not None:
                norm = mpl.colors.Normalize(np.min(im), np.max(im) * darken)
        elif colormap == 'random':
            x = np.random.rand(ncolors, 3)
            cmap =  mpl.colors.LinearSegmentedColormap.from_list('my_colormap', x)
        else:
            cmap = colormap

        # choose default grey scale map for non-color image
        if cmap is None and len(im.shape) == 2:
            cmap = 'gray'

        if len(im.shape) == 3 and darken is not None:
            im = float_image(im) / darken

        if isinstance(cmap, str):
            cmap = cm.get_cmap(cmap, ncolors)

        # set black pixels to non-zero values, used to lighten a binary image
        if black != 0:
            norm = mpl.colors.Normalize(np.min(im), np.max(im))
            im = np.where(im == 0, black, im)

        # print('Colormap is ', cmap)

        # build up options for imshow
        options = {}
        if ynormal:
            options['origin'] = 'lower'

        if xydata is not None:
            options['extent'] = xydata

        # display the image
        if len(im.shape) == 3:
            # reverse the color planes if it's color
            cmapobj = ax.imshow(im[:,:,::-1], norm=norm, cmap=cmap, **options)
        else:
            cmapobj = ax.imshow(im, norm=norm, cmap=cmap, **options)

        # display the color bar
        if cbar:
            fig.colorbar(cmapobj, ax=ax)

        # set title of figure window
        fig.canvas.set_window_title(title_window)

        # set title in figure plot:
        # fig.suptitle(title)  # slightly different positioning
        ax.set_title(title)

        # hide image axes - by default also removes frame
        # back with ax.spines['top'].set_visible(True) ?
        if not axes:
            ax.axis('off')

        # no frame:
        if not frame:
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

        plt.show(block=block)
        return ax
    else:
        ## display using OpenCV

        cv.namedWindow(title, cv.WINDOW_AUTOSIZE)
        cv.imshow(title, im)  # make sure BGR format image

        if block:
            while True:
                k = cv.waitKey(delay=0)  # wait forever for keystroke
                if k == ord('q'):
                    cv.destroyWindow(title)
                    break
            

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
    :param kwargs: key word arguments 
    :return: image and filename
    :rtype: tuple or list of tuples, tuple is (image, filename) where image is
        a 2D, 3D or 4D NumPy array

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


    Extra options include:

        - 'uint8'         return an image with 8-bit unsigned integer pixels in
          the range 0 to 255

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

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox import iread, idisp
        >>> im, file = iread('flowers1.png')
        >>> idisp(im)

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

    if isinstance(filename, str) and (filename.startswith("http://") or filename.startswith("https://")):
        # reading from a URL

        resp = urllib.request.urlopen(filename)
        array = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv.imdecode(array, -1)
        image = convert(image, **kwargs)
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
                path = Path(__file__).parent.parent / "images" / path
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
                path = Path(__file__).parent.parent / "images" / path

                if not path.exists():
                    raise ValueError(f"file {filename} does not exist, and not found in supplied images")

            # read the image
            # TODO not sure the following will work on Windows
            image = cv.imread(path.as_posix())  # default read-in as BGR
            image = convert(image, **kwargs)
            if image is None:
                # TODO check ValueError
                raise ValueError(f"Could not read {filename}")

            print('reading image ', image.shape)
            return (image, str(path))

    elif islistof(filename, (str, Path)):
        # list of filenames or URLs
        # assume none of these are wildcards, TODO should check
        out = []
        for file in filename:
            out.append(iread(file, *kargs))
        return out
    else:
        raise ValueError(filename, 'invalid filename')

def convert(image, grey=False, dtype=None, gamma=None, reduce=None, roi=None):
    """
    Convert image

    :param image: input image
    :type image: ndarray(n,m) or ndarray(n,m,c)
    :param grey: convert to grey scale, default False
    :type grey: bool or 'ITU601' [default] or 'ITU709'
    :param dtype: a NumPy dtype string such as "uint8", "int16", "float32"
    :type dtype: str
    :param reduce: subsample image by this amount in u- and v-dimensions
    :type reduce: int
    :param roi: region of interest: [umin, umax, vmin, vmax]
    :type roi: array_like(4)
    :param gamma: gamma decoding, either the exponent of "sRGB"
    :type gamma: float or str
    :return: converted image
    :rtype: ndarray(n,m) or ndarray(n,m,c)
    """
    if grey and len(image.shape) > 2:
        image = colorconvert(image, 'rgb', 'grey')

    if dtype is not None:
        if 'int' in dtype:
            image = int_image(image, dtype)
        elif 'float' in dtype:
            image = float_image(image, dtype)

    if reduce is not None:
        n = int(reduce)
        if len(image.shape)  == 2:
            image = image[::n, ::n]
        else:
            image = image[::n, ::n, :]
    
    if roi is not None:
        umin, umax, vmin, vmax = roi
        if len(image.shape)  == 2:
            image = image[vmin:vmax, umin:umax]
        else:
            image = image[vmin:vmax, umin:umax, :]
    
    if gamma is not None:
        image = gamma_decode(image, gamma)

    return image

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
        scaled = image * np.float64(np.iinfo(intclass).max)
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
    Write NumPy array as image file

    :param filename: filename to write to
    :type filename: string
    :param kwargs: additional arguments, see ImwriteFlags
    :return: successful write
    :rtype: bool

    - ``iwrite(im, filename, **kwargs)`` writes ``im`` to ``filename`` with
      **kwargs currently for cv.imwrite() options.  The file type is taken 
      from the extension in ``filename``.

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox import iwrite
        >>> import numpy as np
        >>> image = np.zeros((20,20))  # 20x20 black image
        >>> iwrite(image, "black.png")

    .. notes::

        - a color image assumes the planes are in BGR order
        - supports 8-bit greyscale and color images
        - supports uint16 for PNG, JPEG 2000, and TIFF formats
        - supports float32

    :seealso: ``cv2.imwrite``
    """
    return cv.imwrite(filename, im, **kwargs)

if __name__ == "__main__":


    # filename = "~/code/machinevision-toolbox-python/machinevisiontoolbox/images/campus/*.png"

    # im = iread(filename)
    # print(im[0])

    im = np.zeros((100,100))
    # for i in range(100):
    #     im[:,i] = i - 40
    # idisp(im, matplotlib=True, title='default')
    # idisp(im, matplotlib=True, colormap='random', title='random')
    # idisp(im, matplotlib=True, colormap='grey', darken=4, title='dark')
    # idisp(im, matplotlib=True, colormap='signed', title='signed')
    # idisp(im, matplotlib=True, colormap='invsigned', title='invsigned')
    # idisp(im, matplotlib=True, colormap='grey', ncolors=4, title='solarize')
    # idisp(im, matplotlib=True, block=True, colormap='invert', title='grey')
   

    # for i in range(50):
    #     im[:,i] = 10
    # idisp(im, matplotlib=True, block=False, colormap='grey', black=5, title='black=5')
    # idisp(im, matplotlib=True, block=False, colormap='grey', ynormal=True, title='grey')
    # idisp(im, matplotlib=True, block=False, colormap='grey', xydata=np.r_[10,20,30,40], title='grey')
    # idisp(im, matplotlib=True, block=True, colormap='grey', ynormal=True, title='grey')

    im, file = iread('flowers1.png', dtype='float')
    idisp(im, block=True)