import numpy as np
import urllib.request
from pathlib import Path
import warnings

import cv2 as cv
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.backend_tools import ToolBase, ToolToggleBase
from machinevisiontoolbox.base.color import gamma_decode, colorspace_convert
from machinevisiontoolbox.base.types import float_image, int_image
from machinevisiontoolbox.base.data import mvtb_path_to_datafile


# for getting screen resolution
#import pyautogui  # requires pip install pyautogui
from spatialmath.base import islistof

def idisp(im,
          bgr=False,
          matplotlib=True,
          block=False,

          fig=None,
          ax=None,
          reuse=False,

          colormap=None,
          ncolors=None,

          black=0,
          darken=None,
          powernorm=False,
          gamma=None,
          vrange=None,

          badcolor=None,
          undercolor=None,
          overcolor=None,

          title='Machine Vision Toolbox for Python',
          grid=False,
          axes=True,
          gui=True,
          frame=True,
          plain=False,
          colorbar=False,

          square=True,
          width=None,
          height=None,
          flatten=False,
          ynormal=False,
          extent=None,

          savefigname=None,
          colororder="RGB",
          **kwargs):

    """
    Interactive image display tool

    :param im: image to display
    :type im: ndarray(H,W), ndarray(H,W,3)
    :param bgr: image is in BGR (native OpenCV color) order, defaults to False
    :type bgr: bool, optional
    :param matplotlib: plot using Matplotlib (True) or OpenCV (False), defaults to True
    :type matplotlib: bool, optional
    :param block: Matplotlib figure blocks until window closed, defaults to False
    :type block: bool, optional

    :param fig: Matplotlib figure handle to display image on, defaults to new figure
    :type fig: int, optional
    :param ax: Matplotlib axis object to plot on, defaults to new axis
    :type ax: axis object, optional
    :param reuse: plot into current figure, skips setup overhead, defaults to False
    :type reuse: bool, optional

    :param colormap: colormap name or Matplotlib colormap object
    :type colormap: str or matplotlib.colors.Colormap
    :param ncolors: number of colors in colormap
    :type ncolors: int, optional

    :param black: set black (zero) pixels to this value, default 0
    :type black: int or float
    :param darken: darken the image by scaling pixel values by this amount,
        if ``darken`` is True then darken by 0.5
    :type darken: float, bool, optional
    :param powernorm: Matplotlib power-law normalization
    :type powernorm: array_like(2), optional
    :param gamma: gamma correction applied before display
    :type gamma: float, optional
    :param vrange: minimum and maximum values for colormap, defaults to minimum 
        and maximum values from image data.
    :type vrange: array_like(2), optional

    :param badcolor: name of color to display when value is NaN
    :type badcolor: str, optional
    :param undercolor: name of color to display when value is less than colormap minimum
    :type undercolor: str, optional
    :param overcolor: name of color to display when value is less than colormap maximum
    :type overcolor: str, optional

    :param title: title of figure in figure window
    :type title: str, optional
    :param grid: display grid lines over image, default False
    :type gui: bool, optional
    :param axes: display axes on the image, default True
    :type axes: bool, optional
    :param gui: display GUI/interactive buttons, default True
    :type gui: bool, optional
    :param frame: display frame around the image, default True
    :type frame: bool, optional
    :param plain: don't display axes, frame or GUI
    :type plain: bool, optional
    :param colorbar: add colorbar to image, default False
    :type colorbar: bool, optional

    :param width: figure width in millimetres, defaults to Matplotlib default
    :type width: float, optional
    :param height: figure height in millimetres, defaults to Matplotlib default
    :type height: float, optional
    :param square: set aspect ratio so that pixels are square, default True
    :type square: bool, optional
    :param ynormal: y-axis increases upward, default False
    :type ynormal: bool, optional
    :param extent: extent of the image in user units [xmin, xmax, ymin, ymax]
    :type extent: array_like(4), optional

    :param savefigname: if not None, save figure as savefigname (default eps)
    :type savefigname: str, optional
    :param colororder: color order, used for interactive value picker only, defaults to "RGB"
    :type colororder: str
    :param kwargs: additional options passed through to :func:`matplotlib.pyplot.imshow`.

    :return: Matplotlib figure handle and axes handle
    :rtype: figure handle, axes handle

    Display a greyscale or color image using Matplotlib (if ``matplotlib`` is
    True) or OpenCV.  The Matplotlib display is interactive allowing zooming and
    pixel value picking.

    Greyscale images are displayed in indexed mode: the image pixel value is
    mapped through the color map to determine the display pixel value. The
    colormap is specified by a string or else a ``Colormap`` subclass object.
    Valid strings are any valid `matplotlib colormap names <https://matplotlib.org/tutorials/colors/colormaps.html>`_ or

    =========  ===============================================
    Colormap    Meaning
    =========  ===============================================
    grey       zero is black, maximum value is white
    inverted   zero is white, maximum value is black
    signed     negative is red, 0 is white, positive is blue
    invsigned  negative is red, 0 is black, positive is blue
    random     random values
    =========  ===============================================

    Example::

        >>> from machinevisiontoolbox import iread, idisp
        >>> im, file = iread('monalisa.png')
        >>> idisp(im)

    .. note::

        - For grey scale images the minimum and maximum image values are
          mapped to the first and last element of the color map, which by
          default ('grey') is the range black to white. To set your own
          scaling between displayed grey level and pixel value use the ``vrange``
          option.

    :references:
        - Robotics, Vision & Control for Python, Section 10.1, P. Corke, Springer 2023.
    
    :seealso: :func:`matplotlib.imshow` `cv2.imshow <https://docs.opencv.org/4.x/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563>`_
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



        # if flatten:
        #     # either make new subplots for each channel
        #     # or concatenate all into one large image and display
        #     # TODO can we make axes as a list?

        #     # for now, just concatenate:
        #     # first check how many channels:
        #     if im.ndim > 2:
        #         # create list of image channels
        #         imcl = [im[:, :, i] for i in range(im.shape[2])]
        #         # stack horizontally
        #         im = np.hstack(imcl)
        #     # else just plot the regular image - only one channel

        if len(plt.get_fignums()) == 0:
            # there are no figures, create one
            fig, ax = plt.subplots()  # fig creates a new window
        else:
            # there are existing figures

            if reuse:
                # attempt to reuse an axes, saves all the setup overhead
                if ax is None:
                    ax = plt.gca()
                for c in ax.get_children():
                    if isinstance(c, mpl.image.AxesImage):
                        c.set_data(im)
                        try:
                            plt.gcf().canvas.manager.set_window_title(title)  # for 3.4 onward
                        except:
                            pass

                        if isinstance(block, bool):
                            plt.show(block=block)
                        else:
                            plt.pause(block)
                        return

            if fig is not None:
                # make this figure the current one
                plt.figure(fig)
                
            if ax is None:
                fig, ax = plt.subplots()  # fig creates a new window
            

        # aspect ratio:
        if not square:
            mpl.rcParams["image.aspect"] = 'auto'

        # hide interactive toolbar buttons (must be before figure creation)
        if not gui:
            mpl.rcParams['toolbar'] = 'None'

        if darken is True:
            darken = 0.5

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


        # mpl_backend = mpl.get_backend()

        # if mpl_backend == 'Qt5Agg':
        #     from PyQt5 import QtWidgets
        #     app = QtWidgets.QApplication([])
        #     screen = app.primaryScreen()
        #     if screen.name is not None:
        #         print('  Screen: %s' % screen.name())
        #     size = screen.size()
        #     print('  Size: %d x %d' % (size.width(), size.height()))
        #     rect = screen.availableGeometry()
        #     print('  Available: %d x %d' % (rect.width(), rect.height()))
        #     sw = rect.width()
        #     sh = rect.height()
        #     #dpi = screen.physicalDotsPerInch()
        #     dpiscale = screen.devicePixelRatio() # is 2.0 for Mac laptop screen
        # elif mpl_backend == 'TkAgg':
        #     window = plt.get_current_fig_manager().window
        #     sw =  window.winfo_screenwidth()
        #     sh =  window.winfo_screenheight()
        #     print('  Size: %d x %d' % (sw, sh))
        # else:
        #     print('unknown backend, cant find width', mpl_backend)

        # dpi = None  # can make this an input option
        # if dpi is None:
        #     dpi = mpl.rcParams['figure.dpi']  # default is 100


        if width is not None:
            fig.set_figwidth(width / 25.4)  # inches

        if height is not None:
            fig.set_figheight(height / 25.4)  # inches

        ## Create the colormap and normalizer
        norm = None
        cmap = None
        if colormap == 'invert':
            cmap = 'Greys'
        elif colormap == 'signed':
            # signed color map, red is negative, blue is positive, zero is white
            cmap = 'bwr_r'  # blue -> white -> red
            min = np.min(im)
            max = np.max(im)

            # ensure min/max are symmetric about zero, so that zero is white
            if abs(max) >= abs(min):
                min = -max  # lgtm[py/multiple-definition]
            else:
                max = -min  # lgtm[py/multiple-definition]

            if powernorm:
                norm = mpl.colors.PowerNorm(gamma=0.45)
            else:
                # if abs(min) > abs(max):
                #     norm = mpl.colors.Normalize(vmin=min, vmax=abs(min / max) * max)
                # else:
                #     norm = mpl.colors.Normalize(vmin=abs(max / min) * min, vmax=max)
                norm = mpl.colors.CenteredNorm()
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
            if ncolors is None:
                cmap = mpl.colors.LinearSegmentedColormap('signed', cdict)
            else:
                cmap = mpl.colors.LinearSegmentedColormap('signed', cdict, N=ncolors)
            min = np.min(im)
            max = np.max(im)

            # ensure min/max are symmetric about zero, so that zero is black
            if abs(max) >= abs(min):
                min = -max
            else:
                max = -min

            if powernorm:
                norm = mpl.colors.PowerNorm(gamma=0.45)
            else:
                if abs(min) > abs(max):
                    norm = mpl.colors.Normalize(vmin=min, vmax=abs(min / max) * max)
                else:
                    norm = mpl.colors.Normalize(vmin=abs(max / min) * min, vmax=max)
        elif colormap == 'grey':
            cmap = 'gray'
        elif colormap == 'random':
            x = np.random.rand(256 if ncolors is None else ncolors, 3)
            cmap =  mpl.colors.LinearSegmentedColormap.from_list('my_colormap', x)
        else:
            cmap = colormap

        # choose default grey scale map for non-color image
        if cmap is None and len(im.shape) == 2:
            cmap = 'gray'

        # TODO not sure why exclusion for color, nor why float conversion
        if im.ndim == 3 and darken is not None:
            im = float_image(im) / darken

        if isinstance(cmap, str):
            #cmap = cm.get_cmap(cmap, lut=ncolors)
            cmap = mpl.colormaps[cmap]
            if ncolors is not None:
                cmap = cmap.resampled(ncolors)

        # handle values outside of range
        #
        #  - undercolor, below vmin
        #  - overcolor, above vmax
        #  - badcolor, nan, -inf, inf
        #
        # only works for greyscale image
        if im.ndim == 2:
            cmap = copy.copy(cmap)
            if undercolor is not None:
                cmap.set_under(color=undercolor)
            if overcolor is not None:
                cmap.set_over(color=overcolor)
            if badcolor is not None:
                cmap.set_bad(color=badcolor)
        # elif im.ndim == 3:
        #     if badcolor is not None:
        #         cmap.set_bad(color=badcolor)

        if black != 0:
            if np.issubdtype(im.dtype, np.floating):
                m = 1 - black
                c = black
                im = m * im + c
                norm = mpl.colors.Normalize(0, 1)
            elif np.issubdtype(im.dtype, np.bool_):
                norm = mpl.colors.Normalize(0, 1)
                ncolors = 2
            else:
                 max = np.iinfo(im.dtype).max
                 black = black * max
                 c = black
                 m = (max - c) / max
                 im = (m * im + c).astype(im.dtype)
                 norm = mpl.colors.Normalize(0, max)
            # else:
            #     # lift the displayed intensity of black pixels.
            #     # set the greyscale mapping [0,M] to [black,1]
            #     M = np.max(im)
            #     norm = mpl.colors.Normalize(-black * M / (1 - black), M)
        if darken:
            norm = mpl.colors.Normalize(np.min(im), np.max(im) / darken)

        if gamma:
            cmap.set_gamma(gamma)
            
        # print('Colormap is ', cmap)

        # build up options for imshow
        options = kwargs
        if ynormal:
            options['origin'] = 'lower'

        if extent is not None:
            options['extent'] = extent

        # display the image
        if len(im.shape) == 3:
            # reverse the color planes if it's color
            h = ax.imshow(im[:, :, ::-1] if bgr else im, norm=norm, cmap=cmap, **options)
        else:
            if norm is None:
                # exclude NaN values
                if vrange is None:
                    min = np.nanmin(im)
                    max = np.nanmax(im)
                else:
                    min, max = vrange

                if colorbar is not False and ncolors is not None:
                    #  colorbar requested with finite number of colors
                    # adjust range so that ticks fall in middle of color segment
                    min -= 0.5
                    max += 0.5
                norm = mpl.colors.Normalize(vmin=min, vmax=max)

            h = ax.imshow(im, norm=norm, cmap=cmap, **options)

        # display the color bar
        if colorbar is not False:
            cbargs = {}
            if ncolors:
                cbargs['ticks'] = range(ncolors + 1)

            if isinstance(colorbar, dict):
                # passed options have priority
                cbargs = {**cbargs, **colorbar}

            cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, **cbargs)


        # set title of figure window
        try:
            plt.gcf().canvas.manager.set_window_title(title)  # for 3.4 onward
        except:
            pass

        # set title in figure plot:
        # fig.suptitle(title)  # slightly different positioning
        # ax.set_title(title)
        

        # hide image axes - by default also removes frame
        # back with ax.spines['top'].set_visible(True) ?
        if not axes:
            ax.axis('off')

        if extent is None:
            ax.set_xlabel('u (pixels)')
            ax.set_ylabel('v (pixels)')
        if grid is not False:
            # if grid is True:
            #     ax.grid(True)
            # elif isinstance(grid, str):
            ax.grid(color='y', alpha=0.5, linewidth=0.5)


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

        # format the pixel value display
        def format_coord(u, v):
            u = int(u + 0.5)
            v = int(v + 0.5)
            
            try:
                if im.ndim == 2:
                    # monochrome image
                    x = im[v, u]
                    if isinstance(x, np.integer):
                        val = f"{x:d}"
                    elif isinstance(x, np.floating):
                        val = f"{x:.3f}"
                    elif isinstance(x, np.bool_):
                        val = f"{x}"

                    return f"({u}, {v}): {val} {x.dtype}"
                else:
                    # color image
                    x = im[v, u, :]
                    if np.issubdtype(x.dtype, np.integer):
                        val = [f"{_:d}" for _ in x]
                    elif np.issubdtype(x.dtype, np.floating):
                        val = [f"{_:.3f}" for _ in x]
                    val = "[" + ", ".join(val) + "]"

                    return f"({u}, {v}): {val} {colororder}, {x.dtype}"

            except IndexError:
                return ""

        ax.format_coord = format_coord

        # don't display data
        h.format_cursor_data = lambda x: ""

        if block is None:
            pass
        elif isinstance(block, bool):
            plt.show(block=block)
        else:
            plt.pause(block)
        return h
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
    r"""
    Read image from file or URL

    :param file: file name or URL
    :type file: str
    :param kwargs: key word arguments passed to :func:`convert`
    :return: image and filename
    :rtype: tuple (ndarray, str) or list of tuples, image is
        a 2D or 3D NumPy ndarray

    Loads an image from a file or URL, and returns the
    image as a NumPy array, as well as the absolute path name.  The
    image can by greyscale or color in any of the wide range of formats
    supported by the OpenCV ``imread`` function.

    If ``file`` is a list or contains a wildcard, the result will be a list of
    ``(image, path)`` tuples.  They will be sorted by path.

    Extra options can be passsed to perform datatype conversion, color to
    grey scale conversion, gamma correction, image decimation or region of
    interest windowing.  Details are given at :func:`convert`.

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox import iread
        >>> im, file = iread('flowers1.png')
        >>> im.shape
        >>> file[27:]
        >>> imdata = iread('campus/*.png')
        >>> len(imdata)
        >>> imdata[0][0].shape
        >>> imdata[1][0][27:]

    .. note::
        - A greyscale image is returned as an :math:`H \times W` array
        - A color image is returned as an :math:`H \times W \times P` array,
          typically :math:`P=3` for an RGB or BGR image or :math:`P=4` if there
          is an alpha plane, eg. RGBA.
        - wildcard lookup is done using pathlib ``Path.glob()`` and supports
          recursive globbing with the ``**`` pattern.

    :references:
        - Robotics, Vision & Control for Python, Section 10.1, P. Corke, Springer 2023.

    :seealso: :func:`convert` `cv2.imread <https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56>`_
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
                parts = path.parts
                path = mvtb_path_to_datafile('images', Path("").joinpath(*parts[:-1]))
                pathlist = list(path.glob(parts[-1]))
            
            if len(pathlist) == 0:
                raise ValueError("can't expand wildcard")

            # convert to strings
            pathlist = [str(p) for p in pathlist]

            images = []
            pathlist.sort()
            for p in pathlist:
                image = cv.imread(p, -1)  # default read-in as BGR
                images.append(convert(image, **kwargs))
            return images, pathlist

        else:
            # read single file
            path = mvtb_path_to_datafile('images', path)

            # read the image
            # TODO not sure the following will work on Windows
            image = cv.imread(path.as_posix(), -1)  # default read-in as BGR
            image = convert(image, **kwargs)
            if image is None:
                # TODO check ValueError
                raise ValueError(f"Could not read {filename}")

            return (image, str(path))

    else:
        raise ValueError(filename, 'invalid filename')


def convert(image, mono=False, gray=False, grey=False, rgb=True, dtype=None, gamma=None, alpha=False, reduce=None, roi=None, maxintval=None):
    """
    Convert image

    :param image: input image
    :type image: ndarray(H,W), ndarray(H,W,P)
    :param grey: convert to grey scale, default False
    :type grey: bool or 'ITU601' [default] or 'ITU709'
    :param gray: synonym for ``grey``
    :param dtype: a NumPy dtype string such as ``"uint8"``, ``"int16"``, ``"float32"`` or
        a NumPy type like ``np.uint8``.
    :type dtype: str
    :param gamma: gamma decoding, either the exponent of "sRGB"
    :type gamma: float or str
    :param alpha: allow alpha plane, default False
    :type alpha: bool
    :param reduce: subsample image by this amount in u- and v-dimensions
    :type reduce: int
    :param roi: region of interest: [umin, umax, vmin, vmax]
    :type roi: array_like(4)
    :param maxintval: maximum integer value to be used for scaling
    :type maxintval: int
    :return: converted image
    :rtype: ndarray(H,W) or ndarray(H,W,N)

    Peform common image conversion and transformations for NumPy images.

    ``dtype`` controls the resulting pixel data type.  If the image is a floating
    type the pixels are assumed to be in the range [0, 1] and are scaled into
    the range [0, ``maxintval``].  If ``maxintval`` is not given it is taken
    as the maximum value of ``dtype``.

    Gamma decoding specified by ``gamma`` can be applied to float or int
    type images.
    """
    if grey:
        warnings.warn("grey option to Image.Read/iread is deprecated, use mono instead",
        DeprecationWarning)
    if gray:
        warnings.warn("gray option to Image.Read/iread is deprecated, use mono instead",
        DeprecationWarning)
    mono = mono or gray or grey
    if mono and len(image.shape) > 2:
        image = colorspace_convert(image, 'rgb', 'grey')

    if image.ndim == 3 and image.shape[2] >= 3:
        if not alpha:
            image = image[:, :, :3]
        if rgb:
            image = np.copy(image[:, :, ::-1])  # put in RGB color order

    dtype_alias = {
        'int':    'uint8',
        'float':  'float32',
        'double': 'float64',
        'half':   'float16',
    }

    if dtype is not None:
        # default types
        try:
            dtype = dtype_alias[dtype]
        except KeyError:
            pass

        if 'int' in dtype:
            image = int_image(image, intclass=dtype, maxintval=maxintval)
        elif 'float' in dtype:
            image = float_image(image, floatclass=dtype, maxintval=maxintval)
        else:
            raise ValueError(f"unknown dtype: {dtype}")

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


def iwrite(im, filename, bgr=False, **kwargs):
    """          
    Write NumPy array to an image file

    :param filename: filename to write to
    :type filename: string
    :param bgr: image is in BGR (native OpenCV color) order, defaults to False
    :type bgr: bool, optional
    :param kwargs: additional arguments, see ImwriteFlags
    :return: successful write
    :rtype: bool

    Writes the image ``im`` to ``filename`` using cv.imwrite(), with **kwargs
    passed as options.  The file type is taken from the extension in
    ``filename``.

    Example::

        >>> from machinevisiontoolbox import iwrite
        >>> import numpy as np
        >>> image = np.zeros((20,20))  # 20x20 black image
        >>> iwrite(image, "black.png")

    .. note::
        - supports 8-bit greyscale and color images
        - supports uint16 for PNG, JPEG 2000, and TIFF formats
        - supports float32
        - image must be in BGR or RGB format

    :seealso: :func:`iread` `cv2.imwrite <https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce>`_ 
    """
    if bgr:
        # write BGR format image directly
        return cv.imwrite(filename, im, **kwargs)
    elif im.ndim > 2:
        # otherwise, if color, flip the planes
        return cv.imwrite(filename, im[:, :, ::-1], **kwargs)

def pickpoints(self, n=None, matplotlib=True):
    """
    Pick points on image

    :param n: number of points to input, defaults to infinite number
    :type n: int, optional
    :param matplotlib: plot using Matplotlib (True) or OpenCV (False), defaults to True
    :type matplotlib: bool, optional
    :return: Picked points, one per column
    :rtype: ndarray(2,n)

    Allow the user to select points on the displayed image.  
    
    For Matplotlib, a marker is displayed at each point selected with a
    left-click.  Points can be removed by a right-click, like an undo function.
    middle-click or Enter-key will terminate the entry process.  If ``n`` is
    given, the entry process terminates after ``n`` points are entered, but can
    terminated prematurely as above.

    .. note:: Picked coordinates have floating point values.

    :seealso: :func:`idisp`
    """

    if matplotlib:
        points = plt.ginput(n)
        return np.c_[points].T
    else:

        def click_event(event, x, y, flags, params): 

            # checking for left mouse clicks 
            if event == cv2.EVENT_LBUTTONDOWN: 
        
                # displaying the coordinates 
                # on the Shell 
                print(x, ' ', y) 

        cv.setMouseCallback('image', click_event) 
    
        # wait for a key to be pressed to exit 
        cv.waitKey(0)
            
if __name__ == "__main__":


    # filename = "~/code/machinevision-toolbox-python/machinevisiontoolbox/images/campus/*.png"

    # im = iread(filename)
    # print(im[0])

    im = np.zeros((4,4))
    im[:,0] = 0.2
    im[:,3] = -0.4
    idisp(im, colormap='signed', block=True)
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

    im, file = iread('street.png', dtype='float')
    idisp(im, title='Boo!', block=True)
