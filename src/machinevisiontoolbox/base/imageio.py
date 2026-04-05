"""
Low-level image reading from files, URLs, and byte buffers.
"""

from __future__ import annotations

import copy
import time
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Callable

import cv2 as cv
import matplotlib as mpl
import matplotlib.cm as cm

# pyright: reportMissingImports=false
import numpy as np

from machinevisiontoolbox.base.color import colorspace_convert, gamma_decode
from machinevisiontoolbox.base.data import mvtb_path_to_datafile
from machinevisiontoolbox.base.types import float_image, int_image

try:
    import pyclip

    _pyclip = True
except ImportError:
    _pyclip = False

# for getting screen resolution
# import pyautogui  # requires pip install pyautogui

from spatialmath.base import islistof

__last_windowname: str | None = None
__last_window_number: int = 0


def _ensure_mpl_backend() -> None:
    """Select a matplotlib backend appropriate for the current environment.

    Called lazily inside :func:`idisp` so that notebook cells can configure the
    backend (e.g. ``%matplotlib widget``) **before** importing
    :mod:`machinevisiontoolbox`, and so that the choice is deferred to runtime
    rather than fixed at module import time.

    Priority:
    - If a backend is already initialised (pyplot has been imported), do nothing.
    - Pyodide (JupyterLite): use the IPython inline backend (no DOM access needed).
    - Google Colab: use inline.
    - Generic IPython/Jupyter session where no backend is set yet: leave it to
      IPython's ``%matplotlib`` magic or the user's config.
    """
    import sys

    if "matplotlib.pyplot" in sys.modules:
        # pyplot already imported → backend is already chosen, don't interfere
        return

    import matplotlib as mpl

    if mpl.is_interactive() or mpl.get_backend() != mpl.rcParams["backend"]:  # type: ignore[attr-defined]
        # already configured
        return

    current = dict.__getitem__(mpl.rcParams, "backend")  # avoids auto-resolve
    if current != "agg" and current.lower() != "agg":
        # user/environment has already set something explicit
        return

    if sys.platform == "emscripten":
        # Pyodide / JupyterLite — use IPython inline (Agg-based, no DOM)
        mpl.use("module://matplotlib_inline.backend_inline")
        return

    try:
        import google.colab  # noqa: F401

        mpl.use("module://matplotlib_inline.backend_inline")
        return
    except ImportError:
        pass


def _plt():
    """Return :mod:`matplotlib.pyplot`, importing it on first call.

    Importing pyplot triggers backend initialisation, so we call
    :func:`_ensure_mpl_backend` first to give the environment a chance to
    configure the backend before pyplot locks it in.
    """
    import sys

    if "matplotlib.pyplot" not in sys.modules:
        _ensure_mpl_backend()
    import matplotlib.pyplot as plt

    return plt


def idisp(
    im: np.ndarray,
    colororder: str = "RGB",
    matplotlib: bool = True,
    block: bool | float | None = None,
    fps: float | None = None,
    fig: Any = None,
    ax: Any = None,
    reuse: bool = False,
    colormap: str | Any = None,
    ncolors: int | None = None,
    black: int | float = 0,
    darken: float | bool | None = None,
    powernorm: bool = False,
    gamma: float | None = None,
    vrange: tuple[float, float] | list[float] | None = None,
    badcolor: str | None = None,
    undercolor: str | None = None,
    overcolor: str | None = None,
    title: str | None = None,
    grid: bool = False,
    axes: bool = True,
    gui: bool = True,
    frame: bool = True,
    plain: bool = False,
    colorbar: bool | dict[str, Any] = False,
    square: bool = True,
    width: float | None = None,
    height: float | None = None,
    flatten: bool = False,
    ynormal: bool = False,
    extent: tuple[float, float, float, float] | list[float] | None = None,
    coordformat: Callable[[float, float], str] | None = None,
    savefigname: str | None = None,
    **kwargs: Any,
) -> Any:
    r"""
    Interactive image display tool

    :param im: image to display
    :type im: ndarray(H,W), ndarray(H,W,3)
    :param colororder: color order, defaults to "RGB"
    :type colororder: str
    :param matplotlib: plot using Matplotlib (True) or OpenCV (False), defaults to True
    :type matplotlib: bool, optional
    :param block: after display, Matplotlib blocks until window closed or for the specified time period, defaults to False
    :type block: bool, float, optional
    :param fps: frames per second, Matplotlib blocks for 1/fps seconds after display, defaults to None
    :type fps: float, optional

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
    :param flatten: flatten singleton dimensions before display, defaults to False
    :type flatten: bool, optional
    :param ynormal: y-axis increases upward, default False
    :type ynormal: bool, optional
    :param extent: extent of the image in user units [xmin, xmax, ymin, ymax]
    :type extent: array_like(4), optional

    :param coordformat: format coordinates and pixel values for the figure window toolbar
    :type coordformat: callable returning string
    :param savefigname: if not None, save figure as savefigname (default eps)
    :type savefigname: str, optional
    :param kwargs: additional options passed through to :func:`matplotlib.pyplot.imshow`.

    :return: Matplotlib figure handle and axes handle
    :rtype: figure handle, axes handle

    Display a greyscale or color image interactively using OpenCV or Matplotlib (if ``matplotlib``
    is True, default).

    **OpenCV**

    OpenCV is used to display the mage if ``matplotlib`` is False.  The display is
    provided by the "HighGUI" module, and is not interactive.  Overlay graphics cannot
    be displayed, but the ``draw_xxx()`` functions can be used to draw on the image
    prior to display.

    Example::

        >>> from machinevisiontoolbox import iread, idisp
        >>> im, file = iread("monalisa.png", matplotlib=False)
        >>> idisp(im)

    Most of the options apply to the Matplotlib case.

    **Matplotlib**

    The Matplotlib display is interactive allowing zooming and pixel value picking.
    Other graphics can be superimposed on the image using the Matplotlib plotting
    functions ``plot_xxx()``.

    Example::

        >>> from machinevisiontoolbox import iread, idisp
        >>> im, file = iread("monalisa.png")
        >>> idisp(im)

    .. plot::

        from machinevisiontoolbox import iread, idisp
        im, file = iread("monalisa.png")
        idisp(im)

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

    .. note::  For grey scale images the minimum and maximum image values are
        mapped to the first and last element of the color map, which by
        default ('grey') is the range black to white. To set your own
        scaling between displayed grey level and pixel value use the ``vrange``
        option.

    The argument ``block`` has the following functions

    ===================  ==================================================================================
    ``block``            Action after display
    ===================  ==================================================================================
    ``False`` (default)  Call ``plt.show(block=False)``, don't block
    ``True``             Call ``plt.show(block=True)``, block
    ``None``             Don't call ``plt.show()``, don't block, in Jupyter subsequents plots will be added
    ``t`` (numeric)      Block for set time, calls ``plt.pause(t)``. See also ``fps`` option.
    ===================  ==================================================================================

    The ``coordformat`` function is called with (u, v) coordinates and the image is in the variable ``im`` which
    is in scope, but not passed, and is an ndarray(H,W) or ndarray(H,W,P).

    Certain keys can be pressed while the image is displayed:

        - ``c`` will copy the current pixel coordinates to the paste buffer as U,V
        - ``C`` will append the current pixel coordinates to the paste buffer, with a
          newline separator.
        - ``v`` will copy the current pixel coordinates to the paste buffer as U,V,X
        - ``V`` will append the current pixel coordinates and pixel value to the paste
          buffer as U,V,X, with a newline separator.
        - ``x`` will clear the paste buffer
        - ``q`` will quite the window

    .. note::

        - The displayed pixel value is a scalar (int or float), or a tuple of scalars for
          multiplane/color images.
        - The string never ends with a newline, newlines only separate values, eg.
          U1,V1\nU2,V2
        - This functionality requires that ``pyclip`` is installed.

    :references:
        - |RVC3|, Section 10.1.

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

    if not isinstance(block, bool) and isinstance(block, (int, float)):
        fps = 1.0 / block
        block = None

    # if we are running in a Jupyter notebook, print to matplotlib,
    # otherwise print to opencv imshow/new window. This is done because
    # cv.imshow does not play nicely with .ipynb
    if matplotlib:  # _isnotebook() and
        ## display using matplotlib
        plt = _plt()

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

        if title is None:
            title = "Machine Vision Toolbox for Python"

        if len(plt.get_fignums()) == 0:
            # there are no figures, create one
            fig, ax = plt.subplots()  # fig creates a new window
        else:
            # there are existing figures

            if reuse:
                # attempt to reuse an axes, saves all the setup overhead
                if ax is None:
                    ax = plt.gca()

                # look for an image in the axes to update, if there is one
                updated = False
                for c in ax.get_children():
                    if isinstance(c, mpl.image.AxesImage):  # type: ignore
                        c.set_data(im)  # type: ignore
                        updated = True
                if updated:
                    set_window_title(title)

                    if fps is not None:
                        # print("pausing", 1.0 / fps)
                        plt.pause(1.0 / fps)

                    if block is not None:
                        plt.show(block=block)

                    return

            if fig is not None:
                # make this figure the current one
                plt.figure(fig)

            if ax is None:
                fig, ax = plt.subplots()  # fig creates a new window

        if fig is None:
            fig = ax.figure

        # aspect ratio:
        if not square:
            mpl.rcParams["image.aspect"] = "auto"

        # hide interactive toolbar buttons (must be before figure creation)
        if not gui:
            mpl.rcParams["toolbar"] = "None"

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
        # swidth, sheight = pyautogui.size()  # pixels  TODO REPLACE THIS WITH STUFF FROM BDSIM

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
        #     print('unknown backend, can't find width', mpl_backend)

        # dpi = None  # can make this an input option
        # if dpi is None:
        #     dpi = mpl.rcParams['figure.dpi']  # default is 100

        if width is not None:
            fig.set_figwidth(width / 25.4)  # type: ignore  # inches

        if height is not None:
            fig.set_figheight(height / 25.4)  # type: ignore  # inches

        ## Create the colormap and normalizer
        norm = None
        cmap = None
        if colormap == "invert":
            cmap = "Greys"
        elif colormap == "signed":
            # signed color map, red is negative, blue is positive, zero is white
            cmap = "bwr_r"  # blue -> white -> red
            min = np.min(im)
            max = np.max(im)

            # ensure min/max are symmetric about zero, so that zero is white
            if abs(max) >= abs(min):
                min = -max  # lgtm[py/multiple-definition]
            else:
                max = -min  # lgtm[py/multiple-definition]

            if powernorm:
                norm = mpl.colors.PowerNorm(gamma=0.45)  # type: ignore
            else:
                # if abs(min) > abs(max):
                #     norm = mpl.colors.Normalize(vmin=min, vmax=abs(min / max) * max)
                # else:
                #     norm = mpl.colors.Normalize(vmin=abs(max / min) * min, vmax=max)
                norm = mpl.colors.CenteredNorm()  # type: ignore
        elif colormap == "invsigned":
            # inverse signed color map, red is negative, blue is positive, zero is black
            cdict = {
                "red": [(0, 1, 1), (0.5, 0, 0), (1, 0, 0)],
                "green": [(0, 0, 0), (1, 0, 0)],
                "blue": [(0, 0, 0), (0.5, 0, 0), (1, 1, 1)],
            }
            if ncolors is None:
                cmap = mpl.colors.LinearSegmentedColormap("signed", cdict)  # type: ignore
            else:
                cmap = mpl.colors.LinearSegmentedColormap("signed", cdict, N=ncolors)  # type: ignore
            min = np.min(im)
            max = np.max(im)

            # ensure min/max are symmetric about zero, so that zero is black
            if abs(max) >= abs(min):
                min = -max
            else:
                max = -min

            if powernorm:
                norm = mpl.colors.PowerNorm(gamma=0.45)  # type: ignore
            else:
                if abs(min) > abs(max):
                    norm = mpl.colors.Normalize(vmin=min, vmax=abs(min / max) * max)  # type: ignore
                else:
                    norm = mpl.colors.Normalize(vmin=abs(max / min) * min, vmax=max)  # type: ignore
        elif colormap == "grey":
            cmap = "gray"
        elif colormap == "random":
            x = np.random.rand(256 if ncolors is None else ncolors, 3)
            cmap = mpl.colors.LinearSegmentedColormap.from_list("my_colormap", x)  # type: ignore
        else:
            cmap = colormap

        # choose default grey scale map for non-color image
        if cmap is None and len(im.shape) == 2:
            cmap = "gray"

        # TODO not sure why exclusion for color, nor why float conversion
        if im.ndim == 3 and darken is not None:
            im = float_image(im) / darken  # type: ignore

        if isinstance(cmap, str):
            # cmap = cm.get_cmap(cmap, lut=ncolors)
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
            cmap = copy.copy(cmap)  # type: ignore
            if undercolor is not None:
                cmap.set_under(color=undercolor)  # type: ignore
            if overcolor is not None:
                cmap.set_over(color=overcolor)  # type: ignore
            if badcolor is not None:
                cmap.set_bad(color=badcolor)  # type: ignore
        # elif im.ndim == 3:
        #     if badcolor is not None:
        #         cmap.set_bad(color=badcolor)

        if black != 0:
            if np.issubdtype(im.dtype, np.floating):
                m = 1 - black
                c = black
                im = m * im + c
                norm = mpl.colors.Normalize(0, 1)  # type: ignore
            elif np.issubdtype(im.dtype, bool):
                norm = mpl.colors.Normalize(0, 1)  # type: ignore
                ncolors = 2
            else:
                max = np.iinfo(im.dtype).max
                black = black * max
                c = black
                m = (max - c) / max
                im = (m * im + c).astype(im.dtype)
                norm = mpl.colors.Normalize(0, max)  # type: ignore
            # else:
            #     # lift the displayed intensity of black pixels.
            #     # set the greyscale mapping [0,M] to [black,1]
            #     M = np.max(im)
            #     norm = mpl.colors.Normalize(-black * M / (1 - black), M)
        if darken:
            norm = mpl.colors.Normalize(np.min(im), np.max(im) / darken)  # type: ignore

        if gamma:
            cmap.set_gamma(gamma)  # type: ignore

        # print('Colormap is ', cmap)

        # build up options for imshow
        options = kwargs
        if ynormal:
            options["origin"] = "lower"

        if extent is not None:
            options["extent"] = extent

        # display the image
        if len(im.shape) == 3:
            # reverse the color planes if it's color
            if colororder not in ("RGB", "BGR"):
                raise ValueError("unknown colororder ", colororder)
            if colororder == "BGR":
                im = im[:, :, ::-1]
            h = ax.imshow(im, norm=norm, cmap=cmap, **options)
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
                norm = mpl.colors.Normalize(vmin=min, vmax=max)  # type: ignore

            h = ax.imshow(im, norm=norm, cmap=cmap, **options)

        # display the color bar
        if colorbar is not False:
            cbargs = {}
            if ncolors:
                cbargs["ticks"] = range(ncolors + 1)

            if isinstance(colorbar, dict):
                # passed options have priority
                cbargs = {**cbargs, **colorbar}

            cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, **cbargs)  # type: ignore

        # set title of figure window
        set_window_title(title)

        # set title in figure plot:
        # fig.suptitle(title)  # slightly different positioning
        # ax.set_title(title)

        # hide image axes - by default also removes frame
        # back with ax.spines['top'].set_visible(True) ?
        if not axes:
            ax.axis("off")

        if extent is None:
            ax.set_xlabel("u (pixels)")
            ax.set_ylabel("v (pixels)")
        if grid is not False:
            # if grid is True:
            #     ax.grid(True)
            # elif isinstance(grid, str):
            ax.grid(color="y", alpha=0.5, linewidth=0.5)

        # no frame:
        if not frame:
            # NOTE: for frame tweaking, see matplotlib.spines
            # https://matplotlib.org/3.3.2/api/spines_api.html
            # note: can set spines linewidth:
            # ax.spines['top'].set_linewidth(2.0)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)

        if savefigname is not None:
            # TODO check valid savefigname
            # set default save file format
            mpl.rcParams["savefig.format"] = "eps"
            plt.draw()

            # savefig must be called before plt.show
            # after plt.show(), a new fig is automatically created
            plt.savefig(savefigname)

        # format the pixel value display
        def format_coord(u: float, v: float) -> str:
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
                    elif isinstance(x, (np.bool_, bool)):
                        val = f"{x}"
                    else:
                        print(f"unknown pixel type {type(x)}")

                    return f"({u}, {v}): {val}"  # type: ignore
                else:
                    # color image
                    x = im[v, u, :]  # in RGB order
                    if colororder == "BGR":
                        x = x[::-1]
                    if np.issubdtype(x.dtype, np.integer):
                        val = [f"{_:d}" for _ in x]
                    elif np.issubdtype(x.dtype, np.floating):
                        val = [f"{_:.3f}" for _ in x]
                    val = "[" + ", ".join(val) + "]"

                    return f"({u}, {v}): {val} {colororder}, {x.dtype}"

            except IndexError:
                return ""

        def key_press(event: Any) -> None:
            if not _pyclip:
                return

            if event.inaxes is not None:
                u = int(event.xdata + 0.5)
                v = int(event.ydata + 0.5)
            else:
                return

            if event.key == "c":
                # print pixel value at mouse click
                pyclip.copy(f"{u},{v}")
            elif event.key == "C":
                # print pixel value at mouse click
                prev = pyclip.paste()
                if isinstance(prev, bytes):  # type: ignore
                    prev = prev.decode()  # type: ignore
                pyclip.copy(f"{prev}\n{u},{v}")
            elif event.key == "x":
                pyclip.copy("")
            elif event.key == "v":
                val = im[v, u, ...]
                pyclip.copy(f"{u},{v},{tuple(val)}")
            elif event.key == "V":
                val = im[v, u, ...]
                prev = pyclip.paste()
                if isinstance(prev, bytes):  # type: ignore
                    prev = prev.decode()  # type: ignore
                pyclip.copy(f"{prev}\n{u},{v},{tuple(val)}")

        fig.canvas.mpl_connect("key_press_event", key_press)  # type: ignore

        if coordformat is None:
            ax.format_coord = format_coord  # type: ignore
        else:
            ax.format_coord = coordformat  # type: ignore

        # don't display data
        h.format_cursor_data = lambda data: ""  # type: ignore

        if fps is not None:
            # print("pausing", 1.0 / fps)
            plt.pause(1.0 / fps)

        if block is not None:
            plt.show(block=block)

        return h
    else:
        ## display using OpenCV
        global __last_window_number

        if not reuse and title is None:
            # create a unique window name for each call
            title = "idisp." + str(__last_window_number)
            __last_window_number += 1

        # At this point title is guaranteed to be a string
        assert title is not None
        cv.namedWindow(title, cv.WINDOW_AUTOSIZE)
        cv.imshow(title, im)  # make sure BGR format image
        cv.waitKey(1)

        if fps is not None:
            # wait one frame time
            cv.waitKey(round(1000.0 / fps))

        if block is True:
            while True:
                k = cv.waitKey(delay=0)  # wait forever for keystroke
                if k == ord("q"):
                    assert title is not None
                    cv.destroyWindow(title)
                    cv.waitKey(1)
                    break

        # TODO fig, ax equivalent for OpenCV? how to print/plot to the same
        # window/set of axes?


def set_window_title(title: str) -> None:
    try:
        _plt().gcf().canvas.manager.set_window_title(title)  # type: ignore  # for 3.4 onward
    except:
        pass


def cv_destroy_window(title: str | None = None, block: bool = True) -> None:
    if title == "all":
        cv.destroyAllWindows()
    else:
        if block:
            while True:
                k = cv.waitKey(delay=0)  # wait forever for keystroke
                if k == ord("q"):
                    break
        if title is not None:
            cv.destroyWindow(title)
    cv.waitKey(1)  # run the event loop


def _isnotebook() -> bool:
    """
    Determine if code is being run from a Jupyter notebook

    ``_isnotebook`` is True if running Jupyter notebook, else False

    :references:

        - https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-
          is-executed-in-the-ipython-notebook/39662359#39662359
    """
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell in ("Shell", "ZMQInteractiveShell"):
            return True  # Jupyter notebook or qtconsole or CoLab
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def iread(
    filename: str | Path, *args: Any, **kwargs: Any
) -> tuple[np.ndarray, str] | tuple[list[np.ndarray], list[str]]:
    r"""
    Read image from file or URL

    :param filename: file name or URL
    :type filename: str
    :param args: reserved positional arguments
    :type args: tuple
    :param kwargs: key word arguments passed to :func:`convert`
    :type kwargs: dict
    :return: image and filename
    :rtype: tuple (ndarray, str) or list of tuples, image is
        a 2D or 3D NumPy ndarray

    Loads an image from a file or URL, and returns the
    image as a NumPy array, as well as the absolute path name.

    If the path is not absolute it is first searched for relative
    to the current directory, and if not found, it is searched for in
    the ``images`` folder of the
    `mvtb-data package <https://github.com/petercorke/machinevision-toolbox-python/tree/master/mvtb-data>`_.

    If ``file`` is a list or contains a wildcard, the result will be a list of
    ``(image, path)`` tuples.  They will be sorted by path.

    The image can by greyscale or color in any of the wide range of formats
    supported by the OpenCV ``imread`` function. Extra options can be passed to
    perform datatype conversion, color to grey scale conversion, gamma
    correction, image decimation or region of interest windowing.  Details are
    given at :func:`convert`.

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
        - |RVC3|, Section 10.1.

    :seealso: :func:`convert` `cv2.imread <https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56>`_
    """

    if isinstance(filename, str) and (
        filename.startswith("http://") or filename.startswith("https://")
    ):
        # reading from a URL
        import ssl

        ctx = ssl.create_default_context()
        req = urllib.request.Request(
            filename,
            headers={"User-Agent": "machinevisiontoolbox-python/1.0"},
        )
        try:
            resp = urllib.request.urlopen(req, context=ctx)
        except urllib.error.HTTPError as e:
            raise ValueError(f"HTTP {e.code} fetching {filename}") from e
        except urllib.error.URLError as e:
            raise ValueError(f"Could not fetch {filename}: {e.reason}") from e

        if resp.status != 200:
            raise ValueError(f"HTTP {resp.status} fetching {filename}")
        array = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv.imdecode(array, -1)
        if image is not None:
            image = convert(image, **kwargs)
            return (image, filename)
        else:
            raise ValueError(
                f"Could not decode image data from {filename} "
                f"(content-type: {resp.headers.get('Content-Type', 'unknown')})"
            )

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
                path = mvtb_path_to_datafile("images", Path("").joinpath(*parts[:-1]))
                pathlist = list(path.glob(parts[-1]))  # type: ignore

            if len(pathlist) == 0:
                raise ValueError("can't expand wildcard")

            # convert to strings
            pathlist = [str(p) for p in pathlist]

            images = []
            pathlist.sort()
            for p in pathlist:
                image = cv.imdecode(
                    np.fromfile(Path(p).as_posix(), dtype=np.uint8), cv.IMREAD_UNCHANGED
                )
                # image = cv.imread(p, -1)  # default read-in as BGR
                if image is None:
                    raise ValueError(f"Could not decode image: {p}")
                images.append(convert(image, **kwargs))
            return images, pathlist

        else:
            # read single file
            path = mvtb_path_to_datafile("images", path)

            # read the image
            # TODO not sure the following will work on Windows
            image = cv.imdecode(
                np.fromfile(path.as_posix(), dtype=np.uint8), cv.IMREAD_UNCHANGED  # type: ignore
            )
            image = cv.imread(path.as_posix(), -1)  # type: ignore  # default read-in as BGR
            if image is None:
                # TODO check ValueError
                raise ValueError(f"Could not read {filename}")
            image = convert(image, **kwargs)
            return (image, str(path))

    else:
        raise ValueError(filename, "invalid filename")


def convert(
    image: np.ndarray,
    mono: bool = False,
    gray: bool = False,
    grey: bool = False,
    rgb: bool = True,
    dtype: str | type | None = None,
    gamma: float | str | None = None,
    alpha: bool = False,
    reduce: int | None = None,
    roi: tuple[int, int, int, int] | list[int] | None = None,
    maxintval: int | None = None,
    copy: bool = False,
) -> np.ndarray:
    """
    Convert image

    :param image: input image
    :type image: ndarray(H,W), ndarray(H,W,P)
    :param mono: convert to grey scale, synonym for ``grey``
    :type mono: bool, optional
    :param grey: convert to grey scale, default False
    :type grey: bool or 'ITU601' [default] or 'ITU709'
    :param gray: synonym for ``grey``
    :param dtype: a NumPy dtype string such as ``"uint8"``, ``"int16"``, ``"float32"`` or
        a NumPy type like ``np.uint8``.
    :type dtype: str
    :param rgb: force color image to RGB order, otherwise BGR
    :type rgb: bool, optional
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
    :param copy: guarantee that returned image is a copy of the input image, defaults to False
    :type copy: bool
    :return: converted image
    :rtype: ndarray(H,W) or ndarray(H,W,N)

    Perform common image conversion and transformations for NumPy images.

    ``dtype`` controls the resulting pixel data type.  If the image is a floating
    type the pixels are assumed to be in the range [0, 1] and are scaled into
    the range [0, ``maxintval``].  If ``maxintval`` is not given it is taken
    as the maximum value of ``dtype``.

    Gamma decoding specified by ``gamma`` can be applied to float or int
    type images.

    The ``grey``/``gray`` option converts a color image to greyscale and is ignored if the
    image is already greyscale.  Note that this conversions requires knowledge of the color
    plane order specified by ``colororder``.  The planes are inverted by ``invertplanes`` before
    this step.
    """
    if grey:
        warnings.warn(
            "grey option to Image.Read/iread is deprecated, use mono instead",
            DeprecationWarning,
        )
    if gray:
        warnings.warn(
            "gray option to Image.Read/iread is deprecated, use mono instead",
            DeprecationWarning,
        )
    image_original = image

    if image.ndim == 3 and image.shape[2] in (3, 4):
        # is color image RGB, RGBA, BGR, BGRA
        if not alpha:
            # optionally remove the alpha plane
            image = image[:, :, :3]
        if rgb:
            # optionally invert the color planes
            image = np.copy(image[:, :, ::-1])  # reverse the planes
            # np.copy() is required to make torchvision happy
            colororder = "RGB"
        else:
            colororder = "BGR"

    mono = mono or gray or grey
    if mono and len(image.shape) == 3:
        image = colorspace_convert(image, colororder, "grey")  # type: ignore

    dtype_alias = {
        "int": "uint8",
        "float": "float32",
        "double": "float64",
        "half": "float16",
    }

    if dtype is not None:
        # default types
        try:
            dtype = dtype_alias[dtype]  # type: ignore
        except KeyError:
            pass

        if "int" in str(dtype):  # type: ignore
            image = int_image(image, intclass=dtype, maxintval=maxintval)  # type: ignore
        elif "float" in str(dtype):  # type: ignore
            image = float_image(image, floatclass=dtype, maxintval=maxintval)  # type: ignore
        else:
            raise ValueError(f"unknown dtype: {dtype}")

    if reduce is not None:
        n = int(reduce)
        if len(image.shape) == 2:
            image = image[::n, ::n]
        else:
            image = image[::n, ::n, :]

    if roi is not None:
        umin, umax, vmin, vmax = roi
        if len(image.shape) == 2:
            image = image[vmin:vmax, umin:umax]
        else:
            image = image[vmin:vmax, umin:umax, :]

    if gamma is not None:
        image = gamma_decode(image, gamma)  # type: ignore

    if image is not image_original and copy:
        image = image.copy()

    return image


def iwrite(
    im: np.ndarray, filename: str, colororder: str = "RGB", **kwargs: Any
) -> bool:
    """
    Write NumPy array to an image file

    :param im: image to write
    :type im: ndarray(H,W), ndarray(H,W,P)
    :param filename: filename to write to
    :type filename: string
    :param colororder: color order, defaults to "RGB"
    :type colororder: str
    :param kwargs: additional arguments, see ImwriteFlags
    :return: successful write
    :rtype: bool

    Writes the image ``im`` to ``filename`` using cv.imwrite(), passing any
    keyword arguments as options.  The file type is taken from the extension in
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
    if im.ndim > 2 and colororder == "RGB":
        # put image into OpenCV BGR order for writing
        return cv.imwrite(filename, im[:, :, ::-1], **kwargs)
    else:
        return cv.imwrite(filename, im, **kwargs)


def pickpoints(
    self: Any, n: int | None = None, matplotlib: bool = True
) -> np.ndarray | None:
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
        points = plt.ginput(n)  # type: ignore
        return np.c_[points].T
    else:

        def click_event(event: Any, x: int, y: int, flags: Any, params: Any) -> None:
            # checking for left mouse clicks
            if event == cv.EVENT_LBUTTONDOWN:  # type: ignore
                # displaying the coordinates
                # on the Shell
                print(x, " ", y)

        cv.setMouseCallback("image", click_event)

        # wait for a key to be pressed to exit
        cv.waitKey(0)


if __name__ == "__main__":  # type: ignore
    from machinevisiontoolbox import *  # type: ignore
    from machinevisiontoolbox.base import *  # type: ignore

    images = ImageCollection("seq/*.png")  # type: ignore

    im, file = iread("street.png", dtype="float")
    idisp(im, matplotlib=False)  # type: ignore
    idisp(im, matplotlib=False)  # type: ignore

    for image in images:
        image.disp(
            title="sequence", reuse=True, fps=5, matplotlib=False
        )  # do some operation

    # type 'q' in the image animation window to close it
    cv_destroy_window("sequence", block=True)

    # filename = "~/code/machinevision-toolbox-python/machinevisiontoolbox/images/campus/*.png"

    # im = iread(filename)
    # print(im[0])

    # from machinevisiontoolbox import VideoCamera, Image
    # from machinevisiontoolbox.base import idisp
    # import numpy as np

    # c = VideoCamera(0, rgb=False)
    # x = c.grab()

    # x.disp(block=True)
    # x.disp(block=True)
    # idisp(x.image, colororder="BGR", block=True)

    # a = np.eye(2)
    # b = np.dstack([a, a*0, a*0])
    # # idisp(b)
    # # idisp(b, colororder="RGB", title="RGB")
    # # idisp(b, colororder="BGR", title="BGR", block=True)

    # x = Image(b)
    # y = Image(b, colororder="BGR")
    # x.disp()
    # y.disp(block=True)

    # im = np.zeros((4,4))
    # im[:,0] = 0.2
    # im[:,3] = -0.4
    # idisp(im, colormap='signed', block=True)
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

    # im, file = iread("street.png", dtype="float")
    # idisp(im, title="Boo!", block=True)
