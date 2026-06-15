"""
Image file and URL reading/writing, and image-sequence iteration.
"""

import fnmatch
import os
import os.path
import sys
import zipfile
from collections import namedtuple
from pathlib import Path
from typing import TYPE_CHECKING, Any, SupportsFloat, cast

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import PIL.Image
from PIL.ExifTags import TAGS
import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
import numpy as np
import scipy as sp
from scipy import interpolate
from spatialmath.base import argcheck, getvector

if TYPE_CHECKING:
    from machinevisiontoolbox._image_typing import _ImageBase

from machinevisiontoolbox.base import (
    colorname,
    float_image,
    int_image,
    mvtb_path_to_datafile,
)
from machinevisiontoolbox.base.imageio import (
    _pick_imagefile,
    convert,
    idisp,
    iread,
    safe_plt_show,
    iwrite,
)
from machinevisiontoolbox.mvtb_types import Dtype

# from numpy.lib.arraysetops import isin


class ImageIOMixin(_ImageBase if TYPE_CHECKING else object):
    # ======================= image i/io ================================== #

    @classmethod
    def Read(
        cls,
        filename: str | Path | None = None,
        alpha: bool = False,
        rgb: bool = True,
        **kwargs: Any,
    ) -> Self:
        """
        Read image from file

        :param filename: image file name, if not given a file browser is opened
        :type filename: str or Path, optional
        :param alpha: include alpha plane if present, defaults to False
        :type alpha: bool, optional
        :param rgb: force color image to be in RGB order, defaults to True
        :type rgb: bool, optional
        :param kwargs: options applied to image frames, see :func:`~machinevisiontoolbox.base.imageio.convert`
        :raises ValueError: file not found, or no file selected in browser
        :raises ImportError: ``filename`` omitted but tkinter is unavailable
        :return: image from file
        :rtype: :class:`Image`

        Load monochrome or color image from file, many common formats are
        supported.  A number of transformations can be applied to the image
        loaded from the file before it is returned.

        If ``filename`` is omitted a native file-browser dialogue is opened,
        initially showing the ``images`` folder of the ``mvtb_data`` package.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> Image.Read('street.png')
            >>> Image.Read('flowers1.png')
            >>> Image.Read('flowers1.png', grey=True)
            >>> Image.Read('flowers1.png', dtype='float16')
            >>> Image.Read('flowers1.png', reduce=4)
            >>> Image.Read('flowers1.png', gamma='sRGB') # linear tristimulus values

        .. note::  If the path is not absolute it is first searched for relative
            to the current directory, and if not found, it is searched for in
            the ``images`` folder of the `mvtb_data package <https://github.com/petercorke/machinevision-toolbox-python/tree/master/packages/mvtb-data>`_.

        .. warning:: In an ipython environment, eg. ``mvtbtool``, the file-browser dialogue may not work.  In this case, specify the filename as a string.

        :seealso: :func:`~machinevisiontoolbox.base.imageio.iread` :func:`~machinevisiontoolbox.base.imageio.convert`  `cv2.imread <https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56>`_
        """
        if filename is None:
            filename = _pick_imagefile()
        elif not isinstance(filename, (str, Path)):
            raise ValueError("expecting a string or path")

        # read the image
        data = iread(filename, rgb=rgb, alpha=alpha, **kwargs)

        # result is a tuple(image, filename) or a list of tuples

        colororder = None
        if isinstance(data, tuple):
            # singleton image, make it a list
            image_arr, name = data
            assert isinstance(image_arr, np.ndarray)
            if not alpha and image_arr.ndim == 3 and image_arr.shape[2] == 4:
                image_arr = image_arr[:, :, :3]
            if image_arr.ndim > 2:
                colororder = "RGB" if rgb else "BGR"
            if alpha and image_arr.shape[2] == 4:
                colororder += "A"
            return cls(
                image_arr, name=name, colororder=colororder
            )  # OpenCV file read order)
        elif isinstance(data, list):
            raise ValueError("wildcard read not supported, use FileCollection instead")

        raise TypeError("unexpected result type from iread")

    def disp(self, title: str | bool | None = None, **kwargs: Any) -> Any:
        """
        Display image

        :param title: named of window, defaults to image ``name``
        :type title: bool
        :param kwargs: options, see :func:`~machinevisiontoolbox.base.imageio.idisp`

        Display an image using either Matplotlib (default) or OpenCV.  There are
        many display options.  The Matplotlib display is interactive and supports
        zooming, panning, and pixel value inspection.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.disp();

        .. plot::

            from machinevisiontoolbox import Image
            Image.Read('flowers1.png').disp()

        :seealso: :func:`~machinevisiontoolbox.base.imageio.idisp`
        """
        display_title: str | None
        if title is False:
            display_title = None
        elif isinstance(title, str):
            display_title = title
        elif title is None and self.name is not None:
            _, display_title = os.path.split(self.name)
        else:
            display_title = None

        if self.domain is not None:
            # left right top bottom
            kwargs["extent"] = [
                self.domain[0][0],
                self.domain[0][-1],
                self.domain[1][-1],
                self.domain[1][0],
            ]

        if self.colororder_str is not None:
            colororder = self.colororder_str.replace(":", "")
        else:
            colororder = None
        return idisp(
            self._A,
            title=display_title,
            colororder=colororder,
            **kwargs,
        )

    def write(
        self, filename: str | Path, dtype: Dtype = "uint8", **kwargs: Any
    ) -> bool:
        """
        Write image to file

        :param filename: filename to write to
        :type filename: str
        :param dtype: data type to convert to, before writing
        :type dtype: str or NumPy dtype
        :param kwargs: options for :func:`~machinevisiontoolbox.base.iwrite`

        Write image data to a file.  The file format is taken from the extension
        of the filename.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.write('flowers.jpg')

        :seealso: :func:`~machinevisiontoolbox.base.iwrite` `cv2.imwrite <https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce>`_
        """

        # cv2.imwrite can only save 8-bit single channel or 3-channel BGR images
        # with several specific exceptions
        # https://docs.opencv.org/4.4.0/d4/da8/group__imgcodecs.html
        # #gabbc7ef1aa2edfaa87772f1202d67e0ce
        # TODO imwrite has many quality/type flags

        ret = iwrite(self._A.astype(dtype), str(filename), **kwargs)

        return ret

    def metadata(self, key: str | None = None) -> Any:
        """
        Get image EXIF metadata

        :param key: metadata key
        :type key: str, optional
        :return: image metadata
        :rtype: dict, int, float, str

        Get image metadata from EXIF headers.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('church.jpg')
            >>> meta = img.metadata()  # get all metadata as a dict
            >>> len(meta)
            >>> meta
            >>> meta['Make']
            >>> img.metadata('Make')  # get specific metadata item

        .. note::  Metadata items will be converted, where possible, to int or float values.

        """
        if self.name is None:
            return None

        image = PIL.Image.open(self.name)
        exif = {}

        # iterate over the EXIF tags
        meta = image.getexif()
        if meta is None:
            return  # no metadata

        for tag, value in meta.items():
            if tag in TAGS:
                # map tag number to tag name
                exif[TAGS[tag]] = value

        if key is None:
            return exif
        else:
            val = exif[key]
            if isinstance(val, str):
                # attempt to turn string into int or float
                try:
                    return int(val)
                except ValueError:
                    pass
                try:
                    return float(val)
                except ValueError:
                    pass
                return val
            if isinstance(val, int):
                return val
            elif isinstance(val, tuple) and len(val) == 2:
                # old versions of PIL return (numerator, denominator)
                return val[0] / val[1]
            else:
                # float values are actually type PIL.TiffImagePlugin.IFDRational
                if hasattr(val, "__float__"):
                    return float(cast(SupportsFloat, val))
                return val

    def showpixels(
        self,
        fmt: str | None = None,
        ax: Any = None,
        cmap: str = None,
        grid: int | None = 2,
        badcolor: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Display image with pixel values

        :param fmt: format string for displaying pixel values, defaults to None
        :type fmt: str, optional
        :param grid: line width for the grid, defaults to 2. Set to None for no grid.
        :type grid: int | None, optional
        :param ax: The axes on which to display the window, defaults to current axes
        :type ax: axes, optional
        :param cmap: colormap for displaying pixel values, defaults to custom midnight
            colormap
        :type cmap: str, optional
        :param kwargs: additional keyword arguments passed to text annotations
        :rtype: ``Window`` instance

        Display a monochrome image with the pixel values overlaid. This is suitable for
        small images, of order 10x10, used for pedagogical purposes.

        Text parameters can be set using ``kwargs`` with defaults:

        - ``color``: "white"
        - ``fontsize``: 6

        To improve visibility a black stroke is applied to the text, which can be
        configured using additional parameters in ``kwargs``:

        - ``linewidth``: 2, for the path effect stroke
        - ``foreground``: "black", for the path effect stroke
        - ``alpha``: 0.7, for the path effect stroke

        By default, a grid is drawn to show the pixel boundaries, this can
        be turned off by setting ``grid=None``.

        .. plot::

            from machinevisiontoolbox import Image

            img = Image.Random(size=10)
            img.showpixels(color="grey")

        .. plot::

            from machinevisiontoolbox import Image

            img = Image.Random(size=10)
            img.showpixels(color="yellow")

        .. plot::

            from machinevisiontoolbox import Image

            img = Image.Random(size=10)
            img.showpixels(color="yellow", linewidth=0)

        :meth:`showwindow` can be used to superimpose a colored window on the image,
        and to get the pixel values in that window. This can be used to demonstrate
        window operations, such as convolution and morphological operations.

        :seealso: :meth:`print` :meth:`disp`
        """

        if ax is None:
            ax = plt.gca()

        if self.isint:
            if fmt is None:
                fmt = "{:d}"
        elif self.isfloat:
            if fmt is None:
                fmt = "{:.2f}"
        elif self.isbool:
            if fmt is None:
                fmt = "{:d}"
        else:
            raise ValueError("unsupported image type")

        image = self._A

        # Define a gradient from nearly black-blue to white
        if cmap is None:
            # default colormap is a custom "midnight" colormap that is better for
            # displaying pixel values than the default Matplotlib colormaps,
            # dark blue/black to white
            colors = ["#000814", "#001d3d", "#ffffff"]
            cmap = LinearSegmentedColormap.from_list("midnight", colors)

        # display the image
        if ax is None:
            plt.clf()
            ax = plt.gca()

        if badcolor is not None:
            cmap.set_bad(color=badcolor)
        ax.imshow(image, cmap=cmap, zorder=1)

        # label the cells with the pixel values
        for v in range(self.height):
            for u in range(self.width):
                if np.isnan(image[v, u]):
                    continue

                txt_args = {
                    k: kwargs[k]
                    for k in kwargs
                    if k
                    not in ("linewidth", "foreground", "alpha", "color", "fontsize")
                }
                txt = ax.text(
                    u,
                    v,
                    fmt.format(image[v, u]),
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=kwargs.get("color", "white"),
                    fontsize=kwargs.get("fontsize", 6),
                    zorder=5,
                    **txt_args,
                )
                txt.set_path_effects(
                    [
                        path_effects.withStroke(
                            linewidth=kwargs.get("linewidth", 2),
                            foreground=kwargs.get("foreground", "black"),
                            alpha=kwargs.get("alpha", 0.7),
                        )
                    ]
                )

        # Add wide white grid lines to show the pixel boundaries
        if grid is not None:
            for x in np.arange(0.5, self.width - 0.5, 1):
                ax.axvline(x, color="w", linewidth=grid, zorder=2)
            for y in np.arange(0.5, self.height - 0.5, 1):
                ax.axhline(y, color="w", linewidth=grid, zorder=2)

        ax.set_xlabel("u (pixels)")
        ax.set_ylabel("v (pixels)")
        plt.draw()
        return ax

    def showwindow(self, h, ax, **kwargs: Any) -> Any:
        """Superimpose a colored window on the image.

        :param h: The half-width of the window, so the total window size is
            (2h+1)x(2h+1)
        :type h: int
        :param ax: The axes on which to display the window
        :return: The window instance
        :rtype: Any

        Overlay a colored window on the image, and return a ``Window`` instance that can
        be used to manipulate the window and get the pixel values in the window.  This
        is intended for pedagogical purposes, to demonstrate window operations such as
        convolution and morphological operations::

            >> window = img.showwindow(h=1, ax=ax)  # create a window of size 3x3

        The window is a square of size (2h+1)x(2h+1) pixels. It can be centered on the
        pixel at (u,v) by calling::

          >> W = window.move(u, v)

        which also returns the pixel values in the window as 2D Numpy array ``W``. The
        window's visibility can be turned on and off using
        ``window.visible(True|False)``.

        Multiple windows, with different appearances, can be created on the same image,
        and each can be manipulated independently.  The window's appearance can be
        configured by passing additional keyword arguments in ``kwargs`` when the window
        is created, with defaults:

        - ``facecolor``: "#E69F00"
        - ``alpha``: 0.8
        - ``fill``: True
        - ``edgecolor``: "#FFFFFFFF"
        - ``linewidth``: 4
        - ``linestyle``: "--"

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Random(size=10)
            >>> img.showpixels()
            >>> img.showwindow(h=1)  # with 3x3 window
            >>> W = window.move(2,3) # position window at (2,3)
            >>> print(W) # print the pixel values in the window

        .. plot::

            from machinevisiontoolbox import Image
            img = Image.Random(size=10)
            img.showpixels()
            img.showwindow(h=1)  # with 3x3 window
            W = window.move(2,3) # position window at (2,3)


        .. plot::

            from machinevisiontoolbox import Image

            img = Image.Random(size=10)
            img.showpixels()
            window = img.showwindow(h=1)
            window.move(2, 3)

        :seealso: :meth:`showpixels`


        """

        class Window:
            def __init__(self, h=1, ax=None, image=None, **kwargs: Any) -> None:
                self.h = h
                self.image = image
                self.args = kwargs

                w = 2 * h + 1
                patch = mpatches.Rectangle((0, 0), w, w, zorder=4, **kwargs)
                if ax is None:
                    ax = plt.gca()

                ax.add_patch(patch)
                self.patch = patch

            def visible(self, visible: bool = True) -> None:
                self.patch.set_visible(visible)

            def move(
                self,
                u: int | float,
                v: int | float,
                wincolor: Any = None,
                winalpha: float = 0.9,
            ) -> None:

                self.patch.set_x(u - self.h - 0.5)
                self.patch.set_y(v - self.h - 0.5)

                if self.image is None:
                    return None
                else:
                    return self.image._A[
                        v - self.h : v + self.h + 1, u - self.h : u + self.h + 1
                    ]

        default_args = {
            "facecolor": "#E69F00",
            "alpha": 0.8,
            "fill": True,
            "edgecolor": "#FFFFFFFF",
            "linewidth": 4,
            "linestyle": "--",
        }

        return Window(h=h, ax=ax, image=self, **(default_args | kwargs))

    def anaglyph(self, right: Self, colors: str = "rc", disp: int = 0) -> Self:
        """
        Convert stereo images to an anaglyph image

        :param right: right image
        :type right: Image instance
        :param colors: lens colors (left, right), defaults to 'rc'
        :type colors: str, optional
        :param disp: disparity, defaults to 0
        :type disp: int, optional
        :raises ValueErrror: images are not the same size
        :return: anaglyph image
        :rtype: :class:`Image`

        Returns an anaglyph image which combines the two images of a stereo pair
        by coding them in two different colors.  By default the left image is
        red, and the right image is cyan.

        ``colors`` describes the lens color coding as a string with 2 letters,
        the first for left, the second for right, and each is one of:

            ====  ========
            code  color
            ====  ========
            'r'   red
            'g'   green
            'b'   green
            'c'   cyan
            'm'   magenta
            ====  ========

        If ``disp`` is positive the disparity is increased by shifting the
        ``right`` image to the right. If negative disparity is reduced by
        shifting the ``right`` image to the left.  These adjustments are
        achieved by trimming the images.  Use this option to make the images
        more natural/comfortable to view, useful if the images were captured
        with a stereo baseline significantly different to the human eye
        separation (typically 65mm).

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> left = Image.Read("rocks2-l.png", reduce=2)
            >>> right = Image.Read("rocks2-r.png", reduce=2)
            >>> left.anaglyph(right).disp()

        .. plot::

            from machinevisiontoolbox import Image
            left = Image.Read("rocks2-l.png", reduce=2)
            right = Image.Read("rocks2-r.png", reduce=2)
            left.anaglyph(right).disp()

        :references:
            - |RVC3|, Section 14.4.

        :seealso: :meth:`stdisp` :meth:`Overlay`
        """
        if self.size != right.size:
            raise ValueError("images must be same size")
        width, height = self.size

        # ensure the images are greyscale
        left = self.mono()
        right = right.mono()

        if disp > 0:
            left = left.trim(right=disp)
            right = right.trim(left=disp)

        elif disp < 0:
            disp = -disp
            left = left.trim(left=disp)
            right = right.trim(right=disp)

        colordict = {
            "r": (1, 0, 0),
            "g": (0, 1, 0),
            "b": (0, 0, 1),
            "c": (0, 1, 1),
            "m": (1, 0, 1),
            "o": (1, 1, 0),
        }
        return left.colorize(colordict[colors[0]]) + right.colorize(
            colordict[colors[1]]
        )

    def stdisp(self, right: Self, interactive: bool = True) -> None:
        """
        Interactive display of stereo image pair

        :param right: right image
        :type right: :class:`Image`
        :param interactive: if True, clicking in the left image shows the disparity at the corresponding point in the right image, defaults to True
        :type interactive: bool, optional
        :raises ValueError: images are not the same size
        :return: None

        The left and right images are displayed, stacked horizontally.  Clicking
        in the left-hand image sets a crosshair cursor in the right-hand
        image.  Clicking the corresponding point in the right-hand image
        will display the disparity at the top of the right-hand image.

        Example:

            .. code-block:: python

                from machinevisiontoolbox import Image
                left = Image.Read("rocks2-l.png", reduce=2)
                right = Image.Read("rocks2-r.png", reduce=2)
                left.stdisp(right)


        .. plot::

            from machinevisiontoolbox import Image
            left = Image.Read("rocks2-l.png", reduce=2)
            right = Image.Read("rocks2-r.png", reduce=2)
            left.stdisp(right, interactive=False)

        .. note:: The images are assumed to be epipolar aligned.

        :references:
            - |RVC3|, Section 14.4.

        :seealso: :meth:`anaglyph`
        """

        class Cursor:
            """
            A cross hair cursor.
            """

            def __init__(self, ax, ax2):
                self.ax = ax
                self.ax2 = ax2
                self.horizontal_line = ax.axhline(color="k", lw=0.8)
                self.horizontal_line2 = ax2.axhline(color="k", lw=0.8)
                self.vertical_line = ax.axvline(color="k", lw=0.8)
                self.vertical_line2 = ax2.axvline(color="k", lw=0.8)
                self.vertical_line3 = ax2.axvline(color="k", lw=0.8, ls="--")
                self.leftclicked = False
                self.x_left = None

                # text location in axes coordinates
                self.text = self.ax2.text(
                    0.05, 0.95, "", transform=ax2.transAxes, backgroundcolor="w"
                )

            def set_cross_hair_visible(self, visible: bool) -> bool:
                need_redraw = self.horizontal_line.get_visible() != visible
                self.horizontal_line.set_visible(visible)
                self.horizontal_line2.set_visible(visible)

                self.vertical_line.set_visible(visible)
                self.vertical_line2.set_visible(visible)
                self.vertical_line3.set_visible(visible)

                self.text.set_visible(visible)
                return need_redraw

            def on_mouse_move(self, event: Any) -> None:
                if event.inaxes == self.ax2 and self.leftclicked:
                    x, y = event.xdata, event.ydata
                    # update the line positions
                    self.vertical_line3.set_xdata(x)
                    self.text.set_text("d={:.2f}".format(self.x_left - x))
                    self.ax2.figure.canvas.draw()
                # if  event.inaxes:
                #     need_redraw = self.set_cross_hair_visible(False)
                #     if need_redraw:
                #         self.ax.figure.canvas.draw()
                # else:
                #     self.set_cross_hair_visible(True)
                #     x, y = event.xdata, event.ydata
                #     # update the line positions
                #     self.horizontal_line.set_ydata(y)
                #     self.vertical_line.set_xdata(x)
                #     # self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
                #     self.ax.figure.canvas.draw()

            def on_click(self, event: Any) -> None:
                # if not event.inaxes:
                #     need_redraw = self.set_cross_hair_visible(False)
                #     if need_redraw:
                #         self.ax.figure.canvas.draw()
                # else:
                if event.inaxes == self.ax:
                    self.set_cross_hair_visible(True)
                    x, y = event.xdata, event.ydata
                    # update the line positions
                    self.horizontal_line.set_ydata(y)
                    self.vertical_line.set_xdata(x)
                    self.horizontal_line2.set_ydata(y)
                    self.vertical_line2.set_xdata(x)
                    self.ax.figure.canvas.draw()
                    self.leftclicked = True
                    self.x_left = x

        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

        self.disp(ax=ax1, grid=True)
        right.disp(ax=ax2, grid=True)

        if interactive:
            cursor = Cursor(ax1, ax2)
            fig.canvas.mpl_connect("motion_notify_event", cursor.on_mouse_move)
            fig.canvas.mpl_connect("button_press_event", cursor.on_click)
            safe_plt_show(block=True)


if __name__ == "__main__":
    from pathlib import Path

    import pytest

    pytest.main(
        [str(Path(__file__).parent.parent.parent / "tests" / "test_image_io.py"), "-v"]
    )
