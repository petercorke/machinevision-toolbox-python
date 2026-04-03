"""
Core Image class providing pixel storage, arithmetic operators, and colour-plane access.
"""

# pyright: reportMissingImports=false
from __future__ import annotations

import os
import os.path
import urllib
import warnings
from collections.abc import Iterator, Sequence
from math import nan
from pathlib import Path

import cv2 as cv
import numpy as np
import spatialmath.base as smb
from spatialmath import Polygon2
from spatialmath.base import islistof, isscalar

# from numpy.lib.arraysetops import isin
from machinevisiontoolbox.base import (
    draw_box,
    draw_circle,
    draw_labelbox,
    draw_line,
    draw_point,
    draw_text,
    float_image,
    int_image,
)
from machinevisiontoolbox.base.imageio import convert, idisp, iread, iwrite
from machinevisiontoolbox.ImageBlobs import ImageBlobsMixin
from machinevisiontoolbox.ImageColor import ImageColorMixin
from machinevisiontoolbox.ImageConstants import ImageConstantsMixin
from machinevisiontoolbox.ImageFiducials import ImageFiducialsMixin
from machinevisiontoolbox.ImageIO import ImageIOMixin
from machinevisiontoolbox.ImageLineFeatures import ImageLineFeaturesMixin
from machinevisiontoolbox.ImageMorph import ImageMorphMixin
from machinevisiontoolbox.ImageMultiview import ImageMultiviewMixin
from machinevisiontoolbox.ImagePointFeatures import ImagePointFeaturesMixin
from machinevisiontoolbox.ImageTorch import ImageTorchMixin
from machinevisiontoolbox.ImageProcessing import ImageProcessingMixin
from machinevisiontoolbox.ImageRegionFeatures import ImageRegionFeaturesMixin
from machinevisiontoolbox.ImageReshape import ImageReshapeMixin
from machinevisiontoolbox.ImageSpatial import ImageSpatialMixin
from machinevisiontoolbox.Kernel import Kernel
from machinevisiontoolbox.ImageWholeFeatures import ImageWholeFeaturesMixin
from machinevisiontoolbox.mvtb_types import *

# import spatialmath.base.argcheck as argcheck


"""
This class encapsulates a Numpy array containing the pixel values.  The object
supports arithmetic using overloaded operators, as well a large number of methods.
"""


class Image(
    ImageIOMixin,
    ImageConstantsMixin,
    ImageProcessingMixin,
    ImageMorphMixin,
    ImageSpatialMixin,
    ImageColorMixin,
    ImageReshapeMixin,
    ImageBlobsMixin,
    ImageWholeFeaturesMixin,
    ImageRegionFeaturesMixin,
    ImageFiducialsMixin,
    ImageLineFeaturesMixin,
    ImagePointFeaturesMixin,
    ImageMultiviewMixin,
    ImageTorchMixin,
):
    def __init__(
        self,
        image: Image | np.ndarray | None = None,
        colororder: str | dict | None = None,
        copy: bool = False,
        size: tuple | list | None = None,
        dtype: Dtype | bool | None = None,
        name: str | None = None,
        id: int | None = None,
        domain=None,
        binary: bool = False,
        **kwargs,
    ) -> None:
        """
        Create an Image instance

        :param image: image data
        :type image: array_like(H,W), :class:`Image`
        :param colororder: order of color channels
        :type colororder: str, dict
        :param copy: copy the image data, defaults to False
        :type copy: bool, optional
        :param size: new size for the image, defaults to None
        :type size: tuple, optional
        :param dtype: data type for image, defaults to same type as ``image``
        :type dtype: str or NumPy dtype, optional
        :param name: name of image, defaults to None
        :type name: str, optional
        :param id: numeric id of image, typically a sequence number, defaults to None
        :type id: int, optional
        :param domain: domain of image, defaults to None
        :type domain: array_like(W), array_like(H), optional
        :param binary: create binary image, non-zero values are set to True, defaults to False
        :type binary: bool, optional
        :raises TypeError: unknown type passed to constructor

        Create a new :class:`Image` instance which contains pixel values as well as
        information about image size, datatype, color planes and domain.  The pixel
        data is stored in a NumPy array, encapsulated by the object.

        An image is considered to be a two-dimensional (width x height) grid of pixel
        values, that lie within one or more "color" planes.

        The image data can be specified by:

        - a NumPy 2D or 3D array for a greyscale or color image respectively.  For
          the latter case, the last index represents the color plane::

            Image(nd.zeros((100, 200)))
            Image(nd.zeros((100, 200, 4)), colororder="WXYZ")

        - a lists of lists of pixel values, each inner list must have the same
          number of elements (columns)::

                Image([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        - a range of class methods that serve as constructors for common image types such as
          :meth:`Zeros`, :meth:`Constant`, :meth:`Random` and :meth:`String`, or
          read an image from a file :meth:`Read`.

        - an existing :class:`Image` instance.

        **Pixel datatype**

        If ``dtype`` is a string or a NumPy dtype, the pixel data type is set to that type.

        If ``dtype`` is ``True`` the pixel data type is inherited from the input if it is a NumPy array.

        If ``dtype`` is not given and:

          * the input is a NumPy array,the image data type is determined by the dtype of the array:

            - if floating point, then the Image inherits the dtype of the array
            - if integer, then the Image is assigned the smallest integer dtype that can
              contain the values in the array.  If the minimum value is negative, a signed
              integer type is used; otherwise an unsigned type is used.

          * the input is a list of lists then the image data type is:

            - float32 if the list contains any floating point values, otherwise
            - the smallest signed or unsigned int that can represent its
              value span.

        An image can have boolean pixel values which are stored as ``uint8`` values.
        When used in a numerical expression, its values will be cast to integer values
        of 0 or 1 representing False and True respectively.

        **Color planes**

        Images can have multiple planes, typically three (representing the
        primary colors red, green and blue) but *any number* is possible. In
        the underlying Numpy array, these planes are identified by an integer
        plane index (the last dimension of the 3D array).

        Rather than rely on a limiting convention such as planes being in the order
        RGB or BGR, the :class:`Image` contains a dictionary that maps the name of a color plane to
        its index value.  The color plane order can be specified as a dict or a string, eg::

            Image(img, colororder="RGB")
            Image(img, colororder="red:green:blue:alpha")
            Image(img, colororder=dict("R"=2, "G"=1, "B"=0))

        Image planes can be referenced by their index or by their name, eg.::

            img.plane(0)
            img.plane("alpha")

        :seealso: :meth:`colororder` :meth:`colororder_str`

        **Image domain**

        An :class:`Image` has a width and height in units of pixels, but for
        some applications it is useful to specify the pixel coordinates in other
        units, perhaps metres, or latitude/longitude angles, or for a spherical image
        as azimuth and colatitude angles.  The domain is specified by two 1D arrays
        that map the pixel coordinate to the domain variable.

        **Binary images**

        If ``binary`` is True, the image is converted to a binary image, where zero valued
        pixels are set to False and all other values are set to True.  To create an
        image where pixels have integer values of 0 and 1 use the ``dtype`` option::

            Image([[0, 3], [4, 0]])  # pixel values are 0, 3, 4, 0
            Image([[0, 3], [4, 0]], binary=True)  # pixel values are: False, True, True, False
            Image([[0, 3], [4, 0]], binary=True, dtype="uint8")  # pixel values are 0, 1, 1, 0

        **Reshaping**

        Frequently we need to create an image from 1D data, for example::

            1: Y0 Y1 Y2 ...  # (N,)
            2: R0 G0 B0 R1 G1 B1 ... # (3N,)

        Or a 2D array with one or more rows::

            3: Y0 Y1 Y2 ... # (1, N)

            4: R0 G0 B0 R1 G1 B1 ... # (1,3N)

            5: R0 R1 R2 ... # (3,N)
               G0 G1 G2 ...
               B0 B1 B2 ...

        Or a 2D array with one or more columns::

            6: Y0 # (N,1)
               Y1
               Y2
                .
                .

            7: R0 # (3N,1)
               G0
               B0
               R1
               .
               .

            8: R0 G0 B0 # (N,3)
               R1 G1 B1
               R2 G2 B2
                .
                .

        The ``size`` option can be used to reshape the data to the specified size.  The number
        of planes is determined from any of: ``colororder``, the third element of
        ``size``, or the number of row/columns in the data (formats 5 or 8). For formats
        2, 4 or 7 the number of planes must be given explicitly in ``colororder``, the
        third element of ``size``.

        :seealso: :meth:`view1d`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> img = Image([[1, 2], [3, 4]])
            >>> print(img)
            >>> img.print()
            >>> Image(np.array([[1, 2], [3, 4]]))
            >>> Image([[1, 2], [3, 1000]])
            >>> Image([[0.1, 0.2], [0.3, 0.4]])
            >>> Image([[True, False], [False, True]])

        .. warning:: If an image is constructed from an existing :class:`Image` instance or a Numpy array,
            the encapsulated Numpy array is, by default, a *reference* to the passed image data.
            Use the option ``copy=True`` if you want to copy the data.

        """
        self._name = None
        self._colororder = None
        self.id = id
        self.domain = domain

        if isinstance(name, Path):
            name = str(name)

        if isinstance(image, np.ndarray):
            pass

        elif isinstance(image, self.__class__):
            # Image instance
            name = image.name
            colororder = image.colororder
            image = image._A

        elif isinstance(image, list):
            # list of lists

            # attempt to convert it to an ndarray
            try:
                image = np.array(image)
            except ValueError:
                raise ValueError("bad list of lists, check all rows have same length")

        else:
            raise ValueError("bad argument passed to Image constructor")

        if not isinstance(image, np.ndarray):
            raise ValueError(
                "bad argument passed to Image constructor: must be ndarray or list of lists"
            )

        # if dtype is not given, determine the appropriate type for the data
        if dtype is None:
            # no type given, automatically choose it
            if np.issubdtype(image.dtype, np.floating):
                # list contained a float
                dtype = np.float32

            elif np.issubdtype(image.dtype, np.integer):
                # list contained only ints, convert to int/uint8 of smallest
                # size to contain all values
                if image.min() < 0:
                    # value is signed
                    for type in ["int8", "int16", "int32"]:
                        if (image.max() <= np.iinfo(type).max) and (
                            image.min() >= np.iinfo(type).min
                        ):
                            dtype = np.dtype(type)
                            break
                else:
                    # value is unsigned
                    for type in ["uint8", "uint16", "uint32"]:
                        if image.max() <= np.iinfo(type).max:
                            dtype = np.dtype(type)
                            break
        elif dtype is True:
            dtype = image.dtype
        else:
            # dtype is given, convert to a NumPy dtype
            try:
                dtype = np.dtype(dtype)
            except TypeError:
                raise ValueError("bad dtype argument passed to Image constructor")

        if binary:
            image = image > 0

        # change type of array to the determined dtype
        if dtype is not None:
            image = image.astype(dtype, copy=False)

        self.name = name

        color_dict = Image.colororder2dict(colororder)

        if isinstance(size, self.__class__):
            # size is an Image instance, ignore the size/shape and use the image's shape
            size = size.size

        # reshape the image data to match the specified size or shape
        if size is not None:
            newsize = [size[1], size[0]]

            if self.colororder is not None:
                nplanes = len(color_dict)

                if len(size) == 3:
                    if nplanes != size[2]:
                        raise ValueError(
                            "colororder length does not match number of planes in size"
                        )
                newsize.append(nplanes)
            elif len(size) == 3:
                newsize.append(size[2])

            if image.ndim == 1:
                # 1D array
                #   Y0 Y1 Y2 ...

                #   R0 G0 B0 R1 G1 B1 ...
                image = image.reshape(*newsize)

            elif image.ndim == 2:
                if image.shape[1] > image.shape[0]:
                    # wide image, reshape to width x height x nplanes

                    #   Y0 Y1 Y2 ...

                    #   R0 G0 B0 R1 G1 B1 ...

                    #   R0 R1 R2 ...
                    #   G0 G1 G2 ...
                    #   B0 B1 B2 ...
                    if len(newsize) == 3:
                        if image.shape[0] != newsize[2]:
                            raise ValueError(
                                "specified number of color plane (size, colororder) does not match number of rows in data"
                            )
                    else:
                        if image.shape[0] > 1:
                            newsize.append(image.shape[0])

                    image = image.T.reshape(*newsize)
                else:
                    # tall image, reshape to height x width x nplanes

                    # Y0
                    # Y1
                    # Y2
                    #  .
                    #  .

                    # R0 G0 B0
                    # R1 G1 B1
                    # R2 G2 B2
                    #  .
                    #  .

                    if len(newsize) == 3:
                        if image.shape[1] != newsize[2]:
                            raise ValueError(
                                "specified number of color plane (size, colororder) does not match number of columns in data"
                            )
                    else:
                        if image.shape[1] > 1:
                            newsize.append(image.shape[1])

                    image = image.reshape(*newsize)

        if image.ndim not in (2, 3):
            raise ValueError(
                "bad ndarray passed to Image constructor: must be 2D or 3D array"
            )
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]  # squeeze out singleton plane

        # assign the image to the object, copying if requested
        if copy:
            self._A = image.copy()
        else:
            self._A = image

        # final check that colororder length matches number of planes
        if colororder is not None:
            if len(color_dict) != self.nplanes:
                raise ValueError("colororder length does not match number of planes")

        if colororder is None:
            if self.nplanes == 3:
                self.colororder = "RGB"
                warnings.warn("defaulting color to RGB")
        else:
            self.colororder = color_dict

        self.name = name

    __array_ufunc__ = None  # allow Image matrices operators with NumPy values

    def __str__(self) -> str:
        """
        Single line summary of image parameters

        :return: single line summary of image
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> str(img)
        """
        s = f"Image: {self.width} x {self.height} ({self.dtype})"

        if self.colororder is not None:
            co = self.colororder_str or ""
            s += ", " + co
        else:
            s += f", {self.nplanes} anonymous plane{'' if self.nplanes == 1 else 's'}"

        if self.id is not None:
            s += f", id={self.id}"
        if self.name is not None:
            name = self.name
            # if it's a long name, take from rightmost / and add ellipsis
            if len(name) > 20:
                k = [i for i, c in enumerate(name) if c == "/"]
                if len(k) >= 2:
                    name = name[k[-2] :]
                else:
                    name = name[-20:]
                name = "..." + name
            s += f" [{name}]"
        if self.domain is not None:
            s += f", u::{self.domain[0][0]:.3g}:{self.domain[0][-1]:.3g}, v::{self.domain[1][0]:.3g}:{self.domain[1][-1]:.3g}"

        nnan = np.sum(np.isnan(self._A))
        ninf = np.sum(np.isinf(self._A))
        if nnan + ninf > 0:
            s += " (contains "
            if nnan > 0:
                s += f"{nnan}xNaN{'s' if nnan > 1 else ''}"
            if ninf > 0:
                s += f" {ninf}xInf{'s' if ninf > 1 else ''}"
            s += ")"

        return s

    def rprint(self, **kwargs) -> "Image":
        """
        Print image pixels in compact format and return image

        :param fmt: format string, defaults to None
        :type fmt: str, optional
        :param separator: value separator, defaults to single space
        :type separator: str, optional
        :param precision: precision for floating point pixel values, defaults to 2
        :type precision: int, optional
        :param header: print image summary header, defaults to True
        :type header: bool, optional
        :param file: file to print to, defaults to None
        :type file: file, optional
        :return: the printed image
        :rtype: :class:`Image`

        Very compact display of pixel numerical values in grid layout and return the image itself.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Squares(1, 10).rprint()
            >>> print(img)
            >>> img = Image.Squares(1, 10, dtype='float').rprint(precision=1, header=True)
            >>> print(img)

        The function returns the image, which is why we see the image ``repr`` value
        after the printed pixels in this python intrepreter.

        The `rprint` method is particularly useful in a method chain, for example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Random(3).rprint()
            >>> print(img) # return result of print() is the image itself

        .. note::

            - For a boolean image True and False are displayed as 1 and 0
              respectively.
            - For a multiplane images the planes are printed sequentially, along
              with the plane's name.

        :seealso: :meth:`print` :meth:`Image.strhcat` :meth:`Image.showpixels`
        """
        self.print(**kwargs)
        return self

    def print(
        self,
        fmt: str | None = None,
        separator: str = " ",
        precision: int = 2,
        header: bool = False,
        file=None,
    ) -> None:
        """
        Print image pixels in compact format

        :param fmt: format string, defaults to None
        :type fmt: str, optional
        :param separator: value separator, defaults to single space
        :type separator: str, optional
        :param precision: precision for floating point pixel values, defaults to 2
        :type precision: int, optional
        :param header: print image summary header, defaults to True
        :type header: bool, optional
        :param file: file to print to, defaults to None
        :type file: file, optional
        :return: the printed image
        :rtype: :class:`Image`

        Very compact display of pixel numerical values in grid layout.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Squares(1, 10)
            >>> img.print()
            >>> img = Image.Squares(1, 10, dtype='float')
            >>> img.print(precision=1, header=True)

        .. note::

            - For a boolean image True and False are displayed as 1 and 0
              respectively.
            - For a multiplane images the planes are printed sequentially, along
              with the plane's name.

        :seealso: :meth:`rprint` :meth:`Image.strhcat` :meth:`Image.showpixels`
        """

        def format_plane(plane, fmt, indent="  "):
            rows = []
            for v in plane.vspan():
                row = indent
                for u in plane.uspan():
                    row += (fmt or "{}").format(plane._A[v, u])
                rows.append(row)
            return rows

        if fmt is None:
            if self.isint:
                width = max(len(str(self.max())), len(str(self.min())))
                fmt = f"{separator}{{:{width}d}}"
            elif self.isbool:
                width = 1
                fmt = f"{separator}{{:{width}d}}"
            elif self.isfloat:
                ff = f"{{:.{precision}f}}"
                width = max(len(ff.format(self.max())), len(ff.format(self.min())))
                fmt = f"{separator}{{:{width}.{precision}f}}"

        if header:
            print(self, file=file)

        if self.iscolor:
            plane_names = (self.colororder_str or "").split(":")
            for i, plane in enumerate(self.planes()):
                print(f"  plane {plane_names[i]}:")
                print("\n".join(format_plane(plane, fmt, indent="    ")), file=file)
        else:
            print("\n".join(format_plane(self, fmt)), file=file)

    @classmethod
    def strhcat(
        cls,
        *images: "Image",
        widths: int | Sequence[int] = 1,
        arraysep: str = " |",
        labels: Sequence[str] | None = None,
    ) -> str:
        """Format several small images concatenated horizontally

        :param images: one or more images to be formatted horizontally concatenated
        :type images: :class:`Image` instances
        :param widths: number of digits for the formatted array elements, defaults to 1.
            If scalar applies to all images, if list applies to each image.
        :type widths: int or list of ints, optional
        :param arraysep: separator between arrays, defaults to ``" |"``
        :type arraysep: str, optional
        :param labels: list of labels for each array, defaults to None
        :type labels: list of str, optional
        :return: multiline string containing formatted arrays
        :rtype: str
        :raises ValueError: if the arrays have different numbers of rows

        AUTO_EDIT

        For image processing this is useful for displaying small test images.

        The arrays are formatted and concatenated horizontally with a vertical separator.
        Each array has a header row that indicates the column number.  Each row has a
        header column that indicates the row number.

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> A = Image.Random((5,5), maxval=9)
            >>> print(Image.strhcat(A))
            >>> print(Image.strhcat(A, widths=2))
            >>> B = Image.Random((5,5), maxval=9)
            >>> print(Image.strhcat(A, B))
            >>> print(Image.strhcat(A, B, labels=("A:", "B:")))

        The number of rows in each image must be the same, but the number of columns can
        vary.

        :seealso: :meth:`Image.print` :meth:`Image.showpixels` :func:`Image`
        """

        # convert to a list of numpy arrays
        arrays = [image._A for image in images]

        # check that all arrays have the same number of rows
        if len(set([array.shape[0] for array in arrays])) != 1:
            raise ValueError("All arrays must have the same number of rows")

        if isinstance(widths, int):
            widths = [widths] * len(arrays)
        else:
            widths = list(widths)

        # add the header rows, which indicate the column number.  2-digit column
        # numbers are shown with the digits one above the other.
        stitle = ""  # array title row
        s10 = " " * 5  # array column number, 10s digit
        s1 = " " * 5  # array column number, 1s digit
        divider = " " * 5  # divider between column number header and array values
        s = ""
        tens = False  # has a tens row
        for i, array in enumerate(arrays):  # iterate over the input arrays
            width = widths[i]
            # make the pixel value format string based on the passed width
            fmt = f" {{:{width}d}}"

            # build the title row
            if labels is not None:
                stitle += " " * (len(s10) - len(stitle)) + labels[i]

            # build the column number header rows
            for col in range(array.shape[1]):
                # the 10s digits
                if col // 10 == 0:
                    s10 += " " * (width + 1)
                else:
                    s10 += fmt.format(col // 10)
                    tens = True
                # the 1s digits
                s1 += fmt.format(col % 10)
                divider += " " * width + "-"

            s10 += " " * len(arraysep)
            s1 += " " * len(arraysep)
            divider += " " * len(arraysep)

        # concatenate the header rows
        s = stitle + "\n"
        if tens:
            s += s10 + "\n"  # only include if there are 10s digits
        s += s1 + "\n" + divider + "\n"

        # add the element values, row by row
        for row in range(array.shape[0]):
            # add the row number
            s += f"{row:3d}: "

            # for each array, add the elements for this row
            for array, width in zip(arrays, widths):
                # make the pixel value format string based on the width of the value
                fmt = f" {{:{width}d}}"
                for col in range(array.shape[1]):
                    s += fmt.format(array[row, col])
                if array is not arrays[-1]:
                    s += arraysep
            s += "\n"

        return s

    def __repr__(self) -> str:
        """
        Readable representation of image parameters

        :return: summary of image enclosed in angle brackets
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img
        """
        return f"<{str(self)}>"

    def copy(self, copy: bool = True) -> "Image":
        """
        Create image copy

        :param copy: copy the image data
        :type copy: bool
        :return: copy of image
        :rtype: :class:`Image`

        Create a new :class:`Image` instance which contains a copy of the
        original image data. If ``copy`` is False the new :class:`Image`
        instance contains a reference to the original image data.
        """
        return self.__class__(self, copy=copy)

    # ------------------------- properties ------------------------------ #

    @property
    def colororder(self) -> dict[str, int] | None:
        """
        Set/get color order of image

        The color order is a dict where the key is the color plane name, eg. 'R'
        and the value is the index of the plane in the 3D pixel value array.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.colororder
            >>> img.colororder = "BGR"
            >>> img.colororder

        When setting the color order the value can be any of:

        * simple string, one plane per character, eg. ``"RGB"``
        * colon separated string, eg. ``"R:G:B"``, ``"L*:a*:b*"``
        * dictionary, eg. ``dict(R=0, G=1, B=2)``

        For the first two cases the color plane indices are implicit in the
        order in the string.

        .. note:: Changing the color order does not change the order of the planes
            in the image array, it simply changes their label.

        :seealso: :meth:`colororder_str` :meth:`colordict` :meth:`plane` :meth:`red` :meth:`green` :meth:`blue`
        """
        return self._colororder

    @colororder.setter
    def colororder(self, colororder: str | dict[str, int]) -> None:
        cdict = Image.colordict(colororder)

        if self.nplanes is not None and len(cdict) != self.nplanes:
            raise ValueError("colororder length does not match number of planes")
        self._colororder = cdict

    @staticmethod
    def _opencv_type_check(array: np.ndarray, *accepted_types: str) -> None:
        """
        Check that a NumPy array has a type accepted by an OpenCV function.

        :param array: array to check
        :type array: np.ndarray
        :param accepted_types: accepted OpenCV type strings
        :type accepted_types: str
        :raises TypeError: if the array dtype or channel count is not accepted by OpenCV

        Recognised type strings:

        - Depth strings: ``"CV_8U"``, ``"CV_8S"``, ``"CV_16U"``, ``"CV_16S"``,
          ``"CV_32S"``, ``"CV_32F"``, ``"CV_64F"``
        - Channel strings: ``"single-channel"`` (array must be 2D),
          ``"multiple-channel"`` (array may be 2D or 3D)

        Example::

            Image._opencv_type_check(img, "single-channel", "CV_8U")
            Image._opencv_type_check(img, "multiple-channel", "CV_8U", "CV_16S", "CV_32F")
        """
        _dtype_map = {
            "CV_8U": np.uint8,
            "CV_8S": np.int8,
            "CV_16U": np.uint16,
            "CV_16S": np.int16,
            "CV_32S": np.int32,
            "CV_32F": np.float32,
            "CV_64F": np.float64,
        }
        _channel_specs = {"single-channel", "multiple-channel"}

        dtype_specs = [t for t in accepted_types if t not in _channel_specs]
        channel_types = [t for t in accepted_types if t in _channel_specs]

        if channel_types:
            if "single-channel" in channel_types and array.ndim != 2:
                raise TypeError(
                    f"OpenCV requires a single-channel (2D) array, got shape {array.shape}"
                )
            if "multiple-channel" in channel_types and array.ndim not in (2, 3):
                raise TypeError(
                    f"OpenCV requires a 2D or 3D array, got shape {array.shape}"
                )

        if dtype_specs:
            for t in dtype_specs:
                if t not in _dtype_map:
                    raise ValueError(f"unknown OpenCV type string: {t!r}")
            accepted_dtypes = [np.dtype(_dtype_map[t]) for t in dtype_specs]
            # OpenCV Python bindings accept np.bool_ arrays as CV_8U (8-bit)
            if "CV_8U" in dtype_specs:
                accepted_dtypes.append(np.dtype(np.bool_))
            if array.dtype not in accepted_dtypes:
                type_list = ", ".join(dtype_specs)
                raise TypeError(f"OpenCV requires {type_list}, got {array.dtype}")

    @staticmethod
    def colordict(colororder) -> dict[str, int]:
        """
        Parse a color order specification

        :param colororder: order of color channels
        :type colororder: str, dict
        :raises ValueError: ``colororder`` not a string or dict
        :return: dictionary mapping color names to plane indices
        :rtype: dict

        The color order the value can be given in a variety of forms:

        * simple string, one plane per character, eg. ``"RGB"``
        * colon separated string, eg. ``"R:G:B"``, ``"L*:a*:b*"``
        * dictionary, eg. ``dict(R=0, G=1, B=2)``

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> Image.colordict('RGB')
            >>> Image.colordict('red:green:blue')
            >>> Image.colordict({'L*': 0, 'U*': 1, 'V*': 2})

        :deprecated: use :meth:`colororder2dict`
        """
        if isinstance(colororder, dict):
            cdict = colororder
        elif isinstance(colororder, str):
            if ":" in colororder:
                colororder = colororder.split(":")
            else:
                colororder = list(colororder)

            cdict = {}
            for i, color in enumerate(colororder):
                cdict[color] = i
        else:
            raise ValueError("color order must be a dict or string")
        return cdict

    @staticmethod
    def colororder2dict(colororder, start: int = 0) -> dict[str, int]:
        """
        Parse a color order specification to a color dictionary

        :param colororder: order of color channels
        :type colororder: str, dict
        :param start: starting index for plane numbering, defaults to 0
        :type start: int, optional
        :raises ValueError: ``colororder`` not a string or dict
        :return: dictionary mapping color names to plane indices
        :rtype: dict

        The color order the value can be given in a variety of forms:

        AUTO_EDIT

        * simple string, one plane per character, eg. ``"RGB"``
        * colon separated string, eg. ``"R:G:B"``, ``"L*:a*:b*"``
        * dictionary, eg. ``dict(R=0, G=1, B=2)``

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> Image.colororder2dict('RGB')
            >>> Image.colororder2dict('red:green:blue')
            >>> Image.colororder2dict({'L*': 0, 'U*': 1, 'V*': 2})
        """
        if isinstance(colororder, dict):
            # already a dict
            if start > 0:
                # need to renumber the planes
                cdict = {}
                for key, value in colororder.items():
                    cdict[key] = value + start
                return cdict
            else:
                return colororder
        elif isinstance(colororder, str):
            if ":" in colororder:
                colororder = colororder.split(":")
            else:
                colororder = list(colororder)

            cdict = {}
            for i, color in enumerate(colororder):
                cdict[color] = i + start
            return cdict
        elif colororder is None:
            return {}
        else:
            raise ValueError("color order must be a dict or string")

    @staticmethod
    def colordict2list(cdict: dict[str, int]) -> list[str]:
        """
        Convert a color dictionary to a list of color plane names

        :param cdict: dictionary mapping color names to plane indices
        :type cdict: dict
        :return: list of color channels in plane order
        :rtype: list

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> Image.colordict2list({'R': 0, 'G': 1, 'B': 2})

        .. note:: The color planes are sorted by their index value.  There is no
            check that the lowest plane index is zero.

        :seealso: :meth:`colordict2str` :meth:`colororder2dict`
        """
        return [x[0] for x in sorted(cdict.items(), key=lambda x: x[1])]

    @staticmethod
    def colordict2str(cdict: dict[str, int]) -> str:
        """
        Convert a color dictionary to a color order string

        :param cdict: dictionary mapping color names to plane indices
        :type cdict: dict
        :return: color channel names in plane order as a colon-separated string
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> Image.colordict2str({'R': 0, 'G': 1, 'B': 2})

        .. note:: The color planes are sorted by their index value.  There is no
            check that the lowest plane index is zero.

        :seealso: :meth:`colordict2list` :meth:`colororder2dict`

        AUTO_EDIT
        """
        return ":".join(Image.colordict2list(cdict))

    @property
    def colororder_str(self) -> str | None:
        """
        Image color order as a string

        :return: Image color plane order as a colon separated string.
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.colororder_str

        :seealso: :meth:`colororder`
        """
        if self.colororder is not None:
            s = sorted(self.colororder.items(), key=lambda x: x[1])
            return ":".join([x[0] for x in s])
        else:
            return None

    @property
    def name(self) -> str | None:
        """
        Set/get image name

        An image has a string-valued name that can be read and written.
        The name is shown by the Image repr and when images are displayed
        graphically.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.name[-70:]
            >>> img.name = 'my image'
            >>> img.name

        .. note:: Images loaded from a file have their name initially set to
            the full file pathname.

        :seealso: :meth:`Read`
        """
        return self._name

    @name.setter
    def name(self, name: str | None) -> None:
        self._name = name

    # ---- image type ---- #
    @property
    def isfloat(self) -> bool:
        """
        Image has floating point pixel values?

        :return: True if image has floating point values
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.isfloat
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.isfloat

        :seealso: :meth:`isint` :meth:`isbool`
        """
        return np.issubdtype(self.dtype, np.floating)

    @property
    def isint(self) -> bool:
        """
        Image has integer values?

        :return: True if image has integer pixel values
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.isint
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.isint

        :seealso: :meth:`isfloat` :meth:`isbool`
        """
        return np.issubdtype(self.dtype, np.integer)

    @property
    def isbool(self) -> bool:
        """
        Image has bolean values?

        :return: True if image has boolean pixel values
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png') > 200
            >>> img.isint
            >>> img.isbool

        :seealso: :meth:`isint` :meth:`isfloat`
        """
        return np.issubdtype(self.dtype, bool)

    @property
    def dtype(self):
        """
        Datatype of image

        :return: NumPy datatype of image
        :rtype: numpy.dtype

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.dtype
            >>> img = Image.Read('flowers1.png', dtype='float32')
            >>> img.dtype

        :seealso: :meth:`to` :meth:`to_int` :meth:`to_float` :meth:`like` :meth:`cast` :func:`numpy.dtype`
        """
        return self._A.dtype

    @property
    def numnan(self) -> int:
        """
        Number of NaN pixels in image

        :return: number of NaN pixels in image
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> from numpy import nan
            >>> img = Image([[1, 2], [3, nan]], dtype='float32')
            >>> img.numnan

        :seealso: :meth:`numinf`
        """
        return int(np.isnan(self._A).sum())

    @property
    def numinf(self) -> int:
        """
        Number of Inf pixels in image

        :return: number of Inf pixels in image
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> from numpy import inf
            >>> img = Image([[1, 2], [3, inf]], dtype='float32')
            >>> img.numinf

        :seealso: :meth:`numnan`
        """
        return int(np.isinf(self._A).sum())

    # ---- image dimension ---- #

    @property
    def width(self) -> int:
        """
        Image width

        :return: Width of image in pixels
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.width

        :seealso: :meth:`height` :meth:`size` :meth:`umax`
        """
        return self._A.shape[1]

    @property
    def height(self) -> int:
        """
        Image height

        :return: Height of image in pixels
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.height

        :seealso: :meth:`width` :meth:`size` :meth:`vmax`
        """
        return self._A.shape[0]

    @property
    def umax(self) -> int:
        """
        Image maximum u-coordinate

        :return: Maximum u-coordinate in image in pixels
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.umax
            >>> img.width

        :seealso: :meth:`width`
        """
        return self._A.shape[1] - 1

    @property
    def vmax(self) -> int:
        """
        Image maximum v-coordinate

        :return: Maximum v-coordinate in image in pixels
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.vmax
            >>> img.height

        :seealso: :meth:`height`
        """
        return self._A.shape[0] - 1

    def uspan(self, step: int = 1) -> np.ndarray:
        """
        Linear span of image horizontally

        :param step: step size, defaults to 1
        :type step: int, optional
        :return: 1D array of values [0 ... width-1]
        :rtype: ndarray(W)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> img = Image.Read('flowers1.png')
            >>> with np.printoptions(threshold=10):
            >>>     img.uspan()

        .. note:: If the image has a ``domain`` specified the horizontal
            component of this is returned instead.

        .. warning:: Computed using :meth:`numpy.arange` and for ``step>1`` the
            maximum coordinate may not be returned.

        :seealso: :meth:`umax` :meth:`vspan`
        """
        if self.domain is None:
            return np.arange(0, self.width, step)
        else:
            return self.domain[0]

    def vspan(self, step: int = 1) -> np.ndarray:
        """
        Linear span of image vertically

        :param step: step size, defaults to 1
        :type step: int, optional
        :return: 1D array of values [0 ... height-1]
        :rtype: ndarray(H)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> img = Image.Read('flowers1.png')
            >>> with np.printoptions(threshold=10):
            >>>     img.vspan()

        .. note:: If the image has a ``domain`` specified the vertical
            component of this is returned instead.

        .. warning:: Computed using :meth:`numpy.arange` and for ``step>1`` the
            maximum coordinate may not be returned.

        :seealso: :meth:`vmax` :meth:`uspan`
        """
        if self.domain is None:
            return np.arange(0, self.height, step)
        else:
            return self.domain[1]

    @property
    def size(self) -> tuple[int, int]:
        """
        Image size

        :return: Size of image (width, height)
        :rtype: tuple

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.size

        .. note:: The dimensions are in a different order compared to :meth:`shape`.

        :seealso: :meth:`width` :meth:`height` :meth:`shape`
        """
        return (self._A.shape[1], self._A.shape[0])

    @property
    def centre(self) -> tuple[float, float]:
        """
        Coordinate of centre pixel

        :return: Coordinate (u,v) of the centre pixel
        :rtype: tuple

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Zeros((50,50))
            >>> img.centre
            >>> img = Image.Zeros((51,51))
            >>> img.centre

        .. note:: If the image has an even dimension the centre will lie
            between pixels.

        :seealso: :meth:`center` :meth:`centre_int`
        """
        return (self._A.shape[1] / 2, self._A.shape[0] / 2)

    @property
    def center(self) -> tuple[float, float]:
        """
        Coordinate of center pixel

        :return: Coordinate (u,v) of the centre pixel
        :rtype: tuple

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Zeros((50,50))
            >>> img.center
            >>> img = Image.Zeros((51,51))
            >>> img.center

        .. note::
            - If the image has an even dimension the centre will lie
              between pixels.
            - Same as ``centre``, just US spelling

        :seealso: :meth:`center_int`
        """
        return self.centre

    @property
    def centre_int(self) -> tuple[int, int]:
        """
        Coordinate of centre pixel as integer

        :return: Coordinate (u,v) of the centre pixel
        :rtype: tuple

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Zeros((50,50))
            >>> img.centre_int
            >>> img = Image.Zeros((51,51))
            >>> img.centre_int

        .. note:: If the image has an even dimension the centre coordinate will
              be truncated toward zero.

        :seealso: :meth:`centre`
        """
        return (self._A.shape[1] // 2, self._A.shape[0] // 2)

    @property
    def center_int(self) -> tuple[int, int]:
        """
        Coordinate of centre pixel as integer

        :return: Coordinate (u,v) of the centre pixel
        :rtype: tuple

        Integer coordinate of centre pixel.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Zeros((50,50))
            >>> img.center_int
            >>> img = Image.Zeros((51,51))
            >>> img.center_int

        .. note::
            - If the image has an even dimension the centre coordinate will
              be truncated toward zero.
            - Same as ``centre_int``, just US spelling

        :seealso: :meth:`center`
        """
        return self.centre_int

    @property
    def npixels(self) -> int:
        """
        Number of pixels in image plane

        :return: Number of pixels in image plane: width x height
        :rtype: int

        .. note:: Number of planes is not considered.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.npixels
            >>> img.width * img.height

        :seealso: :meth:`size`
        """
        return self._A.shape[0] * self._A.shape[1]

    @property
    def shape(self) -> tuple[int, int] | tuple[int, int, int]:
        """
        Image shape

        :return: Shape of internal NumPy array
        :rtype: 2-tuple, or 3-tuple if color

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.shape
            >>> img = Image.Read('street.png')
            >>> img.shape

        .. note:: The dimensions are in a different order compared to :meth:`size`.

        :seealso: :meth:`size` :meth:`nplanes` :meth:`ndim` :meth:`iscolor`
        """
        return self._A.shape

    @property
    def ndim(self) -> int:
        """
        Number of image array dimensions

        :return: number of image dimensions
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.ndim
            >>> img = Image.Read('street.png')
            >>> img.ndim

        :seealso: :meth:`nplanes` :meth:`shape`
        """
        return self._A.ndim

    def contains(self, p: Array2d) -> bool | np.ndarray:
        """
        Test if coordinate lies within image

        :param p: pixel coordinate
        :type p: array_like(2), ndarray(2,N)
        :return: whether pixel coordinate lies within image bounds
        :rtype: bool, ndarray(N)

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> img = Image.Zeros(10)
            >>> img.contains((4,6))
            >>> img.contains((-1, 7))
            >>> img.contains(np.array([[4, 6], [-1, 7], [10, 10]]).T)
        """
        if isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[0] == 2:
            u = p[0, :]
            v = p[1, :]
        else:
            u = p[0]
            v = p[1]

        return np.logical_and.reduce((u >= 0, v >= 0, u < self.width, v < self.height))

    # ---- color related ---- #
    @property
    def iscolor(self) -> bool:
        """
        Image has color pixels?

        :return: Image is color
        :rtype: bool

        :return: number of image dimensions
        :rtype: int

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.iscolor
            >>> img = Image.Read('street.png')
            >>> img.iscolor

        :seealso: :meth:`isrgb` :meth:`nplanes`
        """
        return self._A.ndim > 2

    @property
    def isbgr(self) -> bool:
        """
        Image has BGR color order?

        :return: Image has BGR color order
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.isbgr

        .. note:: Is False if image is not color.

        :seealso: :meth:`colororder`
        """
        return self.colororder_str == "B:G:R"

    @property
    def isrgb(self) -> bool:
        """
        Image has RGB color order?

        :return: Image has RGB color order
        :rtype: bool

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.isrgb

        .. note:: Is False if image is not color.

        :seealso: :meth:`colororder`
        """
        return self.colororder_str == "R:G:B"

    def to(self, dtype: Dtype) -> "Image":
        """
        Convert image datatype

        :param dtype: Numpy data type
        :type dtype: str or NumPy dtype
        :return: image
        :rtype: :class:`Image`

        Create a new image, same size as input image, with pixels of a different
        datatype.  Integer values are scaled according to the maximum value
        of the datatype, floating values are in the range 0.0 to 1.0.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Random(3)
            >>> img.print()
            >>> img.to('float').print()
            >>> img = Image.Random(3, dtype='float')
            >>> img.print()
            >>> img.to('uint8').print()

        :seealso: :meth:`astype` :meth:`to_int` :meth:`to_float`
        """
        # convert image to different type, does rescaling
        # as just changes type
        dtype = np.dtype(dtype)  # convert to dtype if it's a string
        return self.__class__(self.array_as(dtype), dtype=dtype)

    def astype(self, dtype: Dtype) -> "Image":
        """
        Cast image datatype

        :param dtype: Numpy data type
        :type dtype: str or NumPy dtype
        :return: image
        :rtype: :class:`Image`

        Create a new image, same size as input image, with pixels of a different
        datatype.  Values are retained, only the datatype is changed.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Random(3)
            >>> img.print()
            >>> img.astype('float').print()
            >>> img = Image.Random(3, dtype='float')
            >>> img.print()
            >>> img.astype('uint8').print()

        :seealso: :meth:`to`
        """
        return self.__class__(self._A.astype(dtype), dtype=dtype)

    # ---- NumPy array access ---- #

    @property
    def image(self) -> Array2d | Array3d:
        """
        Image as NumPy array

        :return: image as a NumPy array
        :rtype: ndarray(H,W) or ndarray(H,W,3)

        Return a reference to the encapsulated NumPy array that holds the pixel
        values.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img
            >>> type(img)
            >>> type(img.image)

        .. note:: For a color image the color plane order is given by the
            colororder dictionary.

        .. deprecated:: 1.0.3
            Use :meth:`array` instead.

        :seealso: :meth:`A` :meth:`colororder`
        """
        warnings.warn(
            "image property will be deprecated in v2.0, use .array instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._A  # type: ignore[return-value]

    @property
    def array(self) -> np.ndarray:
        """
        Image as NumPy array

        :return: image as a NumPy array
        :rtype: ndarray(H,W) or ndarray(H,W,3)

        Return a reference to the encapsulated NumPy array that holds the pixel
        values.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> type(img)
            >>> type(img.array)

        .. note:: For a color image the color plane order is given by the
            colororder dictionary.

        .. warning:: This property is for accessing the NumPy image data for
           passing to a NumPy, OpenCV or other function.  It is not intended for direct
           manipulation of the image data -- doing so may make the state of the
           Image instance inconsistent.

        :seealso: :meth:`array_as` :meth:`rgb` :meth:`bgr` :meth:`colororder`
        """
        return self._A

    @property
    def A(self) -> Array2d | Array3d:
        """
        Set/get the NumPy array containing pixel values

        **Getting**

        :return: image as a NumPy array
        :rtype: ndarray(H,W) or ndarray(H,W,3)

        Return a reference to the encapsulated NumPy array that holds the pixel
        values.

        **Setting**

        Replace the encapsulated NumPy array with another.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> img = Image.Read('flowers1.png')
            >>> img
            >>> type(img)
            >>> type(img.A)
            >>> img.A = np.zeros((50,50))
            >>> img

        .. deprecated:: 1.0.3
            Use :meth:`array` instead for accessing the NumPy image data and the
            :meth:`Image` constructor for creating a new ``Image`` with a different array.

        :seealso: :meth:`array` :meth:`image`
        """
        return self._A  # type: ignore[return-value]

    @A.setter
    def A(self, A: Array2d | Array3d) -> None:
        warnings.warn(
            "A setter will be deprecated in v2.0, use Image constructor instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self._A = A

    @property
    def rgb(self) -> np.ndarray:  # type: ignore[return-value]
        """
        Image as NumPy array in RGB color order

        :raises ValueError: image is greyscale
        :return: image as a NumPy array in RGB color order
        :rtype: ndarray(H,W,3)

        The image is guaranteed to be in RGB order irrespective of current color order.

        :seealso: :meth:`image` :meth:`bgr` :meth:`colororder`
        """
        if not self.iscolor:
            raise ValueError("greyscale image has no rgb property")
        if self.isrgb:
            return self._A
        elif self.isbgr:
            return self._A[:, :, ::-1]

    @property
    def bgr(self) -> np.ndarray:  # type: ignore[return-value]
        """
        Image as NumPy array in BGR color order

        :raises ValueError: image is greyscale
        :return: image as a NumPy array in BGR color order
        :rtype: ndarray(H,W,3)

        The image is guaranteed to be in BGR (OpenCV standard) order irrespective of current color order.

        :seealso: :meth:`image` :meth:`rgb` :meth:`colororder`
        """
        if not self.iscolor:
            raise ValueError("greyscale image has no bgr property")
        if self.isbgr:
            return self._A
        elif self.isrgb:
            return self._A[:, :, ::-1]

    # ------------------------- datatype operations ----------------------- #

    def to_int(self, intclass: Dtype = "uint8") -> np.ndarray:
        """
        Image as integer NumPy array

        :param intclass: name of NumPy supported integer class, default is 'uint8'
        :type intclass: str or NumPy dtype, optional
        :return: NumPy array with integer values
        :rtype: ndarray(H,W) or ndarray(H,W,P)

        Return a NumPy array with pixels converted to the integer class
        ``intclass``.  For the case where the input image is:

        * a floating point class, the pixel values are scaled from an
          input range of [0,1] to a range spanning zero to the maximum positive
          value of the output integer class.
        * an integer class, then the pixels are scaled and cast to ``intclass``.
          The scale factor is the ratio of the maximum value of the input and
          output integer classes.
        * boolean class, False is mapped to zero and True is mapped to the
          maximum positive value.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[5_000, 10_000], [30_000, 60_000]])
            >>> img
            >>> img.to_int('uint8')
            >>> img.to_int('uint32')
            >>> img = Image([[0.0, 0.3], [0.5, 1.0]])
            >>> img
            >>> img.to_int('uint8')
            >>> img = Image([[False, True], [True, False]])
            >>> img
            >>> img.to_int('uint8')
            >>> img.to_int('uint16')

        .. note:: Works for greyscale or color (arbitrary number of planes) image

        .. deprecated:: 1.0.3
            Use :meth:`array_as` instead.

        :seealso: :meth:`array_as` :meth:`to_float` :meth:`cast` :meth:`like`
        """
        warnings.warn(
            "to_int property will be deprecated in v2.0, use .array_as(int_type) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return int_image(self._A, intclass)

    def to_float(self, floatclass: Dtype = "float32") -> np.ndarray:
        """
        Image as float NumPy array

        :param floatclass: 'single', 'double', 'float32' [default], 'float64'
        :type floatclass: str or NumPy dtype
        :return: NumPy array with floating point values
        :rtype: ndarray(H,W) or ndarray(H,W,P)

        Return a NumPy array with pixels converted to the floating point class ``floatclass``
        and the values span the range 0 to 1. For the case where the input image
        is:

        * an integer class, the pixel values are scaled from an input range
          spanning zero to the maximum positive value of the integer
          class to [0.0, 1.0]
        * a floating class, the pixels are cast to change type but not their
          value.
        * boolean class, False is mapped to 0.0 and True is mapped to 1.0.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[50, 100], [150, 200]])
            >>> img
            >>> img.to_float()
            >>> img = Image([[0.0, 0.3], [0.5, 1.0]])
            >>> img
            >>> img.to_float('float64')
            >>> img = Image([[False, True], [True, False]])
            >>> img.to_float()

        .. note:: Works for greyscale or color (arbitrary number of planes) image

        .. deprecated:: 1.0.3
            Use :meth:`array_as` instead.

        :seealso: :meth:`array_as` :meth:`to_int` :meth:`cast` :meth:`like`
        """
        warnings.warn(
            "to_float property will be deprecated in v2.0, use .array_as(float_type) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return float_image(self._A, floatclass)

    def array_as(self, dtype: Dtype | None = None) -> np.ndarray:
        """
        Convert Image to NumPy array of specified type

        :param dtype: data type of the output array
        :type dtype: str or np.dtype
        :return: NumPy array with specified type
        :rtype: ndarray(H,W) or ndarray(H,W,P)

        Return a NumPy array with pixels converted to the specified data type.

        =========  ========================================  ====================================
        Input      Output int                                Output float
        =========  ========================================  ====================================
        float      scaled [-1,1] → [min_int, max_int]        cast, values unchanged
        int        scaled by max_int_output/max_int_input    scaled [min_int, max_int] → [-1, 1]
        uint       scaled by max_int_output/max_int_input    scaled [0, max_int] → [0, 1]
        bool       False → 0, True → max_int                 False → 0.0, True → 1.0
        =========  ========================================  ====================================

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[5_000, 10_000], [30_000, 60_000]]) # uint16 image
            >>> img
            >>> img.array_as('uint8')
            >>> img.array_as('uint32')
            >>> img.array_as('float')
            >>> img = Image([[0.0, 0.3], [0.5, 1.0]])
            >>> img
            >>> img.array_as('uint8')
            >>> img = Image([[False, True], [True, False]])
            >>> img
            >>> img.array_int('uint8')
            >>> img.array_int('float')

        .. note:: Works for greyscale or color (arbitrary number of planes) image

        .. warning:: This method assumes that all integer values are unsigned.

        :seealso: :meth:`array` :func:`array_float` :meth:`cast` :meth:`like`
        """
        if dtype is None:
            return self._A
        else:
            dtype = np.dtype(dtype)  # convert to dtype if it's a string
            if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
                return int_image(self._A, dtype)
            elif np.issubdtype(dtype, np.floating):
                return float_image(self._A, dtype)
            else:
                raise ValueError("unsupported dtype specified")

    def cast(self, value: int | float) -> Any:
        """
        Cast value to same type as image

        :param value: value to cast
        :type value: scalar, ndarray
        :return: value cast to same type as image
        :rtype: numpy type, ndarray

        The value, scalar or integer, is **cast** to the same type as the image.
        The result has the same numeric value, but the type is changed.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> x = img.cast(12.5)
            >>> x
            >>> type(x)

        .. note:: Scalars are cast to NumPy types not native Python types.

        :seealso: :meth:`like`
        """
        return self._A.dtype.type(value)

    def like(self, value: int | float, maxint: int | None = None) -> Any:
        """
        Convert value to the same type as image

        :param value: scalar or NumPy array
        :type value: scalar, ndarray
        :param maxint: maximum integer value for an integer image, defaults to
            maximum positive value of the class
        :type maxint: int, optional
        :raises ValueError: [description]
        :return: converted value
        :rtype: NumPy type

        The value, scalar or integer, is **converted** to the same type as the
        image. The result is optionally rescaled and cast:

        * Float to float: values are cast
        * Float to int: values in the interval [0, 1] are scaled to the interval
          [0, ``maxint``] and then cast
        * Int to float: values in the interval [0, ``maxint``] are scaled to the
          interval [0, 1] and then cast.
        * Int to int: values are scaled and cast.

        .. warning:: For integer to integer conversion, integer values  greater
            than the maximum value of the destination class are wrapped not
            clipped.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.like(0.5)

        :seealso: :meth:`cast` :meth:`to_int` :meth:`to_float`

        """
        if self.isint:
            # matching to an integer image
            if (
                isinstance(value, np.ndarray)
                and np.issubdtype(value.dtype, np.integer)
                or isinstance(value, (int, np.integer))
            ):
                # already an integer, cast it to right sort
                return self.cast(value)
            else:
                # it's a float, rescale it then cast
                return self.cast(value * self.maxval)
        else:
            # matching to a float image
            if (
                isinstance(value, np.ndarray)
                and np.issubdtype(value.dtype, np.floating)
                or isinstance(value, (float, np.floating))
            ):
                # already a float of some sort, cast it to the right sort
                return self.cast(value)
            else:
                # it's an int.  We use hints to determine the size, otherwise
                # get it from the type
                if maxint is None:
                    if isinstance(value, np.ndarray):
                        maxint = np.iinfo(value.dtype).max
                    else:
                        maxint = np.iinfo(type(value)).max
                elif isinstance(maxint, int):
                    pass
                elif isinstance(maxint, str) or isinstance(maxint, np.dtype):
                    maxint = np.iinfo(maxint).max
                else:
                    raise ValueError("bad max value specified")
                return self.cast(value / maxint)

    @property
    def minval(self):
        """
        Minimum value of image datatype

        :return: Minimum value
        :rtype: int or float

        For the datatype of the image, return its minimum possible value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Zeros(20, dtype='float32')
            >>> img.minval
            >>> img = Image.Zeros(20, dtype='uint8')
            >>> img.minval

        :seealso: :meth:`maxval`
        """
        if self.isint:
            return np.iinfo(self.dtype).min
        else:
            return np.finfo(self.dtype).min

    @property
    def maxval(self):
        """
        Maximum value of image datatype

        :return: Maximum value
        :rtype: int or float

        For the datatype of the image, return its maximum possible value

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Zeros(20, dtype='float32')
            >>> img.maxval
            >>> img = Image.Zeros(20, dtype='uint8')
            >>> img.maxval

        :seealso: :meth:`minval`
        """
        if self.isint:
            return np.iinfo(self.dtype).max
        else:
            return np.finfo(self.dtype).max

    @property
    def true(self) -> int | float:
        """
        True value for logical image

        :return: Value used as true
        :rtype: int or float

        The true value is 1.0 for a floating point image and maximum integer
        value for an integer value.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Zeros(20, dtype='float32')
            >>> img.true
            >>> img = Image.Zeros(20, dtype='uint8')
            >>> img.true

        :seealso: :meth:`false` :meth:`maxval`
        """
        if self.isint:
            return self.maxval
        else:
            return 1.0

    @property
    def false(self) -> int | float:
        """
        False value for logical image

        :return: Value used as true
        :rtype: int or float

        The false value is 0.0 for a floating point image and 0
        value for an integer value.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Zeros(20, dtype='float32')
            >>> img.false
            >>> img = Image.Zeros(20, dtype='uint8')
            >>> img.false

        :seealso: :meth:`true`
        """
        return 0

    # ------------------------- color plane access -------------------------- #

    @property
    def nplanes(self) -> int:
        """
        Number of color planes

        :return: Number of color planes, or None if image is empty
        :rtype: int

        For a 2D or greyscale image this is one, otherwise it is the third
        dimension of the image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('street.png')
            >>> img.nplanes
            >>> img = Image.Read('flowers1.png')
            >>> img.nplanes

        .. note:: A greyscale image is stored internally as a 2D NumPy array
            which has zero planes, but ``nplanes`` will return 1 in that case.

        :seealso: :meth:`iscolor` :meth:`shape` :meth:`ndim`
        """
        if self._A is None:
            return 0
        elif self._A.ndim == 2:
            return 1
        else:
            return int(self._A.shape[-1])  # type: ignore[return-value]

    def planes(self) -> Iterator[Image]:
        """
        Iterator for image planes

        :return out: image containing a single plane
        :rtype: :class:`Image`

        Iterator that returns sequential image planes.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("flowers4.png") # in BGR order
            >>> for plane in img.planes():
            >>>     print(plane)
        """
        for i in range(self.nplanes):
            yield self.plane(i)

    def plane(self, planes) -> "Image":
        """
        Extract plane(s) from color image

        :param planes: planes to extract
        :type planes: int, list, str
        :raises ValueError: if image is not color
        :return out: image containing only the selected planes
        :rtype: :class:`Image`

        Create a new image from the selected planes of the image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("flowers4.png") # in BGR order
            >>> img.colororder_str
            >>> img.nplanes
            >>> red = img.plane(0) # red plane
            >>> red
            >>> red.iscolor
            >>> red.nplanes
            >>> green_blue = img.plane('G:B') # green and blue planes
            >>> green_blue
            >>> green_blue.iscolor
            >>> green_blue.nplanes
            >>> red_blue = img.plane([0, 2]) # blue and red planes
            >>> red_blue

        .. note::

            - This can also be performed using the overloaded ``__getitem__``
              operator.
            - To select more than one plane, use either a sequence of integers or a string
              of colon separated plane names.
            - For a single-plane image the index must be zero.

        :seealso: :meth:`red` :meth:`green` :meth:`blue` :meth:``__getitem__``
        """

        if isinstance(planes, int):
            if planes == 0 and self.nplanes == 1:
                return self
            if planes < 0 or planes >= self.nplanes:
                raise ValueError("plane index out of range")
            iplanes = [planes]
            colororder = None
        elif isinstance(planes, str):
            if self.colororder is None:
                raise ValueError("plane names require colororder")
            iplanes = []
            colororder = {}
            if ":" in planes:
                planes = planes.split(":")
                planes = [p for p in planes if p != ""]
            for plane in planes:
                try:
                    i = self.colororder[plane]
                    iplanes.append(i)
                    colororder[plane] = len(colororder)  # copy to new dict
                except KeyError:
                    raise ValueError("bad plane name specified")
        elif isinstance(planes, (tuple, list)):
            if self.colororder is None:
                raise ValueError("plane names require colororder")
            colororder = {}
            for plane in planes:
                if not isinstance(plane, int) or plane < 0 or plane >= self.nplanes:
                    raise ValueError("plane index invalid or out of range")
                colorname = [k for k, v in self.colororder.items() if v == plane][0]
                colororder[colorname] = plane
            iplanes = planes
        else:
            raise ValueError("bad plane specified")

        if isinstance(iplanes, list) and len(iplanes) == 1:
            iplanes = iplanes[0]
            colororder = None
        return self.__class__(self._A[:, :, iplanes], colororder=colororder)

    def __getitem__(
        self, keys: int | str | tuple[slice, slice] | tuple[slice, slice, slice]
    ):
        """
        Return pixel value or slice from image

        :param keys: slices to extract
        :type keys: int, str, tuple of int or slice
        :return: slice of image
        :rtype: :class:`Image`

        This is a Swiss-army knife method for accessing subregions of an ``Image``
        by uv-region and/or plane. A ``key`` can be:

        - a 2-tuple of integers, eg. ``img[u,v]``, return this pixel as a scalar.  If the image has
          multiple planes, the result is an ndarray over planes.
        - a 3-tuple of integers, eg, ``img[u,v,p]``, for a multiplane image return this pixel from
          specified plane as a scalar.
        - a 2-tuple containing at least one slice object, eg. ``img[100:110, 200:300]``, return this
          region as an ``Image``. If the image has multiple planes, the result is a multiplane
          ``Image``.
        - a 3-tuple containing at least one slice objects, return this region of uv and planes as an
          ``Image`` with one or more planes.
        - an int, return this plane as an ``Image``.
        - a string, return this named plane or planes as an ``Image``.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("flowers4.png") # in RGB order
            >>> red = img[0] # red plane
            >>> red # greyscale image
            >>> green = img["G"]
            >>> green # greyscale image
            >>> roi = img[100:200, 300:400]
            >>> roi # color image
            >>> roi = img[100:200, 300:400, 1:]
            >>> roi # 2-plane color image
            >>> roi = img[100, :]  # column 100
            >>> roi # color image
            >>> pix = img[100, 200, 1] # green pixel at (100,200) as scalar
            >>> pix
            >>> pix = img[100, 200] # RGB vector at (100,200) as ndarray
            >>> pix

        .. note:: If the result is a single row or column the result is a 1xn or nx1
            ``Image`` instance.  If the result is a single plane the result is a
            greyscale image.

        .. note:: Indexing pixel values this way is slow, use :meth:`pixel(u,v)` for faster
            access, or ``img.A[v,u]`` for direct access to the underlying NumPy array.

        .. warning:: The order of the indices is column, row and plane. This
            is the opposite of the order used for NumPy index on the underlying
            array.  It is consistent with the column-first convention used across the
            Toolbox and is consistent with the :math:`(u,v)` coordinate system for images.

        .. versionadded:: 1.0.0
            The order of the indices changed to column, row, plane.  Previously
            it was row, column, plane.

        :seealso: :meth:`red` :meth:`green` :meth:`blue` :meth:`plane` :meth:`roi` :meth:`pixel`
        """

        def fixdims(out, shape, keys):
            # deal with the fact that some of the keys may have reduced the
            # dimensionality of the array, eg. a slice of span 0 or 1, or an integer
            # key.

            # shape: (nrows, ncols, nplanes)  in NumPy order
            # key: (rowspec, colspec, planespec) in NumPy order

            def lenkey(key, max):
                # compute the span of a particular key
                if isinstance(key, int):
                    # int key has a span on 1
                    return 1
                elif isinstance(key, slice):
                    # slice key has a span depending on the slice and the corresponding
                    # array dimension. we have to essentially replicate the logic of
                    # slice() here.
                    start = key.start if key.start is not None else 0
                    stop = key.stop if key.stop is not None else max
                    step = key.step if key.step is not None else 1
                    n = stop - start
                    if n < step:
                        return 1
                    else:
                        return n // step

            if len(shape) == 2:
                dims = [lenkey(keys[0], shape[0]), lenkey(keys[1], shape[1])]
            elif len(shape) == 3:
                dims = [
                    lenkey(keys[0], shape[0]),
                    lenkey(keys[1], shape[1]),
                    lenkey(keys[2], shape[2]),
                ]
                # ignore loss of color dimension
                if dims[2] == 1:
                    dims = dims[:2]

            return out.reshape(dims)

        if (
            isinstance(keys, tuple)
            and len(keys) == 2
            and isinstance(keys[0], int)
            and isinstance(keys[1], int)
        ):
            # integer keys for row and column, the result is a scalar or vector over planes
            return self._A[keys[1], keys[0]]
        elif (
            isinstance(keys, tuple)
            and len(keys) == 3
            and isinstance(keys[0], int)
            and isinstance(keys[1], int)
            and isinstance(keys[2], int)
        ):
            # integer keys for row and column, the result is a scalar or vector over planes
            return self._A[keys[1], keys[0], keys[2]]
        elif isinstance(keys, int):
            # single integer index, it's a color plane index
            return self.__class__(self._A[..., keys])
        elif isinstance(keys, str):
            # color plane by name
            return self.plane(keys)
        elif isinstance(keys, (list, tuple)):
            # by slices
            if self.iscolor:
                if len(keys) == 2:
                    keys = (keys[1], keys[0], slice(None))
                elif len(keys) == 3:
                    keys = (keys[1], keys[0], keys[2])
                else:
                    raise ValueError("invalid number of slices")
                # slice the data out of the ndarray
                out = self._A[keys]

                if out.ndim < 3:
                    out = fixdims(out, self._A.shape, keys)

            else:
                # greyscale image
                if len(keys) == 2:
                    keys = (keys[1], keys[0])
                else:
                    raise ValueError("invalid number of slices")

                # slice the data out of the ndarray
                out = self._A[keys]

                if out.ndim < 2:
                    out = fixdims(out, self._A.shape, keys)

            # a singleton plane dimensions is a grey scale image
            if out.ndim == 3 and out.shape[2] == 1:
                out = out.squeeze(2)

            colororder = None
            if out.ndim == 3:
                # 3 slices, select uv-region and planes
                colororder = (self.colororder_str or "").split(":")
                colororder = colororder[keys[2]]  # type: ignore[index]
                colororder = ":".join(colororder)
            return self.__class__(out, colororder=colororder)

        else:
            raise ValueError("invalid slice")

    def pixel(self, u: int, v: int) -> int | float | np.ndarray:
        """
        Return pixel value

        :param u: column coordinate
        :type u: int
        :param v: row coordinate
        :type v: int
        :return: pixel value
        :rtype: int, float or ndarray

        Return the specified pixel.  If the image has multiple planes, the result is a vector over planes.

        .. note:: This method is faster than the more general :meth:`__getitem__`
            for the individual pixel case.

        .. warning:: The order of the indices is column, row and plane. This
            is the opposite of the order used for NumPy index on the underlying
            array.  It is consistent with the column-first convention used across the
            Toolbox and is consistent with the :math:`(u,v)` coordinate system for images.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("flowers4.png", mono=True)
            >>> pix = img.pixel(100, 200)
            >>> pix # grey scale pixel value
            >>> img = Image.Read("flowers4.png")
            >>> pix = img.pixel(100, 200)
            >>> pix # color pixel value (R, G, B)

        :seealso: :meth:`__getitem__` :meth:`roi`
        """
        return self._A[v, u]

    def pixels_mask(
        self,
        mask: Image | Polygon2 | list[Polygon2],
        coords: bool = False,
        return_mask: bool = False,
    ):
        """
        Return pixel values at locations specified by a mask

        :param mask: the selection mask as either non-zero pixels in a 2D image or the area covered by a polygon or list of polygons
        :type mask: :class:`Image`, a single :class:`Polygon2` or a list of :class:`Polygon2`
        :param coords: include pixel coordinates in output, defaults to False
        :type coords: bool, optional
        :param return_mask: also return the mask as an Image, defaults to False
        :type return_mask: bool, optional
        :return: array of pixel values and optionally coordinates of pixels selected by the mask, optionally also the mask as an Image
        :rtype: :class:`Array2d`, :class:`Array2d` and :class:`Image`

        For an image with P planes and a mask that selects N pixels, return an PxN array
        of pixel values.  If ``coords`` is True, the result is an (P+2)xN array where
        the first two rows are the u and v coordinates of the selected pixels.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> from spatialmath import Polygon2
            >>> img = Image.Read("flowers4.png")
            >>> polygon = Polygon2([(300, 400), (360, 400), (330, 450)], close=True)
            >>> pixels = img.pixels_mask(polygon, coords=True)
            >>> pixels.shape
            >>> pixels[:,:5] # first 5 pixels, with coordinates

        :seealso: :meth:`pixel` :meth:`__getitem__` :meth:`roi`
        """
        if isinstance(mask, Image):
            if mask.ndim != 2:
                raise ValueError("mask must be a 2D image")
            if mask.shape != self.shape[:2]:
                raise ValueError("mask must be same shape as image")
            mask_array = mask.array
        else:
            # its a Polygon2 or list of Polygon2, we create a mask image from it
            mask_array = Image.Polygons(self.size, mask, color=1, dtype="uint8").array

        v, u = np.where(mask_array > 0)
        # Access the pixel values in the original image
        pixel_values = self._A[v, u, ...].T

        # optionally add the
        if coords:
            pixel_values = np.vstack((u, v, pixel_values))

        if return_mask:
            return pixel_values, Image(mask_array)
        else:
            return pixel_values

    def red(self) -> "Image":
        """
        Extract the red plane of a color image

        :raises ValueError: if image is not color
        :return out: greyscale image representing the red image plane
        :rtype: :class:`Image`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("flowers4.png")
            >>> red = img.red() # red plane
            >>> red.disp()
            >>> red
            >>> red.iscolor

        .. plot::

            from machinevisiontoolbox import Image
            Image.Read("flowers4.png").red().disp()


        :seealso: :meth:`plane` :meth:`green` :meth:`blue`
        """
        return self.plane("R")

    def green(self) -> "Image":
        """
        Extract the green plane of a color image

        :raises ValueError: if image is not color
        :return out: greyscale image representing the green image plane
        :rtype: :class:`Image`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("flowers4.png")
            >>> green = img.green() # green plane
            >>> green.disp()
            >>> green
            >>> green.iscolor

        .. plot::

            from machinevisiontoolbox import Image
            Image.Read("flowers4.png").green().disp()

        :seealso: :meth:`plane` :meth:`red` :meth:`blue`
        """
        return self.plane("G")

    def blue(self) -> "Image":
        """
        Extract the blue plane of a color image

        :raises ValueError: if image is not color
        :return out: greyscale image representing the blue image plane
        :rtype: :class:`Image`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("flowers4.png")
            >>> blue = img.blue() # blue plane
            >>> blue.disp()
            >>> blue
            >>> blue.iscolor

        .. plot::

            from machinevisiontoolbox import Image
            Image.Read("flowers4.png").blue().disp()

        :seealso: :meth:`plane` :meth:`red` :meth:`green`
        """
        return self.plane("B")

    # I think these are not used anymore

    # def astype(self, type):
    #     return self.__class__(self.A, dtype=type)

    # ------------------------- operators ------------------------------ #

    # arithmetic
    def __mul__(self, other) -> "Image":
        """
        Overloaded ``*`` operator

        :return: element-wise product of images
        :rtype: :class:`Image`

        Compute the product of an Image with another image or a scalar.
        Supports:

        * image ``*`` image, element-wise
        * scalar ``*`` image
        * image ``*`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img * img
            >>> z.array
            >>> z = 2 * img
            >>> z.array

        ..warning:: Values will be wrapped not clipped to the range of the
            pixel datatype.
        """
        return self._binop(self, other, lambda x, y: x * y)

    def __rmul__(self, other) -> "Image":
        return self._binop(self, other, lambda x, y: y * x)

    def __imul__(self, other) -> "Image":
        """
        Overloaded in-place ``*=`` operator

        :return: self, modified in place
        :rtype: :class:`Image`

        Multiply an Image in place by another image or a scalar.

        :seealso: :meth:`__mul__`
        """
        self._A = self._binop(self, other, lambda x, y: x * y)._A
        return self

    def __pow__(self, other) -> "Image":
        """
        Overloaded ``**`` operator

        :return: element-wise exponent of image
        :rtype: :class:`Image`

        Compute the element-wise power of an Image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img**3
            >>> z.array

        ..warning:: Values will be wrapped not clipped to the range of the
            pixel datatype.
        """
        if not isscalar(other):
            raise ValueError("exponent must be a scalar")
        return self._binop(self, other, lambda x, y: x**y)

    def __add__(self, other) -> "Image":
        """
        Overloaded ``+`` operator

        :return: element-wise addition of images
        :rtype: :class:`Image`

        Compute the sum of an Image with another image or a scalar.
        Supports:

        * image ``+`` image, element-wise
        * scalar ``+`` image
        * image ``+`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img + img
            >>> z.array
            >>> z = 10 + img
            >>> z.array

        ..warning:: Values will be wrapped not clipped to the range of the
            pixel datatype.
        """
        return self._binop(self, other, lambda x, y: x + y)

    def __radd__(self, other) -> "Image":
        return self._binop(self, other, lambda x, y: y + x)

    def __iadd__(self, other) -> "Image":
        """
        Overloaded in-place ``+=`` operator

        :return: self, modified in place
        :rtype: :class:`Image`

        Add another image or a scalar to an Image in place.

        :seealso: :meth:`__add__`
        """
        self._A = self._binop(self, other, lambda x, y: x + y)._A
        return self

    def __sub__(self, other) -> "Image":
        """
        Overloaded ``-`` operator

        :return: element-wise subtraction of images
        :rtype: :class:`Image`

        Compute the difference of an Image with another image or a scalar.
        Supports:

        * image ``-`` image, element-wise
        * scalar ``-`` image
        * image ``-`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img - img
            >>> z.array
            >>> z = img - 1
            >>> z.array

        ..warning:: Values will be wrapped not clipped to the range of the
            pixel datatype.
        """
        return self._binop(self, other, lambda x, y: x - y)

    def __rsub__(self, other) -> "Image":
        return self._binop(self, other, lambda x, y: y - x)

    def __isub__(self, other) -> "Image":
        """
        Overloaded in-place ``-=`` operator

        :return: self, modified in place
        :rtype: :class:`Image`

        Subtract another image or a scalar from an Image in place.

        :seealso: :meth:`__sub__`
        """
        self._A = self._binop(self, other, lambda x, y: x - y)._A
        return self

    def __truediv__(self, other) -> "Image":
        """
        Overloaded ``/`` operator

        :return: element-wise division of images
        :rtype: :class:`Image`

        Compute the quotient of an Image with another image or a scalar.
        Supports:

        * image ``/`` image, element-wise
        * scalar ``/`` image
        * image ``/`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img / img
            >>> z.array
            >>> z = img / 2
            >>> z.array

        .. note:: The resulting values are floating point.
        """
        return self._binop(self, other, lambda x, y: x / y)

    def __rtruediv__(self, other) -> "Image":
        return self._binop(self, other, lambda x, y: y / x)

    def __itruediv__(self, other) -> "Image":
        """
        Overloaded in-place ``/=`` operator

        :return: self, modified in place
        :rtype: :class:`Image`

        Divide an Image in place by another image or a scalar.

        :seealso: :meth:`__truediv__`
        """
        self._A = self._binop(self, other, lambda x, y: x / y)._A
        return self

    def __floordiv__(self, other) -> "Image":
        """
        Overloaded ``//`` operator

        :return: element-wise floored division of images
        :rtype: :class:`Image`

        Compute the integer quotient of an Image with another image or a scalar.
        Supports:

        * image ``//`` image
        * scalar ``//`` image
        * image ``//`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img / 2
            >>> z.array
            >>> z = img // 2
            >>> z.array
        """
        return self._binop(self, other, lambda x, y: x // y)

    def __rfloordiv__(self, other) -> "Image":
        return self._binop(self, other, lambda x, y: y // x)

    def __ifloordiv__(self, other) -> "Image":
        """
        Overloaded in-place ``//=`` operator

        :return: self, modified in place
        :rtype: :class:`Image`

        Divide an Image in place by another image or a scalar, rounding down.

        :seealso: :meth:`__floordiv__`
        """
        self._A = self._binop(self, other, lambda x, y: x // y)._A
        return self

    def __neg__(self) -> "Image":
        """
        Overloaded unary ``-`` operator

        :return: element-wise negation of image
        :rtype: :class:`Image`


        Compute the element-wise negation of an Image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, -2], [-3, 4]], dtype='int8')
            >>> img.print()
            >>> neg_img = -img
            >>> neg_img.print()
        """
        return self._unop(self, lambda x: -x)

    # bitwise
    def __and__(self, other) -> "Image":
        """
        Overloaded ``&`` operator

        :return: element-wise binary-and of images
        :rtype: :class:`Image`

        Compute the binary-and of an Image with another image or a scalar.
        Supports:

        * image ``&`` image
        * scalar ``&`` image
        * image ``&`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img & Image([[2, 2], [2, 2]])
            >>> z.array
            >>> z = img & 1
            >>> z.array
        """
        return self._binop(self, other, lambda x, y: x & y)

    def __iand__(self, other) -> "Image":
        """
        Overloaded in-place ``&=`` operator

        :return: self, modified in place
        :rtype: :class:`Image`

        Bitwise-AND an Image in place with another image or a scalar.

        :seealso: :meth:`__and__`
        """
        self._A = self._binop(self, other, lambda x, y: x & y)._A
        return self

    def __or__(self, other) -> "Image":
        """
        Overloaded ``|`` operator

        :return: element-wise binary-or of images
        :rtype: :class:`Image`

        Compute the binary-or of an Image with another image or a scalar.
        Supports:

        * image ``|`` image
        * scalar ``|`` image
        * image ``|`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img | Image([[2, 2], [2, 2]])
            >>> z.array
            >>> z = img | 1
            >>> z.array
        """
        return self._binop(self, other, lambda x, y: x | y)

    def __ior__(self, other) -> "Image":
        """
        Overloaded in-place ``|=`` operator

        :return: self, modified in place
        :rtype: :class:`Image`

        Bitwise-OR an Image in place with another image or a scalar.

        :seealso: :meth:`__or__`
        """
        self._A = self._binop(self, other, lambda x, y: x | y)._A
        return self

    def __xor__(self, other) -> "Image":
        """
        Overloaded ``^`` operator

        :return: element-wise binary-xor of images
        :rtype: :class:`Image`

        Compute the binary-xor of an Image with another image or a scalar.
        Supports:

        * image ``^`` image
        * scalar ``^`` image
        * image ``^`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img ^ Image([[2, 2], [2, 2]])
            >>> z.array
            >>> z = img ^ 1
            >>> z.array
        """
        return self._binop(self, other, lambda x, y: x ^ y)

    def __ixor__(self, other) -> "Image":
        """
        Overloaded in-place ``^=`` operator

        :return: self, modified in place
        :rtype: :class:`Image`

        Bitwise-XOR an Image in place with another image or a scalar.

        :seealso: :meth:`__xor__`
        """
        self._A = self._binop(self, other, lambda x, y: x ^ y)._A
        return self

    def __lshift__(self, other) -> "Image":
        """
        Overloaded ``<<`` operator

        :return: element-wise binary-left-shift of images
        :rtype: :class:`Image`

        Left shift pixel values in an Image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img  << 1
            >>> z.array
        """
        if not isinstance(other, int):
            raise ValueError("left shift must be by integer amount")
        return self._binop(self, other, lambda x, y: x << y)

    def __ilshift__(self, other) -> "Image":
        """
        Overloaded in-place ``<<=`` operator

        :return: self, modified in place
        :rtype: :class:`Image`

        Left-shift pixel values of an Image in place.

        :seealso: :meth:`__lshift__`
        """
        if not isinstance(other, int):
            raise ValueError("left shift must be by integer amount")
        self._A = self._binop(self, other, lambda x, y: x << y)._A
        return self

    def __rshift__(self, other) -> "Image":
        """
        Overloaded ``>>`` operator

        :return: element-wise binary-right-shift of images
        :rtype: :class:`Image`

        Right shift pixel values in an Image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img >> 2
            >>> z.array
        """
        if not isinstance(other, int):
            raise ValueError("left shift must be by integer amount")
        return self._binop(self, other, lambda x, y: x >> y)

    def __irshift__(self, other) -> "Image":
        """
        Overloaded in-place ``>>=`` operator

        :return: self, modified in place
        :rtype: :class:`Image`

        Right-shift pixel values of an Image in place.

        :seealso: :meth:`__rshift__`
        """
        if not isinstance(other, int):
            raise ValueError("right shift must be by integer amount")
        self._A = self._binop(self, other, lambda x, y: x >> y)._A
        return self

    def __mod__(self, other) -> "Image":
        """
        Overloaded ``%`` operator

        :return: image with stacked planes
        :rtype: :class:`Image`

        ``img1 % img2`` results in an image with the planes of ``img1`` followed by the
        planes of ``img2``.  The two images must have the same number of rows and
        columns and data type. The color order of the resulting image is the color order of ``img1``
        followed by the color order of ``img2``.

        The operation also supports stacking a scalar as a plane onto an image, in which case the
        resulting image has one more plane than the original image and the new plane is filled with
        the scalar value.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> z = img % Image([[5, 6], [7, 8]])
            >>> print(z.nplanes)
            >>> z = img % 0
            >>> print(z.nplanes)

        :seealso: :meth:`Pstack`
        """
        if smb.isscalar(other):
            other = Image.Constant(self.size, value=other, dtype=str(self._A.dtype))
        return self.Pstack((self, other))

    def __imod__(self, other) -> "Image":
        """
        Overloaded in-place ``%=`` operator

        :return: self, modified in place
        :rtype: :class:`Image`

        Stack the planes of another image (or a scalar plane) onto this Image
        in place.

        :seealso: :meth:`__mod__`
        """
        if smb.isscalar(other):
            other = Image.Constant(self.size, value=other, dtype=str(self._A.dtype))
        self._A = self.Pstack((self, other))._A
        return self

    # relational
    def __eq__(self, other) -> "Image":
        """
        Overloaded ``==`` operator

        :return: element-wise comparison of pixels
        :rtype: bool :class:`Image`

        Compute the equality between an Image and another image or a scalar.
        Supports:

            * image ``==`` image
            * scalar ``==`` image
            * image ``==`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> (img == 2).print()
            >>> (img == Image([[0, 2], [3, 4]])).print()

        :seealso: :meth:`true` :meth:`false`
        """
        return self._binop(self, other, lambda x, y: x == y)

    def __ne__(self, other) -> "Image":
        """
        Overloaded ``!=`` operator

        :return: element-wise comparison of pixels
        :rtype: bool :class:`Image`

        Compute the inequality between an Image and another image or a scalar.
        Supports:

            * image ``!=`` image
            * scalar ``!=`` image
            * image ``!=`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> (img != 2).print()
            >>> (img != Image([[0, 2], [3, 4]])).print()

        :seealso: :meth:`true` :meth:`false`
        """
        return self._binop(self, other, lambda x, y: x != y)

    def __gt__(self, other) -> "Image":
        """
        Overloaded ``>`` operator

        :return: element-wise comparison of pixels
        :rtype: bool :class:`Image`

        Compute the inequality between an Image and another image or a scalar.
        Supports:

            * image ``>`` image
            * scalar ``>`` image
            * image ``>`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> (img > 2).print()
            >>> (img > Image([[0, 2], [3, 4]])).print()

        :seealso: :meth:`true` :meth:`false`
        """
        return self._binop(self, other, lambda x, y: x > y)

    def __ge__(self, other) -> "Image":
        """
        Overloaded ``>=`` operator

        :return: element-wise comparison of pixels
        :rtype: bool :class:`Image`

        Compute the inequality between an Image and another image or a scalar.
        Supports:

            * image ``>=`` image
            * scalar ``>=`` image
            * image ``>=`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> (img >= 2).print()
            >>> (img >= Image([[0, 2], [3, 4]])).print()

        :seealso: :meth:`true` :meth:`false`
        """
        return self._binop(self, other, lambda x, y: x >= y)

    def __lt__(self, other) -> "Image":
        """
        Overloaded ``<`` operator

        :return: element-wise comparison of images
        :rtype: bool :class:`Image`

        Compute the inequality between an Image and another image or a scalar.
        Supports:

            * image ``<`` image
            * scalar ``<`` image
            * image ``<`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> (img < 2).print()
            >>> (img < Image([[0, 2], [3, 4]])).print()

        :seealso: :meth:`true` :meth:`false`
        """
        return self._binop(self, other, lambda x, y: x < y)

    def __le__(self, other) -> "Image":
        """
        Overloaded ``<=`` operator

        :return: element-wise comparison of images
        :rtype: bool :class:`Image`

        Compute the inequality between an Image and another image or a scalar.
        Supports:

            * image ``<=`` image
            * scalar ``<=`` image
            * image ``<=`` scalar

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> (img <= 2).print()
            >>> (img <= Image([[0, 2], [3, 4]])).print()

        :seealso: :meth:`true` :meth:`false`
        """
        return self._binop(self, other, lambda x, y: x <= y)

    def __invert__(self) -> "Image":
        """
        Overloaded ``~`` operator

        :return: element-wise inversion of logical values
        :rtype: boo, :class:`Image`

        Returns logical not operation where image values are interpretted as:

            * floating image: True is 1.0, False is 0.0
            * integer image: True is maximum value, False is 0 True is 1 and False is 0.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[True, False], [False, True]])
            >>> z = ~img
            >>> z.print()
        """

        return self._unop(self, lambda x: ~x)

    @staticmethod
    def _binop(left, right, op, logical=False) -> "Image":
        if isinstance(right, left.__class__):
            # both images
            if left.nplanes == right.nplanes:
                return left.__class__(op(left._A, right._A), colororder=left.colororder)
            elif left.nplanes > 1 and right.nplanes == 1:
                # left image is multiplane, right is singleton
                out = []
                for i in range(left.nplanes):
                    out.append(op(left._A[:, :, i], right._A))
                return left.__class__(np.stack(out, axis=2), colororder=left.colororder)
            elif left.nplanes == 1 and right.nplanes > 1:
                # right image is multiplane, left is singleton
                out = []
                for i in range(right.nplanes):
                    out.append(op(left._A, right._A[:, :, i]))
                return right.__class__(
                    np.stack(out, axis=2), colororder=right.colororder
                )
            else:
                raise ValueError("planes mismatch")
        else:
            # right is a scalar or numpy array
            return left.__class__(op(left._A, right), colororder=left.colororder)

    @staticmethod
    def _logicalop(left, right, op) -> "Image":
        true = left.cast(left.true)
        false = left.cast(left.false)

        if isinstance(right, left.__class__):
            # image OP image
            out = np.where(op(left._A, right._A), true, false)
        else:
            out = np.where(op(left._A, right), true, false)

        return left.__class__(out, colororder=left.colororder)

    @staticmethod
    def _unop(left, op) -> "Image":
        return left.__class__(op(left._A), colororder=left.colororder)

    # ---------------------------- functions ---------------------------- #

    def abs(self) -> "Image":
        """
        Absolute value of image

        :return: element-wise absolute value of image
        :rtype: :class:`Image`

        Return element-wise absolute value of pixel values.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[-1, 2], [3, -4]], dtype='int8')
            >>> img.abs().print()
        """
        return self._unop(self, np.abs)

    def sqrt(self) -> "Image":
        """
        Square root of image

        :return: element-wise square root of image
        :rtype: :class:`Image`

        Return element-wise square root of pixel values.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2], [3, 4]])
            >>> img.sqrt().print()
        """
        return self._unop(self, np.sqrt)

    # ---------------------------- graphics ---------------------------- #

    def draw_line(
        self,
        start: tuple[int, int] | np.ndarray,
        end: tuple[int, int] | np.ndarray,
        **kwargs: Any,
    ) -> None:
        """
        Draw line into image

        :param start: start coordinate (u,v)
        :type start: array_like(2)
        :param end: end coordinate (u,v)
        :type end: array_like(2)
        :param kwargs: parameters passed to :func:`~machinevisiontoolbox.base.graphics.draw_line`

        Example, "burn" a line into the Mona Lisa image::


            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("monalisa.png")
            >>> img.draw_line((20,30), (500,600), thickness=5, color="orange")
            >>> img.disp()

        .. plot::

            from machinevisiontoolbox import Image
            img = Image.Read("monalisa.png")
            img.draw_line((20,30), (500,600), thickness=5, color="orange")
            img.disp()

        .. note:: If ``image`` has multiple planes then ``color`` should have the same number
            of elements as the image has planes. If it is a scalar that value is used
            for each color plane. For a color image ``color`` can be
            a string color name.

        :seealso: :func:`~machinevisiontoolbox.base.graphics.draw_line`
        """
        draw_line(self._A, start, end, **kwargs)

    def draw_circle(
        self,
        centre: tuple[int, int] | np.ndarray,
        radius: int,
        center: tuple[int, int] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Draw circle into image

        :param centre: centre coordinate (u,v)
        :type centre: array_like(2)
        :param radius: circle radius in pixels
        :type radius: int
        :param kwargs: parameters passed to :func:`~machinevisiontoolbox.base.graphics.draw_circle`

        Example, "burn" some circles (confetti) into the Mona Lisa::

            >>> from machinevisiontoolbox import Image
            >>> import matplotlib as mpl
            >>> img = Image.Read("monalisa.png")
            >>> colors = mpl.color_sequences["petroff10"]
            >>> for color in colors:
            ...     u = np.random.randint(20, img.umax-20)
                    v = np.random.randint(20, img.vmax-20)
                    r = np.random.randint(10, 50)
                    img.draw_circle((u, v), r, thickness=-1, color=[255*c for c in color])
            >>> img.disp(img)

        .. plot::

            from machinevisiontoolbox import Image
            import matplotlib as mpl
            img = Image.Read("monalisa.png")
            for color in mpl.color_sequences["petroff10"]:
                u = np.random.randint(20, img.umax-20)
                v = np.random.randint(20, img.vmax-20)
                r = np.random.randint(10, 50)
                img.draw_circle((u, v), r, thickness=-1, color=[255*c for c in color])
            img.disp(img)

        .. note:: If ``image`` has multiple planes then ``color`` should have the same number
            of elements as the image has planes. If it is a scalar that value is used
            for each color plane. For a color image ``color`` can be
            a string color name.

        :seealso: :func:`~machinevisiontoolbox.base.graphics.draw_circle`
        """
        if center is not None and centre is None:
            centre = center
        draw_circle(self._A, centre, radius, **kwargs)

    def draw_box(self, **kwargs: Any) -> None:
        """
        Draw box into image

        :param kwargs: parameters passed to :func:`~machinevisiontoolbox.base.graphics.draw_box`

        There are myriad ways to specify the corners of the box.

        Example, draw boxes over the eyes of the Mona Lisa::

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("monalisa.png")
            >>> img.draw_box(lb=(245,170), rt=(290, 210), color="yellow", thickness=5)
            >>> img.draw_box(lb=(315, 175), rt=(370,205), color="blue", thickness=-1)
            >>> img.disp()

        .. plot::

            from machinevisiontoolbox import Image
            img = Image.Read("monalisa.png")
            img.draw_box(lt=(245,210), rb=(290, 170), color="yellow", thickness=5)
            img.draw_box(lt=(315, 205), rb=(370,175), color="blue", thickness=-1)
            img.disp()

        .. note:: If ``image`` has multiple planes then ``color`` should have the same number
            of elements as the image has planes. If it is a scalar that value is used
            for each color plane. For a color image ``color`` can be
            a string color name.

        :seealso: :func:`~machinevisiontoolbox.base.graphics.draw_box`
        """
        draw_box(self._A, **kwargs)

    def draw_labelbox(self, text: str, **kwargs: Any) -> None:
        """
        Draw label box into image

        :param text: text for the box label
        :type text: str
        :param kwargs: parameters passed to :func:`~machinevisiontoolbox.base.graphics.draw_labelbox` to
            specify the box position, color, etc.

        The box position is specified by the parameters accepted by :func:`~machinevisiontoolbox.base.graphics.draw_box`

        Example::

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("monalisa.png")
            >>> img.draw_labelbox("Face", lb=(243,111), rt=(394,329), color="yellow", fontheight=20)
            >>> img.disp()

        .. plot::

            from machinevisiontoolbox import Image
            img = Image.Read("monalisa.png")
            img.draw_labelbox("Face", lb=(243,111), rt=(394,329), color="yellow", fontheight=20)
            img.disp()

        .. note:: If ``image`` has multiple planes then ``color`` should have the same number
            of elements as the image has planes. If it is a scalar that value is used
            for each color plane. For a color image ``color`` can be
            a string color name.

        :seealso: :func:`~machinevisiontoolbox.base.graphics.draw_labelbox`
        """
        draw_labelbox(self._A, text, **kwargs)

    def draw_text(
        self, pos: tuple[int, int] | np.ndarray, text: str, **kwargs: Any
    ) -> None:
        """
        Draw text into image
        :param pos: text position (u,v)
        :type pos: array_like(2)
        :param text: text to draw
        :type text: str
        :param kwargs: parameters passed to :func:`~machinevisiontoolbox.base.graphics.draw_text`

        Example::

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("monalisa.png")
            >>> img.draw_text((340,290), "Smile!", fontheight=40, color="yellow")
            >>> img.disp()

        .. plot::

            from machinevisiontoolbox import Image
            img = Image.Read("monalisa.png")
            img.draw_text((340,290), "Smile!", fontheight=40, color="yellow")
            img.disp()

        .. note:: If ``image`` has multiple planes then ``color`` should have the same number
            of elements as the image has planes. If it is a scalar that value is used
            for each color plane. For a color image ``color`` can be
            a string color name.

        :seealso: :func:`~machinevisiontoolbox.base.graphics.draw_text`
        """
        draw_text(self._A, pos, text, **kwargs)

    def draw_point(
        self,
        pos: tuple[int, int] | np.ndarray,
        marker: str = "+",
        text: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Draw a marker in image

        :param pos: position of marker
        :type pos: array_like(2), ndarray(2,n), list of 2-tuples
        :param marker: marker character, defaults to "+"
        :type marker: str, optional
        :param text: text label, defaults to None
        :type text: str, optional
        :param kwargs: parameters passed to :func:`~machinevisiontoolbox.base.graphics.draw_point`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read("monalisa.png")
            >>> img.draw_point((270,194), "+", "eye", fontsize=1, color="yellow")
            >>> img.draw_point((293,246), "*", "nose", fontsize=1, color="blue")
            >>> img.disp()

        .. plot::

            from machinevisiontoolbox import Image
            img = Image.Read("monalisa.png")
            img.draw_point((270,194), "+", "eye", fontsize=1, color="yellow")
            img.draw_point((293,246), "*", "nose", fontsize=1, color="blue")
            img.disp()

        .. note:: If ``image`` has multiple planes then ``color`` should have the same number
            of elements as the image has planes. If it is a scalar that value is used
            for each color plane. For a color image ``color`` can be
            a string color name.

        :seealso: :func:`~machinevisiontoolbox.base.graphics.draw_text`
        """
        draw_point(self._A, pos, marker, text, **kwargs)

    @classmethod
    def Pstack(cls, images, colororder=None):
        """
        Concatenation of image planes

        :param images: images to concatenate plane-wise
        :type images: iterable of :class:`Image`
        :param colororder: color order for the result, defaults to None
        :type colororder: str, optional
        :raises ValueError: all images must have the same dtype
        :raises ValueError: all images must have the same size
        :return: plane-stacked image
        :rtype: :class:`Image`

        Create a new image by stacking the planes of the input images.
        All images must have the same width, height and dtype. The resulting
        image has a number of planes equal to the sum of planes in all input images.

        If ``colororder`` is not specified and all images have color orders defined,
        the result's color order is constructed by concatenating the color orders
        of the input images.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> r = Image.Random(size=(100, 120), colororder='R')
            >>> g = Image.Random(size=(100, 120), colororder='G')
            >>> b = Image.Random(size=(100, 120), colororder='B')
            >>> rgb = Image.Pstack((r, g, b))
            >>> rgb.nplanes
            >>> rgb.colororder_str

        :seealso: :meth:`Vstack` :meth:`Hstack`

        AUTO_EDIT
        """
        nplanes = images[0].nplanes
        for image in images[1:]:
            if image.dtype != images[0].dtype:
                raise ValueError("all planes must have the same dtype")
            if image.size != images[0].size:
                raise ValueError("all planes must have the same width x height")
            nplanes += image.nplanes

        if colororder is None:
            if all([im.colororder is not None for im in images]):
                # attempt to create color order from the images
                colororder = images[0].colororder
                ip = len(colororder)
                for image in images[1:]:
                    colororder |= Image.colororder2dict(image.colororder, start=ip)
            else:
                colororder = None
        else:
            if len(Image.colororder2dict(colororder)) != nplanes:
                raise ValueError("colororder does not match number of planes")

        return Image(
            np.concatenate([np.atleast_3d(im._A) for im in images], axis=2),
            colororder=colororder,
        )

    def fliplr(self) -> "Image":
        """
        Flip image left-right

        :return: horizontally flipped image
        :rtype: :class:`Image`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2, 3], [4, 5, 6]])
            >>> img.fliplr().A

        :seealso: :meth:`flipud`
        """
        return self.__class__(np.fliplr(self._A), colororder=self.colororder)

    def flipud(self) -> "Image":
        """
        Flip image up-down

        :return: vertically flipped image
        :rtype: :class:`Image`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image([[1, 2, 3], [4, 5, 6]])
            >>> img.flipud().A

        :seealso: :meth:`fliplr`
        """
        return self.__class__(np.flipud(self._A), colororder=self.colororder)

    def fixbad(self, nan=0.0, posinf=None, neginf=None):
        """
        Fix bad values in image

        :param nan: value to replace NaN values, defaults to 0.0
        :type nan: scalar or array-like(n), optional
        :param posinf: value to replace positive infinity values, defaults to maximum value of image dtype
        :type posinf: scalar or array-like(n), optional
        :param neginf: value to replace negative infinity values, defaults to negative maximum value of image dtype
        :type neginf: scalar or array-like(n), optional
        :return: image with bad values fixed
        :rtype: :class:`~machinevisiontoolbox.ImageCore.Image` instance

        Return an image where any pixel that contains a NaN or Inf values has been
        replaced by the specified replacement values.

        The replacement value can be a scalar, in which case all bad values are replaced
        by the same value, or an array-like of length n, where n is the number of
        channels in the image, in which case each channel is replaced by the
        corresponding value.

        By default NaN values are replaced by 0, positive infinity values are replaced
        by the maximum value of the image dtype, and negative infinity values are
        replaced by the negative maximum value of the image dtype.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> from numpy import nan, inf
            >>> img = Image([[1, 2, 3], [4, nan, 6], [7, 8, inf]])
            >>> img.fixbad(posinf=99).print()

        """
        out = self._A.copy()
        if posinf is None:
            posinf = np.finfo(out.dtype).max
        if neginf is None:
            neginf = -np.finfo(out.dtype).max

        mask = np.isnan(out)
        if out.ndim == 2:
            out[mask] = nan
        else:
            mask = mask.any(axis=2)
            out[mask, :] = nan

        mask = np.isposinf(out)
        if out.ndim == 2:
            out[mask] = posinf
        else:
            mask = mask.any(axis=2)
            out[mask, :] = posinf

        mask = np.isneginf(out)
        if out.ndim == 2:
            out[mask] = neginf
        else:
            mask = mask.any(axis=2)
            out[mask, :] = neginf

        return self.__class__(out, colororder=self.colororder)


if __name__ == "__main__":
    from pathlib import Path

    import pytest

    pytest.main(
        [str(Path(__file__).parent.parent.parent / "tests" / "test_core.py"), "-v"]
    )
