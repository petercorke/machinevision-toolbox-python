"""

vcat
grid(shape=(), list or *pos)


reduce(factor, width, height)
warp
affinemap

pad(halign="<^>", valign="^v-", align="<|>v^-, width=, height=)

samesize by scaling and padding
hcat
scale(factor, width, height)
rotate
"""

import numpy as np
import scipy as sp
import cv2 as cv
from machinevisiontoolbox.base import meshgrid, idisp, name2color
import matplotlib.pyplot as plt
from spatialmath import base as smb
from spatialmath import SE2
from matplotlib.widgets import RectangleSelector

_interp_dict = {

'nearest': cv.INTER_NEAREST, # nearest neighbor interpolation
'linear': cv.INTER_LINEAR, #bilinear interpolation
'cubic': cv.INTER_CUBIC, # bicubic interpolation
'area': cv.INTER_AREA, #esampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire'-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
'Lanczos': cv.INTER_LANCZOS4, #Lanczos interpolation over 8x8 neighborhood
'linear exact': cv.INTER_LINEAR_EXACT, # Bit exact bilinear interpolation
    }

class ImageReshapeMixin:

    def trim(self, left=0, right=0, top=0, bottom=0):
        """
        Trim pixels from the edges of the image

        :param left: number of pixels to trim from left side of image, defaults to 0
        :type left: int, optional
        :param right: number of pixels to trim from right side of image, defaults to 0
        :type right: int, optional
        :param top: number of pixels to trim from top side of image, defaults to 0
        :type top: int, optional
        :param bottom: number of pixels to trim from bottom side of image, defaults to 0
        :type bottom: int, optional
        :return: trimmed image
        :rtype: :class:`Image`

        Trim pixels from the edges of the image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img
            >>> img.trim(left=100, bottom=100)
        """
        image = self.A
        y = slice(top, self.height-bottom)
        x = slice(left, self.width - right)
        if self.iscolor:
            image = image[y, x, :]
        else:
            image = image[y, x]
        
        return self.__class__(image, colororder=self.colororder)

    def pad(self, left=0, right=0, top=0, bottom=0, value=0):
        """
        Pad the edges of the image

        :param left: number of pixels to pad on left side of image, defaults to 0
        :type left: int, optional
        :param right: number of pixels to pad on right side of image, defaults to 0
        :type right: int, optional
        :param top: number of pixels to pad on top side of image, defaults to 0
        :type top: int, optional
        :param bottom: number of pixels to pad on bottom side of image, defaults to 0
        :type bottom: int, optional
        :param value: value of pixels to pad with
        :type value: scalar, str, array_like
        :return: padded image
        :rtype: :class:`Image`

        Pad the edges of the image with pixels equal to ``value``.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png', dtype='float')
            >>> img
            >>> img.pad(left=10, bottom=10, top=10, right=10, value='r')

        """
        pw = ((top,bottom),(left,right))
        if isinstance(value, str):
            value = self.like(name2color(value))
        if self.nplanes != len(value):
            raise ValueError('value not compatible with image')

        planes = []
        for i, v in enumerate(value):
            planes.append(np.pad(self.plane(i).image, pw, constant_values=(v, v)))
        out = np.dstack(planes)

        return self.__class__(out, colororder=self.colororder)

    # TODO rationalize stack and cat methods

    # @classmethod
    # def hcat(cls, *pos, pad=0, return_offsets=False):
    #     """
    #     Horizontal concatenation of images

    #     :param pad: gap between images, defaults to 0
    #     :type pad: int, optional
    #     :param return_offsets: additionally return the horizontal coordinates of each image, defaults to False
    #     :type return_offsets: bool, optional
    #     :raises ValueError: _description_
    #     :raises ValueError: _description_
    #     :return: horizontally stacked images
    #     :rtype: :class:`Image` instance

    #     Example:

    #     .. runblock:: pycon

    #         >>> img = Image.Read('street.png')
    #         >>> Image.hcat(im, im, im)
    #         >>> Image.hcat(im, im, im, return_offsets=True)

    #     :seealso: :meth:`vcat`
    #     """

    #     if isinstance(pos[0], (tuple, list)):
    #         images = pos[0]
    #     else:
    #         images = pos
        
    #     height = max([image.height for image in images])

    #     nplanes = images[0].nplanes
    #     if not all([image.nplanes == nplanes for image in images]):
    #         raise ValueError('all images must have same number of planes')
    #     dtype = images[0].dtype
    #     if not all([image.dtype == dtype for image in images]):
    #         raise ValueError('all images must have same dtype')

    #     u = []
    #     if nplanes == 1:
    #         # single plane case
    #         combo = np.empty(shape=(height,0), dtype=dtype)

    #         for image in images:
    #             if image.height < height:
    #                 image = np.pad(image.image, ((0, height - image.height), (0, 0)),
    #                     constant_values=(pad,0))
    #             else:
    #                 image = image.image
    #             u.append(combo.shape[1])
    #             combo = np.hstack((combo, image))
    #     else:
    #         # multiplane case
    #         combo = np.empty(shape=(height,0, nplanes), dtype=dtype)

    #         for image in images:
    #             if image.height < height:
    #                 image = np.pad(image.image, ((0, height - image.height), (0, 0), (0, 0)),
    #                     constant_values=(pad,0))
    #             else:
    #                 image = image.image
    #             u.append(combo.shape[1])
    #             combo = np.hstack((combo, image))
        
    #     if return_offsets:
    #         return cls(combo), u
    #     else:
    #         return cls(combo)

    # @classmethod
    # def vcat(cls, *pos, pad=0, return_offsets=False):

    #     if isinstance(pos[0], (tuple, list)):
    #         images = pos[0]
    #     else:
    #         images = pos
        
    #     width = max([image.width for image in images])

    #     combo = np.empty(shape=(0, width))

    #     v = []
    #     for image in images:
    #         if image.width < width:
    #             image = np.pad(image.image, ((width - image.width, 0), (0, 0)),
    #                 constant_values=(pad, 0))
    #         else:
    #             image = image.image
    #         v.append(combo.shape[0])
    #         combo = np.vstack((combo, image))
        
    #     if return_offsets:
    #         return cls(combo), v
    #     else:
    #         return cls(combo)

    @classmethod
    def Hstack(cls, images, sep=1, bgcolor=None, return_offsets=False):
        """
        Horizontal concatenation of images

        :param images: images to concatenate horizontally
        :type images: iterable of :class:`Image`
        :param sep: separation between images, defaults to 1
        :type sep: int, optional
        :param bgcolor: color of background, seen in the separation between
            images, defaults to black
        :type bgcolor: scalar, string, array_like, optional
        :param return_offsets: additionally return the horizontal coordinates of
            each input image within the output image, defaults to False
        :type return_offsets: bool, optional
        :raises ValueError: all images must have the same dtype
        :raises ValueError: all images must have the same color order
        :return: horizontally stacked images
        :rtype: :class:`Image`

        Create a new image by stacking the input images horizontally, with a
        vertical separator line of width ``sep`` and color ``bgcolor``.

        The horizontal coordinate of the first column of each image, in the
        composite output image, can be optionally returned if ``return_offsets``
        is True.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('street.png')
            >>> img
            >>> Image.Hstack((img, img, img))
            >>> Image.Hstack((img, img, img), return_offsets=True)

        :seealso: :meth:`Vstack` :meth:`Tile`
        """
        width = (len(images) - 1) * sep
        height = 0
        colororder = None
        for image in images:
            width += image.shape[1]
            if image.shape[0] > height:
                height = image.shape[0]
            if image.iscolor:
                if colororder is not None:
                    if colororder != image.colororder:
                        raise ValueError('all tiles must have same color order')
                colororder = image.colororder
            if image.dtype != images[0].dtype:
                raise ValueError('all tiles must have same dtype')
            #TODO check if colororder matches

        if bgcolor is None:
            if colororder is None:
                bgcolor = 0
            else:
                bgcolor = [0,] * len(colororder)
        canvas = cls.Constant(width, height, bgcolor, dtype=images[0].dtype)
        # if colororder is not None:
        #     canvas = canvas.colorize(colororder=colororder)
        
        width = 0
        u = []
        for image in images:
            u.append(width)
            if colororder is not None and not image.iscolor:
                image = image.colorize(colororder=colororder)
            canvas.paste(image, (width, 0))
            width += image.shape[1] + sep

        if return_offsets:
            return cls(canvas, colororder=colororder), u
        else:
            return cls(canvas, colororder=colororder)

    @classmethod
    def Vstack(cls, images, sep=1, bgcolor=None, return_offsets=False):
        """
        Vertical concatenation of images

        :param images: images to concatenate vertically
        :type images: iterable of :class:`Image`
        :param sep: separation between images, defaults to 1
        :type sep: int, optional
        :param bgcolor: color of background, seen in the separation between
            images, defaults to black
        :type bgcolor: scalar, string, array_like, optional
        :param return_offsets: additionally return the vertical coordinates of
            each input image within the output image, defaults to False
        :type return_offsets: bool, optional
        :raises ValueError: all images must have the same dtype
        :raises ValueError: all images must have the same color order
        :return: vertically stacked images
        :rtype: :class:`Image`

        Create a new image by stacking the input images vertically, with a
        horizontal separator line of width ``sep`` and color ``bgcolor``.

        The vertical coordinate of the first row of each image, in the
        composite output image, can be optionally returned if ``return_offsets``
        is True.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('street.png')
            >>> img
            >>> Image.Vstack((img, img, img))
            >>> Image.Vstack((img, img, img), return_offsets=True)

        :seealso: :meth:`Hstack` :meth:`Tile`
        """
        height = (len(images) - 1) * sep
        width = 0
        colororder = None
        for image in images:
            height += image.shape[0]
            if image.shape[1] > width:
                width = image.shape[1]
            if image.iscolor:
                if colororder is not None:
                    if colororder != image.colororder:
                        raise ValueError('all tiles must have same color order')
                colororder = image.colororder
            if image.dtype != images[0].dtype:
                raise ValueError('all tiles must have same dtype')
            #TODO check if colororder matches

        if bgcolor is None:
            if colororder is None:
                bgcolor = 0
            else:
                bgcolor = [0,] * len(colororder)
        canvas = cls.Constant(width, height, bgcolor, dtype=images[0].dtype)
        # if colororder is not None:
        #     canvas = canvas.colorize(colororder=colororder)
        
        height = 0
        v = []
        for image in images:
            v.append(height)
            if colororder is not None and not image.iscolor:
                image = image.colorize(colororder=colororder)
            canvas.paste(image, (0, height))
            height += image.shape[0] + sep

        if return_offsets:
            return cls(canvas, colororder=colororder), v
        else:
            return cls(canvas, colororder=colororder)
    
    @classmethod
    def Tile(cls, tiles, columns=4, sep=2, bgcolor=None):
        """
        Tile images into a grid

        :param tiles: images to tile
        :type tiles: iterable of :class:`Image`
        :param columns: number of columns in the grid, defaults to 4
        :type columns: int, optional
        :param sep: separation between images, defaults to 1
        :type sep: int, optional
        :param bgcolor: color of background, seen in the separation between images, defaults to black
        :type bgcolor: scalar, string, array_like, optional
        :raises ValueError: all images must have the same size
        :raises ValueError: all images must have the same dtype
        :return: grid of images
        :rtype: :class:`Image` instance

        Construct a new image by tiling the input images into a grid.
        
        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image, ImageCollection
            >>> images = ImageCollection('campus/*.png')  # image iterator
            >>> Image.Tile(images)

        :seealso: :meth:`Hstack` :meth:`Vstack`
        """
        # exemplars, shape=(-1, columns), **kwargs)

        # TODO tile a sequence into specified shape

        shape = tiles[0].shape
        colororder = tiles[0].colororder
        for tile in tiles[1:]:
            if tile.shape != shape:
                raise ValueError('all tiles must be same size')
            if tile.dtype != tiles[0].dtype:
                raise ValueError('all tiles must have same dtype')

        nrows = int(np.ceil(len(tiles) / columns))
        if bgcolor is None:
            if colororder is None:
                bgcolor = 0
            else:
                bgcolor = [0,] * len(colororder)
        canvas = cls.Constant(
                    columns * shape[1] + (columns - 1) * sep,
                    nrows * shape[0] + (nrows - 1) * sep,
                    bgcolor,
                    dtype=tiles[0].dtype)
        if len(shape) == 3:
            canvas = canvas.colorize(colororder=colororder)

        v = 0
        iterator = iter(tiles)
        try:
            while True:
                u = 0  # start new column
                for c in range(columns):
                    # for each column
                    im = next(iterator)
                    canvas.paste(im, (u, v), 'set', 'topleft')
                    u += shape[1] + sep
                v += shape[0] + sep
        except StopIteration:
            # ran out of images
            pass

        return canvas

    def decimate(self, m=2, sigma=None):
        """
        Decimate an image

        :param m: decimation factor
        :type m: int
        :param sigma: standard deviation for Gaussian kernel smoothing, defaults to None
        :type sigma: float, optional
        :raises ValueError: decimation factor m must be an integer
        :return: decimated image
        :rtype: :class:`Image`

        Return a decimated version of the image whose size is
        reduced by subsampling every ``m`` (an integer) pixels in both dimensions.  
        
        The image is smoothed
        with a Gaussian kernel with standard deviation ``sigma``.  If

        - ``sigma`` is None then  a value of ``m/2`` is used,
        - ``sigma`` is zero then no smoothing is performed.

        .. note::

            - If the image has multiple planes, each plane is decimated.
            - Smoothing is used to eliminate aliasing artifacts and the
              standard deviation should be chosen as a function of the maximum
              spatial frequency in the image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Random(6)
            >>> img.print()
            >>> img.decimate(2, sigma=0).print()

        :references:
            - Robotics, Vision & Control for Python, Section 11.7.2, P. Corke, Springer 2023.

        :seealso: :meth:`replicate` :meth:`scale`
        """
        if (m - np.ceil(m)) != 0:
            raise ValueError(m, 'decimation factor m must be an integer')

        if sigma is None:
            sigma = m / 2

        # smooth image
        if sigma > 0:
            ims = self.smooth(sigma)
        else:
            ims = self

        return self.__class__(ims.image[0:-1:m, 0:-1:m, ...], colororder=self.colororder)


    def replicate(self, n=1):
        r"""
        Replicate image pixels

        :param n: replication factor, defaults to 1
        :type n: int, optional
        :return: image with replicated pixels
        :rtype: :class:`Image`

        Create an image where each input pixel becomes an :math:`n \times n` 
        patch of pixel values. This is a simple way of upscaling an image.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Random(5)
            >>> img.print()
            >>> bigger = img.replicate(2)
            >>> bigger.print()

        .. note::
            - Works only for greyscale images.
            - The resulting image is "blocky", apply Gaussian smoothing to 
              reduce this.

        :references:
            - Robotics, Vision & Control for Python, Section 11.7.2, P. Corke, Springer 2023.

        :seealso: :meth:`decimate`
        """
        # TODO merge with other version, handle color
        rowrep = np.empty_like(self.A, shape=(self.shape[0] * n, self.shape[1]))
        for row in range(n):
            rowrep[row::n, :] = self.A
        rowcolrep = np.empty_like(self.A, shape=(self.shape[0] * n, self.shape[1] * n))
        for col in range(n):
            rowcolrep[:, col::n] = rowrep
        return self.__class__(rowcolrep)

    def roi(self, bbox=None):
        """
        Extract region of interest

        :param bbox: region as [umin, umax, vmin, vmax]
        :type bbox: array_like(4)
        :return: region of interest, optional bounding box
        :rtype: :class:`Image`, list

        Return the specified region of the image.  If ``bbox`` is None the image
        is displayed using Matplotlib and the user can interactively select the
        region, returning the image region and the bounding box ``[umin, umax,
        vmin, vmax]``. 

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('monalisa.png')
            >>> smile = img.roi([265, 342, 264, 286])

        """
        if bbox is None:
            # use Rectangle widget to allow user to draw it

            def line_select_callback(eclick, erelease, roi):
                # called on rectangle release
                roi.extend([eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata])
                plt.gcf().canvas.stop_event_loop()  # unblock

            roi = []
            rs = RectangleSelector(plt.gca(), lambda e1, e2: line_select_callback(e1, e2, roi),
                                                drawtype='box', useblit=True,
                                                button=[1, 3],  # don't use middle button
                                                minspanx=5, minspany=5,
                                                spancoords='pixels',
                                                interactive=True)
            rs.set_active(True)
            plt.gcf().canvas.start_event_loop(timeout=-1)  # block till rectangle released
            rs.set_active(False)
            roi = np.round(np.r_[roi]).astype(int)  # roound to nearest int
        else:
            # get passed vector
            roi = smb.getvector(bbox, 4, dtype=int)

        left, right, top, bot = roi
        if left >= right or top >= bot:
            raise ValueError('ROI should be top-left and bottom-right corners')
        # TODO check row/column ordering, and ndim check
        
        if self.ndim > 2:
            roi = self.image[top:bot+1, left:right+1, :]
        else:
            roi = self.image[top:bot+1, left:right+1]

        if bbox is None:
            return self.__class__(roi, colororder=self.colororder), [left, right, top, bot]
        else:
            return self.__class__(roi, colororder=self.colororder)

    def samesize(self, image2, bias=0.5):
        """
        Automatic image trimming

        :param v: image to match size with
        :type image2: :class:`Image` or array_like(2)
        :param bias: bias that controls what part of the image is cropped, defaults to 0.5
        :type bias: float, optional
        :return: resized image
        :rtype out: :class:`Image`

        Return a version of the image that has the same dimensions as ``image2``.
        This is achieved by cropping (to match the aspect ratio) and
        scaling (to match the size).

        ``bias`` controls which part of the image is cropped. ``bias`` = 0.5 is
        symmetric cropping, ``bias`` < 0.5 moves the crop window up or to
        the left, while ``bias``>0.5 moves the crop window down or to the
        right.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> foreground = Image.Read("greenscreen.png", dtype="float")
            >>> foreground
            >>> background = Image.Read("road.png", dtype="float")
            >>> background
            >>> background.samesize(foreground)

        :references:
            - Robotics, Vision & Control for Python, Section 11.4.1.1, P. Corke, Springer 2023.
        
        :seealso: :meth:`trim` :meth:`scale`
        """
        # check inputs
        if bias < 0 or bias > 1:
            raise ValueError(bias, 'bias must be in range [0, 1]')

        im = self.image

        sc = np.r_[image2.shape[:2]] / np.r_[im.shape[:2]]
        o = self.scale(sc.max())

        if o.height > image2.width:  # rows then columns
            # scaled image is too high, so trim rows
            d = o.height - image2.height
            d1 = max(1, int(np.floor(d * bias)))
            d2 = d - d1
            # [1 d d1 d2]
            o = o.image[d1:-d2, :, :]  # TODO check indexing
        if o.width > image2.width:
            # scaled image is too wide, so trim columns
            d = o.width - image2.width
            d1 = max(1, int(np.floor(d * bias)))
            d2 = d - d1
            # [2 d d1 d2]
            out = o.image[:, d1:-d2, :]  # TODO check indexing

        return self.__class__(out, colororder=self.colororder)

    def scale(self, sfactor, sigma=None, interpolation=None):
        """
        Scale an image

        :param sfactor: scale factor
        :type sfactor: scalar
        :param sigma: standard deviation of kernel for image smoothing, in pixels
        :type sigma: float
        :raises ValueError: bad interpolation string
        :raises ValueError: bad interpolation value
        :return: smoothed image
        :rtype: :class:`Image` instance

        Rescale the image. If ``sfactor> 1`` the image is enlarged. 
        
        If ``sfactor < 1`` the image is made smaller and smoothing can be
        applied to reduce sampling artefacts. If ``sigma`` is None, use default
        for scale by sigma=1/sfactor/2. If ``sigma=0`` perform no smoothing.

        =============  ====================================
        interpolation  description
        =============  ====================================
        ``'cubic'``    bicubic interpolation
        ``'linear'``   bilinear interpolation
        ``'area'``     resampling using pixel area relation
        =============  ====================================

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('monalisa.png')
            >>> img.scale(2)
            >>> img.scale(0.5)

        :references:
            - Robotics, Vision & Control for Python, Section 11.7.2, P. Corke, Springer 2023.

        :seealso: `opencv.resize <https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d>`_
        """
        # check inputs
        if not smb.isscalar(sfactor):
            raise TypeError(sfactor, 'factor is not a scalar')

        if interpolation is None:
            if sfactor > 1:
                interpolation = cv.INTER_CUBIC
            else:
                interpolation = cv.INTER_CUBIC
        elif isinstance(interpolation, str):
            if interpolation == 'cubic':
                interpolation = cv.INTER_CUBIC
            elif interpolation == 'linear':
                interpolation = cv.INTER_LINEAR
            elif interpolation == 'area':
                interpolation = cv.INTER_AREA
            else:
                raise ValueError('bad interpolation string')
        else:
            raise TypeError('bad interpolation value')

        if sfactor < 1:
            if sigma is None:
                sigma = 1 / sfactor / 2
            if sigma > 0:
                im = self.smooth(sigma)
        else:
            im = self
        out = cv.resize(im.image, None, fx=sfactor, fy=sfactor, 
            interpolation=interpolation)

        return self.__class__(out, colororder=self.colororder)

    def rotate(self,
               angle,
               centre=None):
        """
        Rotate an image

        :param angle: rotatation angle [radians]
        :type angle: scalar
        :param centre: centre of rotation, defaults to centre of image
        :type centre: array_like(2)
        :return: rotated image
        :rtype: :class:`Image`

        Rotate the image counter-clockwise by angle ``angle`` in radians.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('monalisa.png')
            >>> out = img.rotate(0.5)
            >>> out.disp()

        .. note::
            - Rotation is defined with respect to a z-axis which is into the
              image, therefore counter-clockwise is a positive angle.
            - The pixels in the corners of the resulting image will be
              undefined.

        """
        # TODO note that there is cv.getRotationMatrix2D and cv.warpAffine
        # https://appdividend.com/2020/09/24/how-to-rotate-an-image-in-python-
        # using-opencv/

        if not smb.isscalar(angle):
            raise ValueError(angle, 'angle is not a valid scalar')

        # TODO check optional inputs


        if centre is None:
            centre = (self.width / 2, self.height / 2)
        elif len(centre) != 2:
            raise ValueError('centre must be length 2')

        shape = (self.width, self.height)

        M = cv.getRotationMatrix2D(centre, np.degrees(angle), 1.0)

        out = cv.warpAffine(self.A, M, shape)
        return self.__class__(out, colororder=self.colororder)


    def rotate_spherical(self, R):
        r"""
        Rotate a spherical image

        :param R: an SO(3) rotation matrix
        :type R: :class:`spatialmath.pose3d.SO3`
        :return: rotated spherical image
        :rtype: :class:`Image`

        Rotates pixels in the input spherical image by the SO(3) rotation matrix.

        A spherical image is represented by a rectangular array of pixels with a
        horizintal domain that spans azimuth angle  :math:`\phi \in [0, 2\pi]`
        and a vertical domain that spans colatitude angle :math:`\theta \in [0,
        \pi]`.

        :seealso: :meth:`meshgrid` :meth:`uspan` :meth:`vspan` :func:`scipy.interpolate.griddata`
        """
        Phi, Theta = self.meshgrid(*self.domain)
        nPhi, nTheta = sphere_rotate(Phi, Theta, R)

        # warp the image
        return self.interp2d(nPhi, nTheta, domain=self.domain)
        
    # ======================= interpolate ============================= #

    def meshgrid(self=None, width=None, height=None, step=1):
        """
        Coordinate arrays for image

        :param width: width of array in pixels, defaults to width of image
        :type width: int, optional
        :param height: height of array in pixels, defaults to height of image
        :type height: int, optional
        :return: domain of image
        :rtype u: ndarray(H,W), ndarray(H,W)

        Create a pair of arrays ``U`` and ``V`` that describe the domain of the
        image. The element ``U(u,v) = u`` and ``V(u,v) = v``. These matrices can
        be used for the evaluation of functions over the image such as
        interpolation and warping. 

        Invoking as a class method with ``self=None`` is a convenient way to
        access ``base.meshgrid``.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Zeros(3)
            >>> U, V = img.meshgrid()
            >>> U
            >>> V
            >>> Image(U**2 + V**2).image
            >>> U, V = Image.meshgrid(None, 4, 4)
            >>> U

        :seealso: :func:`~machinevisiontoolbox.base.meshgrid.meshgrid`
        """
        if self is not None:
            u = self.uspan(step)
            v = self.vspan(step)
        else:                
            u = np.arange(0, width, step)
            v = np.arange(0, height, step)
        
        return np.meshgrid(u, v)

    def warp(self, U, V, interp=None, domain=None):
        r"""
        Image warping

        :param U: u-coordinate array for output image
        :type U: ndarray(Wo,Ho)
        :param V: u-coordinate array for output image
        :type V: ndarray(Wo,Ho)
        :param interp: interpolation mode, defaults to None
        :type interp: str, optional
        :param domain: domain of output image, defaults to None
        :type domain: (ndarray(H,W), ndarray(H,W)), optional
        :return: warped image
        :rtype: :class:`Image`

        Compute an image by warping the input image.  The output image is
        :math:`H_o \times W_o` and output pixel (u,v) is interpolated from the
        input image coordinate (U[u,v], V[u,v]):

        .. math:: Y_{u,v} = X_{u^\prime, v^\prime} \mbox{, where } u^\prime = U_{u,v}, v^\prime = V_{u,v}

        .. note::  Uses OpenCV.

        :seealso: :meth:`interp2d` :meth:`meshgrid` `opencv.remap <https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#gab75ef31ce5cdfb5c44b6da5f3b908ea4>`_
        """
        # TODO more interpolation modes
        img = cv.remap(self.A, U.astype("float32"), V.astype("float32"), cv.INTER_LINEAR)
        return self.__class__(img, colororder=self.colororder, domain=domain)

    def interp2d(self, U, V, Ud=None, Vd=None, **kwargs):
        r"""
        Image warping

        :param U: u-coordinate array for output image
        :type U: ndarray(Ho,Wo)
        :param V: u-coordinate array for output image
        :type V: ndarray(Ho,Wo)
        :param Ud: u-coordinate array for domain of input image, defaults to None
        :type Ud: ndarray(H,W), optional
        :param Vd: u-coordinate array for domain of input image, defaults to None
        :type Vd: ndarray(H,W), optional
        :return: warped image
        :rtype: :class:`Image`

        Compute an image by warping the input image.  The output image is
        :math:`H_o \times W_o` and output pixel (u,v) is interpolated from the
        input image coordinate (U[v,u], V[v,u]).

        .. math:: Y_{u,v} = X_{u^\prime, v^\prime} \mbox{, where } u^\prime = U_{u,v}, v^\prime = V_{u,v}

        The coordinates in ``U`` and ``V`` are with respect to the domain
        of the input image but can be overridden by specifying ``Ud`` and ``Vd``.

        .. note::  Uses SciPy

        :seealso: :meth:`warp` :meth:`meshgrid` :meth:`uspan` :meth:`vspan` :func:`scipy.interpolate.griddata`
        """

        if Ud is None and Vd is None:
            if self.domain is None:
                Ud, Vd = self.meshgrid()
            else:
                Ud, Vd = np.meshgrid(*self.domain)

        points = np.array((Ud.flatten(), Vd.flatten())).T
        values = self.image.flatten()
        xi = np.array((U.flatten(), V.flatten())).T
        Zi = sp.interpolate.griddata(points, values, xi)
        
        return self.__class__(Zi.reshape(U.shape), **kwargs)

    # TODO, this should be warp_affine
    def warp_affine(self, M, inverse=False, size=None, bgcolor=None):
        r"""
        Affine warp of image

        :param M: affine matrix
        :type M: ndarray(2,3), SE2
        :param inverse: warp with inverse of ``M``, defaults to False
        :type inverse: bool, optional
        :param size: size of output image, defaults to size of input image
        :type size: array_like(2), optional
        :param bgcolor: background color, defaults to None
        :type bgcolor: scalar, str, array_like, optional
        :return: warped image
        :rtype: :class:`Image`

        Apply an affine warp to the image. Pixels in the output image that
        correspond to pixels outside the input image are set to ``bgcol``.

        .. math:: Y_{u,v} = X_{u^\prime, v^\prime} \mbox{, where } \begin{pmatrix} u^\prime \\ v^\prime \end{pmatrix} = \mat{M} \begin{pmatrix} u \\ v \\ 1 \end{pmatrix}

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> import numpy as np
            >>> from spatialmath import SE2
            >>> img = Image.Read('monalisa.png')
            >>> M = np.diag([0.25, 0.25, 1]) * SE2(100, 200)  # scale and translate
            >>> M
            >>> out = img.warp_affine(M, bgcolor=np.nan)  # unmapped pixels are NaNs
            >>> out.disp(badcolor="r")  # display warped image with NaNs as red

        .. note:: Only the first two rows of ``M`` are used.

        :seealso: :meth:`warp` `opencv.warpAffine <https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983>`_
        """
        flags = cv.INTER_CUBIC
        if inverse:
            flags |= cv.WARP_INVERSE_MAP
        
        # TODO interpolation flags
        
        if size is None:
            size = self.size

        if bgcolor is not None:
            bordermode = cv.BORDER_CONSTANT
            bordervalue = [bgcolor,] * self.nplanes
        else:
            bordermode = None
            bordervalue = None

        if isinstance(M, SE2):
            M = M.A
        out = cv.warpAffine(src=self.image, M=M[:2, :], dsize=size, flags=flags, borderMode=bordermode, borderValue=bordervalue)
        return self.__class__(out, colororder=self.colororder)

    def warp_perspective(self, H, method='linear', inverse=False, tile=False, size=None):
        r"""
        Perspective warp

        :param H: homography
        :type H: ndarray(3,3)
        :param method: interpolation mode: 'linear' [default], 'nearest'
        :type method: str, optional
        :param inverse: use inverse of ``H``, defaults to False
        :type inverse: bool, optional
        :param tile: return minimal enclosing tile, defaults to False
        :type tile: bool, optional
        :param size: size of output image, defaults to size of input image
        :type size: array_like(2), optional
        :raises TypeError: H must be a 3x3 NumPy array
        :return: warped image
        :rtype: :class:`Image`

        Applies a perspective warp to the input image.  
        
        .. math:: Y_{u,v} = X_{u^\prime, v^\prime} \mbox{, where } u^\prime=\frac{\tilde{u}}{\tilde{w}}, v^\prime=\frac{\tilde{v}}{\tilde{w}}, \begin{pmatrix} \tilde{u} \\ \tilde{v} \\ \tilde{w} \end{pmatrix} = \mat{H} \begin{pmatrix} u \\ v \\ 1 \end{pmatrix}

        The resulting image may
        be smaller or larger than the input image.  If ``tile`` is True then
        the output image is the smallest rectangle that contains the warped
        result, and its position with respect to the origin of the input image,
        and the coordinates of the four corners of the input image.

        :references:
            - Robotics, Vision & Control for Python, Section 14.8, P. Corke, Springer 2023.

        :seealso: :meth:`warp` `opencv.warpPerspective <https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87>`_
        """

        if not (isinstance(H, np.ndarray) and H.shape == (3,3)):
            raise TypeError('H must be a 3x3 NumPy array')
        if size is None:
            size = self.size
        
        if tile:
            corners = np.array([
                [0, size[0], size[0],  0],
                [0, 0,        size[1], size[1]]
            ])
            if inverse:
                # can't use WARP_INVERSE_MAP if we want to compute the output
                # tile
                H = np.linalg.inv(H)
                inverse = False
            wcorners = smb.h2e(H @ smb.e2h(corners))
            tl = np.floor(wcorners.min(axis=1)).astype(int)
            br = np.ceil(wcorners.max(axis=1)).astype(int)
            size = br - tl
            H = smb.transl2(-tl)  @ H

        warp_dict = {
            'linear': cv.INTER_LINEAR,
            'nearest': cv.INTER_NEAREST
        }
        flags = warp_dict[method]
        if inverse:
            flags |= cv.WARP_INVERSE_MAP
        out = cv.warpPerspective(src=self.A, M=H, dsize=tuple(size), flags=flags)

        if tile:
            return self.__class__(out), tl, wcorners
        else:
            return self.__class__(out)

    def undistort(self, K, dist):
        r"""
        Undistort image

        :param K: camera intrinsics
        :type K: ndarray(3,3)
        :param dist: lens distortion parameters
        :type dist: array_like(5)
        :return: undistorted image
        :rtype: :class:`Image`

        Remove lens distortion from image.

        The distortion coefficients are :math:`(k_1, k_2, p_1, p_2, k_3)`
        where :math:`k_i` are radial distortion coefficients and :math:`p_i` are
        tangential distortion coefficients.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image, ImageCollection
            >>> import numpy as np
            >>> images = ImageCollection("calibration/*.jpg")
            >>> K = np.array([[ 534.1, 0, 341.5], [ 0, 534.1, 232.9], [ 0, 0, 1]])
            >>> distortion = np.array([ -0.293, 0.1077, 0.00131, -3.109e-05, 0.04348])
            >>> out = images[12].undistort(K, distortion)
            >>> out.disp()

        :seealso: :meth:`~machinevisiontoolbox.CentralCamera.images2C` `opencv.undistort <https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga69f2545a8b62a6b0fc2ee060dc30559d>`_
        """
        undistorted = cv.undistort(self.image, K, dist)
        return self.__class__(undistorted, colororder=self.colororder)   



     # ------------------------- operators ------------------------------ #

    def column(self):
        raise DeprecationWarning('please use view1d')

    def view1d(self):
        """
        Convert image to a column view

        :return: column view
        :rtype: ndarray(N,) or ndarray(N, np)

        A greyscale image is converted to a 1D array in row-major (C) order, 
        ie. row 0, row 1 etc.

        A color image is converted to a 2D array in row-major (C) order, with
        one row per pixel, and each row is the pixel value, the values of its
        planes.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> Image.Read('street.png').view1d().shape
            >>> Image.Read('monalisa.png').view1d().shape

        .. note:: This creates a view of the original image, so operations on
            the column will affect the original image.
        """
        image = self.image
        if image.ndim == 2:
            return image.ravel()
        elif image.ndim == 3:
            return image.reshape((-1, self.nplanes))
            
    # def col2im(col, im):
    #     """
    #     Convert pixel vector to image

    #     :param col: set of pixel values
    #     :type col: numpy array, shape (N, P)
    #     :param im: image
    #     :type im: numpy array, shape (N, M, P), or a 2-vector (N, M)
    #     indicating image size
    #     :return: image of specified shape
    #     :rtype: numpy array

    #     - ``col2im(col, imsize)`` is an image (H, W, P) comprising the pixel
    #         values in col (N,P) with one row per pixel where N=HxW. ``imsize`` is
    #         a 2-vector (N,M).

    #     - ``col2im(col, im)`` as above but the dimensions of the return are the
    #         same as ``im``.

    #     .. note::

    #         - The number of rows in ``col`` must match the product of the
    #             elements of ``imsize``.

    #     :references:

    #         - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
    #     """

    #     # col = argcheck.getvector(col)
    #     col = np.array(col)
    #     if col.ndim == 1:
    #         nc = 1
    #     elif col.ndim == 2:
    #         nc = col.shape[1]
    #     else:
    #         raise ValueError(col, 'col does not have valid shape')

    #     # second input can be either a 2-tuple/2-array, or a full image
    #     im = np.array(im)  # ensure we can use ndim and shape
    #     if im.ndim == 1:
    #         # input is a tuple/1D array
    #         sz = im
    #     elif im.ndim == 2:
    #         im = Image.getimage(im)
    #         sz = im.shape
    #     elif im.ndim == 3:
    #         im = Image.getimage(im)
    #         sz = np.array([im.shape[0], im.shape[1]])  # ignore 3rd channel
    #     else:
    #         raise ValueError(im, 'im does not have valid shape')

    #     if nc > 1:
    #         sz = np.hstack((sz, nc))

    #     # reshape:
    #     # TODO need to test this
    #     return np.reshape(col, sz)

if __name__ == "__main__":

    from machinevisiontoolbox import Image, ImageCollection
    from math import pi
    
    mona = Image.Read("monalisa.png")
    z = Image.Hstack([mona, mona.smooth(sigma=5)]) #.disp(block=True)
    z.disp()
    pass

    # images = ImageCollection('campus/*.png')  # image iterator
    # Image.Tile(images)
    # im = Image.Read('flowers1.png', dtype='float')
    # im.pad(left=10, bottom=10, top=10, right=10, value='r').disp(block=True)

    # im = Image.Read('street.png')
    # Image.Hstack((im, im, im)).disp()
    # print(Image.Hstack((im, im, im), return_offsets=True)[1])

    # img = Image.Read('monalisa.png')
    # img.stats()
    # # img = Image.Read('monalisa.png', reduce=10, grey=False)
    # # print(img)

    # # tiles = [img for i in range(19)]
    # # Image.Tile(tiles).disp(block=True)

    # img.disp()
    # # z = img.roi()[0]
    # # z.disp(block=True)

    # Image.hcat(img, img).disp(block=True)

    # img.scale(.5).disp()

    # im2 = img.scale(2)
    # im2.disp(block=True)

    # img.rotate(pi / 4, centre=(0,0)).disp()

    # im2 = img.rotate(pi / 4)
    # im2.disp(block=True)
