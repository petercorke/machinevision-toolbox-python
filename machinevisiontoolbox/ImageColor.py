#!/usr/bin/env python

import numpy as np
import spatialmath.base.argcheck as argcheck
import cv2 as cv

from machinevisiontoolbox.base import color, name2color
from machinevisiontoolbox.base import imageio

from scipy import interpolate

class ImageColorMixin:
    """
    Image processing color operations on the Image class
    """
    def mono(self, opt='r601'):
        """
        Convert color image to monochrome

        :param opt: greyscale conversion mode, one of: 'r601' [default], 'r709',
          'value' or 'cv'
        :type opt: str, optional
        :return: monochrome image
        :rtype: :class:`Image`

        Return a greyscale image of the same width and height as the color
        image.  Various conversion options are available:

        ===========  =====================================================
        ``opt``      definition
        ===========  =====================================================
        ``'r601'``   ITU Rec. 601, Y' = 0.229 R' + 0.587 G' + 0.114 B'
        ``'r709'``   ITU Rec. 709, Y' =  0.2126 R' + 0.7152 G' + 0.0722 B'
        ``'value'``  V (value) component of HSV space 
        ``'cv'``     OpenCV colorspace() RGB to gray conversion
        ===========  =====================================================

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img
            >>> img.mono()

        .. note:: For a monochrome image returns a reference to the :class:`Image` instance.

        :references:
            - Robotics, Vision & Control for Python, Section 10.2.7, P. Corke, Springer 2023.

        :seealso: :meth:`colorspace` :meth:`colorize`
        """

        if not self.iscolor:
            return self

        if opt == 'r601':
            mono = 0.229 * self.red() + 0.587 * self.green() + \
                0.114 * self.blue()

        elif opt == 'r709':
            mono = 0.2126 * self.red() + 0.7152 * self.green() + \
                0.0722 * self.blue()

        elif opt == 'value':
            # 'value' refers to the V in HSV space, not the CIE L*
            # the mean of the max and min of RGB values at each pixel
            mn = self.image.min(axis=2)
            mx = self.image.max(axis=2)

            mono = mn / 2 + mx / 2

        elif opt == 'cv':
            if self.isrgb:
              return self.colorspace('gray', src="rgb")
            else:
              return self.colorspace('gray', src="bgr")
        else:
            raise TypeError('unknown type for opt')

        return self.__class__(self.cast(mono.image))


    def chromaticity(self, which='RG'):
        r"""
        Create chromaticity image

        :param which: string comprising single letter color plane names, defaults to 'RG'
        :type which: str, optional
        :return: chromaticity image
        :rtype: :class:`Image` instance

        Convert a tristimulus image to a chromaticity image.  For the case of
        an RGB image and ``which='RG'``

        .. math::
            r = \frac{R}{R+G+B}, \, g = \frac{G}{R+G+B}

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('flowers1.png')
            >>> img.chromaticity()
            >>> img.chromaticity('RB')

        .. note:: The chromaticity color planes are the same as ``which`` but
          lower cased.

        :references:
            - Robotics, Vision & Control for Python, Section 10.2.5, P. Corke, Springer 2023.

        :seealso: :func:`~machinevisiontoolbox.base.color.tristim2cc`
        """
        if not self.iscolor:
            raise ValueError('cannot compute chromaticity for greyscale image')
        if self.nplanes != 3:
            raise ValueError('expecting 3 plane image')

        sum = np.sum(self.image, axis=2)
        r = self.plane(which[0]).image / sum
        g = self.plane(which[1]).image / sum

        return self.__class__(np.dstack((r, g)), colororder=which.lower(), dtype="float32")


    def colorize(self, color=[1, 1, 1], colororder='RGB', alpha=False):
        """
        Colorize a greyscale image

        :param color: base color 
        :type color: string, array_like(3)
        :param colororder: order of color channels of resulting image
        :type colororder: str, dict
        :return: color image
        :rtype: :class:`Image` instance

        The greyscale image is colorized by setting each output pixel to the product
        of ``color`` and the input pixel value.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('shark1.png')
            >>> img.colorize([1, 0, 0])  # red shark
            >>> img.colorize('blue')  # blue shark

        :references:
            - Robotics, Vision & Control for Python, Section 11.3, P. Corke, Springer 2023.

        :seealso: :meth:`mono`
        """

        # TODO, colorize all in list
        if isinstance(color, str):
            color = name2color(color)
        else:
            color = argcheck.getvector(color).astype(self.dtype)
        if self.iscolor:
            raise ValueError(self.image, 'Image must be greyscale')

        # alpha can be False, True, or scalar
        if alpha is False:
            out = np.dstack((color[0] * self.image,
                             color[1] * self.image,
                             color[2] * self.image))
        else:
            if alpha is True:
              alpha = 1

            out = np.dstack((color[0] * self.image,
                           color[1] * self.image,
                           color[2] * self.image,
                           alpha * np.ones(self.shape)))

        if self.isint and np.issubdtype(color.dtype, np.floating):
            out = self.cast(out)

        return self.__class__(out, colororder=colororder)


    def kmeans_color(self, k=None, centroids=None, seed=None):
        """
        k-means color clustering

        **Training**

        :param k: number of clusters, defaults to None
        :type k: int, optional
        :param seed: random number seed, defaults to None
        :type seed: int, optional
        :return: label image, centroids and residual
        :rtype: :class:`Image`, ndarray(P,k), float

        The pixels are grouped into ``k`` clusters based on their Euclidean
        distance from ``k`` cluster centroids.  Clustering is iterative and
        the intial cluster centroids are random.

        The method returns a label image, indicating the assigned cluster for
        each input pixel, the cluster centroids and a residual.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> targets = Image.Read("tomato_124.png", dtype="float", gamma="sRGB")
            >>> ab = targets.colorspace("L*a*b*").plane("a*:b*")
            >>> targets_labels, targets_centroids, resid = ab.kmeans_color(k=3, seed=0)
            >>> targets_centroids

        **Classification**

        :param centroids: cluster centroids from training phase
        :type centroids: ndarray(P,k)
        :return: label image
        :rtype: :class:`Image`

        Pixels in the input image are assigned the label of the closest centroid.

        .. note:: The colorspace of the images could a chromaticity space to classify
          objects while ignoring brightness variation.

        :references:
            - Robotics, Vision & Control for Python, Section 12.1.1.2, P. Corke, Springer 2023.

      :seealso: `opencv.kmeans <https://docs.opencv.org/3.4/d5/d38/group__core__cluster.html#ga9a34dc06c6ec9460e90860f15bcd2f88>`_
      """
        # TODO
        # colorspace can be RGB, rg, Lab, ab

        if seed is not None:
            cv.setRNGSeed(seed)
        
        data = self.to_float().reshape((-1, self.nplanes))
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        if k is not None:
            # perform clustering
            ret, label, centres = cv.kmeans(
                    data=data,
                    K= k,
                    bestLabels=None,
                    criteria=criteria,
                    attempts=10,
                    flags=cv.KMEANS_RANDOM_CENTERS
                )
            return self.__class__(label.reshape(self.shape[:2])), centres.T, ret
        
        elif centroids is not None:
            # assign pixels to given cluster centres
            # M x K
            k = centroids.shape[1]
            data = np.repeat(data[..., np.newaxis], k, axis=2)  # N x M x K

            # compute L2 norm over the error
            distance = np.linalg.norm(data - centroids, axis=1)  # N x K

            # now find which cluster centre gave the smallest error 
            label = np.argmin(distance, axis=1)

            return self.__class__(label.reshape(self.shape[:2]))

    def colorspace(self, dst, src=None):
        """
        Transform a color image between color representations

        :param dst: destination color space (see below)
        :type dst: str
        :param src: source color space (see below), defaults to colororder of image
        :type src: str, optional
        :return: color image in new colorspace
        :rtype: :class:`Image`

        Color space names (synonyms listed on the same line) are:

        =======================     ======================
        Color space name            Option string(s)
        =======================     ======================
        grey scale                  'grey', 'gray'
        RGB (red/green/blue)        'rgb'
        BGR (blue/green/red)        'bgr'
        CIE XYZ                     'xyz', 'xyz_709'
        YCrCb                       'ycrcb'
        HSV (hue/sat/value)         'hsv'
        HLS (hue/lightness/sat)     'hls'
        CIE L*a*b*                  'lab', 'l*a*b*'
        CIE L*u*v*                  'luv', 'l*u*v*'
        =======================     ======================

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> im = Image.Read('flowers1.png')
            >>> im.colorspace('hsv')

        .. note:: RGB images are assumed to be linear, or gamma decoded.

        :references:
            - Robotics, Vision & Control for Python, Section 10.2.7, 10.4.1, P. Corke, Springer 2023.

        :seealso: :meth:`mono` :func:`~machinevisiontoolbox.base.color.colorspace_convert`
        """

        # TODO other color cases
        # TODO check conv is valid

        # TODO conv string parsing

        # ensure floats? unsure if cv.cvtColor operates on ints
        # imf = self.to_float()

        if src is None:
            src = self.colororder_str

        # options gamma, on by default if to is RGB or BGR
        # options white on by default

        out = []
        # print('converting from', src, 'to', dst)

        out = color.colorspace_convert(self.image, src, dst)
        # print('conversion done')
        if out.ndim > 2:
            colororder = dst
            colororder = colororder.replace("*", "*:", 2)
        else:
            colororder = None

        return self.__class__(out, dtype=self.dtype, colororder=colororder)


    @classmethod
    def Overlay(cls, im1, im2, colors='rc'):
        """
        Overlay two greyscale images in different colors

        :param im1: first image
        :type im1: :class:`Image`
        :param im2: second image
        :type im2: :class:`Image`
        :param colors: colors for each image, defaults to 'rc''
        :type colors: 2-element string/list/tuple, optional
        :raises ValueError: images must be greyscale
        :return: overlaid images
        :rtype: :class:`Image`

        Two greyscale images are overlaid in different colors.  Useful for
        visualizing disparity or optical flow.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img1 = Image.Read('eiffel-1.png', mono=True)
            >>> img2 = Image.Read('eiffel-2.png', mono=True)
            >>> Image.Overlay(img1, img2)
            >>> Image.Overlay(img1, img2, 'rg')
            >>> Image.Overlay(img1, img2, ((1, 0, 0), (0, 1, 0)))

        .. note:: Images can be different size, the output image size is the 
          maximum of the dimensions of the input images.  Small dimensions are
          zero padded.  The top-left corner of both images are aligned.

        :seealso: :meth:`anaglyph` :meth:`blend` :meth:`stshow`
        """
        if im1.iscolor or im2.iscolor:
            raise ValueError('images must be greyscale')
        h = max(im1.height, im2.height)
        w = max(im1.width, im2.width)
        overlay = cls.Constant(w, h, [0, 0, 0], colororder='RGB')
        im1 = im1.colorize(colors[0]) 
        im2 = im2.colorize(colors[1])
        overlay.paste(im1, (0,0), 'add', copy=False)
        overlay.paste(im2, (0,0), 'add', copy=False)
        return overlay

    def gamma_encode(self, gamma):
        r"""
        Gamma encoding

        :param gamma: gamma value
        :type gamma: str, float
        :return: gamma encoded version of image
        :rtype: :class:`Image`

        Gamma encode the image.  This takes a linear luminance image and
        converts it to a form suitable for display on a non-linear monitor.
        ``gamma`` is either the string 'sRGB' for IEC 61966-2-1:1999 or a float:

        .. math:: \mat{Y}_{u,v} = \mat{X}_{u,v}^\gamma

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image(np.arange(8)[np.newaxis, :])  # create grey step wedge
            >>> img.gamma_encode('sRGB').disp()

        .. note::
            - ``gamma`` is the reciprocal of the value used for gamma decoding
            - Gamma encoding is typically performed in a camera with
              :math:`\gamma=0.45`.
            - For images with multiple planes, the gamma encoding is applied
              to all planes.
            - For floating point images, the pixels are assumed to be in the
              range 0 to 1.
            - For integer images,the pixels are assumed in the range 0 to
              the maximum value of their class.  Pixels are converted first to
              double, processed, then converted back to the integer class.

        :references:
            - Robotics, Vision & Control for Python, Section 10.2.7, 10.3.6, P. Corke, Springer 2023.

        :seealso: :meth:`gamma_encode` :meth:`colorspace`
        """
        out = color.gamma_encode(self.image, gamma)
        return self.__class__(out)

    def gamma_decode(self, gamma):
        r"""
        Gamma decoding

        :param gamma: gamma value
        :type gam: string or float
        :return: gamma decoded version of image
        :rtype: Image instance

        Gamma decode the image.  This takes a gamma-encoded image, as typically
        obtained from a camera or image file, and converts it to a linear
        luminance image.  ``gamma`` is either the string 'sRGB' for IEC
        61966-2-1:1999 or a float:

        .. math:: \mat{Y}_{u,v} = \mat{X}_{u,v}^\gamma

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Image
            >>> img = Image.Read('street.png')
            >>> linear = img.gamma_decode('sRGB')

        .. note::
            - ``gamma`` is the reciprocal of the value used for gamma encoding
            - Gamma decoding should be applied to any color image prior to
              colometric operations.
            - Gamma decoding is typically performed in the display hardware with
              :math:`\gamma=2.2`.
            - For images with multiple planes, the gamma decoding is applied
              to all planes.
            - For floating point images, the pixels are assumed to be in the
              range 0 to 1.
            - For integer images,the pixels are assumed in the range 0 to
              the maximum value of their class.  Pixels are converted first to
              double, processed, then converted back to the integer class.

        :references:
            - Robotics, Vision & Control for Python, Section 10.2.7, 10.3.6, P. Corke, Springer 2023.

        :seealso: :meth:`gamma_encode` :meth:`colorspace`
        """
        out = color.gamma_decode(self.image, gamma)
        return self.__class__(out, colororder=self.colororder)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    import pathlib
    import os.path

    from machinevisiontoolbox import Image

    im1 = Image.Read('eiffel-1.png', mono=True)
    im2 = Image.Read('eiffel-2.png', mono=True)
    Image.Overlay(im1, im2, 'rc').disp(block=True)

    
    exec(open(pathlib.Path(__file__).parent.parent.absolute() / "tests" / "test_color.py").read())  # pylint: disable=exec-used
