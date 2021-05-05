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
from spatialmath import base
from machinevisiontoolbox.base import meshgrid, idisp

class ReshapeMixin:

    @classmethod
    def hcat(cls, *pos, pad=0):

        if isinstance(pos[0], (tuple, list)):
            images = pos[0]
        else:
            images = pos
        
        height = max([image.height for image in images])

        combo = np.empty(shape=(height,0))

        u = []
        for image in images:
            if image.height < height:
                image = np.pad(image.image, ((0,height-image.height),(0,0)), constant_values=(pad,0))
            else:
                image = image.image
            u.append(combo.shape[1])
            combo = np.hstack((combo, image))
        
        return cls(combo), u

    def pad(self, left=0, right=0, top=0, bottom=0, value=0):

        pw = ((top,bottom),(left,right))
        const = (value, value)

        return self.__class__(np.pad(self.image, pw, constant_values=const))


    def samesize(self, im2, bias=0.5):
        """
        Automatic image trimming

        :param im2: image 2
        :type im2: numpy array
        :param bias: bias that controls what part of the image is cropped
        :type bias: float
        :return out: Image with trimmed image
        :rtype out: Image instance

        ``IM.samesize(im2)`` is an image that has the same dimensions as
        ``im2``.  This is achieved by cropping and scaling.

        ``IM.samesize(im2, bias)`` as above but ``bias`` controls which part of
        the image is cropped.  ``bias`` = 0.5 is symmetric cropping, ``bias`` <
        0.5 moves the crop window up or to the left, while ``bias``>0.5 moves
        the crop window down or to the right.

        Example:

        .. runblock:: pycon

        """
        # check inputs
        if bias < 0 or bias > 1:
            raise ValueError(bias, 'bias must be in range [0, 1]')

        out = []
        for im in self:
            sc = np.r_[im2.shape[:2]] / np.r_[im.shape[:2]]
            o = self.scale(sc.max())

            if o.height > im2.width:  # rows then columns
                # scaled image is too high, so trim rows
                d = o.height - im2.height
                d1 = max(1, int(np.floor(d * bias)))
                d2 = d - d1
                # [1 d d1 d2]
                o = o.image[d1:-d2, :, :]  # TODO check indexing
            if o.width > im2.width:
                # scaled image is too wide, so trim columns
                d = o.width - im2.width
                d1 = max(1, int(np.floor(d * bias)))
                d2 = d - d1
                # [2 d d1 d2]
                o = o.image[:, d1:-d2, :]  # TODO check indexing
            out.append(o)

        return self.__class__(out, colororder=self.colororder)

    def scale(self, sfactor, outsize=None, sigma=None, interpolation=None):
        """
        Scale an image

        :param sfactor: scale factor
        :type sfactor: scalar
        :param outsize: output image size (w, h)
        :type outsize: 2-element vector, integers
        :param sigma: standard deviation of kernel for image smoothing
        :type sigma: float
        :return out: Image smoothed image
        :rtype out: Image instance

        - ``IM.scale(sfactor)`` is a scaled image in both directions by
          ``sfactor`` which is a real scalar. ``sfactor> 1`` makes the image
          larger, ``sfactor < 1`` makes it smaller.

        - ``IM.scale(sfactor, outsize)`` as above, with the output image size
          specified as (W, H).

        - ``IM.scale(sfactor, sigma)`` as above, with the initial Gaussian
          smoothing specified as ``sigma``.

        Example:

        .. runblock:: pycon

        """
        # check inputs
        if not base.isscalar(sfactor):
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

        out = []
        for im in self:
            if sfactor < 1 and sigma is not None:
                im = im.smooth(sigma)
            res = cv.resize(im.image, None, fx=sfactor, fy=sfactor, 
                interpolation=interpolation)
            out.append(res)

        return self.__class__(out, colororder=self.colororder)

    _interp_dict = {

'nearest': cv.INTER_NEAREST, # nearest neighbor interpolation
'linear': cv.INTER_LINEAR, #bilinear interpolation
'cubic': cv.INTER_CUBIC, # bicubic interpolation
'area': cv.INTER_AREA, #esampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire'-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
'Lanczos': cv.INTER_LANCZOS4, #Lanczos interpolation over 8x8 neighborhood
'linear exact': cv.INTER_LINEAR_EXACT, # Bit exact bilinear interpolation
    }

    def affine_warp(self, M, inverse=False, size=None, bgcolor=None):
        flags = cv.INTER_CUBIC
        if inverse:
            flags |= cv.WARP_INVERSE_MAP
        
        # TODO interpolation flags
        
        if size is None:
            size = self.shape[:2]

        if bgcolor is not None:
            bordermode = cv.BORDER_CONSTANT
            bordervalue = [bgcolor,] * self.nplanes
        else:
            bordermode = None
            bordervalue = None

        out = cv.warpAffine(src=self.image, M=M, dsize=size, flags=flags, borderMode=bordermode, borderValue=bordervalue)
        return self.__class__(out, colororder=self.colororder)

    def undistort(self, C, dist):
        undistorted = cv.undistort(self.image, C, dist)
        return self.__class__(undistorted, colororder=self.colororder)

    def rotate(self,
               angle,
               crop=False,
               centre=None):
        """
        Rotate an image

        :param angle: rotatation angle [radians]
        :type angle: scalar
        :param crop: output image size (w, h)
        :type crop: 2-element vector, integers
        :param sc: scale factor
        :type sc: float
        :param extrapval: background value of pixels
        :type extrapval: float
        :param sm: smooth (standard deviation of Gaussian kernel, sigma)
        :type sm: float
        :param outsize: output image size (w, h)
        :type outsize: 2-element vector, integers
        :return out: Image with rotated image
        :rtype out: Image instance

        - ``IM.rotate(angle)`` is an image that has been rotated about its
          centre by angle ``angle``.

        - ``IM.rotate(angle, crop)`` as above, but cropped to the same size as
          the original image.

        - ``IM.rotate(angle, scale)`` as above, with scale specified.

        - ``IM.rotate(angle, smooth)`` as above, with initial smoothing
          applied.

        - ``IM.rotate(angle, outsize)`` as above, with size of output image set
          to ``outsize = (H, W)``.

        - ``IM.rotate(angle, extrapval)`` set background pixels to extrapval.
          TODO

        Example:

        .. runblock:: pycon

        .. note::

            - Rotation is defined with respect to a z-axis which is into the
              image.
            - Counter-clockwise is a positive angle.
            - The pixels in the corners of the resulting image will be
              undefined and set to the 'extrapval'.

        """
        # TODO note that there is cv.getRotationMatrix2D and cv.warpAffine
        # https://appdividend.com/2020/09/24/how-to-rotate-an-image-in-python-
        # using-opencv/

        if not base.isscalar(angle):
            raise ValueError(angle, 'angle is not a valid scalar')

        # TODO check optional inputs


        if centre is None:
            centre = (self.width / 2, self.height / 2)
        elif len(centre) != 2:
            raise ValueError('centre must be length 2')

        shape = (self.width, self.height)

        M = cv.getRotationMatrix2D(centre, np.degrees(angle), 1.0)

        out = []
        for im in self:
            res = cv.warpAffine(im.image, M, shape)
            out.append(res)

        return self.__class__(out, colororder=self.colororder)

if __name__ == "__main__":

    from machinevisiontoolbox import Image
    from math import pi

    img = Image.Read('monalisa.png', grey=False)
    print(img)
    img.disp()

    # img.scale(.5).disp()

    # im2 = img.scale(2)
    # im2.disp(block=True)

    img.rotate(pi / 4, centre=(0,0)).disp()

    im2 = img.rotate(pi / 4)
    im2.disp(block=True)
