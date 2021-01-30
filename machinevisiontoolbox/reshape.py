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
            sc = im2.shape / im.shape
            o = self.scale(im, sc.max())

            if o.height > im2.width:  # rows then columns
                # scaled image is too high, so trim rows
                d = out.height - im2.height
                d1 = np.max(1, np.floor(d * bias))
                d2 = d - d1
                # [1 d d1 d2]
                im2 = out[d1:-1-d2-1, :, :]  # TODO check indexing
            if o.width > im2.width:
                # scaled image is too wide, so trim columns
                d = out.width - im2.width
                d1 = np.max(1, np.floor(d * bias))
                d2 = d - d1
                # [2 d d1 d2]
                o = o[:, d1: -1-d2-1, :]  # TODO check indexing
            out.append(o)

        return self.__class__(out)

    def scale(self, sfactor, outsize=None, sigma=None):
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
        if not argcheck.isscalar(sfactor):
            raise TypeError(sfactor, 'factor is not a scalar')

        out = []
        for im in self:
            if np.issubdtype(im.dtype, np.float):
                is_int = False
            else:
                is_int = True
                im = self.float(im)

            # smooth image to prevent aliasing  - TODO should depend on scale
            # factor
            if sigma is not None:
                im = self.smooth(im, sigma)

            nr = im.shape[0]
            nc = im.shape[1]

            # output image size is determined by input size and scale factor
            # else from specified size
            if outsize is not None:
                nrs = np.floor(nr * sfactor)
                ncs = np.floor(nc * sfactor)
            else:
                nrs = outsize[0]
                ncs = outsize[1]

            # create the coordinate matrices for warping
            U, V = self.imeshgrid(im)
            U0, V0 = self.imeshgrid([ncs, nrs])

            U0 = U0 / sfactor
            V0 = V0 / sfactor

            if im.ndims > 2:
                o = np.zeros((ncs, nrs, im.nchannels))
                for k in range(im.nchannels):
                    o[:, :, k] = sp.interpolate.interp2d(U, V,
                                                         im.image[:, :, k],
                                                         U0, V0,
                                                         kind='linear')
            else:
                o = sp.interpolate.interp2d(U, V,
                                            im.image,
                                            U0, V0,
                                            kind='linear')

            if is_int:
                o = self.iint(o)

            out.append(o)

        return self.__class__(out)

    def rotate(self,
               angle,
               crop=False,
               sc=1.0,
               extrapval=0,
               sm=None,
               outsize=None):
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

        if not argcheck.isscalar(angle):
            raise ValueError(angle, 'angle is not a valid scalar')

        # TODO check optional inputs

        out = []
        for im in self:
            if np.issubdtype(im.dtype, np.float):
                is_int = False
            else:
                is_int = True
                im = self.float(im)

            if sm is not None:
                im = self.smooth(im, sm)

            if outsize is not None:
                # output image is determined by input size
                U0, V0 = np.meshgrid(np.arange(0, outsize[0]),
                                     np.arange(0, outsize[1]))
            else:
                outsize = np.array([im.shape[0], im.shape[1]])
                U0, V0 = self.imeshgrid(im)

            nr = im.shape[0]
            nc = im.shape[1]

            # creqate coordinate matrices for warping
            Ui, Vi = self.imeshgrid(im)

            # rotation and scale
            R = cv.getRotationMatrix2D(center=(0, 0), angle=angle, scale=sc)
            uc = nc / 2.0
            vc = nr / 2.0
            U02 = 1.0/sc * (R[0, 0] * (U0 - uc) + R[1, 0] * (V0 - vc)) + uc
            V02 = 1.0/sc * (R[0, 1] * (U0-uc) + R[1, 1] * (V0-vc)) + vc

            if crop:
                trimx = np.abs(nr / 2.0 * np.sin(angle))
                trimy = np.abs(nc/2.0*np.sin(angle))
                if sc < 1:
                    trimx = trimx + nc/2.0*(1.0-sc)
                    trimy = trimy + nr/2.0*(1.0-sc)

                trimx = np.ceil(trimx)  # +1
                trimy = np.ceil(trimy)  # +1
                U0 = U02[trimy:U02.shape[1]-trimy,
                         trimx: U02.shape[0]-trimx]  # TODO check indices
                V0 = V02[trimy: V02.shape[1]-trimy, trimx: V02.shape[0]-trimx]

            if im.ndims > 2:
                o = np.zeros((outsize[0], outsize[1], im.shape[2]))
                for k in range(im.shape[2]):
                    # TODO extrapval?
                    if extrapval:
                        raise ValueError(extrapval,
                                         'extrapval not implemented yet')
                    else:
                        out[:, :, k] = interpolate.interp2(Ui, Vi,
                                                           im.image[:, :, k],
                                                           U02, V02,
                                                           kind='linear')
            else:
                o = sp.interpolate.interp2(Ui, Vi,
                                           im.image,
                                           U02, V02,
                                           kind='linear')

            if is_int:
                o = self.iint(o)

            out.append(o)

        return self.__class__(out)

