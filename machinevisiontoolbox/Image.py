#!/usr/bin/env python
"""
Images class
@author: Dorian Tsai
@author: Peter Corke
"""

import numpy as np
# import cv2 as cv
# import spatialmath.base.argcheck as argcheck
import machinevisiontoolbox as mvt


class Image():  # or inherit from np.ndarray?
    """
    An image class for MVT
    """

    def __init__(self,
                 rawimage=None,
                 numimagechannels=None,
                 numimages=None,
                 colorspace=None,
                 iscolor=None):

        # super().__init__()  # not sure about this

        if rawimage is None:
            # instance-wide attributes
            # init empty image
            self._umax = None
            self._vmax = None
            self._numimagechannels = None
            self._numimages = None
            self._dtype = None
            self._colorspace = None
            self._image = None
            self._iscolor = None
        else:

            # TODO rawimage coming in as a list of images?

            # convert raw image (ndarray, array, already an image class?)
            # check if rawimage is a string - if so, then this is a file
            # location for an image, rather than an image itself:
            if isinstance(rawimage, str):
                # string = name of an image file to read in
                self._image = mvt.iread(rawimage)
                # TODO make it a single image list?
            # elif isinstance(rawimage, list):
                # list of images
            #     self._image = rawimage
            elif mvt.isimage(rawimage):
                # is an actual image or sequence of images compounded into
                # single ndarray
                self._image = mvt.getimage(rawimage)
                # TODO consider moving mvt.getimage and mvt.isimage to Image
                # class
                # TODO make into list of images?

            else:
                raise TypeError(rawimage, 'raw image is not valid image type')

            # ability for user to specify iscolor manually to remove ambiguity
            if iscolor is None:
                self._iscolor = mvt.iscolor(self._image)  # our best guess
            else:
                self._iscolor = iscolor

            self._umax = self._image.shape[0]
            self._vmax = self._image.shape[1]

            # actually, it depends what format rawimage comes in
            # if it comes in as a list of images, everything is much easier!
            if (numimages is not None) and (numimagechannels is not None):
                # TODO check validity of numimages and numimagechannels
                self._numimagechannels = numimagechannels
                self._numimages = numimages

            elif (numimages is not None):
                # TODO check valid
                self._numimages = numimages
                # _numimagechannels based on ndim? find matching shape?
                # ni = np.where(self._image.shape == numimages)
                # here, we assume that N matches what image has:
                # eg, image = [H,W,3,N] or [H,W,N]
                if mvt.iscolor(self._image) or (self._iscolor):  # TODO should print ambiguity?
                    self._numimagechannels = 1
                else:
                    self._numimagechannels = 3

            elif (numimagechannels is not None):
                # TODO check valid
                self._numimagechannels = numimagechannels
                # use this to determine numimages based on ndim
                if mvt.iscolor(self._image) and (self._image.ndim > 3):
                    self._numimages = self._image.shape[3]
                elif not mvt.iscolor(self._image):
                    self._numimages = self._image.shape[2]
                else:
                    raise ValueError(self._image, 'unknown image shape')

            else:
                # (numimages is None) and (numimagechannels is None):
                if (self._image.ndim > 3):
                    # assume [H,W,3,N]
                    self._numimagechannels = 3
                    self._numimages = self._image.shape[3]
                elif self._iscolor and (self._image.ndim == 3):
                    # assume [H,W,3] color
                    self._numimagechannels = self._image.shape[2]
                    self._numimages = 1
                elif not self._iscolor and (self._image.ndim == 3):
                    # asdsume [H,W,N] greyscale
                    # note that if iscolor is None, this also triggers
                    # so in a way, this is the default for the ambiguous case
                    self._numimagechannels = 1
                    self._numimages = self._image.shape[2]
                elif (self._image.ndim < 3):
                    # [H,W]
                    self._numimagechannels = 1
                    self._numimages = 1
                else:
                    raise ValueError(self._image, 'unknown image shape, which \
                        should adhere to (H,W,N,3) or (H,W,N)')

            self._dtype = self._image.dtype

            if (colorspace is not None):
                # TODO check valid
                self._colorspace = colorspace
            else:
                # assume some default: BGR because we import with mvt with
                # opencv's imread()
                self._colorspace = 'BGR'

    def __len__(self):
        return len(self._numimages)

    def __getitem__(self, ind):
        # try to return the ind'th image in an image sequence if it exists
        new = Image()
        new._umax = self._umax
        new._vmax = self._vmax
        new._numimagechannels = self._numimagechannels
        new._dtype = self._dtype
        new._colorspace = self._colorspace
        new._iscolor = self._iscolor

        if isinstance(ind, slice):
            # slice object
            d = np.arange(ind.start, ind.stop, ind.step)
            new._numimages = len(d)
        elif isinstance(ind, tuple) and len(ind) == 3:
            # slice object, note that np.s_ produces a 3-tuple
            d = np.arange(ind[0], ind[1], ind[2])
            new._numimages = len(d)
        else:
            # assume either single index, an array, or a numpy array
            new._numimages = len(ind)

        if self._image.ndim == 4:
            new._image = self._image[0:, 0:, 0:, ind]
        elif self._image.ndim == 3:
            new._image = self._image[0:, 0:, ind]
        elif self._image.ndim < 3 and (np.min(ind) >= 0):
            new._image = self._image
        else:
            raise ValueError(ind, 'invalid image index, ind')

        return new

    # properties
    @property
    def size(self):
        return (self._umax, self._vmax)

    @property
    def nimages(self):
        return self._numimages

    @property
    def nchannels(self):
        return self._numimagechannels

    @property
    def umax(self):
        return self._umax

    @property
    def vmax(self):
        return self._vmax

    @property
    def dtype(self):
        return self._dtype

    @property
    def colorspace(self):
        return self._colorspace

    @property
    def image(self):
        return self._image

    # methods
    @property
    def bgr(self):
        if self._colorspace == 'BGR':
            return self._image
        else:
            # convert to proper colorspace:
            # TODO mvt.colorspace(self._image, '(ctype)->BGR')  # TODO
            # for now, assume we are RGB and simply switch the channels:
            if not self._iscolor:
                return self._image
            else:
                # bgr = np.zeros(self._image.shape)
                # or i in range(self._numimages):
                #    bgr[0:, 0:, 0:, i] = self._image[0:, 0:, ::-1, i]
                # (H,W,3,N) for RGB -> (H,W,3,N) for BGR
                if self._image.ndim > 3:
                    return self._image[0:, 0:, ::-1, 0:]
                else:
                    return self._image[0:, 0:, ::-1]

    @property
    def rgb(self):
        if self._colorspace == 'RGB':
            return self._image
        else:
            # convert to proper colorspace first:
            # return mvt.colorspace(self._image, '(ctype)->RGB')
            # TODO for now, we just assume RGB or BGR
            if not self._iscolor:
                return self._image
            else:
                if self._image.ndim > 3:
                    # (H,W,3,N) for BGR -> (H,W,3,N) for RGB
                    return self._image[0:, 0:, ::-1, 0:]
                else:
                    return self._image[0:, 0:, ::-1]

    def iscolor(self):
        return self._iscolor or mvt.iscolor(self._image)

    def mono(self):
        return mvt.mono(self._image)


if __name__ == "__main__":

    # read im image:

    # test for single colour image
    imfile = 'images/test/longquechen-mars.png'
    rawimage = mvt.iread(imfile)

    # test for image string
    rawimage = imfile

    if False:
        # test for multiple images, stack them first:
        flowers = [str(('flowers' + str(i+1) + '.png')) for i in range(8)]
        print(flowers)

        imlist = [mvt.iread(('images/' + i)) for i in flowers]
        imlist = [np.expand_dims(imlist[i], axis=3) for i in range(len(imlist))]

        rawimage = imlist[0]
        for i in range(1, 8):
            rawimage = np.concatenate((rawimage, imlist[i]), axis=3)
        # I note that it's actually a lot of work to stack these images...
        # easier for the user to provide a list of images!

        # confirm that I have several images stacked:
        import matplotlib.pyplot as plt
        # plt.figure()  # opens a new plot window
        fig0, ax0 = plt.subplots()  # also creates a new window, but with axes for plotting
        ax0.imshow(rawimage[0:, 0:, 0:, 0])

        fig1, ax1 = plt.subplots()
        ax1.imshow(rawimage[0:, 0:, 0:, 6])

    # plt.show()

    im = Image(rawimage)
    # im = Images(imfile) # TODO I would like to code this, but with current inheritance,
    # I have: TypeError: 'str' object cannot be interpreted as an integer
    # because we are trying to create a np.ndarray(imfile)...

    print('image size =', im.size)
    print('num channels = ', im.nimages)
    print('num images = ', im.nchannels)
    mvt.idisp(im.image)

    import code
    code.interact(local=dict(globals(), **locals()))

    mvt.idisp(im.bgr)

    #import code
    #code.interact(local=dict(globals(), **locals()))



