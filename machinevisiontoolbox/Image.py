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
                 colorspace='BGR',
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
            self._imlist = None
            self._iscolor = None
        else:

            # whatever rawimage input type is, try to convert it to a list of
            # numpy array images
            if isinstance(rawimage, str):
                # string = name of an image file to read in
                self._imlist = [mvt.iread(rawimage)]

            elif isinstance(rawimage, list) and isinstance(rawimage[0], str):
                # list of image file names
                self._imlist = [mvt.iread(rawimage[i]) for i in rawimage]

            elif isinstance(rawimage, list) and isinstance(np.asarray(rawimage[0]), np.ndarray):
                # list of images, with each item being a numpy array
                self._imlist = rawimage

            elif mvt.isimage(rawimage):
                # is an actual image or sequence of images compounded into
                # single ndarray
                # make this into a list of images
                # if color:
                rawimage = mvt.getimage(rawimage)
                if rawimage.ndim == 4:
                    # assume (W,H,3,N)
                    self._imlist = [mvt.getimage(rawimage[0:, 0:, 0:, i])
                                    for i in range(rawimage.shape[3])]
                elif rawimage.ndim == 3:
                    # could be single (W,H,3) -> 1 colour image
                    # or (W,H,N) -> N grayscale images
                    if not rawimage.shape[2] == 3:
                        self._imlist = [mvt.getimage(rawimage[0:, 0:, i])
                                        for i in range(rawimage.shape[2])]
                    elif (rawimage.shape[2] == 3) and iscolor:
                        # manually specified iscolor is True
                        # single colour image
                        self._imlist = [mvt.getimage(rawimage)]
                    else:
                        self._imlist = [mvt.getimage(rawimage[0:, 0:, i])
                                        for i in range(rawimage.shape[2])]

                elif rawimage.ndim == 2:
                    # single (W,H)
                    self._imlist = [mvt.getimage(rawimage)]

                else:
                    raise ValueError(rawimage, 'unknown rawimage.shape')

            else:
                raise TypeError(rawimage, 'raw image is not valid image type')
                print('Valid image types: filename string of an image, \
                       list of filename strings, \
                       list of numpy arrays, or a numpy array')

            # check list of images for size consistency

            # assume that the image stack has the same size image for the
            # entire list. TODO maybe in the future, we remove this assumption,
            # which can cause errors, but for now we simply check the shape of
            # each image in the list
            shape = [self._imlist[i].shape for i in range(len(self._imlist))]
            if np.any([shape[i] != shape[0] for i in range(len(self._imlist))]):
                raise ValueError(rawimage, 'inconsistent input image shape')

            self._umax = self._imlist[0].shape[0]
            self._vmax = self._imlist[0].shape[1]

            # ability for user to specify iscolor manually to remove ambiguity
            if iscolor is None:
                self._iscolor = mvt.iscolor(self._imlist[0])  # our best guess
            else:
                self._iscolor = iscolor

            self._numimages = len(self._imlist)

            if self._imlist[0].ndim == 3:
                self._numimagechannels = self._imlist[0].shape[2]
            elif self._imlist[0].ndim == 2:
                self._numimagechannels = 1
            else:
                raise ValueError(self._numimagechannels, 'unknown number of \
                                 image channels')

            # these if statements depracated with the use of lists of images
            """
            # TODO check validity of numimages and numimagechannels
            # wrt self._imlists
            if (numimages is not None) and (numimagechannels is not None):
                self._numimagechannels = numimagechannels
                self._numimages = numimages

            elif (numimages is not None):
                self._numimages = numimages
                # since we have a list of images, numimagechannels is simply the
                # third dimension of the first image shape
                if self._iscolor:  # TODO should print ambiguity?
                    self._numimagechannels = 1
                else:
                    self._numimagechannels = 3

            elif (numimagechannels is not None):
                # TODO check valid
                self._numimagechannels = numimagechannels
                # use this to determine numimages based on ndim
                if mvt.iscolor(self._imlist) and (self._imlist.ndim > 3):
                    self._numimages = self._imlist.shape[3]
                elif not mvt.iscolor(self._imlist):
                    self._numimages = self._imlist.shape[2]
                else:
                    raise ValueError(self._imlist, 'unknown image shape')

            else:
                # (numimages is None) and (numimagechannels is None):
                if (self._imlist.ndim > 3):
                    # assume [H,W,3,N]
                    self._numimagechannels = 3
                    self._numimages = self._imlist.shape[3]
                elif self._iscolor and (self._imlist.ndim == 3):
                    # assume [H,W,3] color
                    self._numimagechannels = self._imlist.shape[2]
                    self._numimages = 1
                elif not self._iscolor and (self._imlist.ndim == 3):
                    # asdsume [H,W,N] greyscale
                    # note that if iscolor is None, this also triggers
                    # so in a way, this is the default for the ambiguous case
                    self._numimagechannels = 1
                    self._numimages = self._imlist.shape[2]
                elif (self._imlist.ndim < 3):
                    # [H,W]
                    self._numimagechannels = 1
                    self._numimages = 1
                else:
                    raise ValueError(self._imlist, 'unknown image shape, which \
                        should adhere to (H,W,N,3) or (H,W,N)')
                """

            self._dtype = self._imlist[0].dtype

            validcolorspaces = ('RGB', 'BGR')
            # TODO add more valid colorspaces
            # assume some default: BGR because we import with mvt with
            # opencv's imread(), which imports as BGR by default
            if colorspace in validcolorspaces:
                self._colorspace = colorspace
            else:
                raise ValueError(colorspace, 'unknown colorspace input')

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

        """
        # if _imlist was an  ndarray
        if self._imlist[0].ndim == 4:
            new._imlist = self._imlist[0:, 0:, 0:, ind]
        elif self._imlist[0].ndim == 3:
            new._imlist = self._imlist[0:, 0:, ind]
        elif self._imlist[0].ndim < 3 and (np.min(ind) >= 0):
            new._imlist = self._imlist
        else:
            raise ValueError(ind, 'invalid image index, ind')
        """
        new._imlist = self.listimages(ind)
        new._numimages = len(new._imlist)

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
    def shape(self):
        return self._imlist[0].shape

    @property
    def imlist(self):
        return self._imlist

    # methods
    # TODO asimagearray - return a numpy array stack
    # TODO would like to be able to do im.imlist[ind]
    # what I currently have is im[ind]?

    def listimages(self, ind):
        if isinstance(ind, int) and (ind >= 0) and (ind <= len(self._imlist)):
            return self._imlist[ind]
        elif isinstance(ind, slice):
            islice = np.arange(ind.start, ind.stop, ind.step)
            return [self._imlist[i] for i in islice]
        # elif isinstance(ind, tuple) and (len(ind) == 3):
        # slice object from numpy as a 3-tuple -> but how can we
        # differentiate between a normal 3-tuple eg (0,1,2) vs a numpy slice
        # (0, 2, 1)? TODO ruminate for later
        #     islice = np.arange()
        elif (len(ind) > 1) and (np.min(ind) >= 0) and (np.max(ind) <= len(self._imlist)):
            return [self._imlist[i] for i in ind]

    def bgr(self, ind=None):
        if ind is None:
            ind = np.arange(0, len(self._imlist))
        imlist = self.listimages(ind)

        if self._colorspace == 'BGR':
            return imlist
        else:
            # convert to proper colorspace:
            # TODO mvt.colorspace(self._imlist, '(ctype)->BGR')  # TODO
            # for now, assume we are RGB and simply switch the channels:
            if not self._iscolor:
                return imlist
            else:
                # bgr = np.zeros(self._imlist.shape)
                # or i in range(self._numimages):
                #    bgr[0:, 0:, 0:, i] = self._imlist[0:, 0:, ::-1, i]
                # (H,W,3,N) for RGB -> (H,W,3,N) for BGR
                if imlist[0].ndim > 3:
                    return [imlist[i][0:, 0:, ::-1, 0:]
                            for i in range(len(imlist))]
                else:
                    return [imlist[i][0:, 0:, ::-1]
                            for i in range(len(imlist))]

    def rgb(self, ind=None):
        if ind is None:
            ind = np.arange(0, len(self._imlist))
        imlist = self.listimages(ind)

        if self._colorspace == 'RGB':
            return imlist
        else:
            # convert to proper colorspace first:
            # return mvt.colorspace(self._imlist, '(ctype)->RGB')
            # TODO for now, we just assume RGB or BGR
            if not self._iscolor:
                return imlist
            else:
                if imlist[0].ndim > 3:
                    # (H,W,3,N) for BGR -> (H,W,3,N) for RGB
                    return [imlist[i][0:, 0:, ::-1, 0:]
                            for i in range(len(imlist))]
                else:
                    return [imlist[i][0:, 0:, ::-1]
                            for i in range(len(imlist))]

    def iscolor(self):
        return self._iscolor or mvt.iscolor(self._imlist)

    def mono(self):
        return mvt.mono(self._imlist)


if __name__ == "__main__":

    # read im image:

    # test for single colour image
    imfile = 'images/test/longquechen-mars.png'
    rawimage = mvt.iread(imfile)

    # test for image string
    rawimage = imfile

    # test for multiple images, stack them first:
    flowers = [str(('flowers' + str(i+1) + '.png')) for i in range(8)]
    print(flowers)

    # list of images
    imlist = [mvt.iread(('images/' + i)) for i in flowers]

    if False:

        # concatenate list of images into a stack of images
        imlista = [np.expand_dims(imlist[i], axis=3) for i in range(len(imlist))]
        rawimage = imlist[0]
        for i in range(1, 8):
            rawimage = np.concatenate((rawimage, imlista[i]), axis=3)
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

    im = Image(imlist)
    # im = Images(imfile) # TODO I would like to code this, but with current inheritance,
    # I have: TypeError: 'str' object cannot be interpreted as an integer
    # because we are trying to create a np.ndarray(imfile)...

    print('image size =', im.size)
    print('num channels = ', im.nimages)
    print('num images = ', im.nchannels)
    # mvt.idisp(im.imlist[0])

    import code
    code.interact(local=dict(globals(), **locals()))

    # mvt.idisp(im.bgr)

    #import code
    #code.interact(local=dict(globals(), **locals()))



