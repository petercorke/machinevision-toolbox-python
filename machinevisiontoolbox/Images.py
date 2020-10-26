#!/usr/bin/env python
"""
Images class
@author: Dorian Tsai
@author: Peter Corke
"""

import numpy as np
import cv2 as cv
import spatialmath.base.argcheck as argcheck
import machinevisiontoolbox as mvt


class Images(np.ndarray):  # or np.ndarray?
    """
    An image class
    """
    # maybe need super.()?

    # list of attributes
    _umax = []              # max number of horizontal pixels
    _vmax = []              # max number of vertical pixels
    _numimagechannels = []  # number of channels per image
    _numimages = []         # number of images
    _image = []             # the image(s) themselves

    _dtype = []             # data type of image (using numpy data types)
    _colorspace = []        # RGB vs BGR, maybe HSV, Lab, etc, so string-based

    _name = []              # image name
    _folder = []            # folder location?

    """
    methods:
    __init__
        define variables given an image or image sequence -> part of getimage?
        ask if ambiguous input, or allow user to specify, otherwise
            automatically set numchannels, numimages

    __len__
    __getitem__  - want to be able to index image wrt coordinates still!
    Still preserve the .shape functionality, etc of numpy arrays (we
    inherit from ndarray).  how would this work? cannot overwrite __getitem__?
    differentiating indexing of image class vs indexing of image coordinates?
    Might be why a lot of opencv have [indobject][1][u,v]?
    __new__?

    isimage
    iscolor
    display (just call idisp)?
    read (just call iread)
    mono - convert to grayscale
    binary - convert to zeros and ones? True and False?
    .rgb()
    .bgr()

    @properties:
    size

    """

    def __init__(self,
                 rawimage=None,
                 numimagechannels=None,
                 numimages=None,
                 colorspace=None):

        super().__init__()  # not sure about this

        if rawimage is None:
            # init empty image
            self._umax = None
            self._vmax = None
            self._numimagechannels = None
            self._numimages = None
            self._dtype = None
            self._colorspace = None
        else:
            # convert raw image (ndarray, array, already an image class?)
            # check if rawimage is a string - if so, then this is a file
            # location for an image, rather than an image itself:
            if isinstance(rawimage, str):
                self._image = mvt.iread(rawimage)
            elif mvt.isimage(rawimage):
                self._image = mvt.getimage(rawimage)
            else:
                raise TypeError(rawimage, 'raw image is not valid image type')

            # self._image = mvt.getimage(rawimage)  # would move this to _getimage()

            self._umax = self._image.shape[0]
            self._vmax = self._image.shape[1]

            if (numimages is not None) and (numimagechannels is not None):
                # TODO check validity of numimages and numimagechannels
                self._numimagechannels = numimagechannels
                self._numimages = numimages
            elif (numimages is not None):
                # TODO check valid
                self._numimages = numimages
                # _numimagechannels based on ndim
                pass
            elif (numimagechannels is not None):
                # TODO check valid
                self._numimagechannels = numimagechannels
                # use this to determine numimages based on ndim
                pass
            else:
                # (numimages is None) and (numimagechannels is None):
                pass

            self._dtype = np.dtype()

            if (colorspace is not None):
                # TODO check valid
                self._colorspace = colorspace
            else:
                # assume some default? BGR?
                self._colorspace = 'BGR'

    def __len__(self):
        return len(self._numimages)

    # TODO how would __getitem__ work?
    # properties
    @property
    def size(self):
        return (self._umax, self._vmax)

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

    # methods
    def bgr(self):
        if self._colorspace == 'BGR':
            return self._image
        else:
            # convert to proper colorspace:
            return mvt.colorspace(self._image, '(ctype)->BGR')  # TODO

    def rgb(self):
        if self._colorspace == 'RGB':
            return self._image
        else:
            # convert to proper colorspace first:
            return mvt.colorspace(self._image, '(ctype)->RGB')

    def iscolor(self):
        return mvt.iscolor(self._image)

    def mono(self):
        return mvt.mono(self._image)


if __name__ == "__main__":

    # read im image:
    imfile = 'images/test/longquechen-mars.png'
    rawimage = mvt.iread(imfile)

    import code
    code.interact(local=dict(globals(), **locals()))


    im = Images(rawimage)
    # im = Images(imfile) # TODO I would like to code this, but with current inheritance,
    # I have: TypeError: 'str' object cannot be interpreted as an integer
    # because we are trying to create a np.ndarray(imfile)...

    print('image size =', im.size)



