.. currentmodule:: machinevisiontoolbox.Image

.. _image_class_label:

The ``Image`` object
====================

The :class:`~machinevisiontoolbox.ImageCore.Image` class is essential for all image
operations and processing within this Toolbox. The class 
encapsulates a NumPy array that contains the pixel values of a greyscale or color image
as a 2D or 3D array respectively. An :class:`~machinevisiontoolbox.ImageCore.Image`
instance has a very large number of methods that perform useful operations on an image
and wrap low-level operations performed using NumPy or OpenCV.

.. autoclass:: machinevisiontoolbox.Image

.. toctree::
   :maxdepth: 1

   kernel
   support-classes
   image-algorithms

.. include:: _image_class_toctree.rst.inc

Image attributes and datatype
-----------------------------

Image attributes
^^^^^^^^^^^^^^^^

Describe the attributes of an :class:`~machinevisiontoolbox.ImageCore.Image`.

.. autosummary::
   :nosignatures:

   ~__repr__
   ~__str__
   ~center
   ~center_int
   ~centre
   ~centre_int
   ~height
   ~name
   ~npixels
   ~size
   ~width

Predicates
^^^^^^^^^^

Test attributes of an ``Image``.

.. autosummary::
   :nosignatures:

   ~isbgr
   ~isbool
   ~iscolor
   ~isfloat
   ~isint
   ~isrgb

Image coordinates
^^^^^^^^^^^^^^^^^

Describe the pixel coordinates of an ``Image``.

.. autosummary::
   :nosignatures:

   ~contains
   ~meshgrid
   ~umax
   ~uspan
   ~vmax
   ~vspan

NumPy pixel data
^^^^^^^^^^^^^^^^

Return ``Image`` pixel data as a NumPy array.

.. autosummary::
   :nosignatures:

   ~array
   ~array_as
   ~bgr
   ~ndim
   ~rgb
   ~shape
   ~to_float
   ~to_int
   ~view1d

Getting and setting pixels
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Access individual pixels or groups of pixels.

.. autosummary::
   :nosignatures:

   ~__getitem__
   ~pixel
   ~pixels_mask

Image datatype
^^^^^^^^^^^^^^

Describe or change the datatype of ``Image`` pixel values.

.. autosummary::
   :nosignatures:

   ~astype
   ~cast
   ~dtype
   ~false
   ~fixbad
   ~like
   ~maxval
   ~minval
   ~numinf
   ~numnan
   ~to
   ~true

Image processing
----------------

Sub images
^^^^^^^^^^

Extract sub-images or planes from an ``Image`` instance.

.. autosummary::
   :nosignatures:

   ~blue
   ~copy
   ~green
   ~plane
   ~red
   ~roi

Color info
^^^^^^^^^^

Return information about the color planes of an ``Image`` instance.

.. autosummary::
   :nosignatures:

   ~colordict
   ~colordict2list
   ~colordict2str
   ~colororder
   ~colororder2dict
   ~colororder_str
   ~nplanes

Color space and gamma
^^^^^^^^^^^^^^^^^^^^^

Convert between color spaces and perform gamma encoding and decoding.

.. autosummary::
   :nosignatures:

   ~chromaticity
   ~colorize
   ~colorspace
   ~gamma_decode
   ~gamma_encode
   ~kmeans_color
   ~mono


Composition
^^^^^^^^^^^

Combine multiple ``Image`` instances into a single ``Image`` instance.

.. autosummary::
   :nosignatures:

   ~anaglyph
   ~blend
   ~Hstack
   ~Overlay
   ~Pstack
   ~stdisp
   ~Tile
   ~Vstack

Monadic functions
^^^^^^^^^^^^^^^^^

Operate elementwise on an ``Image`` instance and returns a new ``Image`` instance.

.. autosummary::
   :nosignatures:

   ~abs
   ~apply
   ~clip
   ~invert
   ~LUT
   ~normhist
   ~sqrt
   ~stretch
   ~threshold
   ~threshold_adaptive
   ~threshold_interactive

   
Dyadic functions
^^^^^^^^^^^^^^^^
Operate elementwise on two ``Image`` instances and return a new ``Image`` instance.

.. autosummary::
   :nosignatures:

   ~apply2
   ~choose
   ~paste


Linear filtering
^^^^^^^^^^^^^^^^

Linear filtering operations including convolution, corner and edge detection.

.. autosummary::
   :nosignatures:

   ~canny
   ~convolve
   ~gradients
   ~Harris_corner_strength
   ~pyramid
   ~scalespace
   ~smooth

Non-linear (morphological) filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Variety of non-linear morphological operations.

.. autosummary::
   :nosignatures:

   ~close
   ~dilate
   ~distance_transform
   ~endpoint
   ~erode
   ~hitormiss
   ~medianfilter
   ~morph
   ~open
   ~rank
   ~thin
   ~thin_animate
   ~triplepoint
   ~window
   ~zerocross

Image labeling
^^^^^^^^^^^^^^

Binary, greyscale and color image segmentation using various algorithms.

.. autosummary::
   :nosignatures:

   ~labels_binary
   ~labels_graphseg
   ~labels_MSER

Image similarity
^^^^^^^^^^^^^^^^

Various scalar image similarity measures.

.. autosummary::
   :nosignatures:

   ~ncc
   ~sad
   ~similarity
   ~ssd
   ~zncc
   ~zsad
   ~zssd

Shape changing
^^^^^^^^^^^^^^

Changing the shape of an ``Image`` instance.

.. autosummary::
   :nosignatures:

   ~decimate
   ~dice
   ~pad
   ~replicate
   ~samesize
   ~scale
   ~trim

Image distortion
^^^^^^^^^^^^^^^^

Distorting the image within an ``Image`` instance.

.. autosummary::
   :nosignatures:

   ~interp2d
   ~roll
   ~rotate
   ~rotate_spherical
   ~undistort
   ~warp
   ~warp_affine
   ~warp_perspective

Multiview operations
^^^^^^^^^^^^^^^^^^^^

Stereo image processing, rectification, and display.

.. autosummary::
   :nosignatures:

   ~DSI_refine
   ~rectify_homographies
   ~stereo_BM
   ~stereo_SGBM
   ~stereo_simple

Operators
^^^^^^^^^

Binary arithmetic and relational operators
""""""""""""""""""""""""""""""""""""""""""

   * ``Image`` ☆ ``Image``
   * ``Image`` ☆ scalar
   * scalar ☆ ``Image``
   
The result is always an ``Image``.

For the first case, the images must have:

   * the same shape,
   * the same width and height but can have different number of color planes.  The image
     with one plane is broadcast across the color planes of the other image.

A scalar value is broadcast across the whole image.

Arithmetic and bitwise logical operations can be performed elementwise on:


.. autosummary::
   :nosignatures:

   ~__add__
   ~__and__
   ~__floordiv__
   __invert__
   ~__lshift__
   ~__mul__
   ~__neg__
   ~__or__
   ~__pow__
   ~__radd__
   ~__rfloordiv__
   ~__rmul__
   ~__rshift__
   ~__rsub__
   ~__rtruediv__
   ~__sub__
   ~__truediv__
   ~__xor__


Logical operations can be performed elementwise on: ``Image`` ☆ ``Image``.
The result is always an ``Image`` with boolean pixel values:

.. autosummary::
   :nosignatures:

   ~__eq__
   ~__ge__
   ~__gt__
   ~__le__
   ~__lt__
   ~__ne__


Inplace arithmetic operators
""""""""""""""""""""""""""""

Arithmetic and bitwise logical operations can be performed elementwise on:

   * ``Image`` ☆= ``Image``
   * ``Image`` ☆= scalar

The result is always an ``Image``.  A scalar value is broadcast across the whole image.

.. autosummary::
   :nosignatures:

   ~__iadd__
   ~__iand__
   ~__ifloordiv__
   ~__ilshift__
   ~__imul__
   ~__ior__
   ~__irshift__
   ~__isub__
   ~__itruediv__
   ~__ixor__

Plane stacking operators
""""""""""""""""""""""""

Stacking operations can be performed on multiple ``Image`` instances.  
A scalar  value is broadcast across the whole image to create a new ``Image`` instance.
In place stacking allows for planes to be appended.

.. autosummary::
   :nosignatures:
   
   ~__imod__
   ~__mod__

Image statistics
----------------

 .. autosummary::
   :nosignatures:

   ~max
   ~mean
   ~median
   ~min
   ~stats
   ~std
   ~var
   ~hist

Image feature extraction
------------------------


Whole image features
^^^^^^^^^^^^^^^^^^^^

Histograms
""""""""""

.. autosummary::
   :nosignatures:

   ~hist


Image moments
"""""""""""""

.. autosummary::
   :nosignatures:

   ~humoments
   ~moments
   ~mpq
   ~npq
   ~upq

Other
"""""

.. autosummary::
   :nosignatures:

   ~flatnonzero
   ~nonzero
   ~otsu
   ~peak2d
   ~sum

Region features
^^^^^^^^^^^^^^^

Find homogeneous regions, text or fiducual tags.

.. autosummary::
   :nosignatures:

   ~blobs
   ~MSER
   ~ocr



Fiducial features
^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~fiducial

Line features
^^^^^^^^^^^^^

Find lines in an image.

.. autosummary::
   :nosignatures:

   ~Hough

Point/corner features
^^^^^^^^^^^^^^^^^^^^^

Find distincitive points in an image.

.. autosummary::
   :nosignatures:

   ~AKAZE
   ~BRISK
   ~ComboFeature
   ~Harris
   ~ORB
   ~SIFT


Image i/o
---------

File
^^^^

.. autosummary::
   :nosignatures:

   ~metadata
   ~Read
   ~write

Graphical
^^^^^^^^^

.. autosummary::
   :nosignatures:

   ~disp
   ~showpixels

Text
^^^^

.. autosummary::
   :nosignatures:
   
   ~print
   ~strhcat

Constant images
---------------

Create images that are constant, random, or have a simple geometric pattern.

.. autosummary::
   :nosignatures:

   ~Chequerboard
   ~Circles
   ~Constant
   ~Polygons
   ~Ramp
   ~Random
   ~Sin
   ~Squares
   ~String
   ~Zeros

Graphical annotation
--------------------

Render simple graphical annotations into an image.  The equivalent functions ``plot_xxx``
from SpatialMath Toolbox create graphical overlays rather than changing the the image data.

.. autosummary::
   :nosignatures:

   ~draw_box
   ~draw_circle
   ~draw_labelbox
   ~draw_line
   ~draw_point
   ~draw_text


Test images
-----------

Sometimes, for pedagogy and unit tests, it is helpful to create, process and numerically
display small example images.