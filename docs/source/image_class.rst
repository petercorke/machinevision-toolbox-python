.. currentmodule:: machinevisiontoolbox.Image

.. _image_class_label:
The ``Image`` object for image operations and processing
========================================================

The :class:`~machinevisiontoolbox.ImageCore.Image` class is a core component of this Toolbox.  It encapsulates a
NumPy array that contains the pixel values of a greyscale or color image as
a 2D or 3D array respectively.
An :class:`~machinevisiontoolbox.ImageCore.Image` instance has a very large number of methods that perform useful
operations on an image and wrap low-level operations performed using NumPy or
OpenCV.

.. autoclass:: machinevisiontoolbox.ImageCore.Image
	:special-members: __init__
   :no-index:

Image attributes and datatype
-----------------------------

Image attributes
^^^^^^^^^^^^^^^^

Describe the attributes of an :class:`~machinevisiontoolbox.ImageCore.Image`.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~width
   ~height
   ~size
   ~npixels
   ~name
   ~centre
   ~centre_int
   ~center
   ~center_int
   ~__str__
   ~__repr__

Predicates
^^^^^^^^^^

Test attributes of an ``Image``.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~isfloat
   ~isint
   ~isbool
   ~iscolor
   ~isbgr
   ~isrgb

Image coordinates
^^^^^^^^^^^^^^^^^

Describe the pixel coordinates of an ``Image``.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~umax
   ~vmax
   ~uspan
   ~vspan
   ~meshgrid
   ~contains

NumPy pixel data
^^^^^^^^^^^^^^^^

Return ``Image`` pixel data as a NumPy array.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~A
   ~rgb
   ~bgr
   ~image
   ~to_int
   ~to_float
   ~view1d
   ~shape
   ~ndim

Image datatype
^^^^^^^^^^^^^^

Describe or change the datatype of ``Image`` pixel values.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~isfloat
   ~isint
   ~isbool
   ~dtype
   ~to
   ~astype
   ~to_int
   ~to_float
   ~cast
   ~like
   ~minval
   ~maxval
   ~true
   ~false

Image processing
----------------

Sub images
^^^^^^^^^^

Extract sub-images or planes from an ``Image`` instance.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~roi
   ~plane
   ~red
   ~green
   ~blue
   ~__getitem__

Color info
^^^^^^^^^^

Return information about the color planes of an ``Image`` instance.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~iscolor
   ~isbgr
   ~isrgb
   ~colororder
   ~colororder_str
   ~colordict
   ~nplanes
   ~plane

Color space and gamma
^^^^^^^^^^^^^^^^^^^^^

Convert between color spaces and perform gamma encoding and decoding.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~mono
   ~colorize
   ~chromaticity
   ~colorspace
   ~gamma_encode
   ~gamma_decode
   ~kmeans_color


Composition
^^^^^^^^^^^

Combine multiple ``Image`` instances into a single ``Image`` instance.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~Hstack
   ~Vstack
   ~Tile
   ~Overlay

Monadic functions
^^^^^^^^^^^^^^^^^

Operate elementwise on an ``Image`` instance and returns a new ``Image`` instance.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~abs
   ~sqrt
   ~LUT
   ~apply
   ~clip
   ~normhist
   ~stretch
   ~threshold
   ~threshold_interactive
   ~threshold_adaptive
   ~invert

   
Dyadic functions
^^^^^^^^^^^^^^^^
Operate elementwise on two ``Image`` instances and return a new ``Image`` instance.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~apply2
   ~blend
   ~choose
   ~paste
   ~direction


Linear filtering
^^^^^^^^^^^^^^^^

Linear filtering operations including convolution, corner and edge detection.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~convolve
   ~smooth
   ~gradients
   ~direction
   ~Harris_corner_strength
   ~scalespace
   ~pyramid
   ~canny

Non-linear (morphological) filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Variety of non-linear morphological operations.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~window
   ~zerocross
   ~rank
   ~medianfilter
   ~distance_transform
   ~erode
   ~dilate
   ~close
   ~open
   ~morph
   ~hitormiss
   ~thin
   ~thin_animate
   ~endpoint
   ~triplepoint

Image labeling
^^^^^^^^^^^^^^

Binary, greyscale and color image segmentation using various algorithms.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~labels_binary
   ~labels_MSER
   ~labels_graphseg
   ~kmeans_color

Image similarity
^^^^^^^^^^^^^^^^

Various scalar image similarity measures.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~sad
   ~ssd
   ~ncc
   ~zsad
   ~zssd
   ~zncc
   ~similarity

Shape changing
^^^^^^^^^^^^^^

Changing the shape of an ``Image`` instance.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~trim
   ~pad
   ~decimate
   ~replicate
   ~roi
   ~samesize
   ~scale
   ~view1d

Image distortion
^^^^^^^^^^^^^^^^

Distorting the image within an ``Image`` instance.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~roll
   ~rotate
   ~rotate_spherical
   ~warp
   ~warp_affine
   ~warp_perspective
   ~interp2d
   ~undistort

Multiview operations
^^^^^^^^^^^^^^^^^^^^

Stereo image processing, rectification, and display.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~stdisp
   ~stereo_simple
   ~DSI_refine
   ~stereo_BM
   ~stereo_SGBM
   ~rectify_homographies

Binary operators
^^^^^^^^^^^^^^^^

Arithmetic and logical operations can be performed elementwise on: ``Image`` ☆ ``Image``, 
``Image`` ☆ scalar, scalar ☆ ``Image``. The result is always an ``Image``.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~__add__
   ~__sub__
   ~__mul__
   ~__pow__
   ~__sub__
   ~__truediv__
   ~__floordiv__
   ~__and__
   ~__or__
   ~__xor__
   ~__lshift__
   ~__rshift__
   ~__eq__
   ~__ne__
   ~__gt__
   ~__ge__
   ~__lt__
   ~__le__

Unary operators
^^^^^^^^^^^^^^^

Operations can be performed elementwise on an ``Image`` and the result is always an ``Image``.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~__minus__
   ~__invert__

Image feature extraction
------------------------


Whole image features
^^^^^^^^^^^^^^^^^^^^

Histograms
""""""""""

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~hist

Statistical measures
""""""""""""""""""""

 .. autosummary::
   :toctree: stubs
   :nosignatures:

   ~mean
   ~std
   ~var
   ~median
   ~stats

Image moments
"""""""""""""

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~mpq
   ~npq
   ~upq
   ~moments
   ~humoments

Other
"""""

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~sum
   ~nonzero
   ~flatnonzero
   ~peak2d
   ~otsu

Region features
^^^^^^^^^^^^^^^

Find homogeneous regions, text or fiducual tags.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~blobs
   ~MSER
   ~ocr
   ~fiducial

Line features
^^^^^^^^^^^^^

Find lines in an image.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~canny
   ~Hough

Point/corner features
^^^^^^^^^^^^^^^^^^^^^

Find distincitive points in an image.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~SIFT
   ~ORB
   ~BRISK
   ~AKAZE
   ~Harris
   ~ComboFeature


Image i/o
---------

File
^^^^

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~Read
   ~write
   ~metadata

Graphical
^^^^^^^^^

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~disp
   ~showpixels
   ~anaglyph
   ~stdisp

Text
^^^^

.. autosummary::
   :toctree: stubs
   :nosignatures:
   
   ~String
   ~print
   ~strhcat

Constant images
---------------

Create images that are constant, random, or have a simple geometric pattern.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~Zeros
   ~Constant
   ~String
   ~Random
   ~Squares
   ~Circles
   ~Ramp
   ~Sin

Graphical annotation
--------------------

Render simple graphical annotations into an image.

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~draw_line
   ~draw_circle
   ~draw_box


Test images
-----------

Sometimes it helpful to create, process and numerically display small example images.  
These functions can help with that

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~String
   ~print
   ~strhcat
   ~showpixels