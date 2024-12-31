.. currentmodule:: machinevisiontoolbox.Image

Image processing and feature extraction
=======================================

The ``Image`` class is a core component of this Toolbox.  It encapsulates a
NumPy array that contains the pixel values of a greyscale or color image as
a 2D or 3D array respectively.
An ``Image`` instance has a very large number of methods that perform useful
operations on an image and wrap low-level operations performed using NumPy or
OpenCV.

Image object
------------

Basic info
^^^^^^^^^^

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

Color
^^^^^

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

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~Hstack
   ~Vstack
   ~Tile
   ~Overlay




Monadic functions
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~abs
   ~sqrt
   ~LUT
   ~apply
   ~clip
   ~roll
   ~normhist
   ~stretch
   ~threshold
   ~threshold_interactive
   ~threshold_adaptive
   ~invert

   
Dyadic functions
^^^^^^^^^^^^^^^^

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

Non-linear filtering
^^^^^^^^^^^^^^^^^^^^

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

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~labels_binary
   ~labels_MSER
   ~labels_graphseg
   ~kmeans_color

Image similarity
^^^^^^^^^^^^^^^^

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
   ~rotate
   ~rotate_spherical
   ~warp
   ~warp_affine
   ~warp_perspective
   ~interp2d
   ~undistort
   ~view1d

Multiview operations
^^^^^^^^^^^^^^^^^^^^

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

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~__minus__
   ~__invert__

Image feature extraction
------------------------


Whole image features
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~hist

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~sum
   ~min
   ~max
   ~nonzero
   ~flatnonzero
   ~peak2d
   ~otsu
 
 .. autosummary::
   :toctree: stubs
   :nosignatures:

   ~mean
   ~std
   ~var
   ~median
   ~stats


.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~mpq
   ~npq
   ~upq
   ~moments
   ~humoments

Region features
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~blobs
   ~MSER
   ~ocr
   ~fiducial

Line features
^^^^^^^^^^^^^

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~canny
   ~Hough

Point/corner features
^^^^^^^^^^^^^^^^^^^^^

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

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~Read
   ~disp
   ~write
   ~metadata
   ~showpixels
   ~anaglyph
   ~stdisp

Constant images
---------------

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

Graphics
--------

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~draw_line
   ~draw_circle
   ~draw_box


Small example images
--------------------

Sometimes it helpful to create, process and numerically display small example images.  
These functions can help with that

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~String
   ~print
   ~strhcat
   ~showpixels