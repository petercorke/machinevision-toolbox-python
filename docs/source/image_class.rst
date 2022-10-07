.. currentmodule:: machinevisiontoolbox.Image

Image object
============

The ``Image`` class is a core component of this Toolbox.  It encapsulates a
NumPy array that contains the pixel values of a greyscale or color image as
a 2D or 3D array respectively.
An ``Image`` instance has a very large number of methods that perform useful
operations on an image and wrap low-level operations performed using NumPy or
OpenCV.

Informational
-------------

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~__str__
   ~__repr__
   ~print
   ~copy
   ~name
   ~width
   ~height
   ~size
   ~shape
   ~npixels
   ~ndim
   ~centre
   ~centre_int
   ~center
   ~center_int

Image coordinates
-----------------

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~umax
   ~vmax
   ~uspan
   ~vspan
   ~meshgrid
   ~contains

Sub images
----------

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~roi
   ~plane
   ~red
   ~green
   ~blue
   ~__getitem__


NumPy pixel data
----------------

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

Datatype
--------

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

Color planes
------------

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
-----

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
-----------

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~Hstack
   ~Vstack
   ~Tile
   ~Overlay

Predicates
----------

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~iscolor
   ~isfloat
   ~isint
   ~isbool
   ~isbgr
   ~isrgb


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

Binary operators
----------------

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
---------------

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~__minus__
   ~__invert__

Monadic functions
-----------------

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
----------------

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~apply2
   ~blend
   ~choose
   ~paste
   ~direction


Linear filtering
----------------

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

Image labeling
--------------

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~labels_binary
   ~labels_MSER
   ~labels_graphseg
   ~kmeans_color

Image similarity
----------------

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

Non-linear filtering
--------------------

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

Shape changing
--------------

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


Image features
--------------

Whole image features
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~sum
   ~min
   ~max
   ~mean
   ~std
   ~var
   ~median
   ~otsu
   ~stats
   ~hist
   ~mpq
   ~npq
   ~upq
   ~moments
   ~humoments
   ~nonzero
   ~flatnonzero
   ~peak2d
   ~ocr
   ~fiducial
   ~MSER
   ~blobs

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

Multiview operations
--------------------

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~stdisp
   ~stereo_simple
   ~DSI_refine
   ~stereo_BM
   ~stereo_SGBM
   ~rectify_homographies



Graphics
--------

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~draw_line
   ~draw_circle
   ~draw_box


