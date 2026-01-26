
1.0.2 January 2026

* `Image` class

  - `warp_affine` can now warp an image into a given output image
  - graphic primitives
  - create an ArUco marker
  - save as PDF

  - all the draw_xxx() functions in base now have wrappers as methods of `Image`
  - draw_xxx() now handle floating point arguments, rounding them to the neares int
  - improvements in name2color to handle image datatypes and colororder

* FiducialCollection class, represent a generalized calibration board with AR tags

* ArUcoBoard class, represent a generalized ArUco calibration board

* changed to src folder layout, code is in src/machinevisiontoolbox
* changed from setuptools to hatch
* works if Open3D is not installed, it's always well behind in Python version support
* removed numpy < 2.0 constraint, OpenCV now suppports numpy 2.x
* working with Python 3.12 (except for Open3D)
* added command line tools:
  * `imtool` for displaying images, exploring pixels, picking points, showing metadata etc.  Works with your own images or those provided with MVTB 
  * `tagtool` for highlighting AR tags in images

1.0.1 March 2025

* `Image` class
  - `.dice` method, chops an image up into tiles, can be overlapping
  - `.Tile(columns=0)` will arrange the tiles into a roughly square layout
  - Constant images, previously many of these could only generate square images, this is now generalized
  - `.Chequerboard` creates a chequerboard pattern
  - set background color for image warp undefined pixels
  - single plane images can have a colorplane name
  - deprecate `colordict()`, use `colororder2dict()` instead
  - `String` now supports two string formats, can create color images
  - fixed bug with gamma="sRGB" which always returned a float image, type is now the same as passed

* Blobs

  - new methods for blob aligned box, plot_aligned_box, 
  - `plot_perimeter` options
  - `perimeter_hull` convex hull
  - fixed bug with runt blobs

* Kernels

  - added `Kernel.HGauss` for Hessian of Gaussian
  - added `Kernel` repr method, `disp` method
ArUcoBoard class

* Camera models
  - Fixed errors in some camera projection models

* Point clouds

  - `PointCloud` now has a "constructor" for depth images, `PointCloud.DepthImage()`

* Miscellaneous

  - Lots more code examples and plots
  - Move decoratores scalar_result and array_result to machinevisiontoolbox.decorators.py
  - improved unit testing



1.0.0 January 2025

* `Kernel` methods now return `Kernel` instances rather than NumPy arrays. Methods that
accept a kernel can accept a `Kernel` instance or a NumPy array.  Methods exist to 
stringify or print a kernel.

* The indexing order of an `Image` object (using square bracket `__getitem__` access) has
  changed and is now `img[u,v]` where `u` is the column coordinate and `v` is the row
  coordinate.  This is consistent with the column-first convention used across the
  Toolbox and is consistent with the $(u,v)$ coordinate system for images. But, this
  is the __opposite__ order to that used for NumPy index on the underlying array, and
  to earlier versions of the Toolbox. 

    - a 2-tuple of integers, select this pixel.  If the image has multiple planes, the
      result is a vector over planes.
    - a 3-tuple of integers, for a multiplane image select this pixel from the specified
      plane.
    - a 2-tuple of slice objects, select this region. If the image has multiple planes,
      the result is a 3D array.
    - a 3-tuple of slice objects, select this region of uv and planes
    - an int, select this plane
    - a string, select this named plane or planes

* added `pixel(u,v)` method for faster access to a single pixel value, scalar or vector.

* the children of a `Blob` is now given as a list of `Blob` objects, not their indices
within the overall list of blobs.  This simplies traversing the blob hierarchy tree.
Similarly, the parent is a reference to the parent `Blob` object rather than an index,
and is `None` if the blob has no parent (its parent is the background).

* Documentation overhaul, both in-code docstrings, and the organization of the overall Sphinx document.

* Additional unit tests

* Myriad minor bug fixes, see commit history.