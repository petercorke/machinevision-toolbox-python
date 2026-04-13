1.1.0 April 2026

* Highlights
  - support for ROS (bag files & streams) and PyTorch
  - major rework across code, documentation, unit tests, and CI
  - type hinting throughout and broad cleanup of lint/suppression issues
  - extensive API and naming consistency work, including deprecation standardization
  - massive use of CoPilot

* Added
  - image sources
    - `ROSBag` reads images and point clouds from ROS 1/2 bag files
    - `ROSTopic` reads images and point clouds from live ROS 1/2 systems via `rosbridge`
    - `ImageSequence`: a sequence of image objects
    - `PointCloudSequence`: a sequence of point cloud objects
    - `TensorStack`: reads images from a batch tensor
    - `LabelMeReader`: returns image and shape data from a LabelMe JSON file
  - CLI tools
    - `bagtool`: animate images or point clouds from a ROS 1/2 bag file
    - `ocrtool`: write OCR text to stdout or JSON
  - documentation
    - ecosystem diagram
    - new Sphinx sections for ROS, PyTorch, NumPy integration, and Jupyter
    - code copy button for examples (strips `>>>` prompts)

* Changed
  - CI and packaging
    - changed `master` branch to `main`
    - reworked and renamed CI files
    - replaced `flake8` with `ruff`
    - added extras: `jupyter`, `pytorch`, `ros`, `ocr`
    - added `all` extra to install all optional extras
    - consolidated documentation and notebooks into `docs`
  - code quality
    - tested with Python 3.13
    - improved behaviour when Open3D is unavailable
    - many more unit tests, with average coverage now > 50%
    - consistent and PEP-conformant `repr()` and `str()` formatting
    - more consistent handling of lazy imports
    - systematic use of `import cv2`
    - consistent naming of test files
    - PEP 8/257-compliant import grouping and module docstrings
    - revisited suppression comment tags for validity and currency
    - normalized deprecation/warning/docstring language with explicit version tagging
  - image I/O and display
    - `iread` now uses `cv2.imdecode` instead of PIL, with internal refactor for corner cases
    - `iread_iter` is an efficient lazy wildcard iterator
    - `idisp` now includes keyboard display and animation controls when `fps` or `animate` is specified
    - fixed `idisp` issues with Jupyter Matplotlib backend selection (for example `%matplotlib widget`)
  - `Image` class and core API
    - uses `_image_typing.py` protocol to resolve `Image` type across mixins
    - improved color order and type logic
    - systematic internal use of `._A` for NumPy array access
    - `Image(..., dtype=True)` forces image dtype to match the ndarray; otherwise the smallest fitting dtype is selected
    - `size` option supports turning pixel row/column data into 2D or 3D images
    - `Image.Tensor()` and `img.tensor()` for PyTorch import/export
    - statistics improvements:
      - `sum`, `min`, `max`, `mean`, `std`, `median` forward arguments to NumPy (for example `axis`)
      - `img.stats` is now a property returning per-plane statistics
      - `img.printstats()` prints formatted per-plane statistics
    - thresholding improvements:
      - native NumPy implementations for `otsu` and `triangle` threshold estimators
      - `threshold_interactive` reworked for Jupyter support
    - NaN/Inf handling:
      - `img.numnan` and `img.numinf` properties
      - `fixbad()` for remediation
      - reporting in `repr()` and `str()`
    - NumPy ufunc integration (for example `np.ceil(img)` returns an `Image`)
    - `%` operator stacks images plane-wise (for example `img1 % img2`, `img1 % 0`)
    - `Image.Random()` supports multi-channel images
    - `img1.sameas(img2)` performs scalar equality checks across datatype/planes/pixels (`img1 == img2` remains element-wise)
  - `Histogram` class
    - now implemented on NumPy rather than OpenCV for wider dtype support
    - works properly in Jupyter
    - histogram computation moved to constructor
    - `clip` controls bin range behaviour
    - plotting options for `span`, `log`, `cursor`, and `stats` markers
  - `Blobs` class
    - major internal refactor; `Blob` is now a dataclass
    - improved handling of runt (single-pixel) blobs
    - added `id` method
    - new attributes/methods: `MER()`/`plot_MER()`, `MEC()`/`plot_MEC()`
  - Image sources
    - all sources are iterators and context managers, reworked the common abstract base class
    - all sources inherit display/animation (`.disp()`) and batch tensor (`.tensor()`) methods
    - all sources accept keyword arguments forwarded to `convert`
    - `VideoCamera` has a new `list` method showing camera name/id mapping
    - `VideoFile`, `WebCam`, and `EarthView` are unchanged
  - CLI
    - improved `--help` strings and option ordering
    - `mvtbtool` supports image preloading into the IPython namespace, autoreload, optional PyTorch import, and `MVTB_OPTIONS` environment variable
    - `tagtool` reports ArUco/April tags to stdout or JSON
  - notebook support
    - notebooks revamped and extended
    - `threshold_interactive` works in Jupyter
    - notebook workflows are unit-testable
    - distributed as a ZIP built by GitHub Actions
    - distributed in a zero-install JupyterLite environment (WASM + Emscripten)

* Deprecated
  - `iread(..., grey=...)` and `iread(..., gray=...)`; use `mono=` instead
  - `img.image` and `img.A`; use `img.array`
  - `img.to_int()` and `img.to_float()`; use `img.array_as()`
  - `column`; use `view1d`
  - `thresh()`; use `threshold()`
  - threshold keyword `t`; use `threshold`
  - threshold keyword `opt`; use `method`
  - `ithresh`; use `threshold_interactive`
  - `adaptive_threshold`; use `threshold_adaptive`
  - legacy `Image.Constant` positional size form (for example migrate from `Image.Constant(10, 20, 30)` to `Image.Constant(30, size=(10, 20))`)
  - `ImageCollection`; use `FileCollection`
  - `ZipArchive`; use `FileArchive`

* Fixed
  - resolved many Sphinx warnings
  - resolved issues in plot/runblock examples
  - improved OpenCV documentation link consistency where intersphinx is not effective
  - fixed issues in `idisp` that confused Jupyter Matplotlib backend selection

* Miscellaneous
  - generated OpenCV guard functions from OpenCV documentation for input validation
  - `Kernel` class refactored to its own file
  - ROS synchronization support via `SyncROSStreams`

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
  - `imtool` for displaying images, exploring pixels, picking points, showing metadata etc.  Works with your own images or those provided with MVTB 
  - `tagtool` for highlighting AR tags in images

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