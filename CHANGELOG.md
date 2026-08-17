# Changelog

## [2.4.0](https://github.com/petercorke/machinevision-toolbox-python/compare/v2.3.0...v2.4.0) (2026-08-17)


### Features

* add __array__ protocol to Image and Kernel, ndim to Kernel ([e762f57](https://github.com/petercorke/machinevision-toolbox-python/commit/e762f578faa269655cbd06b02d48c4c30d48c516))
* relax opencv pin to support both 4.x and 5.x, add CI matrix axis ([2ec69fe](https://github.com/petercorke/machinevision-toolbox-python/commit/2ec69fe4d116bb51f717e0a4a3c8baed95a7101f)), closes [#44](https://github.com/petercorke/machinevision-toolbox-python/issues/44)


### Bug Fixes

* assert hygiene -- unreachable assert, two caller-facing checks ([7afe33e](https://github.com/petercorke/machinevision-toolbox-python/commit/7afe33eb02af65d4e64d52481f668a610f63b807))
* CameraBase.plot() unbound call (self=None) crashed instead of working ([e2d4600](https://github.com/petercorke/machinevision-toolbox-python/commit/e2d460029d6ef1d5bb807168f37b37b3eb10ff1c))
* correct stale open3d-python PyPI name, add py3.13 exclusion marker ([87c1e8f](https://github.com/petercorke/machinevision-toolbox-python/commit/87c1e8f5a2eb3a2b17f2a1f1c182b22fbb025483))
* esttheta() -- broken pickregion() helper and missing theta-sweep body ([50b00cc](https://github.com/petercorke/machinevision-toolbox-python/commit/50b00cc95c15b56b3598505f4cd532a0612f6ff9))
* exclude tests/ from Bandit's assert_used (B101) check ([e62b116](https://github.com/petercorke/machinevision-toolbox-python/commit/e62b116e0b57aef73882ee4689153eda326c399f))
* Histogram.plot(type='ncdf') never actually worked, docstring lied ([6d4152d](https://github.com/petercorke/machinevision-toolbox-python/commit/6d4152d58bb63970863ff0ca49593c21ec815b5b))
* idisp() crashes when reuse=True with no title given ([a05b13b](https://github.com/petercorke/machinevision-toolbox-python/commit/a05b13b93ab4ed03397548348e8de48230b284f3))
* Image.metadata() missing camera-specific Exif sub-IFD tags ([d554ef2](https://github.com/petercorke/machinevision-toolbox-python/commit/d554ef257d2d494c66daaf928114c8097fb8fb91))
* Image() drops maxintval when dtype= is also given, silently truncating ([15f1d32](https://github.com/petercorke/machinevision-toolbox-python/commit/15f1d32576183c862592dca009828ea7e2765d79))
* Image(dtype='float') resolved to float64, not float32 ([18c5d85](https://github.com/petercorke/machinevision-toolbox-python/commit/18c5d85d82ebd2c09a43adb278e4435658f0c2b7))
* ImageConstantsMixin factories silently downcast explicit dtype requests ([977dbf6](https://github.com/petercorke/machinevision-toolbox-python/commit/977dbf6402307f38f695c697d68009ddd914d237))
* mvtbtool's %precision line spuriously echoes with --showassign ([ed43ca6](https://github.com/petercorke/machinevision-toolbox-python/commit/ed43ca64652a7d64a4536ae5665e672ffcc7db4a))
* name2color() colororder reordering returns list, not ndarray ([c90e6fb](https://github.com/petercorke/machinevision-toolbox-python/commit/c90e6fb4db7814707fd1548a7d87b1057b665499))
* PointCloud.Read() double-joins a caller-supplied "data/" prefix ([fff671c](https://github.com/petercorke/machinevision-toolbox-python/commit/fff671cdd28c85a555ee9fc0b7dd6f4bad4eccc9))
* printstats() crashes if .stats hasn't been accessed yet ([3484224](https://github.com/petercorke/machinevision-toolbox-python/commit/34842247c529159364591528ec109deac182899f))
* suppress divide-by-zero RuntimeWarning in BagOfWords idf computation ([05c85e0](https://github.com/petercorke/machinevision-toolbox-python/commit/05c85e0189061019dc55888b33ddb30cc3e9d9ae))
* visual-odometry robustness -- match() metric, points2F(), BundleAdjust ([8691847](https://github.com/petercorke/machinevision-toolbox-python/commit/8691847df58cc3cf003838436f7d94c59d5ae609))


### Documentation

* add AGENTS.md stub linking to rvc-ecosystem conventions ([0827291](https://github.com/petercorke/machinevision-toolbox-python/commit/082729184414b821a9e610f8a409f5cbeabc308c))
* points2F() kwargs -- correct which params apply to which method ([6b10d64](https://github.com/petercorke/machinevision-toolbox-python/commit/6b10d6401925564163568a09e1e504c99eec99fa))

## [2.3.0](https://github.com/petercorke/machinevision-toolbox-python/compare/v2.2.0...v2.3.0) (2026-08-13)

**Highlight: Notebooks working in all environments** Fixed a number of problems with getting notebooks to execute in desktop Jupyter, Colab, and
JupyterLite in browser.  Created nice framework to keep the environment-specific import logic in each notebook consistent at commit time.  Added
CI tests and also CI check that notebooks runs in the pyodide environment.  Currently pinning to pyodide 0.6.1 to sidestep the lack of JSPI support
in Safari and Firefox required by later pyodide kernels.


### Features

* **ci:** add a real-Pyodide dependency-completeness check for JupyterLite ([9072da6](https://github.com/petercorke/machinevision-toolbox-python/commit/9072da6e945b45bf2f04a5994f9dbc9c6d99622d))
* **notebooks:** add shared environment-bootstrap module and drift-check tooling ([44242e8](https://github.com/petercorke/machinevision-toolbox-python/commit/44242e81420e05d2d5656c81c53a33f1071ddac7))
* **notebooks:** migrate all notebooks to the generated bootstrap cell ([75f230c](https://github.com/petercorke/machinevision-toolbox-python/commit/75f230ce7d789e25edcd06694e886e0f4db8a83d))


### Bug Fixes

* **ci:** guard release-please's PyPI-publish trigger against the truthy-string bug ([#74](https://github.com/petercorke/machinevision-toolbox-python/issues/74)) ([0d74276](https://github.com/petercorke/machinevision-toolbox-python/commit/0d742769d2553125a5b2a08a4a8a16f1e9b9f69c))
* **ci:** pass --repo to gh calls in release-please's PyPI-publish trigger ([#73](https://github.com/petercorke/machinevision-toolbox-python/issues/73)) ([6ff01a3](https://github.com/petercorke/machinevision-toolbox-python/commit/6ff01a395402d6363ade8b4f4e07952237089666))
* **docs:** pin jupyterlite-pyodide-kernel==0.6.1, matching RTB and bdsim ([76bd7e1](https://github.com/petercorke/machinevision-toolbox-python/commit/76bd7e130dab5562bb947be9a6b6c1ef3a5e2140))
* **notebooks:** fix JupyterLite bugs found via real-browser testing ([3979404](https://github.com/petercorke/machinevision-toolbox-python/commit/3979404b317a79f2769007a02ff7a24091e1e4b7))
* release-please never triggered the actual PyPI publish ([18cc08b](https://github.com/petercorke/machinevision-toolbox-python/commit/18cc08bc21e38bef2f5ed56e31daee4c9404732a))


### Documentation

* **readme:** move to standard badge layout ([0722f9d](https://github.com/petercorke/machinevision-toolbox-python/commit/0722f9da96c97e2d49e791fd68237063df8d38cf))

## [2.2.0](https://github.com/petercorke/machinevision-toolbox-python/compare/v2.1.0...v2.2.0) (2026-08-10)

**Highlight: OpenCV 5 compatibility.** MVTB now works correctly under both OpenCV 4 and the
newly-released OpenCV 5 — aruco/fiducial pose estimation, HoughLinesP line features,
BRISK/AKAZE point features, and MSER region features have all been fixed for OpenCV 5's API
and shape changes, verified against real OpenCV 4.13.0 and 5.0.0 installs. See #44 for the
full compatibility audit.

### Features

* add mvtbtool --test smoke test, fix silent Open3D banner skip ([6a0bf2c](https://github.com/petercorke/machinevision-toolbox-python/commit/6a0bf2c090b8be4aa2d378495545ca71a55e89b1))

### Bug Fixes

* handle HoughLinesP's (N,4) return shape on OpenCV 5 ([b54f755](https://github.com/petercorke/machinevision-toolbox-python/commit/b54f7552a8c08530173e159d70a84fa5e8f6d486))
* handle MSER's empty-tuple bboxes on OpenCV 5, resolve MSER discrepancy ([aabaf88](https://github.com/petercorke/machinevision-toolbox-python/commit/aabaf885e660c59c4949a5f5706b276ad7c50202))
* normalize aruco ids/matchImagePoints shapes for OpenCV 5 ([6ac5ec0](https://github.com/petercorke/machinevision-toolbox-python/commit/6ac5ec0cc795ad7016f491dbb464e91c4ad34aae))
* replace removed cv2.aruco.estimatePoseSingleMarkers with solvePnP ([635225d](https://github.com/petercorke/machinevision-toolbox-python/commit/635225de657a1484ad526480605d9d234f202ffa))
* resolve BRISK/AKAZE via cv2.xfeatures2d on OpenCV 5 ([fad7247](https://github.com/petercorke/machinevision-toolbox-python/commit/fad7247cf6cfc07f7e8511ffa37ce4ccc928f451))

## [2.1.0](https://github.com/petercorke/machinevision-toolbox-python/compare/v2.0.1...v2.1.0) (2026-08-03)


### Features

* add interactive tools to imtool and bagtool ([b16ab7d](https://github.com/petercorke/machinevision-toolbox-python/commit/b16ab7dbd1f2fe26869a7fcacc710e549735e431))
* **mvtbtool:** add 'tool' extra for IPython/pygments, fail with clear message when missing ([a31655b](https://github.com/petercorke/machinevision-toolbox-python/commit/a31655be13d07acb14548e79c618d62d5bf5589c))


### Bug Fixes

* add missing type annotations, warn on ncdf deprecation ([7e8cabe](https://github.com/petercorke/machinevision-toolbox-python/commit/7e8cabe4ceee1f517b1a9161ab28e9d87137475e))
* add socket timeout to iread() URL fetches ([6ff62b1](https://github.com/petercorke/machinevision-toolbox-python/commit/6ff62b17aa1b556a1823e69ae4950f2786e0ebc1))
* bring 3 mixins in line with the TYPE_CHECKING _ImageBase pattern ([0d31fc5](https://github.com/petercorke/machinevision-toolbox-python/commit/0d31fc52f881df6cfda35c5ab11e4f9753e65aa0))
* colorplane handling in image i/o ([6882967](https://github.com/petercorke/machinevision-toolbox-python/commit/68829677c0f42d4b6e83b61b981f5027efe61ee3))
* gridify() IndexError from numpy float // int staying float64 ([e57597a](https://github.com/petercorke/machinevision-toolbox-python/commit/e57597aa527ab1741ff19066070ecd6968056da5))
* keep single-blob filter() matches ([7b0ab8d](https://github.com/petercorke/machinevision-toolbox-python/commit/7b0ab8d67c67251e12753df198447fdc9255f48a)), closes [#17](https://github.com/petercorke/machinevision-toolbox-python/issues/17)
* replace 6 bare except: clauses with specific exception types ([5a11306](https://github.com/petercorke/machinevision-toolbox-python/commit/5a11306eb838b2c47535389f482ab872b7a22d24))
* un-hide bot-comment notes from HTML comment block ([4a4e943](https://github.com/petercorke/machinevision-toolbox-python/commit/4a4e943fab8e00af404713efb1f3501d1c94f1a3))
* use eigh instead of eig for blob orientation/ellipse moments ([3a74fdf](https://github.com/petercorke/machinevision-toolbox-python/commit/3a74fdf6b558967ae797d19bbb4369275eaf690c))


### Documentation

* add build date to footer copyright notice ([e7de5bf](https://github.com/petercorke/machinevision-toolbox-python/commit/e7de5bf8e68d6251cbc23465befdd4070d7fbc6c))
* add example placeholder to Alternatives considered ([879f086](https://github.com/petercorke/machinevision-toolbox-python/commit/879f086dc0bcde61496c6f227d275dd19c47e53a))
* add PR and issue templates ([d081f7e](https://github.com/petercorke/machinevision-toolbox-python/commit/d081f7e49ae1e1673b00c92e4bd174e3c2629efb))
* ask for a concrete function signature in feature requests ([b25cd12](https://github.com/petercorke/machinevision-toolbox-python/commit/b25cd12efa6eae61a9ef2a6b05a78dbffca82cda))
* fix docstring convention note, add Codacy heads-up ([d68f605](https://github.com/petercorke/machinevision-toolbox-python/commit/d68f605301aaa71795c5cab75acadb910ce6d398))
* fix Image class sidebar, cut over to sphinx-pyrunblock ([20c219d](https://github.com/petercorke/machinevision-toolbox-python/commit/20c219d65fdb68ca14d58777a08f0c29333303dd))
* fix stale rank() reference in card-algorithms.rst ([1e3f6bd](https://github.com/petercorke/machinevision-toolbox-python/commit/1e3f6bdf3cbaa3307985f616af2c178c3d6a5029))
* fix stale/broken references and add footer build date ([c51955b](https://github.com/petercorke/machinevision-toolbox-python/commit/c51955bf14b55ed8dd9a19bf6b9c9d6b97952e6d))
* flesh out PR template with a proper checklist ([f4db44b](https://github.com/petercorke/machinevision-toolbox-python/commit/f4db44b4b21a16514ea0f1de2bceb54d22b6d39c))
* forewarn contributors about Dependabot vulnerability comments ([acfedab](https://github.com/petercorke/machinevision-toolbox-python/commit/acfedabfc285e11e4a505eb06b81c3bec74c0762))
* log ci.yml's conda/micromamba setup as tech debt ([af09d07](https://github.com/petercorke/machinevision-toolbox-python/commit/af09d07cae1df3129df4853d3a672db333f42465))
* log Codacy backlog real numbers to tech-debt ([65b3c5e](https://github.com/petercorke/machinevision-toolbox-python/commit/65b3c5ed3b2a2cbd20ee41a29b9c2b8234db126e))
* log gridify() IndexError bug found while testing bare-except fix ([889d45c](https://github.com/petercorke/machinevision-toolbox-python/commit/889d45c4957f93ec9910ccb960f91a432d1d06a3))
* log Histogram.plot type= docstring/behavior mismatch ([e693eef](https://github.com/petercorke/machinevision-toolbox-python/commit/e693eefa033c5d7a7e95c2787ac3f4dbbce0d2c3))
* log mypy-not-in-CI as high-priority tech debt, with fresh audit ([1b74ae8](https://github.com/petercorke/machinevision-toolbox-python/commit/1b74ae8efc240e43c338dec8415f7198de2a0d3a))
* log pgraph-python CI gap and conda-migration decision to tech-debt ([c5b6aeb](https://github.com/petercorke/machinevision-toolbox-python/commit/c5b6aebdaa42003bc48d0428d3381599f325d1df))
* log PR [#32](https://github.com/petercorke/machinevision-toolbox-python/issues/32)/[#33](https://github.com/petercorke/machinevision-toolbox-python/issues/33) Codacy findings (type shadowing, F405, max param) ([e0ed8db](https://github.com/petercorke/machinevision-toolbox-python/commit/e0ed8db736f1032c2a4fdb78c308c16f1b7e300b))
* migrate tech-debt.md to GitHub Issues (tech-debt label) ([36aae6a](https://github.com/petercorke/machinevision-toolbox-python/commit/36aae6aa9855ac79a4de19fb49fbaf30827b423a))
* remove reference to nonexistent CentralCamera.decomposeF ([c7f1400](https://github.com/petercorke/machinevision-toolbox-python/commit/c7f1400700f2ff5364d2fafe19367cfa49628e8e))
* switch sphinx-codeautolink to released 0.19.0, drop branch pin ([594a9a1](https://github.com/petercorke/machinevision-toolbox-python/commit/594a9a13580cf58868d836adee3a5bbaebc42d12))
* use %pip magic in notebooks, add tagline to README ([843db36](https://github.com/petercorke/machinevision-toolbox-python/commit/843db36441d43ec74e573d9b201f93635213c7ad))

2.0.0 May 2026

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
    - top-level of reST documentation now uses sphinx-design and a card-based TOC

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
    - `idisp` can now put text labels on colorbar ticks
  - `Image` class and core API
    - image NumPy array for pixel data
      - deprecate `.A` property, setting `.A` now raises an exception
      - systematic internal use of `._A` for  array access
      - `.array` for user use, provide a read-only view of the pixel data
    - uses `_image_typing.py` protocol to resolve `Image` type across mixins
    - improved color order and type logic
    - `Image(..., dtype=True)` forces image dtype to match the ndarray; otherwise the smallest fitting dtype is selected
    - `size` option supports turning pixel row/column data into 2D or 3D images
    - `Image.Tensor()` and `img.tensor()` for PyTorch import/export
    - statistics improvements:
      - `sum`, `min`, `max`, `mean`, `std`, `median` forward arguments to NumPy (for example `axis`)
      - `img.stats` is now a property returning per-plane statistics
      - `img.printstats()` prints formatted per-plane statistics
      - all are computed lazily
    - thresholding improvements:
      - native NumPy implementations for `otsu` and `triangle` threshold estimators
      - `threshold_interactive` reworked for Jupyter support
    - NaN/Inf handling:
      - `img.numnan` and `img.numinf` properties
      - `fixbad()` for remediation
      - reporting in `repr()` and `str()`
    - NumPy ufunc integration (for example `np.ceil(img)` returns an `Image`)
    - `%` operator stacks images plane-wise (for example `img1 % img2`, `img1 % 0`)
    - `Image.Random()`
      - supports multi-channel images
      - now has a `pdf` arugment to create an image with arbitary pixel-value distribution.
    - `img1.sameas(img2)` performs scalar equality checks across datatype/planes/pixels (`img1 == img2` remains element-wise)
    - image histogram properties `h`, `pdf`, `cf`, `cdf` are now computed lazily, and each returns a 2-tuple of ndarays (statistic, x).  These property names match those of the `Histogram` class. They were formally called `h`, `cdf`, and `ncdf` and return just a single ndarray.
    - `__str__` now includes image statistics like mean, std, median.
    - all "constant constructors" like `Zeros`, `Constant`, `Random` etc. now have a consistent API with 
      keyword arguments `size`, `dtype`, `colororder`, `like`.
    - `Kernel.Gauss` and `smooth` with a half-width but `sigma=0` will use a rule-of-thumb to estimate a good `sigma` value
    - `showpixels` has a major improvement in visual appearance, some parameters now gone.  The animation window has been refactored into `showwindow`. `examples/morphdemo` shows this in action.
    - `rprint` is a variant of `print` that returns the image object, so `img = Image.Random(size=5).rprint()` does an assignment to `img` and displays the pixel values to stdout in one line.
  - `Histogram` class
    - now has a `pdf` property for an estimated/empirical probability density function
    - cumulative frequency is renamed from `cdf` to `cf`; `ncdf` renamed to `cdf`
    - now implemented using NumPy rather than OpenCV for wider dtype support
    - works properly in Jupyter
    - histogram computation moved to constructor
    - `clip` controls bin range behaviour
    - plotting options for `span`, `log`, `cursor`, and `stats` markers
  - `Blobs` class
    - major internal refactor; `Blob` is now a dataclass
    - improved handling of runt (single-pixel) blobs
    - added `id` method
    - new attributes/methods: `MER()`/`plot_MER()`, `MEC()`/`plot_MEC()`
    - added default color and linestyle to all plot functions.  These all accept Matplotlib format string or keyword arguments for line styling
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
  - `rank`; use `rankfilter`
  - `Image.Zeros(10)` is now `Image.Zeros(size=10)`
  - `Image.Constant(3,4,5)` is now `Image.Constant(5, size=(3,4))`
  - `Image.Random(10)` is now `Image.Random(size=10)`

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
