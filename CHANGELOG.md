# Changelog

## [2.5.0](https://github.com/petercorke/machinevision-toolbox-python/compare/v2.4.1...v2.5.0) (2026-11-27)

**New features**

* `Histogram.compare()`, with seven comparison measures
* `Image.ComboFeature` joins any supported detector to any descriptor, for example FAST with BRIEF
* SURF features (needs an OpenCV build that includes the non-free modules)
* PointCloud and VoxelGrid draw in the notebook cell with `disp(notebook=True)`; PointCloud also gains `concatenate()`, `unpaint()` and painting by color name
* New notebooks: point clouds, and CNN segmentation and detection with Torch

**Improvements**

* Depth images from a ROSBag, or any integer depth image with `depth_units`, come out as float32 metres, with 0 as NaN
* Graphics options are tidier, and MATLAB-style shorthand such as `"r:"` and short Matplotlib names such as `lw` work consistently
* `idisp` has a `?` key that lists its keys, and plotting functions set window titles
* Notebooks in the browser now behave like desktop Jupyter

**Changes**

* `name2color()` no longer accepts wildcards. A name containing `.`, `?` or `*` raises `ValueError`. To list the names that match a pattern, use the new `color_names()`, which takes a regular expression
* `Image.Read` returns C-contiguous pixels, and drawing into a strided image raises an error
* `int_image()` clips floats to 0 to 1
* `gamma_encode()`, `gamma_decode()` and `colorspace_convert()` treat array shape and type consistently. The result has the type of the input unless the new `dtype` argument says otherwise, and 1D arrays work. An 8-bit image converted to HSV, L*a*b* or XYZ now gives float32 in natural units (hue in degrees, L* 0 to 100), where before it had OpenCV's 8-bit encoding. A 3-vector or Nx3 array of integers is scaled to 0 to 1, where before it gave silently wrong colors
* A missing image file raises `FileNotFoundError`, not `ValueError`, with a short traceback and a message that suggests similar file names. Code that caught `ValueError` for a missing file must catch `FileNotFoundError`* `pcd.normals(radius=..., max_nn=...)`, the old method call, still works but is deprecated and warns. Use `estimate_normals()`; `normals` is the property that returns the normals

**Under the hood**

* mypy passes with no errors, `py.typed` is shipped, test coverage is much higher, Codacy rates the code A, and over 60 bugs were found and fixed

**Compatibility**

* Python 3.10 is no longer supported. Python 3.11 to 3.14 are tested
* The in-browser notebooks need a browser with JSPI support: Chrome 137 or later, Safari 27 or later, Firefox 153 or later
* OpenCV 4 and 5 are both supported, including some OpenCV 5.1 edge cases


### Features

* a missing image file raises FileNotFoundError, with a short traceback and suggestions ([#328](https://github.com/petercorke/machinevision-toolbox-python/issues/328)) ([06d2e95](https://github.com/petercorke/machinevision-toolbox-python/commit/06d2e9595067100124d79cee94530a75a8959874))
* add a ? key to idisp that prints the list of keys ([#215](https://github.com/petercorke/machinevision-toolbox-python/issues/215)) ([3e5dcab](https://github.com/petercorke/machinevision-toolbox-python/commit/3e5dcab6096ae54e1446fb67605ee3556226bf5a))
* add a notebook on CNN segmentation and detection with torch ([#206](https://github.com/petercorke/machinevision-toolbox-python/issues/206)) ([4de883f](https://github.com/petercorke/machinevision-toolbox-python/commit/4de883f1a5f387e59949fbe5cf186c6a639bb274))
* add color_names and remove the wildcard from name2color ([#249](https://github.com/petercorke/machinevision-toolbox-python/issues/249)) ([031611f](https://github.com/petercorke/machinevision-toolbox-python/commit/031611fa2043088575495e9a0499fdb1f091b403))
* add Histogram.compare with seven comparison methods ([#203](https://github.com/petercorke/machinevision-toolbox-python/issues/203)) ([9dae5d0](https://github.com/petercorke/machinevision-toolbox-python/commit/9dae5d0f8e308819b38d7e071c6dbe4c9af7ca9a))
* add Image.SURF and SURFFeature ([#201](https://github.com/petercorke/machinevision-toolbox-python/issues/201)) ([09042f4](https://github.com/petercorke/machinevision-toolbox-python/commit/09042f49065f5a87460fbb9d00cf758943695e0e)), closes [#113](https://github.com/petercorke/machinevision-toolbox-python/issues/113)
* add text_style to plot_labelbox, and make the label background follow the box edge color ([#273](https://github.com/petercorke/machinevision-toolbox-python/issues/273)) ([dd6decc](https://github.com/petercorke/machinevision-toolbox-python/commit/dd6decc24bf90183f0f37dbb4b43c44418c9d42d))
* compute the Hough accumulator by voting ([#228](https://github.com/petercorke/machinevision-toolbox-python/issues/228)) ([90e21b4](https://github.com/petercorke/machinevision-toolbox-python/commit/90e21b4245964183e802017ab5a903df0dd00e4d))
* consistent array shape and data type handling in gamma and color conversion ([#204](https://github.com/petercorke/machinevision-toolbox-python/issues/204)) ([860ae3f](https://github.com/petercorke/machinevision-toolbox-python/commit/860ae3f0ed4c4d636d0dd9c1f96c987ae60dacef))
* depth_units turns 0 into NaN in an integer depth image ([#188](https://github.com/petercorke/machinevision-toolbox-python/issues/188)) ([4150231](https://github.com/petercorke/machinevision-toolbox-python/commit/4150231d7f330e8e9ed49c99c11dc29b426da4a7)), closes [#151](https://github.com/petercorke/machinevision-toolbox-python/issues/151)
* Image.ComboFeature joins any supported detector to any descriptor ([#286](https://github.com/petercorke/machinevision-toolbox-python/issues/286)) ([2c58570](https://github.com/petercorke/machinevision-toolbox-python/commit/2c58570f3f6a8ab3354fab5d212c125f92ed809e))
* iwrite creates missing folders in the output path ([#200](https://github.com/petercorke/machinevision-toolbox-python/issues/200)) ([eb34023](https://github.com/petercorke/machinevision-toolbox-python/commit/eb34023ac2f819d2a32279d597a2507f0a777609))
* make sum() work on feature sets and raise clear TypeErrors ([#243](https://github.com/petercorke/machinevision-toolbox-python/issues/243)) ([cb30d19](https://github.com/petercorke/machinevision-toolbox-python/commit/cb30d19f25f654eef3ee6664ac85b94ca09145ca))
* plane() and planes() take copy=True for an independent, C-contiguous result ([#310](https://github.com/petercorke/machinevision-toolbox-python/issues/310)) ([79ec26c](https://github.com/petercorke/machinevision-toolbox-python/commit/79ec26cb0efc00df343e129e13fb1a99f42af057))
* PointCloud unknown names raise AttributeError with a helpful message ([#297](https://github.com/petercorke/machinevision-toolbox-python/issues/297)) ([55a5b9f](https://github.com/petercorke/machinevision-toolbox-python/commit/55a5b9fa7917b1eb35ccb2a1adc23597160e2ec1))
* PointCloud.concatenate, unpaint, paint by color name, and sum() of point clouds ([#287](https://github.com/petercorke/machinevision-toolbox-python/issues/287)) ([58deb9e](https://github.com/petercorke/machinevision-toolbox-python/commit/58deb9e7685f4a9e3cfba17662cfcd72a4a5039b))
* PointCloud.disp(notebook=True) draws in the notebook cell with Plotly ([#279](https://github.com/petercorke/machinevision-toolbox-python/issues/279)) ([829668b](https://github.com/petercorke/machinevision-toolbox-python/commit/829668b4e2b5ef7c7c62840710882a1e475fc31c))
* PointCloud.segment_plane tries 1000 iterations by default, and is documented ([#289](https://github.com/petercorke/machinevision-toolbox-python/issues/289)) ([aebb259](https://github.com/petercorke/machinevision-toolbox-python/commit/aebb25934c1fbd033c2a2a6560e67190ef1ba7b1))
* ROSBag depth_units returns depth images as float32 metres ([#187](https://github.com/petercorke/machinevision-toolbox-python/issues/187)) ([156d787](https://github.com/petercorke/machinevision-toolbox-python/commit/156d787e0a9c77766b0b69a05248a41cf3ade54c))
* run the JupyterLite notebooks with ipympl and follow the plotting recipe ([#193](https://github.com/petercorke/machinevision-toolbox-python/issues/193)) ([fa8de87](https://github.com/petercorke/machinevision-toolbox-python/commit/fa8de87396b9f08f97840d6edcc76bcda52453c5))
* scale, orientation and strength raise for features that do not have them ([#275](https://github.com/petercorke/machinevision-toolbox-python/issues/275)) ([dbe8b4a](https://github.com/petercorke/machinevision-toolbox-python/commit/dbe8b4a551599613b11ffc9c612c8a99a912ff2b))
* style the label of Blobs.plot_centroid and each axis of Blobs.plot_axes ([#269](https://github.com/petercorke/machinevision-toolbox-python/issues/269)) ([d972f53](https://github.com/petercorke/machinevision-toolbox-python/commit/d972f533493b014990a280d518060a73cbf40491))
* use spatialmath's set_window_title and remove MVTB's own ([#190](https://github.com/petercorke/machinevision-toolbox-python/issues/190)) ([df755a4](https://github.com/petercorke/machinevision-toolbox-python/commit/df755a45d101462ee30929697b48f8745061b84e)), closes [#147](https://github.com/petercorke/machinevision-toolbox-python/issues/147)
* VoxelGrid.disp(notebook=True) draws in the notebook cell with Plotly ([#282](https://github.com/petercorke/machinevision-toolbox-python/issues/282)) ([b967cb3](https://github.com/petercorke/machinevision-toolbox-python/commit/b967cb3d35dfb3788db937384948cd41afab156c))
* window titles for plotting functions ([#191](https://github.com/petercorke/machinevision-toolbox-python/issues/191)) ([14d0fb3](https://github.com/petercorke/machinevision-toolbox-python/commit/14d0fb3c88a49599e113b275fcf984ac83ff7b5c))


### Bug Fixes

* accept the short Matplotlib option names (lw, c, ...) in mpl_styling ([#266](https://github.com/petercorke/machinevision-toolbox-python/issues/266)) ([7387141](https://github.com/petercorke/machinevision-toolbox-python/commit/7387141a1e14bf4e5133d582b410505a6c6c20a3))
* add a dtype argument to convolve and keep the sign of gradients ([#246](https://github.com/petercorke/machinevision-toolbox-python/issues/246)) ([bc0f1b3](https://github.com/petercorke/machinevision-toolbox-python/commit/bc0f1b32cf8a75692ad72b73d0f1b70c1fee35f6))
* add py.typed marker and make the package pass mypy ([#176](https://github.com/petercorke/machinevision-toolbox-python/issues/176)) ([49f4606](https://github.com/petercorke/machinevision-toolbox-python/commit/49f4606e6c1446ffbe10602b652ef64abc40b9a3))
* adding an empty feature set no longer discards the descriptors ([#288](https://github.com/petercorke/machinevision-toolbox-python/issues/288)) ([a550c18](https://github.com/petercorke/machinevision-toolbox-python/commit/a550c18e94fbb4064a58a59f8778920520548e14))
* allow a minimum disparity of zero in stereo_simple ([#231](https://github.com/petercorke/machinevision-toolbox-python/issues/231)) ([312e12d](https://github.com/petercorke/machinevision-toolbox-python/commit/312e12d020e3a82943e43f04dc8452bcbd4f5f9f))
* attach the release zips with gh release upload ([#218](https://github.com/petercorke/machinevision-toolbox-python/issues/218)) ([e1a89fb](https://github.com/petercorke/machinevision-toolbox-python/commit/e1a89fbfa1be5fa7a27b91838a3b85e71b90cf0e))
* Blobs.plot_labelbox and plot_box say when a format string has a marker ([#270](https://github.com/petercorke/machinevision-toolbox-python/issues/270)) ([bb4db62](https://github.com/petercorke/machinevision-toolbox-python/commit/bb4db62eeaa118687c8e4d54a8435a72427833d0))
* Camera recognises JupyterLite using spatialmath's isnotebook ([#185](https://github.com/petercorke/machinevision-toolbox-python/issues/185)) ([bbe8342](https://github.com/petercorke/machinevision-toolbox-python/commit/bbe83421d1fbf0a5f01b5401245b125ef1ea2e59)), closes [#166](https://github.com/petercorke/machinevision-toolbox-python/issues/166)
* CannyHistogram.plot shows the figure once, not twice ([#278](https://github.com/petercorke/machinevision-toolbox-python/issues/278)) ([1432117](https://github.com/petercorke/machinevision-toolbox-python/commit/14321172409c9084d844c2f1eb918c637750598f))
* compute the distance between feature sets correctly ([#234](https://github.com/petercorke/machinevision-toolbox-python/issues/234)) ([b3eaf64](https://github.com/petercorke/machinevision-toolbox-python/commit/b3eaf6407bbb34dd3d3b5e5589fbc9c80ef76e3b))
* consume Matplotlib format markers ([#144](https://github.com/petercorke/machinevision-toolbox-python/issues/144)) ([bcf63e4](https://github.com/petercorke/machinevision-toolbox-python/commit/bcf63e40384758ff35271b995c9abd425b65825b))
* disp(notebook=True) names an option that Plotly cannot draw ([#303](https://github.com/petercorke/machinevision-toolbox-python/issues/303)) ([9f4cdf2](https://github.com/petercorke/machinevision-toolbox-python/commit/9f4cdf2341ca85cb771e80780cc0601f328fde58))
* do not change the template in similarity with a z metric ([#237](https://github.com/petercorke/machinevision-toolbox-python/issues/237)) ([08d7184](https://github.com/petercorke/machinevision-toolbox-python/commit/08d718419ebf9b1f914227c8d42565986cec6632))
* draw the orientation hand of a feature in its direction ([#235](https://github.com/petercorke/machinevision-toolbox-python/issues/235)) ([764cc15](https://github.com/petercorke/machinevision-toolbox-python/commit/764cc15e9dcaf277eafa24bf906a1f65501b6066))
* drop Python 3.10 (end of life Oct 2026) ([#174](https://github.com/petercorke/machinevision-toolbox-python/issues/174)) ([8eb814f](https://github.com/petercorke/machinevision-toolbox-python/commit/8eb814fed5649825f9d5fb9407a68052ceb369f1))
* EarthView reads GOOGLE_MAPS_API_KEY first ([#149](https://github.com/petercorke/machinevision-toolbox-python/issues/149)) ([d48a8df](https://github.com/petercorke/machinevision-toolbox-python/commit/d48a8df64eab61281ea3a585914b50342112501d))
* EstPose_p passes its frame argument to estpose ([#210](https://github.com/petercorke/machinevision-toolbox-python/issues/210)) ([3594254](https://github.com/petercorke/machinevision-toolbox-python/commit/3594254bdd5476b3908e483f9e13ab9bcfc1facc))
* examples may open any number of figures in the docs build ([#302](https://github.com/petercorke/machinevision-toolbox-python/issues/302)) ([a4ddb29](https://github.com/petercorke/machinevision-toolbox-python/commit/a4ddb297be5bc080967fc1afe55499dfa3a8b4c0))
* exclude endpoint peaks in findpeaks ([#285](https://github.com/petercorke/machinevision-toolbox-python/issues/285)) ([7b7fc66](https://github.com/petercorke/machinevision-toolbox-python/commit/7b7fc66a2d50eb7542aee3786b6a8914e87522ff))
* FeatureMatch.estimate keeps only an inlier mask as the inlier status ([#272](https://github.com/petercorke/machinevision-toolbox-python/issues/272)) ([6055f56](https://github.com/petercorke/machinevision-toolbox-python/commit/6055f562c48e5b1ac381c00fbf014e56258530c4))
* fetch the monalisa example image from GitHub ([#197](https://github.com/petercorke/machinevision-toolbox-python/issues/197)) ([de3cb1c](https://github.com/petercorke/machinevision-toolbox-python/commit/de3cb1c464adef921b0a7f1532a84bcc391ab05b))
* give rankfilter a default rank of median ([#241](https://github.com/petercorke/machinevision-toolbox-python/issues/241)) ([de734f9](https://github.com/petercorke/machinevision-toolbox-python/commit/de734f9a83871ff95194611f9a654c84a09ac167))
* gridify keeps the descriptors and flags, and copes with features at the far edge ([#277](https://github.com/petercorke/machinevision-toolbox-python/issues/277)) ([0f404f4](https://github.com/petercorke/machinevision-toolbox-python/commit/0f404f4e0447114cacda5c971f7dffa9c6d52f1e))
* Harris(scale=...) reaches the detector instead of being taken as a flag ([#276](https://github.com/petercorke/machinevision-toolbox-python/issues/276)) ([b17c481](https://github.com/petercorke/machinevision-toolbox-python/commit/b17c481e715cc36e0c9c78ec504d383f199b3fc2))
* Histogram.plot(type="cf") works for a greyscale histogram ([#281](https://github.com/petercorke/machinevision-toolbox-python/issues/281)) ([aee8333](https://github.com/petercorke/machinevision-toolbox-python/commit/aee833335a8a4118ec25f7fe2a66726b36b73ace))
* idisp clipboard keys v and V copy plain numbers ([#214](https://github.com/petercorke/machinevision-toolbox-python/issues/214)) ([b36ffc4](https://github.com/petercorke/machinevision-toolbox-python/commit/b36ffc48b23d38d527f9e3291f711e178ce2c2f4))
* Image.Read returns C-contiguous pixels; drawing into a strided image raises ([#162](https://github.com/petercorke/machinevision-toolbox-python/issues/162)) ([dee5e1b](https://github.com/petercorke/machinevision-toolbox-python/commit/dee5e1b385cd985f50ba06c7718136720d484927))
* Image.tensor and Image.Tensor get distinct pages in the docs ([#304](https://github.com/petercorke/machinevision-toolbox-python/issues/304)) ([7d745c9](https://github.com/petercorke/machinevision-toolbox-python/commit/7d745c9b4f3fdee97d79b577f8e27e758d1f73a8))
* Image[...] with a list of planes raises, and a plane index works on a greyscale image ([#309](https://github.com/petercorke/machinevision-toolbox-python/issues/309)) ([70b33bb](https://github.com/petercorke/machinevision-toolbox-python/commit/70b33bba83c873b120766987686e57997fe67e50))
* import mat_struct from the public scipy.io.matlab namespace ([#262](https://github.com/petercorke/machinevision-toolbox-python/issues/262)) ([ae7edef](https://github.com/petercorke/machinevision-toolbox-python/commit/ae7edef3e8de1e8c846bf0bf89eb61d9e170c026))
* index and print MSER features correctly ([#229](https://github.com/petercorke/machinevision-toolbox-python/issues/229)) ([0d4d6c8](https://github.com/petercorke/machinevision-toolbox-python/commit/0d4d6c82845e2e921397708da1c4716fb84bf2d1))
* install Open3D when building the docs ([#294](https://github.com/petercorke/machinevision-toolbox-python/issues/294)) ([efc147b](https://github.com/petercorke/machinevision-toolbox-python/commit/efc147b54881f7c858711a43a0e72962ea1a579c))
* int_image clips floating point data to 0 to 1 instead of letting it wrap ([#325](https://github.com/petercorke/machinevision-toolbox-python/issues/325)) ([b9e42ed](https://github.com/petercorke/machinevision-toolbox-python/commit/b9e42ed8cc3025d444d8b144b2aa9f35b32062e1))
* JupyterLite bootstrap installs the bundled wheel, using the shared template ([#180](https://github.com/petercorke/machinevision-toolbox-python/issues/180)) ([f0584af](https://github.com/petercorke/machinevision-toolbox-python/commit/f0584af9fcfa87dcb71aa8b110c10ef6dfe86efc))
* keep the defaults a MATLAB-style format string does not set in mpl_styling ([#267](https://github.com/petercorke/machinevision-toolbox-python/issues/267)) ([1dda16e](https://github.com/petercorke/machinevision-toolbox-python/commit/1dda16ef284063020d5c4c82009221624725c55f))
* keep the inlier status of a FeatureMatch an array when it is indexed ([#265](https://github.com/petercorke/machinevision-toolbox-python/issues/265)) ([c75544b](https://github.com/petercorke/machinevision-toolbox-python/commit/c75544b3622c9e9d2e8dff4f5fd0c334da95b0ed))
* keep the type of the result of blend ([#238](https://github.com/petercorke/machinevision-toolbox-python/issues/238)) ([fa85a7c](https://github.com/petercorke/machinevision-toolbox-python/commit/fa85a7ca9a2968bf1e2e778866b3db8c055aa16e))
* label the x axis of the Canny histogram "gradient magnitude", and add xlabel to Histogram.plot ([#280](https://github.com/petercorke/machinevision-toolbox-python/issues/280)) ([95daa06](https://github.com/petercorke/machinevision-toolbox-python/commit/95daa0670b347bd48fbe73eaf7ecca45c54e7e34))
* LabelMeReader raises on unreadable input instead of returning None ([#208](https://github.com/petercorke/machinevision-toolbox-python/issues/208)) ([81daa6a](https://github.com/petercorke/machinevision-toolbox-python/commit/81daa6a51148dda42def7495e131efe3d1b488ec)), closes [#150](https://github.com/petercorke/machinevision-toolbox-python/issues/150)
* make canny use its sigma and threshold arguments ([#239](https://github.com/petercorke/machinevision-toolbox-python/issues/239)) ([d5ad31a](https://github.com/petercorke/machinevision-toolbox-python/commit/d5ad31a39229a6a2605706e79086cd24bd7d54f7))
* make filter(minstrength=) keep the stronger features ([#233](https://github.com/petercorke/machinevision-toolbox-python/issues/233)) ([6112160](https://github.com/petercorke/machinevision-toolbox-python/commit/6112160358270ec14b9f5a6f209d242b492d9ca5))
* make morphdemo work with immutable images and Image.Random(size=) ([#195](https://github.com/petercorke/machinevision-toolbox-python/issues/195)) ([586b78b](https://github.com/petercorke/machinevision-toolbox-python/commit/586b78b8bb90550c4980479e4692a66d9287ea8d))
* make stereo_simple return outputs the size of its inputs ([#240](https://github.com/petercorke/machinevision-toolbox-python/issues/240)) ([ce37ae3](https://github.com/petercorke/machinevision-toolbox-python/commit/ce37ae34228c5bacb5b1815c94331e655ee9db3f))
* make the stdisp cross-hair cursor work ([#225](https://github.com/petercorke/machinevision-toolbox-python/issues/225)) ([61f7916](https://github.com/petercorke/machinevision-toolbox-python/commit/61f79162c80c329f1692a737f3445d0f9f7375ae))
* mvtbtool --backend sets the Matplotlib backend instead of stopping with NameError ([#324](https://github.com/petercorke/machinevision-toolbox-python/issues/324)) ([9e831ea](https://github.com/petercorke/machinevision-toolbox-python/commit/9e831ea5f3b24747b3fc72f8b247e308162dfa5d))
* narrow file picker data fallback errors ([#146](https://github.com/petercorke/machinevision-toolbox-python/issues/146)) ([77a3da9](https://github.com/petercorke/machinevision-toolbox-python/commit/77a3da9cda0d87eab9c88b213fc26f993c993664))
* narrow window title exception handling ([#141](https://github.com/petercorke/machinevision-toolbox-python/issues/141)) ([715af60](https://github.com/petercorke/machinevision-toolbox-python/commit/715af60cf0c081b2bf32a3f3a9843143877dd097))
* ocrtool -l no longer crashes when it finds no words ([#300](https://github.com/petercorke/machinevision-toolbox-python/issues/300)) ([ac3d867](https://github.com/petercorke/machinevision-toolbox-python/commit/ac3d867ecefc52bdd2e7c3a349f6e4915a7b0302))
* pickpoints with matplotlib=True no longer raises NameError ([#213](https://github.com/petercorke/machinevision-toolbox-python/issues/213)) ([4ddd9f9](https://github.com/petercorke/machinevision-toolbox-python/commit/4ddd9f94ccefc1bd050cc9435c2fe0f0e0b2f05b))
* PointCloud.normals(radius=..., max_nn=...) still works, with a deprecation warning ([#283](https://github.com/petercorke/machinevision-toolbox-python/issues/283)) ([60e13e6](https://github.com/petercorke/machinevision-toolbox-python/commit/60e13e67ad5650d2cd2ff085e1d7248d7e800fdc))
* preserve unexpected peak refinement failures ([#140](https://github.com/petercorke/machinevision-toolbox-python/issues/140)) ([ca5ee04](https://github.com/petercorke/machinevision-toolbox-python/commit/ca5ee047c72b6873a23482e2a8351dc1c3409283))
* print a FeatureMatch that holds a single match ([#227](https://github.com/petercorke/machinevision-toolbox-python/issues/227)) ([c2244e1](https://github.com/petercorke/machinevision-toolbox-python/commit/c2244e1ce5750f3387bf369ef9fae04d5392a1f0))
* raise for an unknown colorspace in name2color and plot_spectral_locus ([#248](https://github.com/petercorke/machinevision-toolbox-python/issues/248)) ([2ec0f6d](https://github.com/petercorke/machinevision-toolbox-python/commit/2ec0f6d4fee8e0d01bdbdf48a8d78301410bf674))
* raise NotImplementedError for VISJAC_P depthest=True ([#252](https://github.com/petercorke/machinevision-toolbox-python/issues/252)) ([833eb61](https://github.com/petercorke/machinevision-toolbox-python/commit/833eb613cc382bbead195dfea3ccdfa0cc271854))
* raise when the pose estimate of ESTPOSE_P fails ([#264](https://github.com/petercorke/machinevision-toolbox-python/issues/264)) ([7eb53fb](https://github.com/petercorke/machinevision-toolbox-python/commit/7eb53fbd493c8f9762c8ca5151948e2a96792bc4))
* refuse a wildcard in Image.Read with its own error ([#226](https://github.com/petercorke/machinevision-toolbox-python/issues/226)) ([a90deaa](https://github.com/petercorke/machinevision-toolbox-python/commit/a90deaadcd221d3f2708bab1d0760c35a28a9c43))
* remove the errors and warnings found by running the notebooks ([#205](https://github.com/petercorke/machinevision-toolbox-python/issues/205)) ([af799b1](https://github.com/petercorke/machinevision-toolbox-python/commit/af799b1ae827ad50c21248981221614b77169f6b))
* replace DSI_refine drange with dmin and ignore NaN planes ([#247](https://github.com/petercorke/machinevision-toolbox-python/issues/247)) ([56d26b4](https://github.com/petercorke/machinevision-toolbox-python/commit/56d26b4151557c2de15c4b1d2e3a9c57d8b3bbd8))
* report valid color lookup values clearly ([#138](https://github.com/petercorke/machinevision-toolbox-python/issues/138)) ([637a787](https://github.com/petercorke/machinevision-toolbox-python/commit/637a787d2dc459ac6f3186998dd625e7aad5f677))
* report video file initialisation failures ([#153](https://github.com/petercorke/machinevision-toolbox-python/issues/153)) ([0138f72](https://github.com/petercorke/machinevision-toolbox-python/commit/0138f720a1f41d51699ae02e9aa06ed8a15129b6))
* return None from the descriptor property when there are none ([#230](https://github.com/petercorke/machinevision-toolbox-python/issues/230)) ([7950cce](https://github.com/petercorke/machinevision-toolbox-python/commit/7950cce3dc4e57110b599470698e1649dbcbff6c))
* ROSBag detects the ROS release by default (release="auto") ([#164](https://github.com/petercorke/machinevision-toolbox-python/issues/164)) ([60ef838](https://github.com/petercorke/machinevision-toolbox-python/commit/60ef838a9341cd31a72903ffdca762fcc427e648))
* ROSBag does not try the ROS 1 wire format on a ROS 2 bag when it detects the release ([#316](https://github.com/petercorke/machinevision-toolbox-python/issues/316)) ([c3ab74e](https://github.com/petercorke/machinevision-toolbox-python/commit/c3ab74ea8cf4fbdc893a52517991fca04864f122))
* ROSBag finds the end of image data from the message layout, not a guess ([#163](https://github.com/petercorke/machinevision-toolbox-python/issues/163)) ([20c6efe](https://github.com/petercorke/machinevision-toolbox-python/commit/20c6efe383a929c3651fa8524862b5027a4c3c7a))
* ROSBag keeps the BGR color order of a CompressedImage unless told otherwise ([#315](https://github.com/petercorke/machinevision-toolbox-python/issues/315)) ([e862924](https://github.com/petercorke/machinevision-toolbox-python/commit/e8629243c831df3f1f470356e76356aa897ffe09))
* ROSBag skips a truncated message with a warning, instead of stopping with a struct.error ([#319](https://github.com/petercorke/machinevision-toolbox-python/issues/319)) ([5de13cd](https://github.com/petercorke/machinevision-toolbox-python/commit/5de13cd621a652c60eaaa0f5c671b450d201a024))
* say what is wrong with the marker string of draw_point ([#268](https://github.com/petercorke/machinevision-toolbox-python/issues/268)) ([86a6592](https://github.com/petercorke/machinevision-toolbox-python/commit/86a6592a8b675e7cc31ba79df42306e936de19b8))
* say why an OpenCV contrib module is missing ([#182](https://github.com/petercorke/machinevision-toolbox-python/issues/182)) ([e7c2ae5](https://github.com/petercorke/machinevision-toolbox-python/commit/e7c2ae57a0c9aee3b8ddf76077e164dc6653a5f5))
* **sources:** read EarthView key from GOOGLE_MAPS_API_KEY first ([d48a8df](https://github.com/petercorke/machinevision-toolbox-python/commit/d48a8df64eab61281ea3a585914b50342112501d))
* stop sad, ssd and ncc wrapping around on uint8 images ([#236](https://github.com/petercorke/machinevision-toolbox-python/issues/236)) ([df7ab0a](https://github.com/petercorke/machinevision-toolbox-python/commit/df7ab0ae622409ca0cd12c56fc966bb958b73c73))
* stop the Hough accumulator looping forever with OpenCV 5.1 ([#224](https://github.com/petercorke/machinevision-toolbox-python/issues/224)) ([8e38010](https://github.com/petercorke/machinevision-toolbox-python/commit/8e380108f1e6328549d46ffb74e6900902777499))
* the blocks module docstring writes the arrow, not an undefined |rarr| ([#301](https://github.com/petercorke/machinevision-toolbox-python/issues/301)) ([6c721c6](https://github.com/petercorke/machinevision-toolbox-python/commit/6c721c62ff597dc29dd3c84be84c54620deeab06))
* the docs build installs libegl1 and libgl1 so Open3D imports ([#313](https://github.com/petercorke/machinevision-toolbox-python/issues/313)) ([3437b52](https://github.com/petercorke/machinevision-toolbox-python/commit/3437b521a21b1a60613677154a8a22b8776cd86f))
* the notebook display of a point cloud or voxel grid uses the same scale on all axes ([#284](https://github.com/petercorke/machinevision-toolbox-python/issues/284)) ([d4256c9](https://github.com/petercorke/machinevision-toolbox-python/commit/d4256c9a9e980e510447934b9f6ce07136b4f0a1))
* the notebooks index links to homography.ipynb, not homographies.ipynb ([#291](https://github.com/petercorke/machinevision-toolbox-python/issues/291)) ([748c963](https://github.com/petercorke/machinevision-toolbox-python/commit/748c9636ab1b1bf1dcaec7f0327220604414bf59))
* treat any non-zero pixel as foreground in distance_transform ([#242](https://github.com/petercorke/machinevision-toolbox-python/issues/242)) ([5c51cf2](https://github.com/petercorke/machinevision-toolbox-python/commit/5c51cf205fe9cd7a3d628ac3e3a42dc1c1cead32))
* undefined names in IBVS_sph.Example and the camera block ([#198](https://github.com/petercorke/machinevision-toolbox-python/issues/198)) ([ab53a94](https://github.com/petercorke/machinevision-toolbox-python/commit/ab53a94065c91a09d9d2f26b009dd072b0122897))
* unpin the Pyodide kernel and warn browsers that lack JSPI ([#181](https://github.com/petercorke/machinevision-toolbox-python/issues/181)) ([1e935a1](https://github.com/petercorke/machinevision-toolbox-python/commit/1e935a1e99292cd4e0ae2c0d8d897e428945f1f9)), closes [#170](https://github.com/petercorke/machinevision-toolbox-python/issues/170)
* update ImagePlane to the bdsim 1.5 graphics block interface ([#251](https://github.com/petercorke/machinevision-toolbox-python/issues/251)) ([76e6d2a](https://github.com/petercorke/machinevision-toolbox-python/commit/76e6d2a708a3133f3dda8707480cc45909ac7f26))
* use Colormap.with_extremes instead of set_bad, set_under and set_over ([#263](https://github.com/petercorke/machinevision-toolbox-python/issues/263)) ([8e4c7b9](https://github.com/petercorke/machinevision-toolbox-python/commit/8e4c7b9e4e80116bc0689322ceccc130b4b87af6))
* use the primaries argument of shadow_invariant ([#232](https://github.com/petercorke/machinevision-toolbox-python/issues/232)) ([c7277b9](https://github.com/petercorke/machinevision-toolbox-python/commit/c7277b99f15065dc013dfebeb510f65a01a4613d))


### Documentation

* add the point clouds notebook ([#326](https://github.com/petercorke/machinevision-toolbox-python/issues/326)) ([11bd36f](https://github.com/petercorke/machinevision-toolbox-python/commit/11bd36ff9da2e0e689125f9663ba32d090f8a1ac))
* bare &gt;&gt;&gt; examples get an Examples heading, code-block and plot ([#306](https://github.com/petercorke/machinevision-toolbox-python/issues/306)) ([7516838](https://github.com/petercorke/machinevision-toolbox-python/commit/751683854e445d139dda17229d01b87eaf7d7933))
* bring AGENTS.md in line with RTB's ([#179](https://github.com/petercorke/machinevision-toolbox-python/issues/179)) ([5e2e6d1](https://github.com/petercorke/machinevision-toolbox-python/commit/5e2e6d1bd44fd22cb589b0bac58174ccb590e344))
* describe the PointCloud wrapper of Open3D on the point cloud page ([#298](https://github.com/petercorke/machinevision-toolbox-python/issues/298)) ([c53ceb2](https://github.com/petercorke/machinevision-toolbox-python/commit/c53ceb25d30fedb228571d6d27127502af56be2e))
* explain the JupyterLite sidebar and that its files are volatile ([#184](https://github.com/petercorke/machinevision-toolbox-python/issues/184)) ([f73b3cb](https://github.com/petercorke/machinevision-toolbox-python/commit/f73b3cbfe84e874c97898b70eaffb5d88c055d88)), closes [#168](https://github.com/petercorke/machinevision-toolbox-python/issues/168) [#173](https://github.com/petercorke/machinevision-toolbox-python/issues/173)
* fail the docs build when a runblock example fails ([#175](https://github.com/petercorke/machinevision-toolbox-python/issues/175)) ([3f6897f](https://github.com/petercorke/machinevision-toolbox-python/commit/3f6897f453832cc887fd38e74f781e916e05b216)), closes [#159](https://github.com/petercorke/machinevision-toolbox-python/issues/159)
* fix broken runblock examples; showwindow defaults to current axes ([#161](https://github.com/petercorke/machinevision-toolbox-python/issues/161)) ([d770ddd](https://github.com/petercorke/machinevision-toolbox-python/commit/d770dddb02f885f0e0f7724618c42cc8d706ad9f))
* fix Sphinx warnings and resolve cross-references ([#177](https://github.com/petercorke/machinevision-toolbox-python/issues/177)) ([8f265d9](https://github.com/petercorke/machinevision-toolbox-python/commit/8f265d950d7d86b1d73dd2a2d3273331cf242bd6))
* fix the broken bdsim logo on the blocks page ([#183](https://github.com/petercorke/machinevision-toolbox-python/issues/183)) ([7e7b736](https://github.com/petercorke/machinevision-toolbox-python/commit/7e7b7365739631ac171dfcf6a9799fb260c36443)), closes [#169](https://github.com/petercorke/machinevision-toolbox-python/issues/169)
* Image() says that integer arrays are narrowed to the smallest type that holds them ([#318](https://github.com/petercorke/machinevision-toolbox-python/issues/318)) ([51c7ee7](https://github.com/petercorke/machinevision-toolbox-python/commit/51c7ee7dcd0643175d9a48dc6a15dd3cd76c9339))
* Image() says that integer arrays are narrowed to the smallest type that holds them, and how to keep the dtype ([51c7ee7](https://github.com/petercorke/machinevision-toolbox-python/commit/51c7ee7dcd0643175d9a48dc6a15dd3cd76c9339))
* JupyterLite troubleshooting section, linked from the README ([#134](https://github.com/petercorke/machinevision-toolbox-python/issues/134)) ([97c3eb3](https://github.com/petercorke/machinevision-toolbox-python/commit/97c3eb348f41f745b2dd60a90b041a32a8df90f3))
* remove black from CONTRIBUTING, it is no longer configured ([#178](https://github.com/petercorke/machinevision-toolbox-python/issues/178)) ([5c57843](https://github.com/petercorke/machinevision-toolbox-python/commit/5c57843001b17d82b63dc3d7863f96ba4af6920d))
* say what Image keys are valid, and when indexing gives a view or a non-contiguous result ([#308](https://github.com/petercorke/machinevision-toolbox-python/issues/308)) ([17b2b80](https://github.com/petercorke/machinevision-toolbox-python/commit/17b2b8086be0abdd17fc73382bf4dd8e01d25e90))
* the name2color example uses a color name that exists ([#299](https://github.com/petercorke/machinevision-toolbox-python/issues/299)) ([b0b2059](https://github.com/petercorke/machinevision-toolbox-python/commit/b0b2059111e5db13b4a12c5aae620eb5d31e57f0))
* the VideoFile examples use traffic_sequence.mp4, the file that exists ([#305](https://github.com/petercorke/machinevision-toolbox-python/issues/305)) ([7b3aed7](https://github.com/petercorke/machinevision-toolbox-python/commit/7b3aed77ddc5e5d8be624572aca0dc80eef191ca))
* tidy the notebooks index logo and copyright line ([#189](https://github.com/petercorke/machinevision-toolbox-python/issues/189)) ([fd5e3a4](https://github.com/petercorke/machinevision-toolbox-python/commit/fd5e3a449ce83bb7af83d52005d48893a8a2ef97))
* uniform notebook header and Bootstrap alert boxes ([#192](https://github.com/petercorke/machinevision-toolbox-python/issues/192)) ([d0ac8da](https://github.com/petercorke/machinevision-toolbox-python/commit/d0ac8da40b80131796f9f140a107b143deb2b055))
* use .. rubric:: Examples in place of Example: headings ([#295](https://github.com/petercorke/machinevision-toolbox-python/issues/295)) ([d02e955](https://github.com/petercorke/machinevision-toolbox-python/commit/d02e955235a3b7e1d8f3261029fd8e33d14442a7))
* use .. rubric:: References in place of the :references: field ([#296](https://github.com/petercorke/machinevision-toolbox-python/issues/296)) ([20e21ec](https://github.com/petercorke/machinevision-toolbox-python/commit/20e21ec6b1f93dc537dc6d25753e0429c9505779))
* use the .. seealso:: directive in place of the :seealso: field ([#293](https://github.com/petercorke/machinevision-toolbox-python/issues/293)) ([8346c35](https://github.com/petercorke/machinevision-toolbox-python/commit/8346c35f970db30abba47ec65530cb160ae8d404))

## [2.4.1](https://github.com/petercorke/machinevision-toolbox-python/compare/v2.4.0...v2.4.1) (2026-10-13)


### Bug Fixes

* BundleAdjust fixed landmarks and pgraph 1.0 Vertex ([1467091](https://github.com/petercorke/machinevision-toolbox-python/commit/14670915f109ea1b57ea3f3adc2d2f3805e53296))
* BundleAdjust fixed landmarks and pgraph 1.0 Vertex ([195e895](https://github.com/petercorke/machinevision-toolbox-python/commit/195e895aa8240c722472fbd31a3be4b91ca9524d))
* **docs:** install spatialmath and pgraph with deps=False in JupyterLite ([201f234](https://github.com/petercorke/machinevision-toolbox-python/commit/201f23487f9f7628d0df2b0cef386175740972a2))
* **docs:** install spatialmath and pgraph with deps=False in JupyterLite ([5615b3b](https://github.com/petercorke/machinevision-toolbox-python/commit/5615b3b7d147e44ae1f2d032deec13bfc69f548a))
* don't pass a positional argument to pgraph.UGraph ([f5590e0](https://github.com/petercorke/machinevision-toolbox-python/commit/f5590e03b6d35d2bcbf62519c1de82d231790169))
* don't pass a positional argument to pgraph.UGraph ([6fb62e5](https://github.com/petercorke/machinevision-toolbox-python/commit/6fb62e50a8d79dd748fab62cbd3054e98af57375))
* estimatePose misaligned ids and corners when other markers are in view ([43c5980](https://github.com/petercorke/machinevision-toolbox-python/commit/43c5980c94b42c861b1f6b441df3b39733412414))
* FiducialCollection.estimatePose() misaligned ids and corners ([#52](https://github.com/petercorke/machinevision-toolbox-python/issues/52)) ([c054867](https://github.com/petercorke/machinevision-toolbox-python/commit/c0548678c81b2ba46b279946e00a5276a2f9e2ae))
* remove labeltool console script that points at a module that was never committed ([#117](https://github.com/petercorke/machinevision-toolbox-python/issues/117)) ([ef39fa9](https://github.com/petercorke/machinevision-toolbox-python/commit/ef39fa96b89ac7f9723442230baf27e1a5b6140e))
* remove labeltool console script whose module was never committed ([30b5c6a](https://github.com/petercorke/machinevision-toolbox-python/commit/30b5c6a9ad19b1b73ad969897abee61c0ad556a5))


### Documentation

* add a script and README for the block icons ([1185da9](https://github.com/petercorke/machinevision-toolbox-python/commit/1185da946260c675561405c4c6059eeace2d1d6d))
* add a script and README for the block icons ([2206b53](https://github.com/petercorke/machinevision-toolbox-python/commit/2206b532e03af6ea8a6cf85c0a0855fcb95b2a52))
* add Codacy grade badge to README ([f91ed2c](https://github.com/petercorke/machinevision-toolbox-python/commit/f91ed2c8a74daf8baf01632548d3d3c23fc06293))
* add CONTRIBUTING.md ([1885fc1](https://github.com/petercorke/machinevision-toolbox-python/commit/1885fc153bf885fa317473cb5b1f7d887cbe0f97))
* note how to use a phone as a camera in WebCam and VideoCamera ([76cea5e](https://github.com/petercorke/machinevision-toolbox-python/commit/76cea5e6f035b558330c0aab1bf7a036100dbf92))
* note how to use a phone as a camera in WebCam and VideoCamera ([72ca04b](https://github.com/petercorke/machinevision-toolbox-python/commit/72ca04b30f0f87af1118659d74e59b32978902d0))

## [2.4.0](https://github.com/petercorke/machinevision-toolbox-python/compare/v2.3.0...v2.4.0) (2026-08-25)


### Features

* add __array__ protocol to Image and Kernel, ndim to Kernel ([e762f57](https://github.com/petercorke/machinevision-toolbox-python/commit/e762f578faa269655cbd06b02d48c4c30d48c516))
* relax opencv pin to support both 4.x and 5.x, add CI matrix axis ([2ec69fe](https://github.com/petercorke/machinevision-toolbox-python/commit/2ec69fe4d116bb51f717e0a4a3c8baed95a7101f)), closes [#44](https://github.com/petercorke/machinevision-toolbox-python/issues/44)


### Bug Fixes

* assert hygiene -- unreachable assert, two caller-facing checks ([7afe33e](https://github.com/petercorke/machinevision-toolbox-python/commit/7afe33eb02af65d4e64d52481f668a610f63b807))
* CameraBase.plot() unbound call (self=None) crashed instead of working ([e2d4600](https://github.com/petercorke/machinevision-toolbox-python/commit/e2d460029d6ef1d5bb807168f37b37b3eb10ff1c))
* catch cv2.error in points2F(), not just F/mask=None ([c4197cb](https://github.com/petercorke/machinevision-toolbox-python/commit/c4197cbd22784b8c002df8a02bb95764dc589b50))
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
