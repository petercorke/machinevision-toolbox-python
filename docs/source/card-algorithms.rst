Machine vision algorithms
=========================

This section describes provides a non-exhaustive list of entry points for common machine vision algorithms supported
by the toolbox.  Consider these as some starting points for exploration.


.. dropdown:: Linear 2D filtering
   :animate: fade-in-slide-down
   :icon: filter

   * Convolutional filtering: :meth:`~machinevisiontoolbox.Image.convolve`, :meth:`~machinevisiontoolbox.Image.smooth`
   * Filter kernels: :meth:`~machinevisiontoolbox.ImageSpatial.Kernel.Gauss`, :meth:`~machinevisiontoolbox.ImageSpatial.Kernel.DGauss`, :meth:`~machinevisiontoolbox.ImageSpatial.Kernel.DoG`, :meth:`~machinevisiontoolbox.ImageSpatial.Kernel.Laplace`, :meth:`~machinevisiontoolbox.ImageSpatial.Kernel.LoG`, :meth:`~machinevisiontoolbox.ImageSpatial.Kernel.Sobel`
   * Edge detection: :meth:`~machinevisiontoolbox.Image.canny`
   * Scale-space: :meth:`~machinevisiontoolbox.Image.scalespace`, :meth:`~machinevisiontoolbox.Image.pyramid`
   * Gradient estimation: :meth:`~machinevisiontoolbox.Image.gradients`, :meth:`~machinevisiontoolbox.Image.direction`

.. dropdown:: Nonlinear 2D filtering
   :animate: fade-in-slide-down
   :icon: sliders

   * Mathematical morphology: :meth:`~machinevisiontoolbox.Image.morph`, :meth:`~machinevisiontoolbox.Image.open`, :meth:`~machinevisiontoolbox.Image.close`, :meth:`~machinevisiontoolbox.Image.dilate`, :meth:`~machinevisiontoolbox.Image.erode`
   * Hit or Miss filtering: :meth:`~machinevisiontoolbox.Image.hitormiss`, :meth:`~machinevisiontoolbox.Image.thin`, :meth:`~machinevisiontoolbox.Image.triplepoint`
   * Rank filter: :meth:`~machinevisiontoolbox.Image.rank`, :meth:`~machinevisiontoolbox.Image.medianfilter`
   * Distance transform: :meth:`~machinevisiontoolbox.Image.distance_transform`
   * Image similarity: :meth:`~machinevisiontoolbox.Image.similarity`

.. dropdown:: Feature extraction
   :animate: fade-in-slide-down
   :icon: telescope

   * Whole image features
       - Simple statistics: :meth:`~machinevisiontoolbox.Image.stats`, :meth:`~machinevisiontoolbox.Image.hist`, :meth:`~machinevisiontoolbox.Image.mean`, :meth:`~machinevisiontoolbox.Image.median`, :meth:`~machinevisiontoolbox.Image.std`, :meth:`~machinevisiontoolbox.Image.var`, etc.
       - Histogram analysis: :meth:`~machinevisiontoolbox.Image.hist`, :meth:`~machinevisiontoolbox.Histogram.h`, :meth:`~machinevisiontoolbox.Histogram.cf`, :meth:`~machinevisiontoolbox.Histogram.cdf`, :meth:`~machinevisiontoolbox.Histogram.pdf`, :meth:`~machinevisiontoolbox.Histogram.peaks`
   
   * Region features
       - Segmentation:
          - Thresholding: :meth:`~machinevisiontoolbox.Image.otsu`, :meth:`~machinevisiontoolbox.Image.triangle`, :meth:`~machinevisiontoolbox.Image.threshold`, :meth:`~machinevisiontoolbox.Image.threshold_adaptive`, :meth:`~machinevisiontoolbox.Image.threshold_interactive`
          - MSER features: :meth:`~machinevisiontoolbox.Image.MSER`
          - Color k-means: :meth:`~machinevisiontoolbox.Image.kmeans_color`
       - Connected component (blob) analysis: :meth:`~machinevisiontoolbox.Image.blobs`

   * Line features
       - Hough lines: :meth:`~machinevisiontoolbox.Image.Hough`, :class:`~machinevisiontoolbox.ImageLineFeatures.HoughFeature`
   

   * Point features:
       - Harris corners: :meth:`~machinevisiontoolbox.Image.Harris`
       - Scale-orientation invariant features: :meth:`~machinevisiontoolbox.Image.SIFT`, :meth:`~machinevisiontoolbox.Image.SURF`, :meth:`~machinevisiontoolbox.Image.BRISK`, :meth:`~machinevisiontoolbox.Image.ORB`, etc.
       - Feature matching: :class:`~machinevisiontoolbox.ImagePointFeatures.FeatureMatch`


   * Text features (OCR): :meth:`~machinevisiontoolbox.Image.ocr`
   * Fiducial features (AR tags, AprilTags, etc.): :meth:`~machinevisiontoolbox.Image.fiducial`, :class:`~machinevisiontoolbox.Fiducial`

.. dropdown:: Image retrieval
   :animate: fade-in-slide-down
   :icon: search

   * Bag of words matching: :class:`~machinevisiontoolbox.BagOfWords`


.. dropdown:: Camera models
   :animate: fade-in-slide-down
   :icon: device-camera

   
   - Central-projection (aka pinhole) camera: :class:`~machinevisiontoolbox.Camera.CentralCamera`
      - Camera calibration: :meth:`~machinevisiontoolbox.Camera.CentralCamera.images2C`, :meth:`~machinevisiontoolbox.Camera.CentralCamera.decomposeC`
      - Pose estimation: :meth:`machinevisiontoolbox.Camera.CentralCamera.estpose`
      - Projection: :meth:`~machinevisiontoolbox.Camera.CentralCamera.project_point`, :meth:`~machinevisiontoolbox.Camera.CentralCamera.project_line`, :meth:`~machinevisiontoolbox.Camera.CentralCamera.project_conic`, :meth:`~machinevisiontoolbox.Camera.CentralCamera.project_quadric`
      - Epipolar geometry: :meth:`~machinevisiontoolbox.Camera.CentralCamera.E`, :meth:`~machinevisiontoolbox.Camera.CentralCamera.F`, :meth:`~machinevisiontoolbox.Camera.CentralCamera.epiline`, CentralCamera.decomposeF, :meth:`~machinevisiontoolbox.Camera.CentralCamera.decomposeE`
      - Homography: :meth:`~machinevisiontoolbox.Camera.CentralCamera.H`
   - Fisheye camera: :class:`~machinevisiontoolbox.Camera.FishEyeCamera`
   - Catadioptric (omnidirectional) camera: :class:`~machinevisiontoolbox.Camera.CatadioptricCamera`
   - Spherical camera: :class:`~machinevisiontoolbox.Camera.SphericalCamera`


.. dropdown:: Multiview geometry
   :animate: fade-in-slide-down
   :icon: project


   * Stereo vision: :meth:`~machinevisiontoolbox.Image.stereo_simple`, :meth:`~machinevisiontoolbox.Image.stereo_BM`, :meth:`~machinevisiontoolbox.Image.stereo_SGBM`
   * Rectification: :meth:`~machinevisiontoolbox.Image.rectify_homographies`
   * Bundle adjustment: :class:`~machinevisiontoolbox.BundleAdjust`


.. dropdown:: Point cloud processing
   :animate: fade-in-slide-down
   :icon: graph

   * Downsampling: :meth:`~machinevisiontoolbox.PointCloud.PointCloud.downsample_voxel`, :meth:`~machinevisiontoolbox.PointCloud.PointCloud.downsample_random`
   * Transform: :meth:`~machinevisiontoolbox.PointCloud.PointCloud.transform`
   * ICP (Iterative Closest Point): :meth:`~machinevisiontoolbox.PointCloud.PointCloud.ICP`


.. dropdown:: Visual servoing
   :animate: fade-in-slide-down
   :icon: sync

   * Position-based: PBVS
   * Image-based: IBVS

