# isort: skip_file
# classes
from machinevisiontoolbox.ImageCore import Image
from machinevisiontoolbox.PointCloud import PointCloud
from machinevisiontoolbox.Sources import (
    VideoFile,
    VideoCamera,
    FileCollection,
    ImageCollection,
    ImageSequence,
    FileArchive,
    ZipArchive,
    EarthView,
    WebCam,
    ROSTopic,
    ROSMessage,
    SyncROSStreams,
    ROSBag,
    PointCloudSequence,
    TensorStack,
    LabelMeReader,
)
from machinevisiontoolbox import base

mvb = base

from machinevisiontoolbox.Kernel import Kernel
from machinevisiontoolbox.ImageBlobs import Blobs, Blob
from machinevisiontoolbox.ImageWholeFeatures import (
    ImageWholeFeaturesMixin,
    Histogram,
    Polygon,
)
from machinevisiontoolbox.ImageRegionFeatures import (
    ImageRegionFeaturesMixin,
    MSERFeature,
    OCRWord,
)
from machinevisiontoolbox.ImageFiducials import (
    ImageFiducialsMixin,
    Fiducial,
    FiducialCollection,
    ArUcoBoard,
)
from machinevisiontoolbox.ImagePointFeatures import FeatureMatch, BaseFeature2D
from machinevisiontoolbox.Camera import (
    CameraBase,
    CentralCamera,
    FishEyeCamera,
    CatadioptricCamera,
    SphericalCamera,
)
from machinevisiontoolbox.BagOfWords import BagOfWords
from machinevisiontoolbox.BundleAdjust import BundleAdjust
from machinevisiontoolbox.VisualServo import (
    VisualServo,
    PBVS,
    IBVS,
    IBVS_l,
    IBVS_e,
    IBVS_sph,
    IBVS_polar,
)

from machinevisiontoolbox.base import (
    blackbody,
    loadspectrum,
    lambda2rg,
    cmfrgb,
    tristim2cc,
    lambda2xy,
    cmfxyz,
    luminos,
    rluminos,
    ccxyz,
    name2color,
    color2name,
    colorname,
    cie_primaries,
    colorspace_convert,
    gamma_encode,
    gamma_decode,
    XYZ2RGBxform,
    plot_chromaticity_diagram,
    plot_spectral_locus,
    shadow_invariant,
    esttheta,
    plot_labelbox,
    draw_box,
    draw_labelbox,
    draw_point,
    draw_text,
    draw_line,
    draw_circle,
    int_image,
    float_image,
    mvtb_path_to_datafile,
    mvtb_load_data,
    mvtb_load_matfile,
    mvtb_load_jsonfile,
    mkcube,
    mksphere,
    mkcylinder,
    mkgrid,
    meshgrid,
    spherical_rotate,
    findpeaks,
    findpeaks2d,
    findpeaks3d,
    mvtb_version,
    mpl_styling,
)

# Import image I/O symbols from their defining module to avoid static-analysis
# false positives seen when resolving these names via machinevisiontoolbox.base.
from machinevisiontoolbox.base.imageio import (
    idisp,
    iread,
    iread_iter,
    iwrite,
    convert,
    cv_destroy_window,
    set_window_title,
    pickpoints,
)

__all__ = [
    # core
    "Image",
    "PointCloud",
    # image sources
    "VideoFile",
    "VideoCamera",
    "FileCollection",
    "ImageCollection",
    "ImageSequence",
    "FileArchive",
    "ZipArchive",
    "EarthView",
    "WebCam",
    # ROS
    "ROSTopic",
    "ROSMessage",
    "SyncROSStreams",
    "ROSBag",
    "PointCloudSequence",
    # ML / batch
    "TensorStack",
    "LabelMeReader",
    # processing helpers
    "Kernel",
    "Histogram",
    "Polygon",
    # blobs / regions
    "Blobs",
    "Blob",
    "MSERFeature",
    "OCRWord",
    # point features
    "FeatureMatch",
    "BaseFeature2D",
    # fiducials
    "Fiducial",
    "FiducialCollection",
    "ArUcoBoard",
    # cameras
    "CameraBase",
    "CentralCamera",
    "FishEyeCamera",
    "CatadioptricCamera",
    "SphericalCamera",
    # algorithms
    "BagOfWords",
    "BundleAdjust",
    # visual servo
    "VisualServo",
    "PBVS",
    "IBVS",
    "IBVS_l",
    "IBVS_e",
    "IBVS_sph",
    "IBVS_polar",
    # base module (access base functions via mvb.iread, mvb.idisp, etc.)
    "mvb",
]

try:
    import importlib.metadata

    __version__ = importlib.metadata.version("machinevision-toolbox-python")
except:
    pass
