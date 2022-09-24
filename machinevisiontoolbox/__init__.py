# classes
from machinevisiontoolbox.ImageCore import Image
from machinevisiontoolbox.Sources import VideoFile, VideoCamera, \
    ImageCollection, ZipArchive, EarthView, WebCam
from machinevisiontoolbox.ImageIO import *
from machinevisiontoolbox import base
from machinevisiontoolbox.ImageSpatial import Kernel
from machinevisiontoolbox.ImageBlobs import Blobs
from machinevisiontoolbox.ImageWholeFeatures import *
from machinevisiontoolbox.ImageRegionFeatures import *
from machinevisiontoolbox.ImagePointFeatures import FeatureMatch, BaseFeature2D
from machinevisiontoolbox.Camera import CameraBase, CentralCamera, \
    FishEyeCamera, CatadioptricCamera, SphericalCamera
from machinevisiontoolbox.PointCloud import PointCloud
from machinevisiontoolbox.BagOfWords import BagOfWords
from machinevisiontoolbox.BundleAdjust import BundleAdjust
from machinevisiontoolbox.VisualServo import *

from machinevisiontoolbox.base import *

from machinevisiontoolbox.morphdemo import morphdemo


# next one pollutes name space with SMTB base
# from machinevisiontoolbox.image_feature import *
