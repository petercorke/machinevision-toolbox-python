Region features
===============

These methods extract features such as homogenous regions, text and fiducials
from the image.

.. autoclass:: machinevisiontoolbox.ImageRegionFeatures.ImageRegionFeaturesMixin
   :members:

Region feature classes
----------------------

.. autoclass:: machinevisiontoolbox.ImageRegionFeatures.MSERFeature
   :members:
   :special-members: __len__, __str__, __getitem__

.. autoclass:: machinevisiontoolbox.ImageRegionFeatures.OCRWord
   :members:
   :special-members: __len__, __str__,__getitem__

.. autoclass:: machinevisiontoolbox.ImageRegionFeatures.Fiducial
   :members:
   :special-members: __len__, __str__,__getitem__
