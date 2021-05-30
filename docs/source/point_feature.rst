Corner/point features
=====================

This is a three step process:

1. :ref:`find features<Extract feature points>`
2. :ref:`sort/filter features<Feature representation>`
3. :ref:`match features<Feature matching>`

Extract feature points
----------------------

Find point (corner) features in a grey-scale image.

.. autoclass:: machinevisiontoolbox.ImagePointFeatures.ImagePointFeaturesMixin
   :members:

Feature representation
----------------------

The base class for all point features, eg. SIFT, ORB etc.

.. autoclass:: machinevisiontoolbox.ImagePointFeatures.BaseFeature2D
   :members:
   :special-members: __len__, __getitem__

Feature matching
----------------

The match object performs and describes point feature correspondence.

.. autoclass:: machinevisiontoolbox.ImagePointFeatures.Match
   :members:
   :special-members: __len__, __getitem__

