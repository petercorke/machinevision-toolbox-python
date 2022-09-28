Corner/point features
=====================

Image feature detection, description and matching is key to many algorithms used
multi-view geometry. The key steps are:

1. :ref:`find features<Extract feature points>`
2. :ref:`sort/filter features<Feature representation>`
3. :ref:`match features<Feature matching>`

Extract feature points
----------------------

Find point (corner) features in a grey-scale image.

.. autoclass:: machinevisiontoolbox.ImagePointFeatures.ImagePointFeaturesMixin
   :members:


Feature matching
----------------

The match object performs and describes point feature correspondence.

.. autoclass:: machinevisiontoolbox.ImagePointFeatures.FeatureMatch
   :members:
   :special-members: __len__, __getitem__


Feature representation
----------------------

.. inheritance-diagram:: machinevisiontoolbox.ImagePointFeatures.SIFTFeature
      machinevisiontoolbox.ImagePointFeatures.ORBFeature
      machinevisiontoolbox.ImagePointFeatures.BRISKFeature
      machinevisiontoolbox.ImagePointFeatures.AKAZEFeature
      machinevisiontoolbox.ImagePointFeatures.HarrisFeature
      machinevisiontoolbox.ImagePointFeatures.FREAKFeature
      machinevisiontoolbox.ImagePointFeatures.BOOSTFeature
      machinevisiontoolbox.ImagePointFeatures.BRIEFFeature
      machinevisiontoolbox.ImagePointFeatures.LATCHFeature
      machinevisiontoolbox.ImagePointFeatures.LUCIDFeature
    :top-classes: machinevisiontoolbox.ImagePointFeatures.BaseFeature2D
    :parts: 1

The base class for all point features.

.. autoclass:: machinevisiontoolbox.ImagePointFeatures.SIFTFeature
   :members:
   :inherited-members:
   :special-members: __len__, __getitem__

.. autoclass:: machinevisiontoolbox.ImagePointFeatures.ORBFeature
   :members:
   :inherited-members:
   :special-members: __len__, __getitem__

.. autoclass:: machinevisiontoolbox.ImagePointFeatures.BRISKFeature
   :members:
   :inherited-members:
   :special-members: __len__, __getitem__

.. autoclass:: machinevisiontoolbox.ImagePointFeatures.AKAZEFeature
   :members:
   :inherited-members:
   :special-members: __len__, __getitem__

.. autoclass:: machinevisiontoolbox.ImagePointFeatures.HarrisFeature
   :members:
   :inherited-members:
   :special-members: __len__, __getitem__

.. autoclass:: machinevisiontoolbox.ImagePointFeatures.FREAKFeature
   :members:
   :inherited-members:
   :special-members: __len__, __getitem__

.. autoclass:: machinevisiontoolbox.ImagePointFeatures.BOOSTFeature
   :members:
   :inherited-members:
   :special-members: __len__, __getitem__