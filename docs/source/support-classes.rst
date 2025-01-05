.. currentmodule:: machinevisiontoolbox

Supporting classes
==================

Convolution kernel class
------------------------

.. autoclass:: ImageSpatial.Kernel
    :members:
    :special-members:
    
Image feature classes
---------------------

Whole image features
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ImageWholeFeatures.Histogram
    :members:
    :special-members:


Fiducial features
^^^^^^^^^^^^^^^^^

.. autoclass:: ImageRegionFeatures.Fiducial
    :members:

Line features
^^^^^^^^^^^^^

.. autoclass:: ImageLineFeatures.HoughFeature
    :members:

Point features
^^^^^^^^^^^^^^

.. inheritance-diagram:: ImagePointFeatures.SIFTFeature
      ImagePointFeatures.ORBFeature
      ImagePointFeatures.BRISKFeature
      ImagePointFeatures.AKAZEFeature
      ImagePointFeatures.HarrisFeature
      ImagePointFeatures.FREAKFeature
      ImagePointFeatures.BOOSTFeature
      ImagePointFeatures.BRIEFFeature
      ImagePointFeatures.LATCHFeature
      ImagePointFeatures.LUCIDFeature
    :top-classes: ImagePointFeatures.BaseFeature2D
    :parts: 1

.. autosummary::
    :toctree: stubs
    :nosignatures:

    ~ImagePointFeatures.SIFTFeature
    ~ImagePointFeatures.ORBFeature
    ~ImagePointFeatures.BRISKFeature
    ~ImagePointFeatures.AKAZEFeature
    ~ImagePointFeatures.FREAKFeature
    ~ImagePointFeatures.BOOSTFeature
    ~ImagePointFeatures.BRIEFFeature
    ~ImagePointFeatures.DAISYFeature
    ~ImagePointFeatures.LATCHFeature
    ~ImagePointFeatures.LUCIDFeature
    ~ImagePointFeatures.HarrisFeature

.. autoclass:: ImagePointFeatures.BaseFeature2D
    :members:

