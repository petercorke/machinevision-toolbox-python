.. currentmodule:: machinevisiontoolbox

Supporting classes
==================

    
Image feature classes
---------------------

Whole image features
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Histogram
    :members:
    :special-members:


Fiducial features
^^^^^^^^^^^^^^^^^

.. autoclass:: Fiducial
    :members:
    :special-members:

Blob features
^^^^^^^^^^^^^

.. autoclass:: Blob
    :members:
    :special-members:

.. autoclass:: Blobs
    :members:
    :special-members:

Line features
^^^^^^^^^^^^^

.. autoclass:: HoughFeature
    :members:

Point features
^^^^^^^^^^^^^^

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

.. autoclass:: ImagePointFeatures.BaseFeature2D
    :members:

Convolution kernel class
------------------------

.. autoclass:: Kernel
    :members:
    :special-members:
