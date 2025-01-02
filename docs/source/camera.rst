.. currentmodule:: machinevisiontoolbox.Camera

Camera models and calibration
=============================

Camera models
-------------

A set of classes that model the projective geometry of cameras.

.. inheritance-diagram:: CentralCamera FishEyeCamera CatadioptricCamera SphericalCamera
    :top-classes: Camera
    :parts: 1

.. autosummary::
    :toctree: stubs/camera
    :nosignatures:
    :template: mvtbtemplate.rst

    ~CentralCamera 
    ~FishEyeCamera 
    ~CatadioptricCamera 
    ~SphericalCamera


Calibration
-----------

Intrinsic calibration of camera from multiple images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: CentralCamera
    :members: images2C

Extrinsic calibration of camera from marker ArUcoBoard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: machinevisiontoolbox.ImageRegionFeatures.ArUcoBoard
    :members:

