.. currentmodule:: machinevisiontoolbox.PointCloud.PointCloud
    

.. _pointcloud_class_label:

The ``PointCloud`` object
=========================

The :class:`~machinevisiontoolbox.PointCloud.PointCloud` class is essential for all point cloud
operations and processing within this Toolbox. The class 
encapsulates a NumPy array that contains the point coordinates and optionally colors
as a 3D array respectively. An :class:`~machinevisiontoolbox.PointCloud.PointCloud`
instance has many methods that perform useful operations on a point cloud
and wrap low-level operations performed using NumPy or Open3d.

.. autoclass:: machinevisiontoolbox.PointCloud.PointCloud
   :special-members: __init__


Informational
-------------

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~__str__
   ~__repr__
   ~__len__

Access to point cloud data
--------------------------

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~pcd
   ~points
   ~colors

Point cloud i/o
---------------

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~Read
   ~write
   ~disp

Point cloud operations
----------------------

.. autosummary::
   :toctree: stubs
   :nosignatures:

    ~copy
    ~transform
    ~downsample_voxel
    ~downsample_random
    ~voxel_grid
    ~normals
    ~remove_outlier
    ~segment_plane
    ~select
    ~paint
    ~ICP
    ~__getattr__

Overloaded operators
--------------------

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~__rmul__
   ~__imul__
   ~__add__