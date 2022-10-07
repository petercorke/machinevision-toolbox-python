.. currentmodule:: machinevisiontoolbox.PointCloud.PointCloud
    
Point clouds
============   


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