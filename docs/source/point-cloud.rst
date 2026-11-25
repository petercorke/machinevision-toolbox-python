.. currentmodule:: machinevisiontoolbox.PointCloud.PointCloud
    

.. _pointcloud_class_label:

``PointCloud`` object
=====================

The :class:`~machinevisiontoolbox.PointCloud.PointCloud` class is essential for all
point cloud operations and processing within this Toolbox.  It wraps an
`Open3D <https://www.open3d.org/docs/release/>`_ point cloud, which can hold the
point coordinates and, optionally, their colors and surface normals.

Open3D is an open-source library that supports rapid development of software that
deals with 3D data.  The Open3D frontend exposes a set of carefully selected data
structures and algorithms in both C++ and Python.  The backend is highly optimized and
is set up for parallelization.

The ``PointCloud`` class provides a subset of commonly used methods but does not
attempt to wrap every Open3D method.  Instead, any method or property not known to the
``PointCloud`` class, such as ``pc.some_method()``, falls through to the native Open3D
method ``some_method()`` on the encapsulated Open3D object, and all parameters are
passed through.  If the return value is an Open3D point cloud, or a tuple that contains
one, it is wrapped and returned as a ``PointCloud`` instance.  A name that neither class
knows raises :class:`AttributeError`.

.. note::

   Open3D uses NumPy arrays in which points are rows.  This is different to the
   Toolbox convention of points as columns.  The Toolbox wrappers handle this
   transposition, but anything that falls through to Open3D, or that is reached through
   :attr:`pcd`, uses the Open3D convention.

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
   ~iscolor

Point cloud i/o
---------------

.. autosummary::
   :toctree: stubs
   :nosignatures:

   ~Read
   ~DepthImage
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
    ~estimate_normals
    ~normals
    ~remove_outlier
    ~segment_plane
    ~select
    ~paint
    ~unpaint
    ~concatenate
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
   ~__radd__