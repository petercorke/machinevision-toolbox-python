****************************************
``machinevisiontoolbox.base``
****************************************

The ``base`` sub-package provides the foundational primitives that underpin the Machine
Vision Toolbox, offering a streamlined interface for essential vision operations. It
centralises core functionality for camera geometry, spatial transformations, and color
space management, alongside high-performance utilities for image acquisition and
visualization. By abstracting the low-level complexities of data representation and I/O,
base establishes the robust computational bedrock required for advanced vision-based
control and scene analysis.

The functions operate on, and return, NumPy arrays.

.. toctree::
   :maxdepth: 2

   imageio
   types
   color
   graphics
   peak
   data
   shapes
   cm
