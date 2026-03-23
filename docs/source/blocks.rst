.. image:: https://raw.githubusercontent.com/petercorke/bdsim/master/figs/BDSimLogo_NoBackgnd@2x.png
    :width: 400

.. |bdsim| replace:: ``bdsim``
.. _bdsim: https://github.com/petercorke/bdsim
    
bdsim blocks
============

|bdsim|_ is a block diagram simulation environment for Python. It is used for many
examples in the book |RVC3| for both control and computer vision.  

``bdsim`` is:

* similar in principle to Simulink, but it is open source,
  exploits the full power of Python and its ecosystem, and is compatible with standard
  software development tools and practices.  

* designed to be extensible, and this section describes a set
  of block definitions that add computer vision capability to the block diagram simulation
  environment. 
  
The following blocks add computer vision capability to the block diagram
simulation environment by wrapping the functionality of the `Machine Vision Toolbox for Python <https://github.com/petercorke/machinevision-toolbox-python>`_.
The block documentation follows the ``bdsim`` practice and conventions.

.. toctree::
   :maxdepth: 2

   blocks-camera