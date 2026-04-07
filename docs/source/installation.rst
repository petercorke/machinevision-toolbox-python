Installing the Toolbox
======================

To install the latest stable release from PyPI, simply run:

.. code-block:: bash

		$ pip install machinevision-toolbox-python

This will install the Toolbox as well as images and data files used in the documentation and examples. 

You can also install the latest development version from GitHub using:

.. code-block:: bash

		$ pip install git+https://github.com/machinevision-toolbox/machinevision-toolbox-python.git

.. warning:: The GitHub version may be unstable and is not recommended for production use. It may contain bugs or incomplete features. Use it at your own risk.


Support files
-------------

The images and data files used for documentation and examples are available in a
separate package [mvtb-data package](https://pypi.org/project/mvtb-data). This avoids
bloating the core installation and hitting PyPI storage limits -- it only chages
infrequently.  The package is installed automatically with the main toolbox,
but if you want to install just the images and data files you can run:

.. code-block:: bash

		$ pip install mvtb-data

Installation environment
------------------------

If you are installing into an existing local Python environment then you must have Numpy
2.x. and OpenCV is a critical dependency which does not yet work with Numpy 2.x. Python 3.9
or newer is recommended.

It is highly recommend that you use Miniconda and create an environment for your machine
vision work. This will allow you to manage dependencies and avoid conflicts with other
code. You can use tools such as ``venv`` or ``conda`` to create and manage your environment. For example, with conda you could run:

.. code-block:: bash
        :linenos:

        $ conda create -n MVTB python=3.12
        $ conda activate MVTB
        $ pip install machinevision-toolbox-python

Line 2 activates the environment you just created, and typically this will
modify your shell prompt to indicate the active environment. Line 3 installs
the toolbox and its dependencies into that environment. You can then use this
environment for your machine vision projects.

.. important::

    NumPy 2.x was not backwards compatible with NumPy 1.x. OpenCV did not support NumPy 2.x until 4.10.0.84.
    Open3D did not support NumPy 2.x until 0.18.1, and is typically 6 months behind in its support for new versions of Python.

If you want to install the Toolbox into an existing environment with NumPy 1.x and a compatible version of OpenCV, you can tell ``pip``
to avoid dependency resolution:

.. code-block:: bash

        $ pip install --no-deps machinevision-toolbox-python

You will have to manage the dependencies yourself, and conflicts may arise with the many
other Toolbox package dependencies. You can check the required dependencies in the
``pyproject.toml`` file in the source code.

Installation extras
-------------------

``pip`` also supports installing optional dependencies, known as "extras". These are
additional packages that provide extra functionality but are not required for the core
functionality of the toolbox. You can specify which extras you want to install by
including them in square brackets after the package name.

.. code-block:: bash

		$ pip install machinevision-toolbox-python[extra]

The available extras are:

+--------------+-------------------------------------------+
| Extra        | Purpose                                   |
+==============+===========================================+
| ``dev``      | Development and test tools                |
+--------------+-------------------------------------------+
| ``ros``      | ROS bag and ROS bridge support            |
+--------------+-------------------------------------------+
| ``docs``     | Documentation build toolchain             |
+--------------+-------------------------------------------+
| ``jupyter``  | Jupyter notebook widgets and plotting     |
|              | support                                   |
+--------------+-------------------------------------------+
| ``open3d``   | Open3D support                            |
+--------------+-------------------------------------------+
| ``torch``    | PyTorch support                           |
+--------------+-------------------------------------------+
| ``ocr``      | OCR support via Tesseract bindings        |
+--------------+-------------------------------------------+
| ``labelme``  | LabelMe JSON annotation reader support    |
+--------------+-------------------------------------------+
| ``all``      | All of the above                          |
+--------------+-------------------------------------------+

For example, to install the ROS support extras, you would run:


.. code-block:: bash

        $ pip install machinevision-toolbox-python[ros]

To install ROS and PyTorch support, you would run:

.. code-block:: bash

        $ pip install machinevision-toolbox-python[ros,torch]

To install all extras, you would run:

.. code-block:: bash

        $ pip install machinevision-toolbox-python[all]

