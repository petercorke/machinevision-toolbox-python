Jupyter notebooks
===================

The Toolbox folder ``docs/notebooks`` includes a number of Jupyter notebooks that demonstrate the use of the
use of the Toolbox for specific applications.  There are a number of different ways to
run a Jupyter notebook.

Running a notebook
---------------------

Locally from the command line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You need to load some packages first.  You can do this at install time, and no harm in installing again.

.. code-block:: bash

    pip install machinvision-toolbox-python [jupyter]

or just pull the packages you need for the notebooks

.. code-block:: bash

    pip install jupyter ipympl ipywidgets

Then run Jupyter

.. code-block:: bash

    jupyter notebook yourfile.ipynb

which will start a server and open a new browser tab with the Jupyter GUI.  The UI is a bit clunky, but it works.  

JupyterLab is a more modern interface to Jupyter notebooks, and it is available as an
option when you install Jupyter.  It has a more modern interface and better support for
multiple notebooks. You would:

.. code-block:: bash

    pip install jupyterlab
    jupyter lab yourfile.ipynb

.. note:: 
    A powerful feature of Jupyter is that the server can run on a different machine
    than your browser interface.  This allows you to run the notebooks on a more powerful
    computer on your network -- that machine must have the Toolbox and Jupyter
    installed and have the notebooks available in its filesystem.  To do this, you would start a
    Jupyter server on the remote machine ``$ jupyter server``, note the URL it is serving
    on, and connect to it from your local browser.

Locally using Visual Studio Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a very convenient way to work and highly recommended -- the interface is rather
more polished than the standard Jupyter interface.  You need to install
some packages first.  You can do this at install time, and no harm in installing again.

.. code-block:: bash

    pip install machinvision-toolbox-python [jupyter]

or

.. code-block:: bash

    pip install jupyter ipympl ipywidgets

Then install the [Jupyter extension for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter). This will allow you to open
and run the notebooks directly in Visual Studio Code -- just open the file from the file explorer view.
This provides a much nicer interface, supports multiple notebooks, and is great for debugging.

.. note:: 
    You can also use the Visual Studio Code interface to run the notebooks in a Jupyter
    server running on your local machine or on a remote machine.  If you are running the
    notebooks on a remote machine then you will need to set up a Jupyter server on that
    machine ``$ jupyter server``, note the URL it is serving on, and connect to it from 
    Visual Studio Code.  This is a bit more work to set up, but it allows you to run 
    the notebooks on a more powerful machine than your local laptop -- that machine 
    must have the Toolbox and Jupyter installed and the notebooks available on its filesystem.

In the browser using Jupyter lite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Jupyter lite is a version of Jupyter that runs entirely in the browser, without the need
for a server.  The environment supports NumPy, SciPy and OpenCV all compiled for the web.
It is a great option for quickly trying out the notebooks without
installing anything on your local machine.  To use it, just click the "Launch in Jupyter
Lite" button at the top of each notebook page, and it will open the notebook in a new
browser tab running Jupyter lite.


In the cloud using Google Colab
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. |colab_image| image:: https://colab.research.google.com/img/colab_favicon_256px.png
   :height: 30px
   :alt: Open In Colab

.. |jupyterlite_image| image:: https://img.shields.io/badge/Jupyter-Lite-orange?logo=jupyter
   :height: 20px
   :alt: Open In JupyterLite


Just click the |colab_image| buttons below to open the notebooks in Colab.  Colab will
pull the notebook from GitHub, and allow you to interact with it in a browser tab. If
you modify the notebook you have the option of saving it in your own GitHub account, or
on Google Drive.

While this is theoretically a convenient approach with zero install on your computer, it is unfortunate that each notebook is quite slow to startup because the toolboxes need to be installed into the Colab environment, and they are only cached there for a short amount of time.


Provided notebooks by category
--------------------------------


Quick Start
~~~~~~~~~~~

.. csv-table::
   :header-rows: 1
   :widths: 28, 16, 16, 40

   "Notebook", "|jupyterlite_image|", "|colab_image|", "Description"
   "`Introduction <https://github.com/petercorke/machinevision-toolbox-python/blob/main/docs/lite/files/intro.ipynb>`_", "`open <https://petercorke.github.io/machinevision-toolbox-python/lite/lab/index.html?path=intro.ipynb>`_", "`open <https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/lite/files/intro.ipynb>`_", "A quick introduction to the toolbox."

Image Fundamentals
~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header-rows: 1
   :widths: 28, 16, 16, 40

   "Notebook", "|jupyterlite_image|", "|colab_image|", "Description"
   "`Exploring Images <https://github.com/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/exploring-images.ipynb>`_", "`open <https://petercorke.github.io/machinevision-toolbox-python/lite/lab/index.html?path=exploring-images.ipynb>`_", "`open <https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/exploring-images.ipynb>`_", "The basics of images, pixels, and bit-depth."
   "`Grey scale images <https://github.com/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/greyscale-images.ipynb>`_", "`open <https://petercorke.github.io/machinevision-toolbox-python/lite/lab/index.html?path=greyscale-images.ipynb>`_", "`open <https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/greyscale-images.ipynb>`_", "The basics of grey scale images."
   "`Color images <https://github.com/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/color-images.ipynb>`_", "`open <https://petercorke.github.io/machinevision-toolbox-python/lite/lab/index.html?path=color-images.ipynb>`_", "`open <https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/color-images.ipynb>`_", "The basics of color images."
   "`Image Processing <https://github.com/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/image-processing.ipynb>`_", "`open <https://petercorke.github.io/machinevision-toolbox-python/lite/lab/index.html?path=image-processing.ipynb>`_", "`open <https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/image-processing.ipynb>`_", "Fundamentals of filtering, kernels, and spatial operations."
   "`Understanding gamma <https://github.com/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/gamma.ipynb>`_", "`open <https://petercorke.github.io/machinevision-toolbox-python/lite/lab/index.html?path=gamma.ipynb>`_", "`open <https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/gamma.ipynb>`_", "The most misunderstood image transform in computer vision."

Camera Geometry and Projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header-rows: 1
   :widths: 28, 16, 16, 40

   "Notebook", "|jupyterlite_image|", "|colab_image|", "Description"
   "`Camera Animation <https://github.com/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/camera_animation.ipynb>`_", "`open <https://petercorke.github.io/machinevision-toolbox-python/lite/lab/index.html?path=camera_animation.ipynb>`_", "`open <https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/camera_animation.ipynb>`_", "Interactive: visualise point projection on a central sensor."
   "`The Central Camera <https://github.com/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/camera.ipynb>`_", "`open <https://petercorke.github.io/machinevision-toolbox-python/lite/lab/index.html?path=camera.ipynb>`_", "`open <https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/camera.ipynb>`_", "Introduction to the ``CentralCamera`` object and properties."
   "`Homogeneous Coords <https://github.com/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/homogeneous-coords.ipynb>`_", "`open <https://petercorke.github.io/machinevision-toolbox-python/lite/lab/index.html?path=homogeneous-coords.ipynb>`_", "`open <https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/homogeneous-coords.ipynb>`_", "Math refresher on spatial transforms with interactive demo."
   "`Homography <https://github.com/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/homography.ipynb>`_", "`open <https://petercorke.github.io/machinevision-toolbox-python/lite/lab/index.html?path=homography.ipynb>`_", "`open <https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/homography.ipynb>`_", "Planar projections, warping, and estimation."
   "`Camera calibration <https://github.com/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/calibration.ipynb>`_", "`open <https://petercorke.github.io/machinevision-toolbox-python/lite/lab/index.html?path=calibration.ipynb>`_", "`open <https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/calibration.ipynb>`_", "Calibrating a camera from a set of images."

Advanced Vision Features
~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header-rows: 1
   :widths: 28, 16, 16, 40

   "Notebook", "|jupyterlite_image|", "|colab_image|", "Description"
   "`Finding Blobs <https://github.com/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/finding-blobs.ipynb>`_", "`open <https://petercorke.github.io/machinevision-toolbox-python/lite/lab/index.html?path=finding-blobs.ipynb>`_", "`open <https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/finding-blobs.ipynb>`_", "Region segmentation and shape analysis."
   "`Image convolution <https://github.com/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/image-convolution.ipynb>`_", "`open <https://petercorke.github.io/machinevision-toolbox-python/lite/lab/index.html?path=image-convolution.ipynb>`_", "`open <https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/image-convolution.ipynb>`_", "Basis of smoothing, edge detection, and point features."
   "`Image Features <https://github.com/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/image-features.ipynb>`_", "`open <https://petercorke.github.io/machinevision-toolbox-python/lite/lab/index.html?path=image-features.ipynb>`_", "`open <https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/image-features.ipynb>`_", "Point and corner detection (SIFT, ORB, etc.)."
   "`Fiducial Markers <https://github.com/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/fiducials.ipynb>`_", "`open <https://petercorke.github.io/machinevision-toolbox-python/lite/lab/index.html?path=fiducials.ipynb>`_", "`open <https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/fiducials.ipynb>`_", "Detecting ArUco markers and QR-like codes."
   "`Image Motion <https://github.com/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/image-motion.ipynb>`_", "`open <https://petercorke.github.io/machinevision-toolbox-python/lite/lab/index.html?path=image-motion.ipynb>`_", "`open <https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/image-motion.ipynb>`_", "Relationship between camera motion (3D) and image motion (2D)."
   "`Visual Servoing <https://github.com/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/IBVS.ipynb>`_", "`open <https://petercorke.github.io/machinevision-toolbox-python/lite/lab/index.html?path=IBVS.ipynb>`_", "`open <https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/IBVS.ipynb>`_", "Image-Based Visual Servoing (IBVS)."


Obtaining the notebooks
-----------------------

If you want to run the notebooks locally, you will need to obtain them from the GitHub repository.  
You can download all the notebooks `directly from the GitHub web interface as a zip file <https://petercorke.github.io/machinevision-toolbox-python/mvtb_notebooks.zip>`_, and extract the notebooks from the zip file.

Alternatively, you can clone the repository

.. code-block:: shell

    git clone https://github.com/petercorke/machinevision-toolbox-python.git

The notebooks are located in the ``docs/notebooks`` folder.  
You can also navigate to `that folder on GitHub <https://github.com/petercorke/machinevision-toolbox-python/tree/main/docs/notebooks>`_ and download the notebooks individually.



Other Jupyter notebook tools
----------------------------

* ``jupyter nbconvert``  executes a notebook and saves the result in HTML, PDF, Markdown, LaTeX, or other formats. It can also turn the notebook into a Python script, with the documentation cells as comment blocks.
*  ``papermill`` allows parameterizing and executing Jupyter Notebooks programmatically. This is useful for running the same notebook with different parameters, for example to run a training notebook with different hyperparameters.  It can also be used to execute a notebook and save the result in a new notebook file, which is useful for keeping a record of the executed notebook with the results.
