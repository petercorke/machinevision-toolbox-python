Jupyter notebooks
=================

The Toolbox folder ``notebooks`` includes a number of Jupyter notebooks that demonstrate the use of the
use of the Toolbox for specific applications.  There are a number of different ways to
execute a Jupyter notebook.

## Jupyter Notebooks

I provide a selection of Jupyter/Python notebooks that will help to embed the knowledge from each lecture.

Alternatively, you can run them locally on your laptop, and that requires that you first install the [Machine Vision Toolbox for Python](https://github.com/petercorke/machinevision-toolbox-python)
```
pip install machinevisiontoolbox
```
This will install all the required dependencies (including OpenCV) as well as example images for the exercises.

If you are installing into an existing local Python environment then you must have Numpy 1.x.  OpenCV is a critical dependency which  does not yet work with Numpy 2.x.  Python 3.9 or newer is recommended.  

I would highly recommend that you use [Miniconda](https://docs.conda.io/projects/miniconda/en/latest) and create an environment for your RVSS code.
```
conda create -n RVSS python=3.10
conda activate RVSS
pip install machinevision-toolbox-python
```

If you run into any issues with Conda or local install talk to me, Tobi or Don.

To render images nicely within the provided notebooks you will also need to install
```
pip install ipywidgets  # interactive controls for Jupyter notebooks
pip install ipympl  # enables matplotlib interactive features in Jupyter notebooks
```

<site>/lite/lab/index.html?path=files/notebooks/<notebook>.ipynb

https://petercorke.github.io/machinevision-toolbox-python/mvtb_notebooks.zip

Locally from command line
--------------------------

You need to load some packages first.  You can do this at install time, and no harm in installing again.

```
pip install machinvision-toolbox-python [jupyter]
```

or just pull the packages you need for the notebooks.

```
pip install jupyter ipympl ipywidgets
```

then run Jupyter
```
jupyter notebook file.ipynb
```
which will open a new browser tab with the Jupyter GUI.  The UI is a bit clunky, but it works.  The Visual Studio version below is much more slick.  These notebooks have not been tested with JupyterLab.

JupyterLab is a more modern interface to Jupyter notebooks, and it is available as an
option when you install Jupyter.  It has a more modern interface and better support for
multiple notebooks. You would
install JupyterLab with
```
pip install jupyterlab
```
and then run Jupyter
```
jupyter lab file.ipynb
```

Locally using Visual Studio Code
--------------------------------

This is a very convenient way to work and highly recommended. You need to install
some packages first.  You can do this at install time, and no harm in installing again.

```
pip install machinvision-toolbox-python [jupyter]
```

or just pull the packages you need for the notebooks.

```
pip install jupyter ipympl ipywidgets
```

Then install the [Jupyter extension for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter). This will allow you to open
and run the notebooks directly in Visual Studio Code -- just open the file from the file explorer view.
This provides a much nicer interface, supports multiple notebooks, and is great for debugging.

.. note:: You can also use the Visual Studio Code interface to run the notebooks in a Jupyter
    server running on your local machine or on a remote machine.  If you are running the
    notebooks on a remote machine then you will need to set up a Jupyter server on that
    machine ``$ jupyter server``, note the URL it is serving on, and connect to it from 
    Visual Studio Code.  This is a bit more work to set up, but it allows you to run 
    the notebooks on a more powerful machine than your local laptop -- that machine 
    must have the Toolbox and Jupyter installed and the notebooks available on its filesystem.

In the browser using Jupyter lite
---------------------------------

.. note:: JupyterLab

In the cloud using Google Colab
-------------------------------

This is theoretically a convenient approach with zero install on your computer, but unforutunately each notebook is quite slow to startup because the toolboxes need to be installed into the Colab environment, and they are only cached there for a short amount of time.

Just click the <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> buttons below.
Colab will pull the notebook from GitHub, and allow you to interact with it in a browser tab. If you modify the notebook you have the option of saving it in your own GitHub account, or on Google Drive.  



Jupyter notebook tools
----------------------

* using the command line tool ``jupyter nbconvert`` to execute the notebook and save the result as a new notebook or as an HTML file
* using the command line tool ``papermill``

Notebooks by category
---------------------
