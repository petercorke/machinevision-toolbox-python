# Jupyter Notebooks

First install the [Machine Vision Toolbox for Python](https://github.com/petercorke/machinevision-toolbox-python)
```
pip install machinevisiontoolbox[jupyter]
```
This will install all the required dependencies (including OpenCV), example images, and Jupyter with interactive notebook graphics.


You can run Jupyter notebooks a few different ways as discussed below.

### Jupyter from inside Visual Studio

This is a very convenient way to work and highly recommended. Use the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).  You need to do the local software installs as mentioned above.

### Install the Jupyter app
```
pip install jupyter
```
then run Jupyter
```
jupyter notebook
```
which will open a new browser tab with the Jupyter GUI.  The UI is a bit clunky, but it works.  The Visual Studio version, above, is much more slick.  These notebooks have not been tested with JupyterLab.


### Google Colab

This is theoretically a convenient approach with zero install on your computer, but unforutunately each notebook is quite slow to startup because the toolboxes need to be installed into the Colab environment, and they are only cached there for a short amount of time.

Just click the <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> buttons below.
Colab will pull the notebook from GitHub, and allow you to interact with it in a browser tab. If you modify the notebook you have the option of saving it in your own GitHub account, or on Google Drive.  



* Jupyter/Python Notebooks
Welcome to the interactive companion for the **Machine Vision Toolbox for Python**. These notebooks run entirely in your browser via **JupyterLite**—no installation, no configuration, just code.

---

## Quick Introduction

* [**Introduction**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/notebooksintro.ipynb) - A quick introduction.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
---

## 🖼️ Image Fundamentals
Learn how to manipulate pixels, colorspaces, and basic filters using the high-level `Image` class.

* [**Exploring Images**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/notebooksexploring-images.ipynb) — The basics of images, pixels, and bit-depth.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Grey scale images**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/notebooksgreyscale-images.ipynb) - The basics of grey scale images.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Color images**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/notebookscolor-images.ipynb) - The basics of color images.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Image Processing**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/notebooksimage-processing.ipynb) — Fundamentals of filtering, convolution kernels, and spatial operations.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Understanding gamma**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/notebooksgamma.ipynb) - The most misunderstood image transform that is everywhere in image processing.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---

## 📷 Camera Geometry & Projection
Explore how the 3D world is mapped onto 2D sensors using comprehensive camera models.

* [**Camera Animation**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/notebookscamera_animation.ipynb) — **Interactive:** Visualize how points project onto a central perspective sensor.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**The Central Camera**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/notebookscamera.ipynb) — Introduction to the Toolbox `CentralCamera` object and its properties.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Homogeneous Coordinates**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/notebookshomogeneous-coords.ipynb) — A refresher on the math behind spatial transforms with an interactive animation.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Homography**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/notebookshomographies.ipynb) — Computing planar projections, image warping, and homography estimation.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Camera calibration**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/notebookscalibration.ipynb) - Calibrating a camera from a set of images.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---

## 🔍 Advanced Vision Features
Moving beyond pixels to robust features and 3D reasoning.

* [**Finding Blobs**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/notebooksfinding-blobs.ipynb) — Region segmentation, binary shape analysis, and blob parameters.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Image convolution**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/notebooksimage-convolution.ipynb) - The basis of smoothing, edge detection and point features.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Image Features**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/notebooksimage-features.ipynb) — Fundamentals of point and corner detection (SIFT, ORB, etc.) as discussed in the lectures.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Fiducial Markers**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/notebooksfiducials.ipynb) — Detecting ArUco markers and QR-like codes in real-world scenes.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Image Motion**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/notebooksimage-motion.ipynb) - The relationship between camera motion (3D) and image plane motion (2D).
* [**Visual Servoing**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/notebooksIBVS.ipynb) - Image-Based Visual Servoing.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
---
<p align="center">
  <img src="https://github.com/petercorke/machinevision-toolbox-python/raw/main/figs/VisionToolboxLogo_NoBackgnd@2x.png" width="200">
  <br>
  <font size="2">Created by Peter Corke | QUT Centre for Robotics</font>
</p>


