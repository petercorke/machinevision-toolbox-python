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

* [**Introduction**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/intro.ipynb) - A quick introduction.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
---

## 🖼️ Image Fundamentals
Learn how to manipulate pixels, colorspaces, and basic filters using the high-level `Image` class.

* [**Exploring Images**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/exploring-images.ipynb) — The basics of images, pixels, and bit-depth.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Grey scale images**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/greyscale-images.ipynb) - The basics of grey scale images.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Color images**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/color-images.ipynb) - The basics of color images.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Image Processing**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/image-processing.ipynb) — Fundamentals of filtering, convolution kernels, and spatial operations.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Understanding gamma**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/gamma.ipynb) - The most misunderstood image transform that is everywhere in image processing.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---

## 📷 Camera Geometry & Projection
Explore how the 3D world is mapped onto 2D sensors using comprehensive camera models.

* [**Camera Animation**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/camera_animation.ipynb) — **Interactive:** Visualize how points project onto a central perspective sensor.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**The Central Camera**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/camera.ipynb) — Introduction to the Toolbox `CentralCamera` object and its properties.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Homogeneous Coordinates**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/homogeneous-coords.ipynb) — A refresher on the math behind spatial transforms with an interactive animation.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Homography**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/homographies.ipynb) — Computing planar projections, image warping, and homography estimation.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Camera calibration**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/calibration.ipynb) - Calibrating a camera from a set of images.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---

## 🔍 Advanced Vision Features
Moving beyond pixels to robust features and 3D reasoning.

* [**Finding Blobs**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/finding-blobs.ipynb) — Region segmentation, binary shape analysis, and blob parameters.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Image convolution**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/image-convolution.ipynb) - The basis of smoothing, edge detection and point features.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Image Features**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/image-features.ipynb) — Fundamentals of point and corner detection (SIFT, ORB, etc.) as discussed in the lectures.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Fiducial Markers**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/fiducials.ipynb) — Detecting ArUco markers and QR-like codes in real-world scenes.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
* [**Image Motion**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/image-motion.ipynb) - The relationship between camera motion (3D) and image plane motion (2D).
* [**Visual Servoing**](https://colab.research.google.com/github/petercorke/machinevision-toolbox-python/blob/main/docs/notebooks/IBVS.ipynb) - Image-Based Visual Servoing.<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
---
<p align="center">
  <img src="https://github.com/petercorke/machinevision-toolbox-python/raw/main/docs/figs/VisionToolboxLogo_NoBackgnd@2x.png" width="200">
  <br>
  <font size="2">Created by Peter Corke | QUT Centre for Robotics</font>
</p>

---

## Maintainer notes: how these notebooks install themselves

*(Written 2026-08 - only relevant if you're editing the install machinery itself.)*

Every notebook here has to work in three different environments, each of which needs the toolbox installed a different way:

- **Locally** (VS Code, `jupyter notebook`, `nbmake` in CI) — the toolbox is assumed already installed (dev env or `pip install`); nothing to do.
- **Google Colab** — Colab's "Open in Colab" link fetches *only that one `.ipynb` file* from GitHub, no sibling files, so the toolbox has to be `pip install`ed fresh into the Colab VM each session.
- **JupyterLite** (the in-browser "Try it Now" version published alongside the Sphinx docs) — runs on Pyodide/WASM, where packages install via `micropip` rather than `pip`, and the toolbox wheel is installed with `deps=False` (to stop micropip re-resolving compiled packages like numpy/scipy/opencv that Pyodide already provides its own WASM builds of) — which means every *other* runtime dependency (tqdm, requests, ...) has to be listed and installed explicitly.

**All three cases are handled by one file: `_mvtb_nb_bootstrap.py`.** It detects which environment it's in and does whatever that environment needs, ending with a one-line sanity check so a broken environment-detection is obvious at a glance rather than failing mysteriously three cells later:
```
Running locally using machinevision-toolbox-python v2.2.0
Running on Colab using machinevision-toolbox-python v2.2.0
Running in browser using machinevision-toolbox-python v2.2.0
```

Every notebook's first code cell is a copy of that file's content, marked with a `# MVTB_BOOTSTRAP_CELL` comment. It's a *copy*, not an import — Colab can't see sibling files, so the cell has to be fully self-contained, the same way Colab notebooks in this space normally are (this is deliberate; a fancier design that fetched the shared code over the network at runtime was considered and rejected — it would have made every notebook depend on GitHub being reachable just to install itself, which is a worse failure mode than a bit of duplication).

**The copy is machine-generated, not hand-maintained**, which is the actual point: edit `_mvtb_nb_bootstrap.py` once and the `notebooks-sync-bootstrap` hook regenerates every notebook's marked cell. `_mvtb_nb_bootstrap.py` is a copy of the canonical template in [rvc-ecosystem](https://github.com/petercorke/rvc-ecosystem/blob/main/notebooks/bootstrap_template.py) with only its repo-specific constants changed, so RTB, MVTB and bdsim install the same way. The tools are shared too: they are [pre-commit](https://pre-commit.com) hooks from rvc-ecosystem, configured in `.pre-commit-config.yaml`, and nothing from them is copied into this repo.

| Hook | Does |
| :--- | :--- |
| `notebooks-clear-outputs` | clears outputs and execution counts, sets the same kernel in every notebook, and keeps only the `mvtb_bootstrap` metadata |
| `notebooks-sync-bootstrap` | regenerates the bootstrap cell of every notebook from `_mvtb_nb_bootstrap.py` |
| `notebooks-first-cell` | the first cell is markdown, and its logo has a width, a height and `style="height:auto"` |

### One-time setup (per clone)

```
make setup
```
which runs
```
pip install pre-commit
pre-commit install
```
Git does not copy hooks with a clone, so do this in every new clone of the repo.

After that the hooks run on `git commit`, and fix a notebook in place (re-add it and commit again). To run them by hand on every notebook:
```
pre-commit run --all-files
```
Nothing is enforced on a machine where the hooks are not installed. CI runs the same command on every PR (the docs workflow), so a notebook that is out of step fails there.

**Why this exists at all:** before this, 13 of the 16 notebooks each had their own hand-pasted, slightly-drifted copy of the Colab-install snippet, and the JupyterLite demo notebook had its own bespoke version with a manually-maintained dependency list. That's exactly how machinevision-toolbox-python 2.2.0 shipped a broken "Try it Now" demo — `tqdm` was added as a real dependency but nobody updated the one notebook's hand-typed install list, and nothing tested it before release. One template, generated everywhere, checked by CI, closes that gap.

