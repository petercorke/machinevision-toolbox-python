************
Introduction
************

Rationale
=========

The goal of this package is to simplify the expression of computer vision algorithms in
Python.  Images can be represented as 2D or 3D arrays which are the domain of `NumPy
<https://numpy.org>`_ but many powerful image and point cloud specific operations are
provided by other popular packages such as `OpenCV <https://opencv.org>`_, `Pillow <https://pillow.readthedocs.io/en/stable/>`_,
`SciPy <https://scipy.org>`_, `scikit-image <https://scikit-image.org>`_, and `Open3D <open3d.org>`_.
OpenCV does an adequate job of displaying images but is nowhere nearly as powerful
`matplotlib <https://matplotlib.org>`_ which can display a wide range of 2D graphics,
but for 3D graphics Open3D is the go-to.

In practice, using these various packages together, to exploit their individual strengths,
is complex -- each have their own way of working, similar options are accessed
differently and some function require image pixels to have particular types. None of
them consider the image as an object with a set of useful image and vision processing
methods and operators.  

For example, to read an image using OpenCV, smooth it, and display it is::

    import cv2
    import numpy

    # read image
    src = cv2.imread(".../flowers1.png", cv2.IMREAD_UNCHANGED)
    
    # apply Gaussian blur on src image
    dst = cv2.GaussianBlur(src, (5,5), cv2.BORDER_DEFAULT)
    
    # display input and output image
    cv2.imshow("Gaussian Smoothing",numpy.hstack((src, dst)))
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image

Using this toolbox we would write instead::

    from machinevisiontoolbox import Image

    img = Image.Read("flowers1.png") # read the image
    smooth = img.smooth(hw=2)  # apply a Gaussian blur
    smooth.disp(block=True)  # display and block until window dismissed

or even::

    from machinevisiontoolbox import Image

    img = Image.Read("flowers1.png").smooth(hw=2).disp(block=True)

which exploits the power of Python's method chaining -- allowing a processing pipeline
to be expressed in a single line of very readable code.

While the merits (or demerits) of these different approaches is  subjective, you get the idea that the Toolbox allows 
succinct coding without the need for lots of OpenCV flags like ``cv2.IMREAD_UNCHANGED`` in the example above.

In summary, the `Machine Vision Toolbox for Python (MVTB-P) <https://github.com/petercorke/machinevision-toolbox-python>`_:

* provides many functions that are useful in machine vision and vision-based control.  
* provides a simple, yet powerful and consistent, object-oriented wrapper of OpenCV
  functions. It supports operator overloading and handles the gnarly details of
  OpenCV-like conversion to/from float32 and the BGR color order.
* leverages the power of NumPy and OpenCV, and inherits their efficiency, portability
  and maturity.
* has similar, but not identical, functionality to the older `Machine Vision Toolbox for MATLAB <https://github.com/petercorke/machinevision-toolbox-matlab>`_. 
* includes over 100 functions such as image file reading and writing, acquisition,
  display, filtering, blob, point and line feature extraction,  mathematical morphology,
  homographies, visual Jacobians, camera calibration and color space conversion. With
  input from a web camera and output to a robot (not provided) it would be possible to
  implement a visual servo system entirely in Python.  
* includes functionality spanning photometry, photogrammetry, colorimetry; while also being sufficient to 
  support the book `Robotics, Vision & Control <https://petercorke.com/rvc3p>`_. 


Image objects
=============

The key element of the Toolbox is the :class:`~machinevisiontoolbox.ImageCore.Image` class.
This sections provides some examples, but full details are given in :ref:`image_class_label`.
The remainder of this section provides a brief overview of the key features of the
:class:`~machinevisiontoolbox.ImageCore.Image` class with examples.

Firstly, there are lots of ways to create an image.  We can read an image from a file::

    img = Image.Read("street.png")

or create it from code::

	img = Image.Zeros(100, dtype="uint8")

Under the hood the :class:`~machinevisiontoolbox.ImageCore.Image` object contains some image parameters, a lot
of methods, and a reference to a 2D or 3D NumPy ndarray containing the pixel data.

:class:`~machinevisiontoolbox.ImageCore.Image` object methods generally consider pixel coordinates with the horizontal coordinate
first and the vertical coordinate second -- consistent with the way we write about
algorithms but the opposite to the way that NumPy indexes an array.

An image object has a lot of useful attributes that describe the image, including:

* ``img.width``, the width of the image in pixels
* ``img.height``, the height of the image in pixels
* ``img.size``, the size of the image (width, height) in pixels
* ``img.nplanes``, the number of planes in the image

as well as a number of useful predicates including:

* ``img.iscolor``, is the image multichannel?
* ``img.ismono``, is the image single channel?
* ``img.isfloat``, does the image have floating point pixels?


Accessing the pixel array
-------------------------

We can access the array of pixel values by
either the ``A`` or ``image`` attribute, or by using the object as if it were a
NumPy array, for example::

	np.mean(img.A)
	np.mean(img.image)
	np.mean(img)

We can slice the image using the same syntax as a NumPy array::

	img[10:20, 30:40]

but only for reading, not for assignment. The result is another :class:`~machinevisiontoolbox.ImageCore.Image object`.


Multi-plane images
------------------

Color images are handled a bit more sensibly than raw OpenCV.  A multi-channel
or multi-plane image is a NumPy ndarray with an arbitrary number of planes and a
dictionary that maps channel names to an integer index.  For instance, to create multi-plane images 
we can write any of the following::

	img = Image.Zeros(100, colororder="RGB")
	img = Image.Zeros(100, colororder="XYZ")
	img = Image.Zeros(100, colororder="red:green:blue")
	img = Image.Zeros(100, colororder="PQRST")  # 5 channel image

which create 100x100 images with 3, 3, 3 and 5 planes respectively, with all pixel values set to zero.
Rather than have the meaning of the plane implicit (ie. plane 0 is red), it is explicit, for example::

	img.plane("R")
	img.plane("Y")
	img.plane("blue")

A more common example is to read a color image::

    img = Image.Read("flowers1.png")
    img.red().disp()  # display the red plane of the image, whether RGB or BGR format
    img.colorspace("hsv").plane("h").disp()  # display the hue plane of an HSV image



Image iterators
---------------

Frequently we want to use images that form a seqeuence -- consecutive frames from a camera
or a video file, a web camera, image files in a folder or zip file.
Rather than build this capability into the `Image` object we provide a number of
iterator objects::
	
	for img in ZipArchive("holidaypix.zip"):
		# process the image
		


Getting started
===============

Using pip
---------

Install a snapshot from PyPI::

	% pip install machinevision-toolbox-python


From GitHub source
------------------

Install the current code base from GitHub and pip install a link to that cloned copy::

	% git clone https://github.com/petercorke/machinevision-toolbox-python.git
	% cd machinevision-toolbox-python
	% pip install -e .


Examples
========

MVTB tool
---------

An interactive IPython session with all the MVTB tools loaded. Start a session from
the shell::

	% mvtbtool
	_  _ ____ ____ _  _ _ _  _ ____    _  _ _ ____ _ ____ _  _ 
	|\/| |__| |    |__| | |\ | |___    |  | | [__  | |  | |\ | 
	|  | |  | |___ |  | | | \| |___     \/  | ___] | |__| | \| 
															
	___ ____ ____ _    ___  ____ _  _                          
	|  |  | |  | |    |__] |  |  \/                           
	|  |__| |__| |___ |__] |__| _/\_  

	for Python

	You're running: MVTB==0.9.7, SMTB==1.1.13, NumPy==1.26.4, SciPy==1.14.1,
                    Matplotlib==3.10.0, OpenCV==4.10.0, Open3D==0.18.0

	from machinevisiontoolbox import *
	from spatialmath import *

	matplotlib interactive mode on

	func/object?       - show brief help
	help(func/object)  - show detailed help
	func/object??      - show source code

		
	Python 3.10.16 (main, Dec 11 2024, 10:22:29) [Clang 14.0.6 ]
	Type 'copyright', 'credits' or 'license' for more information
	IPython 8.31.0 -- An enhanced Interactive Python. Type '?' for help.
	Using matplotlib backend: macosx

	>>> im = Image.Read("monalisa.png")

	>>> im.disp()
	Out[2]: <matplotlib.image.AxesImage at 0x1690e9720>

It has the advantage of command history, tab completion, and inline help.

Binary blobs
------------

We load a binary image of two sharks and find the blobs in the image.  We then display the image with the blobs
marked by bounding boxes and centroids.

.. code-block:: python

	import machinevisiontoolbox as mvtb
	import matplotlib.pyplot as plt
	im = mvtb.Image("shark2.png")   # read a binary image of two sharks
	fig = im.disp();   # display it with interactive viewing tool
	f = im.blobs()  # find all the white blobs
	print(f)

which will display::

	┌───┬────────┬──────────────┬──────────┬───────┬───────┬─────────────┬────────┬────────┐
	│id │ parent │     centroid │     area │ touch │ perim │ circularity │ orient │ aspect │
	├───┼────────┼──────────────┼──────────┼───────┼───────┼─────────────┼────────┼────────┤
	│ 0 │     -1 │ 371.2, 355.2 │ 7.59e+03 │ False │ 557.6 │       0.341 │  82.9° │  0.976 │
	│ 1 │     -1 │ 171.2, 155.2 │ 7.59e+03 │ False │ 557.6 │       0.341 │  82.9° │  0.976 │
	└───┴────────┴──────────────┴──────────┴───────┴───────┴─────────────┴────────┴────────┘

.. code-block:: python

	f.plot_box(fig, color='g')  # put a green bounding box on each blob
	f.plot_centroid(fig, 'o', color='y')  # put a circle+cross on the centroid of each blob
	f.plot_centroid(fig, 'x', color='y')
	plt.show(block=True)  # display the result

![Binary image showing bounding boxes and centroids](https://github.com/petercorke/machinevision-toolbox-python/raw/master/figs/shark2+boxes.png)


Binary blob hierarchy
---------------------

We load a binary image with nested objects

.. code-block:: python

	im = mvtb.Image("multiblobs.png")
	im.disp()

.. image:: https://github.com/petercorke/machinevision-toolbox-python/raw/master/figs/multi.png
	:alt: Binary image showing bounding boxes and centroids

.. code-block:: python

	f  = im.blobs()
	print(f)

which will display::

	┌───┬────────┬───────────────┬──────────┬───────┬────────┬─────────────┬────────┬────────┐
	│id │ parent │      centroid │     area │ touch │  perim │ circularity │ orient │ aspect │
	├───┼────────┼───────────────┼──────────┼───────┼────────┼─────────────┼────────┼────────┤
	│ 0 │      1 │  898.8, 725.3 │ 1.65e+05 │ False │ 2220.0 │       0.467 │  86.7° │  0.754 │
	│ 1 │      2 │ 1025.0, 813.7 │ 1.06e+05 │ False │ 1387.9 │       0.769 │ -88.9° │  0.739 │
	│ 2 │     -1 │  938.1, 855.2 │ 1.72e+04 │ False │  490.7 │       1.001 │  88.7° │  0.862 │
	│ 3 │     -1 │  988.1, 697.2 │ 1.21e+04 │ False │  412.5 │       0.994 │ -87.8° │  0.809 │
	│ 4 │     -1 │  846.0, 511.7 │ 1.75e+04 │ False │  496.9 │       0.992 │ -90.0° │  0.778 │
	│ 5 │      6 │  291.7, 377.8 │  1.7e+05 │ False │ 1712.6 │       0.810 │ -85.3° │  0.767 │
	│ 6 │     -1 │  312.7, 472.1 │ 1.75e+04 │ False │  495.5 │       0.997 │ -89.9° │  0.777 │
	│ 7 │     -1 │  241.9, 245.0 │ 1.75e+04 │ False │  496.9 │       0.992 │ -90.0° │  0.777 │
	│ 8 │      9 │ 1228.0, 254.3 │ 8.14e+04 │ False │ 1215.2 │       0.771 │ -77.2° │  0.713 │
	│ 9 │     -1 │ 1225.2, 220.0 │ 1.75e+04 │ False │  496.9 │       0.992 │ -90.0° │  0.777 │
	└───┴────────┴───────────────┴──────────┴───────┴────────┴─────────────┴────────┴────────┘

We can display a label image, where the value of each pixel is the label of the blob that the pixel
belongs to

.. code-block:: python

	out = f.labelImage(im)
	out.stats()
	out.disp(block=True, colormap="jet", cbar=True, vrange=[0,len(f)-1])

and request the blob label image which we then display


.. image:: https://github.com/petercorke/machinevision-toolbox-python/raw/master/figs/multi_labelled.png
	:alt: Binary image showing bounding boxes and centroids

Camera modelling
----------------

.. code-block:: python

	cam = mvtb.CentralCamera(f=0.015, rho=10e-6, imagesize=[1280, 1024], pp=[640, 512], name="mycamera")
	print(cam)
				Name: mycamera [CentralCamera]
		focal length: (array([0.015]), array([0.015]))
		  pixel size: 1e-05 x 1e-05
		principal pt: (640.0, 512.0)
		  image size: 1280.0 x 1024.0
		focal length: (array([0.015]), array([0.015]))
				pose: t = 0, 0, 0; rpy/zyx = 0°, 0°, 0°

and its intrinsic parameters are

.. code-block:: python

	print(cam.K)
		[[1.50e+03 0.00e+00 6.40e+02]
		[0.00e+00 1.50e+03 5.12e+02]
		[0.00e+00 0.00e+00 1.00e+00]]

We can define an arbitrary point in the world

.. code-block:: python

	P = [0.3, 0.4, 3.0]

and then project it into the camera

.. code-block:: python

	p = cam.project(P)
	print(p)
		[790. 712.]

which is the corresponding coordinate in pixels.  If we shift the camera slightly the image plane coordinate will also change

.. code-block:: python

	p = cam.project(P, T=SE3(0.1, 0, 0) )
	print(p)
	[740. 712.]

We can define an edge-based cube model and project it into the camera's image plane

.. code-block:: python

	X, Y, Z = mkcube(0.2, pose=SE3(0, 0, 1), edge=True)
	cam.mesh(X, Y, Z)

.. image:: https://github.com/petercorke/machinevision-toolbox-python/raw/master/figs/cube.png
	:alt: Perspective camera view


Color space
-----------

Plot the CIE chromaticity space

.. code-block:: python

	showcolorspace("xy")

.. image:: https://github.com/petercorke/machinevision-toolbox-python/raw/master/figs/colorspace.png
	:alt: CIE chromaticity space

Load the spectrum of sunlight at the Earth's surface and compute the CIE xy chromaticity coordinates

.. code-block:: python

	nm = 1e-9
	lam = np.linspace(400, 701, 5) * nm # visible light
	sun_at_ground = loadspectrum(lam, 'solar')
	xy = lambda2xy(lambda, sun_at_ground)
	print(xy)
		[[0.33272798 0.3454013 ]]
	print(colorname(xy, 'xy'))
		khaki


