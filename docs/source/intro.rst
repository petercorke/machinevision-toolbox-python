************
Introduction
************


The goal of this package is to simplify the expression of computer vision
algorithms in Python.  Images can be represented as 2D or 3D arrays which are
the domain of `NumPy <https://numpy.org>`_` but many power image specific operations are provided by 
`OpenCV  <https://opencv.org>`_, `SciPy <https://scipy.org>`_` and `Open3D <open3d.org>`_.  `matplotlib <https://matplotlib.org>`_ is a portable and powerful
way to display graphical data, including images, whereas OpenCV does a great job of displaying
images but other graphics, not so much.

In practice, using these various
packages is complex, each have their own way of working, similar options are
accessed differently and some function require image pixels to have particular types.
None of them consider the image as an object with a set of useful image and vision
processing methods and operators.  

For example, to read an image using OpenCV and display it using matplotlib::

    import matplotlib.pyplot as plt
    import cv2

    img = cv2.Read()
    plt.imshow(img)

Using this toolbox we would write instead::

    from machinevisiontoolbox import Image

    img = Image.Read('flowers1.png')
    img.disp()

While this particular example is no shorter it is perhaps a bit clearer about what is going on 
(yes, this is very subjective!).

Image.Read('flowers1.png').disp()

and using Python method chaining we could write::

    Image.Read('flowers1.png').smooth(5).disp()

In essence an image is an array with one or more planes or channels.  The underlying representation
can be a 2D or 3D NumPy array.  The planes have specific meaning, for example red or hue which is typically 
implicit in the array.  This package keeps the color plane semantic information with the image data::

    img = Image.Read('flowers1.png')
    img.red().disp()
    img.colorspace('hsv').plane('h').disp()

img.nplanes
img.iscolor
img.ismono
img.isfloat
img.to_float().isfloat

image[i]
for plane in image:
