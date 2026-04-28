``NumPy`` interface
=====================

The Toolbox is built on top of NumPy, and all :class:`Image` instances are designed to
be fully compatible with NumPy operations.

Accessing the pixels as a NumPy array
-------------------------------------

The image pixels are stored in a NumPy array *within* the :class:`Image` object. 
For an image that is W pixels wide, H pixels high, and with C channels, the NumPy array is:

- 2D for a single-channel (greyscale) image, shape (H, W)
- 3D for a multi-channel (color) image, shape (H, W, C)

:class:`Image` instances, and their encapsulated pixel data, are designed to be immutable, so the underlying NumPy array is not directly accessible.
You can access a *read-only view* of this array using the ``array`` attribute::

	arr = img.array

This means that you cannot modify the pixel values
directly through this array reference. If you need to modify the image, you should create a copy
of it first, and then modify the copy.
For example:

.. code-block:: pycon

    >>> img = Image.Read("myimage.png")
    >>> arr = img.array.copy()  # create a writeable copy of the read-only array
    >>> arr[0, 0] = 255  # modify the top-left pixel
    >>> new_img = Image(arr)  # create a new Image instance from the modified array

We can slice the image using NumPy-style indexing on the underlying array. For example,
if we want to get a 10x10 pixel region of the image starting at column 10 and row 30, we
can do:

.. code-block:: pycon

    >>> slice = arr[30:40, 10:20]

and the result would be a NumPy array with shape (10,10).

Using the Toolbox :class:`Image` class, we can use a similar syntax:

.. code-block:: pycon

    >>> slice = img[10:20, 30:40]

and the result would be an :class:`~machinevisiontoolbox.Image` object that contains the
pixel values of the specified region. This is the *same pixel region* as in the previous
example, but now represented as an :class:`~machinevisiontoolbox.Image` instead of a
NumPy array. This is a powerful feature of the Toolbox, as it allows us to easily
manipulate and process images using familiar NumPy-style indexing, while still
benefiting from the rich set of methods and functionality provided by the :class:`Image`
class.

The Toolbox can also return individual pixel values

    >>> img[u, v]

which is a NumPy scalar for a single-channel image, or a NumPy array (C,) for a multi-channel image, representing the pixel value at column u and row v. 
This equivalent to the syntax

.. code-block:: pycon

    >>> img.array[v, u]

The real power of the Toolbox comes from applying fast vectorized functions to whole images, not writing loops to process individual pixels.

.. important:: 
    In the examples above you will have noticed transposition of the coordinates for the NumPy and :class:`Image` classes examples.
    The Toolbox is very consistent about always putting the horizontal coordinate or dimension, before
    the vertical one. This is opposite to the convention used by NumPy arrays and OpenCV, but it is more
    consistent with the way we think about images (width x height) and how the Cartesian coordinates of the plot when an image is displayed (x/u axis horizontal, y/v axis vertical).

    Toolbox methods generally refer to an image ``size`` (W,H) to be distinct from the ``shape`` (H,W) used by NumPy.
    An image has both  ``size`` and ``shape`` properties. For an image
    that is W pixels wide, H pixels high, and with C channels:

    - the ``shape`` is the dimensions of the underlying NumPy array (H,W) or (H,W,C).  The :meth:`Image.shape` property is equivalent to ``img.array.shape``.
    - the ``size`` is always a 2-tuple representing the number of pixels (W, H) in the image. The number of channels (C) is not
      included in the size but is returned by the ``nplanes`` property, which will be 1 for a single-channel image even though 
      the underlying array has no third dimension.


Using the NumPy ufunc protocol
------------------------------

A universal function (ufunc) is a NumPy function that operates on ndarrays in
an element-wise fashion, leveraging vectorized C code to efficiently handle broadcasting
and type casting across multi-dimensional data.

The Toolbox implements the NumPy ufunc protocol, which allows it to seamlessly integrate
with NumPy's universal functions (ufuncs). 

Consider the following example:

.. code-block:: pycon

    >>> img.mean()

which returns the mean value of the image.  This is a method of the :class:`Image` class that
computes the mean pixel value using NumPy under the hood. An alternative way to compute the mean using NumPy explicitly would be:

.. code-block:: pycon

    >>> np.mean(img.array)

where we access the underlying NumPy array and apply the ufunc directly to it.

However, we can also do the operation using NumPy ufuncs:

.. code-block:: pycon

    >>> np.mean(img)

which invokes the ufunc protocol:

* convert the :class:`Image` instance to a NumPy array 
* pass it to the ufunc, and 
* convert any 2D or 3D NumPy arrays in the result back to :class:`Image` instances.

While a number of common operations such as ``mean``, ``median``, ``min``, ``max``, ``std``, ``var``, etc. are implemented
as methods of the :class:`Image` class, the ufunc protocol allows us to use *any* NumPy ufunc
directly on the image, for example::

    np.ceil(img) # computes the ceiling of each pixel value
    np.arctan2(img1, img2) # computes arctan(img1/img2) elementwise
    np.hypot(img1, img2)  # computes sqrt(img1**2 + img2**2) elementwise

all of which will return an :class:`Image` instance containing the result of the operation.


Example:

.. runblock:: pycon

    >>> from machinevisiontoolbox import Image
    >>> import numpy as np
    >>> img = Image([[1.2, 2.8], [3.1, 4.9]], dtype='float32')
    >>> np.ceil(img).array
    >>> a = Image([[0.0, 1.0], [2.0, 3.0]], dtype='float32')
    >>> b = Image([[1.0, 1.0], [1.0, 1.0]], dtype='float32')
    >>> np.arctan2(a, b).array


To clarify, the mode of operation, consider the following two different ways of
adding a pair of images:

.. code-block:: pycon
    :linenos:

    >>> im1 = Image([[1, 2], [3, 4]])
    >>> im2 = Image([[5, 6], [7, 8]])
    >>> sum1 = np.add(im1, im2)
    >>> sum2 = im1 + im2

Line 3 uses NumPy ufunc dispatch, so it extracts the underlying arrays from the
:class:`Image` instances, calls NumPy’s ``add`` ufunc, then wraps the ndarray result
back into an :class:`Image`. 

In contrast, line 4 uses Python operator dispatch which is
handled by the Toolbox and implements the ``__add__`` method. That method has some nice features to 
handle various cases that plain NumPy wouldn't do, such as image + image, image + scalar, multi-channel-image + image, etc.

Both lines will produce the same result, but the ufunc approach is more flexible as it
allows us to use *any* of NumPy's nearly 90 ufuncs. The second approach only supports the functions and
operators that are implemented by the :class:`Image` class. 

.. note::

    - Only the ``"__call__"`` ufunc mode is supported. Reduction-style ufunc methods
      such as ``reduce`` or ``accumulate`` are delegated back to NumPy.
    - The NumPy ``out=`` argument is intentionally not supported. Images are
      treated as immutable values, so in-place ufunc writes are rejected.