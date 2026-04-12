NumPy integration
=================

The Toolbox is built on top of NumPy, and all :class:`Image` instances are designed to
be fully compatible with NumPy operations. This means that you can use any NumPy
function or ufunc on an :class:`Image` instance, and it will behave as expected. The
Toolbox implements the NumPy ufunc protocol, which allows it to seamlessly integrate
with NumPy's universal functions (

ufuncs). When you apply a ufunc to an :class:`Image`, the Toolbox will automatically
convert the image to its underlying NumPy array, apply the ufunc, and then convert the
result back into an :class:`Image` if the output is an array.

Accessing the pixels as a NumPy array
-------------------------------------

The image pixels are stored in a 2D or 3D NumPy array. 2D for a single-channel (greyscale) image,
and 3D for a multi-channel (color) image.

You can access this array by
the ``array`` attribute::

	arr = img.array

It is important to note that ``img.size`` is different to ``img.shape``. For an image
that is W pixels wide and H pixels high, with C channels:

- the ``shape`` is the dimensions of the underlying array (H,W) or (H,W,C).
- the ``size`` is the number of pixels (W, H) in the image. The number of channels (C) is not
  included in the size but is found by ``nplanes`` which will be 1 for a single-channel image even though 
  the underlying array has no third dimension.

The toolbox is very consistent about always putting the horizontal coordinate or dimension, before
the vertical one. This is the opposite of the usual convention for NumPy arrays, but it is more
consistent with the way we think about images and how they are displayed. 

We can slice the image using the same syntax as a NumPy array::

	img[10:20, 30:40]

which is the region of pixels in columns [10,20) and rows [30,40). The result is another :class:`~machinevisiontoolbox.ImageCore.Image object`.

Using NumPy this would be:

    arr[30:40, 10:20]

and the result would be a NumPy array.

Using the NumPy ufunc protocol
------------------------------

The Toolbox implements the NumPy ufunc protocol, which allows it to seamlessly integrate
with NumPy's universal functions (ufuncs). When you apply a ufunc to an :class:`Image`, the Toolbox will automatically
convert the image to its underlying NumPy array, apply the ufunc, and then convert the
result back into an :class:`Image` if the output is an array.

Consider the following example:

    img.mean

which returns the mean value of the image.  This is a method of the :class:`Image` class that computes the mean pixel value. However, if you want to use NumPy's ufuncs, you can do so directly on the image:

    np.mean(img)

which also allows us to specify the axis along which to compute the mean, for example:

    np.mean(img, axis=0)


We can access the array of pixel values by the ``array`` attribute::

	np.mean(img.array)

.. warning:: The ``array`` attribute returns a view of the underlying pixel data. 
    Modifying this array will modify the image and if the array dimensions were changed
    it may invalidate some of the image image metadata.

but only for reading, not for assignment. The result is 

This method supports ordinary ufunc calls such as ``np.ceil(img)`` or
``np.arctan2(img1, img2)``. Any :class:`Image` input arguments are converted to
their underlying NumPy arrays, the ufunc is applied, and any ndarray results are
wrapped back into :class:`Image` instances.

Only the ``"__call__"`` ufunc mode is supported. Reduction-style ufunc methods
such as ``reduce`` or ``accumulate`` are delegated back to NumPy.

.. important:: The NumPy ``out=`` argument is intentionally not supported. Images are
    treated as immutable values, so in-place ufunc writes are rejected.

Example:

.. runblock:: pycon

    >>> from machinevisiontoolbox import Image
    >>> import numpy as np
    >>> img = Image([[1.2, 2.8], [3.1, 4.9]], dtype='float32')
    >>> np.ceil(img).array
    >>> a = Image([[0.0, 1.0], [2.0, 3.0]], dtype='float32')
    >>> b = Image([[1.0, 1.0], [1.0, 1.0]], dtype='float32')
    >>> np.arctan2(a, b).array


To clarify the mode of operation, consider the following two different ways of
adding two images.

.. code-block:: pycon
    :linenos:

    >>> im1 = Image([[1, 2], [3, 4]])
    >>> im2 = Image([[5, 6], [7, 8]])
    >>> im3 = np.add(im1, im2)
    >>> im3 = im1 + im2

Line 3 uses NumPy ufunc dispatch, so it comes to this method which unwraps
both ``Image`` instances to ndarrays, calls NumPy’s ``add`` ufunc, then wraps the ndarray result
back into an ``Image``.

Line 4 uses Python operator dispatch which are handled by the Toolbox.  These
perform scalar and plane broadcasting, but they do not support arbitrary ufuncs
such as ``np.ceil`` or ``np.arctan2``.  If you want to use those ufuncs, you
must call them directly on the images, which will invoke this method.