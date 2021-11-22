import numpy as np

def int_image(image, intclass='uint8', maxintval=None):
    """
    Convert image to integer type

    :param image: input image
    :type image: ndarray(h,w,nc) or ndarray(h,w,nc)
    :param intclass: either 'uint8', or any integer class supported by np
    :type intclass: str
    :return: image with integer pixel types
    :rtype: ndarray(h,w,nc) or ndarray(h,w,nc)

    - ``int_image(image)`` is a copy of image with pixels converted to unsigned
        8-bit integer (uint8) elements in the range 0 to 255.

    - ``int_image(intclass)`` as above but the output pixels are converted to
        the integer class ``intclass``.

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox import iread, idisp, int_image
        >>> im, file = iread('flowers1.png')
        >>> idisp(int_image(im, 'uint16'))

    .. note::

        - Works for greyscale or color (arbitrary number of planes) image
        - If the input image is floating point (single or double) the
            pixel values are scaled from an input range of [0,1] to a range
            spanning zero to the maximum positive value of the output integer
            class.
        - If the input image is an integer class then the pixels are cast
            to change type but not their value.

    :references:

        - Robotics, Vision & Control, Section 12.1, P. Corke,
            Springer 2011.
    """

    if np.issubdtype(image.dtype, np.bool_):
        return image.astype(intclass) * np.iinfo(intclass).max

    elif np.issubdtype(image.dtype, np.floating):
        # rescale to integer
        scaled = image * np.float64(np.iinfo(intclass).max)
        return np.rint(scaled).astype(intclass)

    elif np.issubdtype(image.dtype, np.integer):
        # scale and cast to different integer type
        if maxintval is None:
            maxintval = np.iinfo(image.dtype).max
        image = image * (np.iinfo(intclass).max / maxintval)
        return image.astype(intclass)
 

def float_image(image, floatclass='float32', maxintval=None):
    """
    Convert image to float type

    :param image: input image
    :type image: ndarray(h,w,nc) or ndarray(h,w,nc)
    :param floatclass: 'single', 'double', 'float32' [default], 'float64'
    :type floatclass: str
    :return: image with floating point pixel types
    :rtype: ndarray(h,w,nc) or ndarray(h,w,nc)

    - ``float_image()`` is a copy of image with pixels converted to
        ``float32`` floating point values spanning the range 0 to 1. The
        input integer pixels are assumed to span the range 0 to the maximum
        value of their integer class.

    - ``float_image(im, floatclass)`` as above but with floating-point pixel
        values belonging to the class ``floatclass``.

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox import iread, idisp, float_image
        >>> im, file = iread('flowers1.png')
        >>> idisp(float_image(im))

    .. note::

        - Works for greyscale or color (arbitrary number of planes) image
        - If the input image is integer the
          pixel values are scaled from an input range
          spanning zero to the maximum positive value of the output integer
          class to [0,1]
        - If the input image is a floating class then the pixels are cast
            to change type but not their value.

    :references:

        - Robotics, Vision & Control, Section 12.1, P. Corke,
            Springer 2011.
    """

    if floatclass in ('float', 'single', 'float32', 'float64'):
        # convert to float pixel values
        if np.issubdtype(image.dtype, np.integer):
            # rescale the pixel values
            if maxintval is None:
                maxintval = np.iinfo(image.dtype).max
            return image.astype(floatclass) / maxintval
        elif np.issubdtype(image.dtype, np.floating):
            # cast to different float type
            return image.astype(floatclass)
    else:
        raise ValueError('bad float type')

    def image_to_dtype(image, dtype):
        dtype = np.dtype(dtype)  # convert to dtype if it's a string

        if np.issubdtype(dtype, np.integer):
           return int_image(dtype)
        elif np.issubdtype(dtype, np.floating):
            return float_image(dtype)