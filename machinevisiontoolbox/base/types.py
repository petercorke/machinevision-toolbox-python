import numpy as np

def int_image(image, intclass='uint8', maxintval=None):
    """
    Convert image to integer type

    :param image: input image
    :type image: ndarray(H,W), ndarray(H,W,P)
    :param intclass: integer class to convert to, the name of any integer 
        class supported by NumPy, defaults to ``'uint8'``
    :type intclass: str
    :param maxintval: maximum value of integer, defaults to maximum positive
        value of ``image`` datatype
    :type maxintval: int
    :return: image with integer pixel types
    :rtype: ndarray(H,W), ndarray(H,W,P)

    Return a copy of the image as a NumPy array with pixel values scaled and
    converted to the integer class ``intclass``. If the input image is:

    * a floating point class, the pixel values are scaled from an input range of
      [0.0, 1.0] to a range spanning zero to the maximum positive value of
      ``intclass``.
    * an integer class, the pixels are scaled and cast to ``intclass``. The
      scale factor is the ratio of ``maxintval`` to the maximum positive value
      of ``inclass``.
    * the boolean class, False is mapped to zero and True is mapped to the
      maximum positive value of ``inclass``.

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox import int_image
        >>> import numpy as np
        >>> im = np.array([[1,2],[3,4]], 'uint8')
        >>> int_image(im, 'int16')
        >>> im = np.array([[False, True],[True, False]])
        >>> int_image(im)

    .. note:: Works for greyscale or color (arbitrary number of planes) image

    :references:
        - Robotics, Vision & Control for Python, Section 10.1, P. Corke, Springer 2023.

    :seealso: :func:`float_image`
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
    :type image: ndarray(H,W), ndarray(H,W,P)
    :param floatclass: 'single', 'double', 'float32' [default], 'float64'
    :type floatclass: str
    :param maxintval: maximum value of integer, defaults to maximum positive
        value of ``image`` datatype
    :type maxintval: int
    :return: image with floating point pixel types
    :rtype: ndarray(H,W), ndarray(H,W,P)

    Return a copy of the image as a NumPy array with pixels scaled and converted
    to the float class ``floatclass`` with pixel values spanning the range 0.0 to
    1.0. If the input image is:

    * an integer class, the pixel values are scaled from an input range spanning
      zero to ``maxintval`` to [0.0, 1.0]
    * a floating point class, the pixels are cast to change type but not their
      value.
    * the boolean class, False is mapped to 0.0 and True is mapped to 1.0.

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox import float_image
        >>> import numpy as np
        >>> im = np.array([[1,2],[3,4]], 'uint8')
        >>> float_image(im)
        >>> im = np.array([[False, True],[True, False]])
        >>> float_image(im)

    .. note:: Works for greyscale or color (arbitrary number of planes) image

    :references:
        - Robotics, Vision & Control for Python, Section 10.1, P. Corke, Springer 2023.

    :seealso: :func:`int_image`
    """

    if floatclass not in ('float', 'single', 'double', 'half', 'float16', 'float32', 'float64'):
        raise ValueError('bad float type')

    if np.issubdtype(image.dtype, np.integer):
        # rescale the pixel values
        if maxintval is None:
            maxintval = np.iinfo(image.dtype).max
        return image.astype(floatclass) / maxintval
    elif np.issubdtype(image.dtype, np.floating):
        # cast to different float type
        return image.astype(floatclass)
    elif np.issubdtype(image.dtype, np.bool_):
        return image.astype(floatclass)

    def image_to_dtype(image, dtype):
        dtype = np.dtype(dtype)  # convert to dtype if it's a string

        if np.issubdtype(dtype, np.integer):
           return int_image(dtype)
        elif np.issubdtype(dtype, np.floating):
            return float_image(dtype)