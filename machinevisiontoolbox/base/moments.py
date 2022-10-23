
import numpy as np
from typing import Union
from machinevisiontoolbox.base.meshgrid import meshgrid

def mpq(im: np.ndarray, p: int, q: int) -> Union[int, float]:
    r"""
    Image moments

    :param im: image
    :param p: u exponent
    :param q: v exponent
    :return: moment

    Computes the pq'th moment of the image:
    
    .. math::
    
        m(I) = \sum_{uv} I_{uv} u^p v^q

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox.base import iread, mpq
        >>> img, _ = iead('shark1.png')
        >>> mpq(img, 1, 0)

    :note:
        - Supports single channel images only.
        - ``mpq(im, 0, 0)`` is the same as ``np.sum(im)`` but less efficient.

    :references:
        - Robotics, Vision & Control for Python, Section 12.1.3.4, P. Corke, Springer 2023.

    :seealso: :func:`npq` :func:`upq`
    """

    if not isinstance(p, int) or not isinstance(q, int):
        raise TypeError(p, 'p, q must be an int')

    X, Y = meshgrid(im.shape[1], im.shape[0])
    return np.sum(im * (X ** p) * (Y ** q))

def upq(im: np.ndarray, p: int, q: int) -> Union[int, float]:
    r"""
    Central image moments

    :param im: image
    :param p: u exponent
    :param q: v exponent
    :return: central moment

    Computes the pq'th central moment of the image:
    
    .. math::
    
        \mu(I) = \sum_{uv} I_{uv} (u-u_0)^p (v-v_0)^q

    where :math:`u_0 = m_{10}(I) / m_{00}(I)` and :math:`v_0 = m_{01}(I) / m_{00}(I)`.

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox.base import iread, upq
        >>> img, _ = iead('shark1.png')
        >>> upq(img, 2, 2)

    :note:
        - The central moments are invariant to translation.
        - Supports single channel images only.

    :references:
        - Robotics, Vision & Control for Python, Section 12.1.3.4, P. Corke, Springer 2023.

    :seealso: :func:`mpq` :func:`npq`
    """

    if not isinstance(p, int) or not isinstance(q, int):
        raise TypeError(p, 'p, q must be an int')

    m00 = mpq(im, 0, 0)
    xc = mpq(im, 1, 0) / m00
    yc = mpq(im, 0, 1) / m00

    X, Y = meshgrid(im.shape[1], im.shape[0])

    return np.sum(im * ((X - xc) ** p) * ((Y - yc) ** q))


def npq(im: np.ndarray, p: int, q: int) -> Union[int, float]:
    r"""
    Normalized central image moments

    :param im: image
    :param p: u exponent
    :param q: v exponent
    :return: normalized central moment

    Computes the pq'th normalized central moment of the image:
    
    .. math::
    
        \nu(I) = \frac{\mu_{pq}(I)}{m_{00}(I)} = \frac{1}{m_{00}(I)} \sum_{uv} I_{uv} (u-u_0)^p (v-v_0)^q 

    where :math:`u_0 = m_{10}(I) / m_{00}(I)` and :math:`v_0 = m_{01}(I) / m_{00}(I)`.

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox.base import iread, npq
        >>> img, _ = iead('shark1.png')
        >>> npq(img, 2, 2)

    :note:
        - The normalized central moments are invariant to translation and
            scale.
        - Supports single channel images only.

    :references:
        - Robotics, Vision & Control for Python, Section 12.1.3.4, P. Corke, Springer 2023.

    :seealso: :func:`mpq` :func:`upq`
    """
    if not isinstance(p, int) or not isinstance(q, int):
        raise TypeError(p, 'p, q must be an int')
    if (p+q) < 2:
        raise ValueError(p+q, 'normalized moments only valid for p+q >= 2')

    g = (p + q) / 2 + 1

    return upq(im, p, q) / mpq(im, 0, 0) ** g