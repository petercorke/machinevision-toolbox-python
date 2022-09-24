

import numpy as np
import spatialmath.base as base
import scipy as sp
from numpy.polynomial import Polynomial

def findpeaks(y, x=None, npeaks=None, scale=1, interp=0, return_poly=False):
    r"""
    Find peaks in a 1D signal

    :param y: 1D-signal :math:`y(x)`
    :type y: ndarray(N)
    :param x: corresponding dependent variable, defaults to the integer
        sequence :math:`0 \ldots N-1`.
    :type x: ndarray(N), optional
    :param npeaks: number of peaks ``P`` to return, defaults to all
    :type npeaks: int, optional
    :param scale: peak scale, defaults to 1
    :type scale: int, optional
    :param interp: interpolate the peak value, defaults to False
    :type interp: bool or int, optional
    :param return_poly: return interpolation polynomial, defaults to False
    :type return_poly: bool, optional
    :raises ValueError: ``interp`` must be > 2 if given
    :return: peak positions and values
    :rtype: ndarray(P), ndarray(P)

    Find the peak in a 1D signal.

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox.base import *
        >>> import numpy as np
        >>> y = np.array([0, 0, 1, 2, 0, 0, 0, 3, 1, 0, 0, 0, 0])
        >>> findpeaks(y, scale=3)
        >>> findpeaks(y, scale=3, interp=True)

    .. note::
        - A maxima is defined as an element that is larger than its ``scale`` 
          neighbours on either side.  This is the largest value in a 2*scale+1
          sliding window.
        - The first and last ``scale`` elements will never be returned as maxima.
        - To find minima, use ``findpeak(-y)``.
        - The ``interp`` options fits points in the neighbourhood about the peak with
          an M'th order polynomial and its peak position is returned.  Typically
          choose M to be even. Setting ``interp`` to True uses ``M=2``.
          Alternatively set ``interp`` to M.
        - If ``return_poly`` is True then an additional value is returned
          which is a list of polynomial coefficients for each peak, see
          :obj:`numpy.polynomial`.

    :seealso: :func:`findpeaks2d` :func:`findpeaks3d`
    """

    # if second argument is a matrix we take this as the corresponding x
    # coordinates
    y = base.getvector(y)
    if x is None:
        x = np.arange(len(y))
    else:
        x = base.getvector(x, len(y))

    # find the maxima
    if scale > 0:
        dx = x[1] - x[0]
        # compare to a moving window max filtered version
        n = round(scale / dx) * 2 + 1
        kernel = np.ones((n,))
        k, = np.nonzero(y == sp.signal.order_filter(y, kernel, n - 1))
        k = np.array([kk for kk in k if kk >= scale and kk <= len(y)-scale])
    else:
        # take the zero crossings
        dy = np.diff(y)
        # TODO: I think itertools can do this?
        k, = np.nonzero([v and w for v, w in zip(np.r_[dy, 0] < 0, np.r_[0, dy] > 0)])

    # sort the maxima into descending magnitude
    i = np.argsort(-y[k])
    k = k[i]    # indices of the maxima

    if npeaks is not None:
        k = k[:npeaks]

    # interpolate the peaks if required
    if interp is True:
        interp = 2

    if interp > 0:
        if interp < 2:
            raise ValueError('interpolation polynomial must be at least second order')
        
        n2 = round(interp / 2)

        # for each previously identified peak x(i), y(i)
        refined_x = []
        refined_y = []
        polys = []
        for i in k:
            # fit a polynomial to the local neighbourhood
            try:
                poly, *_ = Polynomial.fit(x[i-n2:i+n2+1], y[i-n2:i+n2+1], interp, full=True)
            except:
                #handle situation where neighbourhood falls off the data
                #vector
                print(f"Peak at {x[i]} couldn't be fitted, skipping")
                continue

            # find the roots of the polynomial closest to the coarse peak
            r = poly.deriv(1).roots()
            if len(r) == 0:
                # no roots found
                continue
            j = np.argmin(abs(r - x[i]))
            xp = r[j]
            
            #store x, y for the refined peak
            refined_x.append(xp)
            refined_y.append(poly(xp))
            polys.append(poly)
        
        if return_poly:
            return np.array(refined_x), np.array(refined_y), polys
        else:
            return np.array(refined_x), np.array(refined_y)
    else:
        return x[k], y[k]
    

def findpeaks2d(z, npeaks=2, scale=1, interp=False, positive=True):
    r"""
    Find peaks in a 2D signal

    :param z: 2D-signal :math:`z(x,y)`
    :type z: ndarray(H,W)
    :param npeaks: number of peaks to return (default all)
    :type npeaks: int
    :param scale: scale of peaks to consider
    :type scale: float
    :param interp: interpolate the peak value, defaults to False
    :type interp: bool, optional
    :param positive: peak must be > 0, defaults to False
    :type positive: bool, optional
    :return: peak positions and magnitudes, one per row
    :rtype: ndarray(P,3)

    Find the maximum of a 2D signal, typically a greyscale image.

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox.base import *
        >>> import numpy as np
        >>> z = np.zeros((10,10))
        >>> z[3,4] = 2
        >>> z[4,4] = 1
        >>> findpeaks2d(z)
        >>> findpeaks2d(z, interp=True)

    .. note::

        - A maxima is defined as an element that larger than its neighbours
          in a 2*scale+1 square window. 
        - Elements where the window falls off the edge of the input array
          will never be returned as maxima.
        - To find minima, use ``findpeaks2d(-image)``.
        - The interp options fits points in the neighbourhood about the
          peak with a paraboloid and its peak position is returned.

    :seealso: :func:`findpeaks` :func:`findpeaks3d`
    """

    # TODO check valid input

    # create a neighbourhood mask for non-local maxima suppression

    # scale is taken as half-width of the window
    w = 2 * scale + 1
    M = np.ones((w, w), dtype='uint8')
    M[scale, scale] = 0  # set middle pixel to zero

    # compute the neighbourhood maximum
    # znh = self.window(self.float(z), M, 'max', 'wrap')
    # image = self.asint()
    # nh_max = cv.morphologyEx(image, cv.MORPH_DILATE, M)
    nhood_max = sp.ndimage.maximum_filter(z, footprint=M)

    # find all pixels greater than their neighbourhood
    
    if positive:
        k = np.flatnonzero((z > nhood_max) & (z > 0))
    else:
        k = np.flatnonzero(z > nhood_max)

    # sort these local maxima into descending order
    image_flat = z.ravel()

    maxima = image_flat[k]

    ks = np.argsort(-maxima)
    k = k[ks]

    if npeaks is not None:
        npks = min(len(k), npeaks)
        k = k[0:npks]

    y, x = np.unravel_index(k, z.shape)

    # interpolate peaks if required
    if interp:
        refined = []
        for xk, yk in zip(x, y):
        # now try to interpolate the peak over a 3x3 window
            try:
                zc = z[yk,   xk]
                zn = z[yk-1, xk]
                zs = z[yk+1, xk]
                ze = z[yk,   xk+1]
                zw = z[yk,   xk-1]
            except IndexError:
                continue

            dx = (ze - zw) / (2 * (2 * zc - ze - zw))
            dy = (zs - zn) / (2 * (2 * zc - zn - zs))

            zest = zc - (ze - zw)**2 / (8 * (ze - 2 * zc + zw)) \
                - (zn - zs)**2 / (8 * (zn - 2 * zc + zs))
            
            aest = np.min(np.abs(np.r_[ze/2 - zc + zw/2, zn/2 - zc + zs/2]))
            refined.append([xk + dx, yk + dy, zest, aest])
        return np.array(refined)
    else:
        # xy = np.stack((y, x), axis=0)
        return np.column_stack((x, y, image_flat[k]))


def findpeaks3d(v, npeaks=None):
    r"""
    Find peaks in a 3D signal

    :param z: 3D-data :math:`v(x,y,z)`
    :type z: ndarray(H,W,D)
    :param npeaks: number of peaks to return (default all)
    :type npeaks: int
    :return: peak position and magnitude, one per row
    :rtype: ndarray(N,4)

    Find the maximum of a 3D signal, typically volumetric data such as a
    scale-space image stack.

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox.base import *
        >>> import numpy as np
        >>> z = np.zeros((10,10,10))
        >>> z[3,4,5] = 1
        >>> findpeaks3d(z)

    .. note:: A maxima is defined as an element that larger than its 26
            neighbours. Edges elements will never be returned as maxima.


    :seealso: :func:`findpeaks` :func:`findpeaks2d`
    """
    # absolute value of Laplacian as a 3D matrix, with scale along axis 2

    # find maxima within all 26 neighbouring pixels
    # create 3x3x3 structuring element and maximum filter
    se_nhood = np.ones((3, 3, 3))
    se_nhood[1, 1, 1] = 0
    eps = np.finfo(np.float64).eps
    maxima = (v > sp.ndimage.maximum_filter(v, footprint=se_nhood, mode='nearest'))

    # find the locations of the minima
    i, j, k = np.nonzero(maxima)
    
    # create result matrix, one row per feature: i, j, k, L
    # where k is index into scale
    result = np.column_stack((i, j, k, v[i, j, k]))

    # sort the rows on strength column, descending order
    k = np.argsort(-result[:, 3])
    result = result[k, :]

    if npeaks is not None:
        result = result[:npeaks, :]

    return result

if __name__ == "__main__":

    from machinevisiontoolbox.base import *
    import numpy as np

    y = np.array([0, 0, 1, 2, 0, 0, 0, 3, 1, 0, 0, 0, 0])
    print(findpeaks(y, scale=3))
    print(findpeaks(y, scale=3, interp=True))

    z = np.zeros((10,10))
    z[3,4] = 2
    z[4,4] = 1
    print(findpeaks2d(z))
    print(findpeaks2d(z, interp=True))

    # y = mvtb_load_matfile('data/peakfit.mat')["y"]
    # plt.plot(y, '-o')
    # xmax, ymax = findpeaks(y, interp=2)
    # print(xmax, ymax)

    # a = [1, 1, 1, 1, 1]
    # print(findpeaks(a))

    # a = [5, 1, 1, 1, 1]
    # print(findpeaks(a))

    # a = [1, 1, 5, 1, 1]
    # print(findpeaks(a))

    # a = [1, 1, 5, 1, 1]
    # print(findpeaks(a, [10, 11, 12, 13, 14]))


    # a = [1, 2, 3, 4, 3, 2, 1]
    # print(findpeaks(a))

    # a.extend(a)
    # print(findpeaks(a))

    # a = [1, 1, 1, 2, 1, 4, 1, 1, 2]
    # print(findpeaks(a))