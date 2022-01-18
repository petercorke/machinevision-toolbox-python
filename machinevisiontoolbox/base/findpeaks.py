"""
Find peaks in vector

YP = PEAK(Y, OPTIONS) are the values of the maxima in the vector Y.

[YP,I] = PEAK(Y, OPTIONS) as above but also returns the indices of the maxima
in the vector Y.

[YP,XP] = PEAK(Y, X, OPTIONS) as above but also returns the corresponding 
x-coordinates of the maxima in the vector Y.  X is the same length as Y
and contains the corresponding x-coordinates.

Options::
'npeaks',N    Number of peaks to return (default all)
'scale',S     Only consider as peaks the largest value in the horizontal 
              range +/- S points.
'interp',M    Order of interpolation polynomial (default no interpolation)
'plot'        Display the interpolation polynomial overlaid on the point data

Notes::
- A maxima is defined as an element that larger than its two neighbours.
  The first and last element will never be returned as maxima.
- To find minima, use PEAK(-V).
- The interp options fits points in the neighbourhood about the peak with
  an M'th order polynomial and its peak position is returned.  Typically
  choose M to be even.  In this case XP will be non-integer.

See also PEAK2.

"""

import numpy as np
import spatialmath.base as base
import scipy as sp

def findpeaks(y, x=None, npeaks=None, scale=1, interp=0):
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
    else:
        # take the zero crossings
        dy = np.diff(y)
        # TODO: I think itertools can do this?
        k, = np.nonzero([v and w for v, w in zip(np.r_[dy, 0] < 0, np.r_[0, dy] > 0)])

    # sort the maxima into descending magnitude
    i = np.argsort(y[k])
    k = k[i]    # indices of the maxima

    if npeaks is not None:
        k = k[:npeaks]

    # interpolate the peaks if required
    if interp > 0:
        raise RuntimeError('not implemented yet')
        # if interp < 2:
        #     raise ValueError('interpolation polynomial must be at least second order')
        
        # xp = []
        # yp = []
        # N2 = round(interp / 2)

        # # for each previously identified peak x(i), y(i)
        # for i in k:
        #     # fit a polynomial to the local neighbourhood
        #     try:
        #         pp = polyfit(x(i-N2:i+N2), y(i-N2:i+N2), N);
        #     except:
        #         #handle situation where neighbourhood falls off the data
        #         #vector
        #         warning('Peak at %f too close to start or finish of data, skipping', x(i));

            
        #     # find the roots of the polynomial closest to the coarse peak
        #     r = roots( polydiff(pp) );
        #     [mm,j] = min(abs(r-x(i)));
        #     xx = r(j);
            
        #     #store x, y for the refined peak
        #     xp = [xp; xx];
        #     yp = [y; polyval(pp, xx)];
    #     pass

    # else:
    #     xp = x(k)
    
    
    # return values
    # yp = y(k)';
    # if nargout > 1
    #     xpout = xp';
    # end

    #[yp,xpout] = 

    return x[k], y[k]


def peak2(image, npeaks=2, scale=1, interp=False, positive=True):
    """
    Find peaks in a matrix

    :param npeaks: number of peaks to return (default all)
    :type npeaks: scalar
    :param sc: scale of peaks to consider
    :type sc: float
    :param interp:  interpolation done on peaks
    :type interp: boolean
    :return: peak position and magnitude, one per row
    :rtype: ndarray(npeaks,3)

    - ``IM.peak2()`` are the peak values in the 2-dimensional signal
        ``IM``. Also returns the indices of the maxima in the matrix ``IM``.
        Use SUB2IND to convert these to row and column.

    - ``IM.peak2(npeaks)`` as above with the number of peaks to return
        specifieid (default all).

    - ``IM.peak2(sc)`` as above with scale ``sc`` specified. Only consider
        as peaks the largest value in the horizontal and vertical range +/- S
        units.

    - ``IM.peak2(interp)`` as above with interp specified. Interpolate peak
        (default no peak interpolation).

    Example:

    .. runblock:: pycon

    .. note::

        - A maxima is defined as an element that larger than its eight
            neighbours. Edges elements will never be returned as maxima.
        - To find minima, use PEAK2(-V).
        - The interp options fits points in the neighbourhood about the
            peak with a paraboloid and its peak position is returned.  In
            this case IJ will be non-integer.

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
    nhood_max = sp.ndimage.maximum_filter(image, footprint=M)

    # find all pixels greater than their neighbourhood
    
    if positive:
        k = np.flatnonzero((image > nhood_max) & (image > 0))
    else:
        k = np.flatnonzero(image > nhood_max)

    # sort these local maxima into descending order
    image_flat = image.ravel()

    maxima = image_flat[k]

    ks = np.argsort(-maxima)
    k = k[ks]

    if npeaks is not None:
        npks = min(len(k), npeaks)
        k = k[0:npks]

    x, y = np.unravel_index(k, image.shape)
    # xy = np.stack((y, x), axis=0)
    return np.column_stack((y, x, image_flat[k]))

    # interpolate peaks if required
    # if interp:
    #     # TODO see peak2.m, line 87-131
    #     raise ValueError(interp, 'interp not yet supported')
    # else:
    #     xyp = xy
    #     zp = image_flat[k]
    #     ap = []


def peak3(L, npeaks=None):

    # absolute value of Laplacian as a 3D matrix, with scale along axis 2

    # find maxima within all 26 neighbouring pixels
    # create 3x3x3 structuring element and maximum filter
    se_nhood = np.ones((3, 3, 3))
    se_nhood[1, 1, 1] = 0
    eps = np.finfo(np.float64).eps
    maxima = (L > sp.ndimage.maximum_filter(L, footprint=se_nhood, mode='nearest'))

    # find the locations of the minima
    i, j, k = np.nonzero(maxima)
    
    # create result matrix, one row per feature: i, j, k, |L|
    # where k is index into scale
    result = np.column_stack((j, i, k, L[i, j, k]))

    # sort the rows on strength column, descending order
    k = np.argsort(-result[:, 3])
    result = result[k, :]

    if npeaks is not None:
        result = result[:npeaks, :]

    return result

if __name__ == "__main__":

    a = [1, 1, 1, 1, 1]
    print(findpeaks(a))

    a = [5, 1, 1, 1, 1]
    print(findpeaks(a))

    a = [1, 1, 5, 1, 1]
    print(findpeaks(a))

    a = [1, 1, 5, 1, 1]
    print(findpeaks(a, [10, 11, 12, 13, 14]))


    a = [1, 2, 3, 4, 3, 2, 1]
    print(findpeaks(a))

    a.extend(a)
    print(findpeaks(a))

    a = [1, 1, 1, 2, 1, 4, 1, 1, 2]
    print(findpeaks(a))