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