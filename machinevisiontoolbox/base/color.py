#!/usr/bin/env python
# import io as io
import numpy as np
from spatialmath import base 
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.path as mpath

import urllib.request

from scipy import interpolate
from collections import namedtuple
from pathlib import Path


# TODO
# need to remove references to image class here
# bring col2im from .. into here
# perhaps split out colorimetry and put ..

def blackbody(lam, T):
    """
    Compute blackbody emission spectrum

    :param lam: wavelength 𝜆 [m]
    :type lam: float or array_like
    :param T: blackbody temperature [K]
    :type T: float

    ``blackbody(𝜆, T)`` is the blackbody radiation power density [W/m^3]
    at the wavelength 𝜆 [m] and temperature T [K].

    If 𝜆 is a vector (N,), then the result is a vector (N,) of
    blackbody radiation power density at the corresponding elements of 𝜆.

    Example::

        l = np.linspace(380, 700, 10) * 1e-9  # visible spectrum
        e = blackbody(l, 6500)                # emission of sun
        plt.plot(l, e)

    :references:

        - Robotics, Vision & Control, Section 10.1,
          P. Corke, Springer 2011.
    """

    # physical constants
    c = 2.99792458e8   # m/s         (speed of light)
    h = 6.626068e-34   # m2 kg / s   (Planck's constant)
    k = 1.3806503e-23  # J K-1      (Boltzmann's constant)

    lam = base.getvector(lam)

    e = 2.0 * h * c**2 / (lam**5 * (np.exp(h * c / k / T / lam) - 1))
    if len(e) == 1:
        return e[0]
    else:
        return e


def _loaddata(filename, verbose=False, **kwargs):
    """
    Load data from filename

    :param filename: filename
    :type filename: string
    :param **kwargs: keyword arguments for numpy.genfromtxt
    :type **kwargs: keyword arguments in name, value pairs
    :return: data
    :rtype: numpy array

    ``_loaddata(filename)`` returns ``data`` from ``filename``, otherwise
    returns None

    Example:


    .. note::

        - Comments are assumed to be as original data files were part of the
          MATLAB machine vision toolbox, which can be changed using kwargs.

    """

    if not isinstance(filename, str):
        raise ValueError('filename is not a valid string')

    # reading from a file

    if not ("." in filename):
        filename += '.dat'

    path = Path(filename).expanduser()

    if not path.exists():
        # file doesn't exist, look in MVTB data folder instead

        path = Path(__file__).parent.parent / 'data' / filename

        if not path.exists():
            raise ValueError(f"Cannot open {filename}")

    try:
        # import filename, which we expect to be a .dat file
        # columns for wavelength and spectral data
        # assume column delimiters are whitespace, so for .csv files,
        # replace , with ' '
        with open(path.as_posix()) as file:
            clean_lines = (line.replace(',', ' ') for line in file)
            # default delimiter whitespace
            data = np.genfromtxt(clean_lines, **kwargs)
    except IOError:
        raise ValueError(f"Cannot open {filename}")

    if verbose:
        print(f"_loaddata: {path}, {data.shape}")

    if data is None:
        raise ValueError('Could not read the specified data filename')

    return data


def loadspectrum(lam, filename, verbose=True, method='linear',**kwargs):
    """
    Load spectrum data

    :param lam: wavelength 𝜆 [m]
    :type lam: array_like(n)
    :param filename: filename
    :type filename: str
    :param kwargs**: keyword arguments for scipy.interpolate.interp1d
    :return: interpolated spectral data and corresponding wavelength
    :rtype: ndarray(n)

    ``loadspectrum(𝜆, filename, **kwargs)`` is spectral data (N,D) from file
    filename interpolated to wavelengths [meters] specified in 𝜆 (N).
    The spectral data can be scalar (D=1) or vector (D>1) valued.

    Example:

    .. runblock:: pycon

    .. note::

        - The file is assumed to have its first column as wavelength in metres,
          the remainding columns are linearly interpolated and returned as
          columns of S.
        - The files are kept in the private folder inside the MVTB folder.
        - Default interpolation mode is linear, to change this use ``kind=``
          a string such as "slinear", "quadratic", "cubic", etc.  See
          `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_
          for more info.

    :references:

        - Robotics, Vision & Control, Section 10.1, P. Corke, Springer 2011.
    """

    # check valid input
    lam = base.getvector(lam)
    data = _loaddata(filename, comments='%', verbose=verbose, **kwargs)

    # interpolate data
    data_wavelength = data[0:, 0]
    data_s = data[0:, 1:]

    f = interpolate.interp1d(data_wavelength, data_s, axis=0, kind=method,
                             bounds_error=False, fill_value=0, **kwargs)

    return f(lam)


def lambda2rg(lam, e=None, **kwargs):
    """
    RGB chromaticity coordinates

    :param lam: wavelength 𝜆 [m]
    :type lam: float or array_like
    :param e: illlumination spectrum defined at the wavelengths 𝜆
    :type e: numpy array (N,1)
    :return: rg rg-chromaticity
    :rtype: numpy array, shape (N,2)

    ``lambda2rg(𝜆)`` is the rg-chromaticity coordinate (1,2) for
    illumination at the specific wavelength 𝜆 [m]. If 𝜆 is a
    numpy array (N,1), then P (N,2) is a vector whose elements are the
    chromaticity coordinates at the corresponding elements of 𝜆.

    ``lambda2rg(𝜆, e)`` is the rg-chromaticity coordinate (1,2) for an
    illumination spectrum ``e`` (N,1) defined at corresponding wavelengths
    𝜆 (N,1).

    Example::
        #TODO

    .. note::

        - Data from http://cvrl.ioo.ucl.ac.uk
        - From Table I(5.5.3) of Wyszecki & Stiles (1982). (Table 1(5.5.3) of
          Wyszecki & Stiles (1982) gives the Stiles & Burch functions in
          250 cm-1 steps, while Table I(5.5.3) of Wyszecki & Stiles (1982)
          gives them in interpolated 1 nm steps.).
        - The Stiles & Burch 2-deg CMFs are based on measurements made on
          10 observers. The data are referred to as pilot data, but probably
          represent the best estimate of the 2 deg CMFs, since, unlike the CIE
          2 deg functions (which were reconstructed from chromaticity data),
          they were measured directly.
        - These CMFs differ slightly from those of Stiles & Burch (1955). As
          noted in footnote a on p. 335 of Table 1(5.5.3) of Wyszecki &
          Stiles (1982), the CMFs have been "corrected in accordance with
          instructions given by Stiles & Burch (1959)" and renormalized to
          primaries at 15500 (645.16), 19000 (526.32), and 22500 (444.44) cm-1.

    :references:

        - Robotics, Vision & Control, Section 10.2, P. Corke, Springer 2011.
    """

    # check input
    lam = base.getvector(lam)

    if e is None:
        rgb = cmfrgb(lam, **kwargs)
    else:
        e = base.getvector(e)
        if len(e) != len(lam):
            raise ValueError('number of wavelengths and intensities must match')
        rgb = cmfrgb(lam, e, **kwargs)

    cc = tristim2cc(rgb)
    # r = cc[0:, 0]
    # g = cc[0:, 1]

    return cc[0:, 0:2]


def cmfrgb(lam, e=None, **kwargs):
    """
    RGB color matching function

    :param lam: wavelength 𝜆 [m]
    :type lam: float or array_like
    :param e: illlumination spectrum defined at the wavelengths 𝜆
    :type e: numpy array (N,1)
    :return: rg-chromaticity
    :rtype: numpy array, shape = (N,3)

    ``rgb = cmfrgb(𝜆)`` is the CIE color matching function (N,3)
    for illumination at wavelength 𝜆 (N,1) [m]. If 𝜆 is a vector
    then each row of RGB is the color matching function of the
    corresponding element of 𝜆.

    ``rgb = cmfrgb(𝜆, e)`` is the CIE color matching (1,3) function for an
    illumination spectrum e (N,1) defined at corresponding wavelengths
    𝜆 (N,1).

    Example:

    .. runblock:: pycon

    :references:

        - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
    """

    lam = base.getvector(lam)  # lam is (N,1)

    ret = loadspectrum(lam, 'cmfrgb.dat', **kwargs)

    # approximate rectangular integration
    if e is not None:
        e = base.getvector(e)  # e is a vector Nx1
        e = np.expand_dims(e, 1)
        dlam = lam[1] - lam[0]
        ret = np.dot(e.T, ret.T) / ret.shape[0] * dlam

    return ret


def tristim2cc(tri):
    """
    Tristimulus to chromaticity coordinates

    :param tri: xyz as an array, (N,3) or (N,M,3)
    :type tri: float or array_like
    :return cc: chromaticity coordinates
    :rtype: numpy array, shape = (N,2) or (N,M,2)

    ``tristim2cc(tri)`` is the chromaticity coordinate (1x2) corresponding
    to the tristimulus ``tri`` (1,3). Multiple tristimulus values can be
    given as rows of ``tri`` (N,3), in which case the chromaticity
    coordinates are the corresponding rows (N,2).

    ``tristim2cc(im)`` is the chromaticity coordinates corresponding to every
    pixel in the tristimulus image ``im`` (H,W,3). The return is (H,W,2) that
    has planes corresponding to r and g, or x and y (depending on whether the
    input image was rgb or xyz).

    Example:

    .. runblock:: pycon

    :references:

        - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
    """

    # TODO check if tri is correct shape? can be vector or matrix
    tri = np.array(tri)

    if tri.ndim == 2 and tri.shape[-1] == 3:
        # N x 3 case
        # each row is R G B, or X Y Z
        s = np.sum(tri, axis=1)
        s = base.getvector(s)
        ss = np.stack((s, s), axis=-1)
        cc = tri[0:, 0:2] / ss

    elif tri.ndim == 3 and tri.shape[-1] == 3:
        # N x M x 3 case

        # tri is given as an image
        s = np.sum(tri, axis=2)  
        ss = np.stack((s, s), axis=-1)  # could also use np.tile
        cc = tri[0:, 0:, :2] / ss

    elif base.isvector(tri, 3):
        tri = base.getvector(tri)
        cc = tri[:2] / tri[2]

    else:
        raise ValueError('bad shape input')

    return cc


def lambda2xy(lam, *args):
    """
    XY-chromaticity coordinates for a given wavelength 𝜆 [meters]

    :param lam: wavelength 𝜆 [m]
    :type lam: float or array_like
    :return xy: xy-chromaticity coordinates
    :rtype: numpy array, shape = (N,2)

    ``lambda2xy(𝜆)`` is the xy-chromaticity coordinate (1,2) for
    illumination at the specific wavelength 𝜆 [metres]. If 𝜆 is a
    vector (N,1), then the return is a vector (N,2) whose elements

    Example:

    .. runblock:: pycon

    :references:

        - Robotics, Vision & Control, Section 10.2, P. Corke, Springer 2011.
    """

    # argcheck
    lam = base.getvector(lam)

    cmf = cmfxyz(lam, *args)
    xy = tristim2cc(cmf)

    return xy


def cmfxyz(lam, e=None, **kwargs):
    """
    Color matching function for xyz tristimulus

    :param lam: wavelength 𝜆 [m]
    :type lam: float or array_like
    :param e: illlumination spectrum defined at the wavelengths 𝜆
    :type e: numpy array (N,1)
    :return: xyz-chromaticity
    :rtype: numpy array, shape = (N,3)

    The color matching function is the XYZ tristimulus required to match a
    particular wavelength excitation.

    ``cmfxyz(𝜆)`` is the CIE XYZ color matching function (N,3) for illumination
    at wavelength 𝜆 (N,1) [m].  If 𝜆 is a vector then each row of XYZ
    is the color matching function of the corresponding element of 𝜆.

    ``cmfxzy(𝜆, e)`` is the CIE XYZ color matching (1,3) function for an
    illumination spectrum e (N,1) defined at corresponding wavelengths 𝜆 (N,1).

    Example:

    .. runblock:: pycon

    .. note::

        - CIE 1931 2-deg XYZ CMFs from cvrl.ioo.ucl.ac.uk .

    :references:

        - Robotics, Vision & Control, Chapter 14.3, P. Corke, Springer 2011.
    """

    lam = base.getvector(lam)
    xyz = _loaddata('cmfxyz.dat', comments='%')

    XYZ = interpolate.pchip_interpolate(
        xyz[:, 0], xyz[:, 1:], lam, axis=0, **kwargs)

    if e is not None:
        # approximate rectangular integration
        dlam = lam[1] - lam[0]
        XYZ = e.reshape((1,-1)) @ XYZ * dlam

    return XYZ


def luminos(lam, **kwargs):
    """
    Photopic luminosity function

    :param lam: wavelength 𝜆 [m]
    :type lam: float or array_like
    :return lum: luminosity
    :rtype: numpy array, shape = (N,1)

    ``luminos(𝜆)`` is the photopic luminosity function for the wavelengths in
    𝜆 (N,1) [m]. If 𝜆 is a vector then ``lum`` is a vector whose elements are
    the luminosity at the corresponding 𝜆.

    Example:

    .. runblock:: pycon

    .. note::

        - Luminosity has units of lumens, which are the intensity with which
          wavelengths are perceived by the light-adapted human eye.

    ::references:

        - Robotics, Vision & Control, Chapter 10.1, P. Corke, Springer 2011.

    :seealso: :func:`~rluminos`
    """

    lam = base.getvector(lam)
    data = _loaddata('photopicluminosity.dat', comments='%')

    flum = interpolate.interp1d(data[0:, 0], data[0:, 1],
                                bounds_error=False, fill_value=0, **kwargs)
    lum = flum(lam)

    return lum  # photopic luminosity is the Y color matching function


def rluminos(lam, **kwargs):
    """
    Relative photopic luminosity function

    :param lam: wavelength 𝜆 [m]
    :type lam: float or array_like
    :return lum: relative luminosity
    :rtype: numpy array, shape = (N,1)

    ``rluminos(𝜆)`` is the relative photopic luminosity function for the
    wavelengths in 𝜆 (N,1) [m]. If 𝜆 is a vector then ``p`` is a vector
    whose elements are the luminosity at the corresponding 𝜆.

    Example:

    .. runblock:: pycon

    .. note::

        - Relative luminosity lies in t he interval 0 to 1, which indicate the
          intensity with which wavelengths are perceived by the light-adapted
          human eye.

    :references:

        - Robotics, Vision & Control, Chapter 10.1, P. Corke, Springer 2011.
    """

    lam = base.getvector(lam)
    xyz = cmfxyz(lam, **kwargs)
    return xyz[0:, 1]  # photopic luminosity is the Y color matching function


def ccxyz(lam, e=None):
    """
    Chromaticity coordinates

    :param lam: wavelength 𝜆 [m]
    :type lam: float or array_like
    :param e: illlumination spectrum defined at the wavelengths 𝜆
    :type e: numpy array (N,1)
    :return xyz: xyz-chromaticity coordinates
    :rtype: numpy array, shape = (N,3)

    Example:

    .. runblock:: pycon

    :references:

        - Robotics, Vision & Control, Chapter 14.3, P. Corke, Springer 2011.
    """

    lam = base.getvector(lam)
    xyz = cmfxyz(lam)

    if e is None:
        cc = xyz / (np.sum(xyz, axis=1) * np.ones((3, 1))).T
    else:
        e = base.getvector(e)
        xyz = xyz / (e * np.ones((1, 3)))
        xyz = np.sum(xyz)
        cc = xyz / (np.sum(xyz) * np.ones((1, 3)))

    return cc


def _loadrgbdict(fname):
    """
    Load file as rgb dictionary

    :param fname: filename
    :type fname: string
    :return: rgbdict
    :rtype: dictionary

    ``_loadrgbdict(fname)`` returns ``rgbdict`` from ``fname``, otherwise
    returns Empty

    .. note::
    
        - Assumes the file is organized as four columns: R, G, B, name.
        - Color names are converted to lower case
        - # comment lines and blank lines are ignored
        - Values in the range [0,255] are mapped to [0,1.0]


    """

    if not isinstance(fname, str):
        raise ValueError(fname, 'file name must be a string')

    data = _loaddata(fname, comments='#',
                     dtype=None, encoding='ascii')

    # result is an ndarray of tuples
    # convert data to a dictionary
    rgbdict = {}
    for *rgb, name in data:
        rgbdict[str(name).lower()] = [x / 255.0 for x in rgb]

    return rgbdict


_rgbdict = None

def color_bgr(color):
    rgb = colorname(color)
    return [int(x * 255) for x in reversed(rgb)]

def colorname(arg, colorspace='rgb'):
    """
    Map between color names and RGB values

    :param name: name of a color or name of a 3-element color array
    :type name: string or (numpy, tuple, list)
    :param colorspace: name of colorspace (eg 'rgb' or 'xyz' or 'xy' or 'ab')
    :type colorspace: string
    :return out: output
    :rtype out: named tuple, name of color, numpy array in colorspace

    - ``name`` is a string/list/set of color names, then ``colorname`` returns
      a 3-tuple of rgb tristimulus values.

    Example:

    .. runblock:: pycon

    .. note::

        - Color name may contain a wildcard, eg. "?burnt"
        - Based on the standard X11 color database rgb.txt
        - Tristiumuls values are [0,1]

    :references:

        - Robotics, Vision & Control, Chapter 14.3, P. Corke, Springer 2011.
    """
    # I'd say str in, 3 tuple out, or 3-element array like (numpy, tuple, list)
    #  in and str out

    # load rgbtable (rbg.txt as a dictionary)
    global _rgbdict

    if _rgbdict is None:
        _rgbdict = _loadrgbdict('rgb.txt')

    if isinstance(arg, str) or base.islistof(arg, str):
        # string, or list of strings
        if isinstance(arg, str):
            return _rgbdict[arg]
        else:
            return [_rgbdict[name] for name in arg]

    elif isinstance(arg, (np.ndarray, tuple, list)):
        # map numeric tuple to color name

        n = np.array(arg).flatten()  # convert tuple or list into np array
        table = np.vstack([rgb for rgb in _rgbdict.values()])

        if colorspace in ('rgb', 'xyz', 'lab'):
            if len(n) != 3:
                raise ValueError('color value must have 3 elements')
            if colorspace in ('xyz', 'lab'):
                table = colorconvert(table, 'rgb', colorspace)
            dist = np.linalg.norm(table - n, axis=1)
            k = np.argmin(dist)
            return list(_rgbdict.keys())[k]

        elif colorspace in ('xy', 'ab'):
            if len(n) != 2:
                raise ValueError('color value must have 2 elements')

            if colorspace == 'xy':
                table = colorconvert(table, 'rgb', 'xyz')
                with np.errstate(divide='ignore',invalid='ignore'):
                    table = table[:,0:2] / np.tile(np.sum(table, axis=1), (2,1)).T
            elif colorspace == 'ab':
                table = colorconvert(table, 'rgb', 'Lab')
                table = table[:,1:3]
            
            dist = np.linalg.norm(table - n, axis=1)
            k = np.nanargmin(dist)
            return list(_rgbdict.keys())[k]
        else:
            raise ValueError('unknown colorspace')
    else:
        raise ValueError('arg is of unknown type')

def showcolorspace(cs='xy', N=501, L=90, *args):
    """
    Display spectral locus

    :param cs: 'xy', 'lab', 'ab' or None defines which colorspace to show
    :type xy: string
    :param N: number of points to sample in the x- and y-directions
    :type N: integer, N > 0, default 501
    :param L: length of points to sample for Lab colorspace
    :type L: integer, L > 0, default 90
    :return color: colorspace image
    :rtype color: Image instance

    TODO for now, just return Image of plot

    Example:

    .. runblock:: pycon

    .. note::

        - The colors shown within the locus only approximate the true
            colors, due to the gamut of the display device.

    :references:

        - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
    """

    # TODO check valid inputs
    # TODO cslist = [None, 'xy', 'ab', 'Lab']
    # which should be defined by cases (showcolorspace.m)

    if not isinstance(cs, str):
        raise ValueError(cs, 'cs must be a string')

    if cs == 'xy':
        #   create axes
        #   create meshgrid
        #   convert xyY to XYZ
        #   convert XYZ to RGB (requires colorspace function)
        #   define boundary
        ex = 0.8
        ey = 0.9
        Nx = round(N * ex)
        Ny = round(N * ey)
        e = 0.01
        # generate colors in xyY color space
        ax = np.linspace(e, ex - e, Nx)
        ay = np.linspace(e, ey - e, Ny)
        xx, yy = np.meshgrid(ax, ay)
        iyy = 1.0 / (yy + 1e-5 * (yy == 0).astype(float))

        # convert xyY to XYZ
        Y = np.ones((Ny, Nx))
        X = Y * xx * iyy
        Z = Y * (1.0 - xx - yy) * iyy
        XYZ = np.dstack((X, Y, Z))

        # TODO replace with color.colorspace(im,conv,**kwargs)
        # (replace starts here)
        # NOTE using cv.COLOR_XYZ2RGB does not seem to work properly
        # it does not do gamma corrections

        XYZ = mvt.Image(XYZ)  # TODO
        BGR = XYZ.colorspace('xyz2bgr')  # TODO

        # define the boundary
        nm = 1e-9
        lam = np.arange(400, 700, step=5) * nm
        xyz = ccxyz(lam)

        xy = xyz[0:, 0:2]

        # make a smooth boundary with spline interpolation
        irange = np.arange(0, xy.shape[0]-1, step=0.1)
        drange = np.linspace(0, xy.shape[0]-1, xy.shape[0])
        fxi = interpolate.interp1d(drange, xy[:, 0], kind='cubic')
        fyi = interpolate.interp1d(drange, xy[:, 1], kind='cubic')
        xi = fxi(irange)
        yi = fyi(irange)
        # add the endpoints
        xi = np.append(xi, xi[0])
        yi = np.append(yi, yi[0])

        # determine which points from xx, yy, are contained within polygon
        # defined by xi, yi
        p = np.stack((xi, yi), axis=-1)
        polypath = mpath.Path(p)

        xxc = xx.flatten('F')
        yyc = yy.flatten('F')
        pts_in = polypath.contains_points(np.stack((xxc, yyc), axis=-1))
        # same for both xx and yy
        colors_in = np.reshape(pts_in, xx.shape, 'F')
        # colors_in_yy = pts_in.reshape(yy.shape)

        # set outside pixels to white
        colorsin3 = np.dstack((colors_in, colors_in, colors_in))
        BGR.image[colorsin3 == 0] = 1.0

        out = BGR.rgb
        labels = ('x', 'y')

    elif cs in ('ab', 'Lab'):
        ax = np.linspace(-100, 100, N)
        ay = np.linspace(-100, 100, N)
        aa, bb = np.meshgrid(ax, ay)

        # convert from Lab to RGB
        avec = base.getvector(aa)
        bvec = base.getvector(bb)

        Lab = np.stack((L * np.ones(avec.shape), avec, bvec), axis=1)
        # TODO currently does not work. OpenCV
        # out = cv.cvtColor(Lab, cv.COLOR_Lab2BGR)

        Lab = mvt.Image(Lab)  # TODO

        BGR = Lab.colorspace('Lab2bgr')  # TODO

        bgr2d = np.squeeze(BGR.image)
        from machinevisiontoolbox.Image import col2im  # TODO
        out = col2im(bgr2d, [N, N])  # TODO 
        out = mvt.Image(out)  # TODO 
        out = out.float()
        out = out.pixelswitch(BGR.kcircle(np.floor(N / 2)),
                                np.r_[1.0, 1.0, 1.0])
    else:
        raise ValueError('no or unknown color space provided')

    _, ax = plt.subplots()
    ax.imshow(np.flipud(out), extent=(0, ex, 0, ey))
    ax.grid(True)
    # ax.invert_yaxis()
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(cs + ' colorspace ')
    xy_addticks(ax)
    return ax

def rg_addticks(ax):
    """
    Label spectral locus

    :param ax: axes reference for plotting
    :type ax: Matplotlib.pyplot axes object

    ``rg_addticks(ax)`` adds wavelength ticks to the spectral locus.
    """

    # well-spaced points around the locus
    lam = np.arange(460, 550, 10)
    lam = np.hstack((lam, np.arange(560, 620, 20)))

    rgb = cmfrgb(lam * 1e-9)
    r = rgb[0:, 0] / np.sum(rgb, axis=1)
    g = rgb[0:, 1] / np.sum(rgb, axis=1)

    ax.plot(r, g, 'ko')

    for i in range(len(lam)):
        ax.text(r[i], g[i], '  {0}'.format(lam[i]))

def xy_addticks(ax):
    """
    Label spectral locus

    :param ax: axes reference for plotting
    :type ax: Matplotlib.pyplot axes object

    ``rg_addticks(ax)`` adds wavelength ticks to the spectral locus.
    """

    # well-spaced points around the locus
    lam = np.arange(460, 550, 10)
    lam = np.hstack((lam, np.arange(560, 620, 20)))

    xyz = cmfxyz(lam * 1e-9)
    x = xyz[0:, 0] / np.sum(xyz, axis=1)
    y = xyz[0:, 1] / np.sum(xyz, axis=1)

    ax.plot(x, y, 'ko')

    for i in range(len(lam)):
        ax.text(x[i], y[i], '  {0}'.format(lam[i]))

def cie_primaries():
    """
    Define CIE primary colors

    ``cie_primaries`` is a 3-vector with the wavelengths [m] of the
    IE 1976 red, green and blue primaries respectively.

    """
    return np.array([700, 546.1, 435.8]) * 1e-9

def colorconvert(image, src, dst):

    flag = _convertflag(src, dst)

    if isinstance(image, np.ndarray) and image.ndim == 3:
        # its a color image
        return cv.cvtColor(image, flag)
    elif base.ismatrix(image, (None, 3)):
        # not an image, see if it's Nx3
        image = base.getmatrix(image, (None, 3), dtype=np.float32)
        image = image.reshape((-1, 1, 3))
        return cv.cvtColor(image, flag).reshape((-1, 3))

def _convertflag(src, dst):

    if src == 'rgb':
        if dst in ('grey', 'gray'):
            return cv.COLOR_RGB2GRAY
        elif dst in ('xyz', 'xyz_709'):
            return cv.COLOR_RGB2XYZ
        elif dst == 'ycrcb':
            return cv.COLOR_RGB2YCrCb
        elif dst == 'hsv':
            return cv.COLOR_RGB2HSV
        elif dst == 'hls':
            return cv.COLOR_RGB2HLS
        elif dst == 'lab':
            return cv.COLOR_RGB2Lab
        elif dst == 'luv':
            return cv.COLOR_RGB2Luv
        else:
            raise ValueError(f"destination colorspace {dst} not known")
    elif src == 'bgr':
        if dst in ('grey', 'gray'):
            return cv.COLOR_BGR2GRAY
        elif dst in ('xyz', 'xyz_709'):
            return cv.COLOR_BGR2XYZ
        elif dst == 'ycrcb':
            return cv.COLOR_BGR2YCrCb
        elif dst == 'hsv':
            return cv.COLOR_BGR2HSV
        elif dst == 'hls':
            return cv.COLOR_BGR2HLS
        elif dst == 'lab':
            return cv.COLOR_BGR2Lab
        elif dst == 'luv':
            return cv.COLOR_BGR2Luv
        else:
            raise ValueError(f"destination colorspace {dst} not known")
    elif src in ('xyz', 'xyz_709'):
        if dst == 'rgb':
            return cv.COLOR_XYZ2RGB
        elif dst == 'bgr':
            return cv.COLOR_XYZ2BGR
        else:
            raise ValueError(f"destination colorspace {dst} not known")
    elif src == 'ycrcb':
        if dst == 'rgb':
            return cv.COLOR_YCrCb2RGB
        elif dst == 'bgr':
            return cv.COLOR_YCrCbBGR
        else:
            raise ValueError(f"destination colorspace {dst} not known")
    elif src == 'hsv':
        if dst == 'rgb':
            return cv.COLOR_HSVRGB
        elif dst == 'bgr':
            return cv.COLOR_HSV2BGR
        else:
            raise ValueError(f"destination colorspace {dst} not known")
    elif src == 'hls':
        if dst == 'rgb':
            return cv.COLOR_HLS2RGB
        elif dst == 'bgr':
            return cv.COLOR_HLS2BGR
        else:
            raise ValueError(f"destination colorspace {dst} not known")
    elif src == 'lab':
        if dst == 'rgb':
            return cv.COLOR_Lab2RGB
        elif dst == 'bgr':
            return cv.COLOR_Lab2BGR
        else:
            raise ValueError(f"destination colorspace {dst} not known")
    elif src == 'luv':
        if dst == 'rgb':
            return cv.COLOR_Luv2RGB
        elif dst == 'bgr':
            return cv.COLOR_Luv2BGR
        else:
            raise ValueError(f"destination colorspace {dst} not known")
    else:
        raise ValueError(f"source colorspace {src} not known")

def gamma_encode(image, gamma):
    """
    Inverse gamma correction

    :param image: input image
    :type image: ndarray(h,w) or ndarray(h,w,n)
    :param gamma: string identifying srgb, or scalar to raise the image power
    :type gamma: string or float TODO: variable input seems awkward
    :return out: gamma corrected version of image
    :rtype out: ndarray(h,w) or ndarray(h,w,n)

    - ``gamma_encode(image, gamma)`` maps linear tristimulus values to a gamma encoded 
      image.

    Example:

    .. runblock:: pycon

    .. note::

        - Gamma decoding should be applied to any color image prior to
            colometric operations.
        - The exception to this is colorspace conversion using COLORSPACE
            which expects RGB images to be gamma encoded.
        - Gamma encoding is typically performed in a camera with
            GAMMA=0.45.
        - Gamma decoding is typically performed in the display with
            GAMMA=2.2.
        - For images with multiple planes the gamma correction is applied
            to all planes.
        - For images sequences the gamma correction is applied to all
            elements.
        - For images of type double the pixels are assumed to be in the
            range 0 to 1.
        - For images of type int the pixels are assumed in the range 0 to
            the maximum value of their class.  Pixels are converted first to
            double, processed, then converted back to the integer class.

    :references:

        - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
    """

    if not (base.isscalar(gamma) or isinstance(gamma, str)):
        raise ValueError('gamma must be string or scalar')

    if gamma == 'srgb':

        imagef = ifloat(image)

        if imagef.ndims == 2:
            # greyscale
            return _srgb(imagef.image)
        elif imagef.ndims == 3:
            # multi-dimensional
            out = np.alloc(imagef.shape, dtype=imagef.dtype)
            for p in range(imagef.ndims):
                out[:,:,p] = _srgb(imagef[:,:,p])
        else:
            raise ValueError('expecting 2d or 3d image')

        if np.issubdtype(image.dtype, np.floating):
            # original image was float, convert back
            return iint(out)

    else:
        # normal power law:
        # import code
        # code.interact(local=dict(globals(), **locals()))
        if np.issubdtype(image.dtype, np.float):
            return image ** gamma
        else:
            # int image
            maxg = np.float32((np.iinfo(image.dtype).max))
            return ((image.astype(np.float32) / maxg) ** gamma) * maxg

def gamma_decode(image, gamma):
    """
    Gamma decoding

    :param image: input image
    :type image: ndarray(h,w) or ndarray(h,w,n)
    :param gamma: string identifying srgb, or scalar to raise the image power
    :type gamma: string or float TODO: variable input seems awkward
    :return out: gamma corrected version of image
    :rtype out: ndarray(h,w) or ndarray(h,w,n)

    - ``gamma_decode(image, gamma)`` is the image with an inverse gamma correction based
        on ``gamma`` applied.

    Example:

    .. runblock:: pycon

    .. note::

        - Gamma decoding should be applied to any color image prior to
            colometric operations.
        - The exception to this is colorspace conversion using COLORSPACE
            which expects RGB images to be gamma encoded.
        - Gamma encoding is typically performed in a camera with
            GAMMA=0.45.
        - Gamma decoding is typically performed in the display with
            GAMMA=2.2.
        - For images with multiple planes the gamma correction is applied
            to all planes.
        - For images sequences the gamma correction is applied to all
            elements.
        - For images of type double the pixels are assumed to be in the
            range 0 to 1.
        - For images of type int the pixels are assumed in the range 0 to
            the maximum value of their class.  Pixels are converted first to
            double, processed, then converted back to the integer class.

    :references:

        - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
    """

    if not (base.isscalar(gamma) or isinstance(gamma, str)):
        raise ValueError('gamma must be string or scalar')

    if gamma == 'srgb':

        imagef = ifloat(image)

        if imagef.ndims == 2:
            # greyscale
            return _srgb_inv(imagef.image)
        elif imagef.ndims == 3:
            # multi-dimensional
            out = np.alloc(imagef.shape, dtype=imagef.dtype)
            for p in range(imagef.ndims):
                out[:,:,p] = _srgb_inv(imagef[:,:,p])
        else:
            raise ValueError('expecting 2d or 3d image')

        if np.issubdtype(image.dtype, np.float):
            # original image was float, convert back
            return iint(out)

    else:

        # normal power law:
        if np.issubdtype(image.dtype, np.float):
            return image ** (1.0 / gamma)
        else:
            # int image
            maxg = np.float32((np.iinfo(image.dtype).max))
            return ((image.astype(np.float32) / maxg) ** (1 / gamma)) * maxg # original
            # return ((image.astype(np.float32) / maxg) ** gamma) * maxg
        


    def _srgb_inverse(self, Rg):
        """
        Inverse sRGB gamma correction

        :param Rg: 2D image
        :type Rg: numpy array, shape (N,M)
        :return: R
        :rtype: numpy array

        - ``_srgb_imverse(Rg)`` maps an sRGB gamma encoded image to linear
          tristimulus values.

        Example:

        .. runblock:: pycon

        .. note::

            - Based on code from Pascal Getreuer 2005-2010
            - And colorspace.m from Peter Corke's Machine Vision Toolbox
        """

        R = np.alloc(Rg.shape, dtype=np.float32)
        a = 0.0404482362771076
        i = np.where(Rg <= a)
        noti = np.where(Rg > a)
        R[i] = Rg[i] / 12.92
        R[noti] = np.real(((Rg[noti] + 0.055) / 1.055) ** 2.4)
        return R

    def _srgb(self, R):
        """
        sRGB Gamma correction

        :param R: 2D image
        :type R: numpy array, shape (N,M)
        :return: Rg
        :rtype: numpy array

        - ``_srgb(R)`` maps linear tristimulus values to an sRGB gamma encoded 
          image.

        Example:

        .. runblock:: pycon

        .. note::

            - Based on code from Pascal Getreuer 2005-2010
            - And colorspace.m from Peter Corke's Machine Vision Toolbox
        """

        Rg = np.alloc(R.shape, dtype=np.float32)
        a = 0.0031306684425005883
        b = 0.416666666666666667
        i = np.where(R <= a)
        noti = np.where(R > a)
        Rg[i] = R[i] * 12.92
        Rg[noti] = np.real(1.055 * (R[noti] ** b) - 0.055)
        return Rg

if __name__ == '__main__':  # pragma: no cover

    # import pathlib
    # import os.path

    # exec(open(os.path.join(pathlib.Path(__file__).parent.absolute(),
    # "test_color.py")).read())
    # import machinevisiontoolbox.color as color

    # rg = color.lambda2rg(555e-9)
    # print(rg)

    # wcc = color.tristim2cc(np.r_[1, 1, 1])
    # print(wcc)

    # import code
    # code.interact(local=dict(globals(), **locals()))

    print(colorname('red'))
    img = np.float32(np.r_[0.5, 0.2, 0.1]).reshape((1,1,3))
    print(img.shape)
    # print(cv.cvtColor(img, cv.COLOR_RGB2HSV))
    print(cv.cvtColor(img, _convertflag('rgb', 'hsv')))
    print(colorname([0.5,0.2, 0.5]))
    print(colorname([0.5,0.2], 'xy'))


    rg = lambda2rg(lam=np.array([555e-9, 666e-9]),
                             e=np.array([4, 2]))