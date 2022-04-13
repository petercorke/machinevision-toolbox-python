#!/usr/bin/env python
# import io as io
from machinevisiontoolbox.base.data import mvtb_path_to_datafile
import numpy as np
import re
from spatialmath import base 
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as colors
import warnings

import urllib.request

from scipy import interpolate
from collections import namedtuple
from pathlib import Path
from machinevisiontoolbox.base.types import float_image, int_image

# TODO
# need to remove references to image class here
# bring col2im from .. into here
# perhaps split out colorimetry and put ..

def blackbody(λ, T):
    """
    Compute blackbody emission spectrum

    :param λ: wavelength 𝜆 [m]
    :type λ: float or array_like
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

    λ = base.getvector(λ)

    e = 2.0 * h * c**2 / (λ**5 * (np.exp(h * c / k / T / λ) - 1))
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
    path = mvtb_path_to_datafile(filename, folder='data')

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

_spectra = {}

def loadspectrum(λ, filename, verbose=False, method='linear', **kwargs):
    """
    Load spectrum data

    :param λ: wavelength 𝜆 [m]
    :type λ: array_like(n)
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
        - The files are kept in the private folder inside the MVTB folder with
          extension .dat
        - Default interpolation mode is linear, to change this use ``kind=``
          a string such as "slinear", "quadratic", "cubic", etc.  See
          `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_
          for more info.

    :references:

        - Robotics, Vision & Control, Section 10.1, P. Corke, Springer 2011.
    """
    global _spectra

    if filename not in _spectra:
        # save an interpolator for every spectrum
        _spectra[filename] = _loaddata(filename + '.dat',
            comments='%', verbose=verbose, **kwargs)

    # check valid input
    λ = base.getvector(λ)
    
    # interpolate data
    data = _spectra[filename]
    f = interpolate.interp1d(data[:, 0], data[:, 1:],
                        axis=0, kind=method, 
                        bounds_error=False, 
                        fill_value=0, **kwargs)

    spectrum = f(λ)
    if spectrum.shape[1] == 1:
        return spectrum.flatten()
    else:
        return spectrum


def lambda2rg(λ, e=None, **kwargs):
    """
    RGB chromaticity coordinates

    :param λ: wavelength 𝜆 [m]
    :type λ: float or array_like
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
    λ = base.getvector(λ)

    if e is None:
        rgb = cmfrgb(λ, **kwargs)
    else:
        e = base.getvector(e)
        if len(e) != len(λ):
            raise ValueError('number of wavelengths and intensities must match')
        rgb = cmfrgb(λ, e, **kwargs)

    cc = tristim2cc(rgb)

    if cc.shape[0] == 1:
        return cc[0, :]
    else:
        return cc

def cmfrgb(λ, e=None, **kwargs):
    """
    RGB color matching function

    :param λ: wavelength 𝜆 [m]
    :type λ: array_like(n)
    :param e: illlumination spectrum defined at the wavelengths 𝜆
    :type e: array_like(n)
    :return: rg-chromaticity
    :rtype: ndarray(n,3)

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

    λ = base.getvector(λ)  # λ is (N,1)

    cmf = loadspectrum(λ, 'cmfrgb', **kwargs)
    # approximate rectangular integration
    # assume steps are equal sized
    if e is not None:
        e = base.getvector(e, out='row')  # e is a vector Nx1
        dλ = λ[1] - λ[0]
        ret = (e @ cmf) / cmf.shape[0] * dλ
    else:
        ret = cmf

    if ret.shape[0] == 1:
        ret = ret[0, :]
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
        s = base.getvector(s)  # ?? TODO
        ss = np.stack((s, s), axis=-1)
        cc = tri[0:, 0:2] / ss

        #  / np.sum(XYZ, axis=1)[..., np.newaxis];

    elif tri.ndim == 3 and tri.shape[-1] == 3:
        # N x M x 3 case

        # tri is given as an image
        s = np.sum(tri, axis=2)  
        ss = np.stack((s, s), axis=-1)  # could also use np.tile
        cc = tri[0:, 0:, :2] / ss

    elif base.isvector(tri, 3):
        tri = base.getvector(tri)
        cc = tri[:2] / np.sum(tri)

    else:
        raise ValueError('bad shape input')

    return cc


def lambda2xy(λ, *args):
    """
    XY-chromaticity coordinates for a given wavelength 𝜆 [meters]

    :param λ: wavelength 𝜆 [m]
    :type λ: array_like(n) or float
    :return xy: xy-chromaticity coordinates
    :rtype: nudarray(N,2) or ndarray(2)

    ``lambda2xy(𝜆)`` is the xy-chromaticity coordinate (1,2) for
    illumination at the specific wavelength 𝜆 [metres]. If 𝜆 is a
    vector (N,1), then the return is a vector (N,2) whose elements

    Example:

    .. runblock:: pycon

    :references:

        - Robotics, Vision & Control, Section 10.2, P. Corke, Springer 2011.
    """

    # argcheck
    λ = base.getvector(λ)

    cmf = cmfxyz(λ, *args)
    xy = tristim2cc(cmf)

    if xy.shape[0] == 1:
        return xy[0, :]
    else:
        return xy

def cmfxyz(λ, e=None, **kwargs):
    """
    Color matching function for xyz tristimulus

    :param λ: wavelength 𝜆 [m]
    :type λ: float or array_like
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
    λ = base.getvector(λ)

    cmfxyz = loadspectrum(λ, 'cmfxyz')

    if e is not None:
        # approximate rectangular integration
        dλ = λ[1] - λ[0]
        XYZ = e.reshape((1,-1)) @ cmfxyz * dλ
        return XYZ
    else:
        return cmfxyz

def luminos(λ, **kwargs):
    """
    Photopic luminosity function

    :param λ: wavelength 𝜆 [m]
    :type λ: float or array_like
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
    λ = base.getvector(λ)

    luminos = loadspectrum(λ, 'photopicluminosity')

    return luminos * 683  # photopic luminosity is the Y color matching function


def rluminos(λ, **kwargs):
    """
    Relative photopic luminosity function

    :param λ: wavelength 𝜆 [m]
    :type λ: float or array_like
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

    λ = base.getvector(λ)
    xyz = cmfxyz(λ, **kwargs)
    return xyz[:, 1]  # photopic luminosity is the Y color matching function


def ccxyz(λ, e=None):
    """
    Chromaticity coordinates

    :param λ: wavelength 𝜆 [m]
    :type λ: float or array_like
    :param e: illlumination spectrum defined at the wavelengths 𝜆
    :type e: numpy array (N,1)
    :return xyz: xyz-chromaticity coordinates
    :rtype: numpy array, shape = (N,3)

    Example:

    .. runblock:: pycon

    :references:

        - Robotics, Vision & Control, Chapter 14.3, P. Corke, Springer 2011.
    """

    λ = base.getvector(λ)
    xyz = cmfxyz(λ)

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

def name2color(name, colorspace='RGB'):
    """
    Map color name to value

    :param name: name of a color
    :type name: str
    :param colorspace: name of colorspace (eg 'rgb' or 'xyz' or 'xy' or 'ab')
    :type colorspace: string
    :return out: color tristimulus value
    :rtype out: ndarray(3) or ndarray(2)

    Looks up the RGB tristimulus for this color using ``matplotlib.colors`` and
    converts it to the desired ``colorspace``.  RGB tristimulus values are in
    the range [0,1].

    If a Python-style regexp is passed, then the return value is a list
    of matching colors.

    Example:

    .. runblock:: pycon

        >>> name2color('r')
        >>> name2color('r', 'xy')
        >>> name2color('lime green)
        >>> name2color('.*burnt.*')

    :references:

        - Robotics, Vision & Control, Chapter 14.3, P. Corke, Springer 2011.
    
    :seealso: :func:`color2name`
    """
    colorspace = colorspace.lower()

    def csconvert(name, cs):

        rgb = colors.to_rgb(name)

        if cs == 'rgb':
            return np.r_[rgb]
        elif cs in ('xyz', 'lab', 'l*a*b*'):
            return colorspace_convert(rgb, 'rgb', cs)
        elif cs == 'xy':
            xyz = colorspace_convert(rgb, 'rgb', 'xyz')
            return xyz[:2] / np.sum(xyz)
        elif cs == 'ab':
            Lab = colorspace_convert(rgb, 'rgb', 'lab', 'l*a*b*')
            return Lab[1:]
        else:
            raise ValueError('unknown colorspace')

    if any([c in ".?*" for c in name]):
        # has a wildcard
        return list(filter(re.compile(name).match, [key for key in colors.get_named_colors_mapping().keys()]))
    else:
        try:
            return csconvert(name, colorspace)
        except ValueError:
            return None

def color2name(color, colorspace='RGB'):
    """
    Map color value to color name

    :param color: color value
    :type color: array_like(3) or array_like(2)
    :param colorspace: name of colorspace (eg 'rgb' or 'xyz' or 'xy' or 'ab')
    :type colorspace: string
    :return out: color name
    :rtype out: str

    Converts the given value from the specified ``colorspace`` to RGB and finds
    the closest value in ``matplotlib.colors``.

    Example:

    .. runblock:: pycon

        >>> color2name(([0 ,0, 1]))
        >>> color2name((0.2, 0.3), 'xy')

    .. note::

        - Color name may contain a wildcard, eg. "?burnt"
        - Based on the standard X11 color database rgb.txt
        - Tristiumuls values are [0,1]

    :references:

        - Robotics, Vision & Control, Chapter 14.3, P. Corke, Springer 2011.

    :seealso: :func:`name2color`
    """

    # map numeric tuple to color name
    colorspace = colorspace.lower()

    color = np.array(color).flatten()  # convert tuple or list into np array
    table = np.vstack([colors.to_rgb(color) for color in colors.get_named_colors_mapping().keys()])

    if colorspace in ('rgb', 'xyz', 'lab', 'l*a*b*'):
        if len(color) != 3:
            raise ValueError('color value must have 3 elements')
        if colorspace in ('xyz', 'lab'):
            table = colorspace_convert(table, 'rgb', colorspace)
        dist = np.linalg.norm(table - color, axis=1)
        k = np.argmin(dist)
        return list(colors.get_named_colors_mapping())[k]

    elif colorspace in ('xy', 'ab', 'a*b*'):
        if len(color) != 2:
            raise ValueError('color value must have 2 elements')

        if colorspace == 'xy':
            table = colorspace_convert(table, 'rgb', 'xyz')
            with np.errstate(divide='ignore',invalid='ignore'):
                table = table[:,0:2] / np.tile(np.sum(table, axis=1), (2,1)).T
        elif colorspace in ('ab', 'a*b*'):
            table = colorspace_convert(table, 'rgb', 'Lab')
            table = table[:,1:3]
        
        dist = np.linalg.norm(table - color, axis=1)
        k = np.nanargmin(dist)
        return list(colors.get_named_colors_mapping())[k]
    else:
        raise ValueError('unknown colorspace')


def colorname(arg, colorspace='RGB'):
    raise DeprecationWarning('please use name2color or color2name')


_white = {
    'd65': [0.3127, 0.3290],  #D65 2 deg
    'e':   [0.33333, 0.33333]  # E
}

_xy_primaries = {
    'itu-709': np.array([
        [0.64, 0.33],
        [0.30, 0.60],
        [0.15, 0.06]]),
    'cie': np.array([
        [0.6400, 0.3300], 
        [0.3000, 0.6000], 
        [0.1500, 0.0600]]),
    'srgb': np.array([
        [0.6400, 0.3300], 
        [0.3000, 0.6000],
        [0.1500, 0.0600]])
}

def XYZ2RGBxform(white='D65', primaries='sRGB'):
    """
    Transformation matrix from XYZ to RGB colorspace

    :param white: illuminant: 'E' or 'D65' [default]
    :type white: str
    :param primaries: xy coordinates of primaries to use: 'CIE', ITU=709' or
        'sRGB' [default]
    :type primaries: str
    :raises ValueError: bad 

    :return: [description]
    :rtype: [type]

    .. note::
    
        - Use the inverse of the transform for RGB to XYZ.
        - Works with linear RGB colorspace, not gamma encoded
    """
    
    if isinstance(white, str):
        try:
            white = _white[white.lower()]
        except:
            raise ValueError('unknown white value, must be one of'
                ', '.join(_white.keys()))
    else:
        white = base.getvector(white, 2)

    if isinstance(primaries, str):
        try:
            primaries = _xy_primaries[primaries.lower()]
        except:
            raise ValueError('unknown primary value, must be one of'
                ', '.join(_xy_primaries.keys()))
    else:
        white = base.getmatrix(primaries, (3,2))

    def column(primaries, i):
        primaries = base.getmatrix(primaries, (None, 2))
        return np.array([
            primaries[i,0] / primaries[i,1],
            1,
            (1 - primaries[i,0] - primaries[i,1]) / primaries[i,1]
        ])

    # build the columns of the inverse transform
    Xr = column(primaries, 0)
    Xg = column(primaries, 1)
    Xb = column(primaries, 2)
    M = np.array([Xr, Xg, Xb]).T

    # determine the white point
    Xw = column(white, 0)
    J = np.linalg.inv(M) @ Xw
    M = np.array([Xr, Xg, Xb]).T @ np.diag(J)

    return M

def xy_chromaticity_diagram(N = 500, Y=1):
    ex = 0.8
    ey = 0.9
    e0 = 0.0

    Nx = round(N * (ex - e0))
    Ny = round(N * (ey - e0))
    # generate colors in xyY color space
    x, y = np.meshgrid(np.linspace(e0, ex, Nx), np.linspace(e0, ey, Ny))
    # hack to prevent divide by zero errors
    y[0,:] = 1e-3

    # convert xyY to XYZ
    Y = np.ones((Ny, Nx)) * Y
    X = Y * x / y
    Z = Y * (1.0 - x - y) /  y
    XYZ = np.dstack((X, Y, Z)).astype(np.float32)

    RGB = colorspace_convert(XYZ, 'xyz', 'rgb')
    RGB = _normalize(RGB)  # fit to interval [0, 1]
    RGB = gamma_encode(RGB)  # gamma encode

    # define the spectral locus boundary as xy points, Mx2 matrix
    nm = 1e-9
    λ = np.arange(400, 700, step=5) * nm
    xyz = ccxyz(λ)
    xy_locus = xyz[:, :2]

    ## make a smooth boundary with spline interpolation

    # set up interpolators for x and y
    M = xy_locus.shape[0]
    drange = np.arange(M)
    fxi = interpolate.interp1d(drange, xy_locus[:, 0], kind='cubic')
    fyi = interpolate.interp1d(drange, xy_locus[:, 1], kind='cubic')

    # interpolate in steps of 0.1 
    irange = np.arange(0, M - 1, step=0.1)
    xi = fxi(irange)
    yi = fyi(irange)

    # close the path
    xi = np.append(xi, xi[0])
    yi = np.append(yi, yi[0])

    ## determine which points from xx, yy, are contained within polygon
    ## defined by xi, yi

    # create a polygon
    p = np.stack((xi, yi), axis=-1)
    polypath = mpath.Path(p)

    # flatten x/y grids into array columnwise
    xc = x.flatten('F')
    yc = y.flatten('F')

    # check if these xy points are contained in the polygon
    # returns a bool array
    pts_in = polypath.contains_points(np.stack((xc, yc), axis=-1))
    # reshape it to size of original grids
    outside = np.reshape(~pts_in, x.shape, 'F')

    # set outside pixels to white
    outside3 = np.dstack((outside, outside, outside))
    RGB[outside3] = 1.0

    return np.flip(RGB, axis=0)  # flip top to bottom

def ab_chromaticity_diagram(L=100, N=256):
    a, b = np.meshgrid(np.linspace(-128, 127, N), np.linspace(-128, 127, N))

    # convert from Lab to RGB
    ac = a.flatten('F')
    bc = b.flatten('F')

    L = np.ones(a.shape) * L
    Lab = np.dstack((L, a, b)).astype(np.float32)

    # TODO currently does not work. OpenCV
    # out = cv.cvtColor(Lab, cv.COLOR_Lab2BGR)

    RGB = colorspace_convert(Lab, 'lab', 'rgb')
    RGB = _normalize(RGB)  # fit to interval [0, 1]
    RGB = gamma_encode(RGB)  # gamma encode

    outside = np.sqrt(a**2 + b**2) > 128
    # set outside pixels to white
    outside3 = np.dstack((outside, outside, outside))
    RGB[outside3] = 1.0

    return np.flip(RGB, axis=0)  # flip top to bottom

def plot_chromaticity_diagram(colorspace='xy', brightness=1, alpha=1, block=False):
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
    if colorspace.lower() == 'xy':
        RGB = xy_chromaticity_diagram(Y=brightness)
        plt.imshow(RGB, extent=(0,0.8, 0, 0.9), alpha=alpha)
        plt.xlabel('x')
        plt.ylabel('y')
    elif colorspace.lower() in ('ab', 'l*a*b*', 'ab', 'a*b*'):
        RGB = ab_chromaticity_diagram(L=brightness*100)
        plt.imshow(RGB, extent=(-128, 127, -128, 127), alpha=alpha)
        plt.xlabel('a*')
        plt.ylabel('b*')
    else:
        raise ValueError('bad colorspace')

    plt.show(block=block)

def plot_spectral_locus(colorspace='xy', labels=True, ax=None, block=False,
    lambda_ticks=None):

    nm = 1e-9

    λ = np.arange(400, 700) * nm

    if colorspace in ('XY', 'xy'):
        locus = ccxyz(λ)[:, :2]
    elif colorspace in ('rg'):
        locus = lambda2rg(λ, method='cubic')

    if ax is None:
        ax = plt.subplot(1, 1, 1)

    ax.plot(locus[:, 0], locus[:, 1])

    if labels:
        ## add ticks

        # well-spaced points around the locus
        if lambda_ticks is None:
            λ = np.arange(460, 550, 10)
            λ = np.hstack((λ, np.arange(560, 620, 20)))
        else:
            λ = lambda_ticks

        if colorspace in ('XY', 'xy'):
            xyz = cmfxyz(λ * 1e-9)
            x = xyz[:, 0] / np.sum(xyz, axis=1)
            y = xyz[:, 1] / np.sum(xyz, axis=1)
        elif colorspace in ('rg',):
            rgb = cmfrgb(λ * 1e-9)
            x = rgb[:, 0] / np.sum(rgb, axis=1)
            y = rgb[:, 1] / np.sum(rgb, axis=1)
        else:
            raise ValueError('bad colorspace')

        ax.plot(x, y, 'ko')

        for i in range(len(λ)):
            ax.text(x[i], y[i], '  {0}'.format(λ[i]))

    plt.show(block=block)


def cie_primaries():
    """
    Define CIE primary colors

    ``cie_primaries`` is a 3-vector with the wavelengths [m] of the
    CIE-1976 red, green and blue primaries respectively.

    """
    return np.array([700, 546.1, 435.8]) * 1e-9

def colorspace_convert(image, src, dst):

    operation = _convertflag(src, dst)

    if isinstance(image, np.ndarray) and image.ndim == 3:
        # its a color image
        return cv.cvtColor(image, code=operation)
    else:
        # not an image, see if it's Nx3
        image = base.getmatrix(image, (None, 3), dtype=np.float32)
        image = image.reshape((-1, 1, 3))
        converted = cv.cvtColor(image, code=operation)
        if converted.shape[0] == 1:
            return converted.flatten().astype(np.float64)
        else:
            return converted.reshape((-1, 3)).astype(np.float64)

def _convertflag(src, dst):

    src = src.replace(':', '').lower()
    dst = dst.replace(':', '').lower()

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
        elif dst in ('lab', 'l*a*b*'):
            return cv.COLOR_RGB2Lab
        elif dst in ('luv', 'l*u*v*'):
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
        elif dst in ('lab', 'l*a*b*'):
            return cv.COLOR_BGR2Lab
        elif dst in ('luv', 'l*u*v*'):
            return cv.COLOR_BGR2Luv

    elif src in ('xyz', 'xyz_709'):
        if dst == 'rgb':
            return cv.COLOR_XYZ2RGB
        elif dst == 'bgr':
            return cv.COLOR_XYZ2BGR

    elif src == 'ycrcb':
        if dst == 'rgb':
            return cv.COLOR_YCrCb2RGB
        elif dst == 'bgr':
            return cv.COLOR_YCrCbBGR

    elif src == 'hsv':
        if dst == 'rgb':
            return cv.COLOR_HSVRGB
        elif dst == 'bgr':
            return cv.COLOR_HSV2BGR

    elif src == 'hls':
        if dst == 'rgb':
            return cv.COLOR_HLS2RGB
        elif dst == 'bgr':
            return cv.COLOR_HLS2BGR

    elif src in ('lab', 'l*a*b*'):
        if dst == 'rgb':
            return cv.COLOR_Lab2RGB
        elif dst == 'bgr':
            return cv.COLOR_Lab2BGR

    elif src in ('luv', 'l*u*v*'):
        if dst == 'rgb':
            return cv.COLOR_Luv2RGB
        elif dst == 'bgr':
            return cv.COLOR_Luv2BGR

    raise ValueError(f"unknown conversion {src} -> {dst}")

def gamma_encode(image, gamma='sRGB'):
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

    if isinstance(gamma, str) and gamma.lower() == 'srgb':

        imagef = float_image(image)

        if imagef.ndim == 2:
            # greyscale
            return _srgb(imagef)
        elif imagef.ndim == 3:
            # multi-dimensional
            out = np.empty(imagef.shape, dtype=imagef.dtype)
            for p in range(imagef.ndim):
                out[:,:,p] = _srgb(imagef[:,:,p])
        else:
            raise ValueError('expecting 2d or 3d image')

        if np.issubdtype(image.dtype, np.integer):
            # original image was float, convert back
            return int_image(out)
        else:
            return out

    else:
        # normal power law:
        # import code
        # code.interact(local=dict(globals(), **locals()))
        if np.issubdtype(image.dtype, np.floating):
            return image ** gamma
        else:
            # int image
            maxg = np.float32((np.iinfo(image.dtype).max))
            return ((image.astype(np.float32) / maxg) ** gamma) * maxg

def gamma_decode(image, gamma='sRGB'):
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

    if isinstance(gamma, str) and gamma.lower() == 'srgb':

        imagef = float_image(image)

        if imagef.ndim == 2:
            # greyscale
            return _srgb_inverse(imagef)
        elif imagef.ndim == 3:
            # multi-dimensional
            out = np.empty(imagef.shape, dtype=imagef.dtype)
            for p in range(imagef.ndim):
                out[:,:,p] = _srgb_inverse(imagef[:,:,p])
        else:
            raise ValueError('expecting 2d or 3d image')

        if np.issubdtype(image.dtype, np.float):
            # original image was float, convert back
            return int_image(out)
        else:
            return out

    else:

        # normal power law:
        if np.issubdtype(image.dtype, np.float):
            return image ** (1.0 / gamma)
        else:
            # int image
            maxg = np.float32((np.iinfo(image.dtype).max))
            return ((image.astype(np.float32) / maxg) ** (1 / gamma)) * maxg # original
            # return ((image.astype(np.float32) / maxg) ** gamma) * maxg
        


def _srgb_inverse(Rg):
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

    R = np.empty(Rg.shape, dtype=np.float32)
    Rg = np.clip(Rg, 0, 1)
    a = 0.0404482362771076
    i = np.where(Rg <= a)
    noti = np.where(Rg > a)
    R[i] = Rg[i] / 12.92
    R[noti] = ((Rg[noti] + 0.055) / 1.055) ** 2.4
    return R

def _srgb(R):
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

    Rg = np.empty(R.shape, dtype=np.float32)
    a = 0.0031306684425005883
    b = 0.416666666666666667
    i = np.where(R <= a)
    noti = np.where(R > a)
    Rg[i] = R[i] * 12.92
    Rg[noti] = np.real(1.055 * (R[noti] ** b) - 0.055)
    return Rg

def _normalize(rgb):
    """
    Normalize the pixel values

    :param rgb: [description]
    :type rgb: [type]

    Normalize pixel values into the range [0, 1].  After colorspace transformations
    pixel values can be outside this interval, both negative and positive.

    For every pixel (r, g, b):

    - If ``w = min(r, g, b)`` < 0 we must add (w, w, w) to make a displayable
      value.  This corresponds to adding some amount of white which will desaturate the color.
      ``r' = r + w, g' = g + w, b' = b + w``.
    - If ``s = max(r', g', b')`` > 1 we must scale the value by ``s`` to make it
      displayable: ``r'' = r' / s, g'' = g' / s, b'' = b' / s``
    """

    # find minimum of (r, g, b, 0)
    mn = np.minimum(np.amin(rgb, axis=2), 0)
    # and substract, effectively blending it with white (desaturation)
    rgb = rgb - mn[..., np.newaxis]

    # find maximum of (r, g, b, 1)
    mx = np.maximum(np.amax(rgb, axis=2), 1)
    # and scale the pixel
    rgb = rgb / mx[..., np.newaxis]
    return rgb

def shadow_invariant(image, θ=None, geometricmean=True, exp=False, sharpen=None, primaries=None):
    """
    Shadow invariant image

    :param image: linear color image (gamma decoded)
    :type image: ndarray(H,W,3) float
    :param geometricmean: normalized with geometric mean of color channels, defaults to True
    :type geometricmean: bool, optional
    :param exp: exponentiate the logarithmic image, defaults to False
    :type exp: bool, optional
    :param sharpen: a sharpening transform, defaults to None
    :type sharpen: ndarray(3,3), optional
    :param primaries: camera peak filter responses (nm), defaults to None
    :type primaries: array_like(3), optional
    :return: greyscale shadow invariant image
    :rtype: ndarray(H,W)

    ``shadow_invariant(image)`` is the greyscale invariant image (HxW)
    computed from the color image ``im`` (HxWx3) with a projection line of slope ``θ``.

    If ``image`` (Nx3) it is assumed to have one row per pixel and GS is similar
    (Nx3).

    If IM (HxWx3xN) it is assumed to be a sequence of color images and GS is
    also a sequence (HxWxN).

    If ``θ`` is not provided then the slope is computed from the camera spectral
    characteristics ``primaries`` a vector of the peak response of the camera's
    filters in the order red, green, blue.  If these aren't provided they
    default to 610, 538, 460nm.

    Example:

            >>> im = iread('parks.png', gamma='sRGB', dtype='double')
            >>> gs = shadow_invariant(im, 0.7)
            >>> idisp(gs)

    :references:

    - “Dealing with shadows: Capturing intrinsic scene appear for image-based outdoor localisation,”
      P. Corke, R. Paul, W. Churchill, and P. Newman
      Proc. Int. Conf. Intelligent Robots and Systems (IROS), pp. 2085–2 2013.
    """


    # Convert the image into a vector (h*w,channel)
    if image.ndim == 3 and image.shape[2] == 3:
        im = image.reshape(-1, 3).astype(float)
    else:
        raise ValueError('must pass an RGB image')

    # compute chromaticity

    if sharpen is not None:
        im = im @ opt.sharpen
        im = max(0, im);

    if geometricmean:
        # denom = prod(im, 2).^(1/3);
        A = np.prod(im, axis=1)
        denom = np.abs(A) ** (1.0 / 3)
    else:
        denom = im[:, 1]

    # this next bit will generate divide by zero errors, suppress any 
    # error messages. The results will be nan which we can deal with later.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        r_r = im[:, 0] / denom
        r_b = im[:, 2] /denom

    # Take the log
    r_rp = np.log(r_r)
    r_bp = np.log(r_b)

    # figure the illumination invariant direction shape=(2,)

    # create an array of logs (2,N)
    d = np.array([r_rp, r_bp])

    if θ is None:
        # compute direction from the spectral peaks of the camera filters
        if primaries is None:
            # Planckian illumination constant 
            c2 = 1.4388e-2

            #  spectral peaks of the Sony ICX204 sensor used in BB2 camera
            primaries = [610, 538, 460] * 1e-9

            e_r = -c2 / primaries[0]
            e_b = -c2 / primaries[2]
            c = np.r_[e_b, -e_r]
            c /= np.linalg.norm(c)
    else:
        # otherwise use the provided angle
        c = np.array([np.cos(θ), np.sin(θ)])

    gs = d.T @ c

    # optionally exponentiate the image
    if exp:
        gs = np.exp(gs)

    # reshape the gs vector to the original image size
    if image.ndim == 3:
        gs = gs.reshape(image.shape[:2])

    return gs

def est_theta(im, sharpen=None):

    def pickregion(im):

        im.disp()

        clicks = plt.ginput(n=-1)

        xy = np.array(clicks)
        print(xy)
        
        base.plot_poly(xy.T, 'g', close=True)

        polygon = Polygon2(xy.T)
        polygon.plot('g')
        
        X, Y = im.meshgrid()
        inside = polygon.contains(np.c_[X.ravel(), Y.ravel()].T)
        
        return inside

    k_region = pickregion(im)

    imcol = im.column()

    z = imcol[k_region, :]
    print(z.shape)
    # k = find(in);
    plt.show(block=True)


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

    # print(colorname('red'))
    # img = np.float32(np.r_[0.5, 0.2, 0.1]).reshape((1,1,3))
    # print(img.shape)
    # # print(cv.cvtColor(img, cv.COLOR_RGB2HSV))
    # print(cv.cvtColor(img, _convertflag('rgb', 'hsv')))
    # print(colorname([0.5,0.2, 0.5]))
    # print(colorname([0.5,0.2], 'xy'))


    # rg = lambda2rg(λ=np.array([555e-9, 666e-9]),
    #                          e=np.array([4, 2]))

    # z = colorname('chocolate', 'xy')
    # print(z)
    # bs = colorname('burntsienna', 'xy')
    # print(bs)

    # colorname('?burnt')
    
    # z = colorname('burntsienna')
    # print(z)
    # bs = colorname('burntsienna', 'xy')
    # print(bs)

    # green_cc = lambda2rg(500 * 1e-9)
    # print(green_cc)

    # print(name2color('r'))
    # print(name2color('r', 'lab'))
    # print(name2color('.*burnt.*'))
    # print(color2name([0,0,1]))

    nm = 1e-9;
    lmbda = np.arange(300, 1_001, 10) * nm;

    sun_ground = loadspectrum(lmbda, 'solar');

    print(name2color('orange', 'xy'))
    print(name2color('.*coral.*'))
    print(color2name([0.45, 0.48], 'xy'))

    print(cmfrgb([500*nm, 600*nm]))
    green_cc = lambda2rg(500 * nm)

    pass