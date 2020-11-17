#!/usr/bin/env python
# import io as io
import numpy as np
import spatialmath.base.argcheck as argcheck
# import cv2 as cv
# import matplotlib.path as mpath

import machinevisiontoolbox as mvt
from machinevisiontoolbox import Image

from scipy import interpolate
from collections import namedtuple
from pathlib import Path


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

    lam = argcheck.getvector(lam)

    e = 2.0 * h * c**2 / (lam**5 * (np.exp(h * c / k / T / lam) - 1))
    if len(e) == 1:
        return e[0]
    else:
        return e


def _loaddata(filename, **kwargs):
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

    # check filename is a string
    check_filename_isstr = isinstance(filename, str)
    if check_filename_isstr is False:
        print('Warning: input variable "filename" is not a valid string')

    if not ("." in filename):
        filename = filename + '.dat'

    try:
        # import filename, which we expect to be a .dat file
        # columns for wavelength and spectral data
        # assume column delimiters are whitespace, so for .csv files,
        # replace , with ' '
        with open(filename) as file:
            clean_lines = (line.replace(',', ' ') for line in file)
            # default delimiter whitespace
            data = np.genfromtxt(clean_lines, **kwargs)
    except IOError:
        print('An exception occurred: Spectral file {} not found'.format(
              filename))
        data = None

    return data


def loadspectrum(lam, filename, **kwargs):
    """
    Load spectrum data

    :param lam: wavelength 𝜆 [m]
    :type lam: float or array_like
    :param filename: filename
    :type filename: string
    :param kwargs**: keyword arguments for scipy.interpolate.interp1d
    :return: interpolated spectral data and corresponding wavelength
    :rtype: collections.namedtuple

    ``loadspectrum(𝜆, filename, **kwargs)`` is spectral data (N,D) from file
    filename interpolated to wavelengths [meters] specified in 𝜆 (N,1).
    The spectral data can be scalar (D=1) or vector (D>1) valued.

    Example:

    .. autorun:: pycon

    .. note::

        - The file is assumed to have its first column as wavelength in metres,
          the remainding columns are linearly interpolated and returned as
          columns of S.
        - The files are kept in the private folder inside the MVTB folder.

    :references:

        - Robotics, Vision & Control, Section 10.1, P. Corke, Springer 2011.
    """

    # check valid input
    lam = argcheck.getvector(lam)
    data = _loaddata(filename, comments='%')

    # interpolate data
    data_wavelength = data[0:, 0]
    data_s = data[0:, 1:]

    # TODO default is currently linear interpolation
    # perhaps make default slinear, or quadratic?
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.
    # interp1d.html
    f = interpolate.interp1d(data_wavelength, data_s, axis=0,
                             bounds_error=False, fill_value=0, **kwargs)
    s = f(lam)

    return namedtuple('spectrum', 's lam')(s, lam)


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
    lam = argcheck.getvector(lam)

    if e is None:
        rgb = cmfrgb(lam, **kwargs)
    else:
        e = argcheck.getvector(e)
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

    .. autorun:: pycon

    :references:

        - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
    """

    lam = argcheck.getvector(lam)  # lam is (N,1)

    cmfrgb_data = Path('data') / 'cmfrgb.dat'
    rgb = loadspectrum(lam, cmfrgb_data.as_posix(), **kwargs)
    ret = rgb.s

    # approximate rectangular integration
    if e is not None:
        e = argcheck.getvector(e)  # e is a vector Nx1
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

    .. autorun:: pycon

    :references:

        - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
    """

    # TODO check if tri is correct shape? can be vectror or matrix
    tri = np.array(tri)
    if tri.ndim < 2:
        # we to make tri at least a 2D vector
        tri = argcheck.getvector(tri)
        tri = np.expand_dims(tri, axis=0)
    else:
        # currently, Image.getimage returns a numpy array
        # TODO consider using Image class
        tri = mvt.Image.getimage(tri)

    if tri.ndim < 3:
        # each row is R G B, or X Y Z
        s = np.sum(tri, axis=1)
        s = argcheck.getvector(s)
        ss = np.stack((s, s), axis=-1)
        cc = tri[0:, 0:2] / ss
    else:
        # tri is given as an image
        s = np.sum(tri, axis=2)  # could also use np.tile
        ss = np.stack((s, s), axis=-1)
        cc = tri[0:, 0:, 0:2] / ss

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

    .. autorun:: pycon

    :references:

        - Robotics, Vision & Control, Section 10.2, P. Corke, Springer 2011.
    """

    # argcheck
    lam = argcheck.getvector(lam)

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

    .. autorun:: pycon

    .. note::

        - CIE 1931 2-deg XYZ CMFs from cvrl.ioo.ucl.ac.uk .

    :references:

        - Robotics, Vision & Control, Chapter 14.3, P. Corke, Springer 2011.
    """

    lam = argcheck.getvector(lam)
    cmfxyz_data_name = Path('data') / 'cmfxyz.dat'
    xyz = _loaddata(cmfxyz_data_name.as_posix(), comments='%')

    XYZ = interpolate.pchip_interpolate(
        xyz[:, 0], xyz[:, 1:], lam, axis=0, **kwargs)

    if e is not None:
        # approximate rectangular integration
        dlam = lam[1] - lam[0]
        XYZ = e * XYZ * dlam

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

    .. autorun:: pycon

    .. note::

        - Luminosity has units of lumens, which are the intensity with which
          wavelengths are perceived by the light-adapted human eye.

    ::references:

        - Robotics, Vision & Control, Chapter 10.1, P. Corke, Springer 2011.

    :seealso: :func:`~rluminos`
    """

    lam = argcheck.getvector(lam)
    data = _loaddata((Path('data') / 'photopicluminosity.dat').as_posix(),
                     comments='%')

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

    .. autorun:: pycon

    .. note::

        - Relative luminosity lies in t he interval 0 to 1, which indicate the
          intensity with which wavelengths are perceived by the light-adapted
          human eye.

    :references:

        - Robotics, Vision & Control, Chapter 10.1, P. Corke, Springer 2011.
    """

    lam = argcheck.getvector(lam)
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

    .. autorun:: pycon

    :references:

        - Robotics, Vision & Control, Chapter 14.3, P. Corke, Springer 2011.
    """

    lam = argcheck.getvector(lam)
    xyz = cmfxyz(lam)

    if e is None:
        cc = xyz / (np.sum(xyz, axis=1) * np.ones((3, 1))).T
    else:
        e = argcheck.getvector(e)
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

    Example::

        # TODO

    """

    if not isinstance(fname, str):
        raise ValueError(fname, 'file name must be a string')

    data = _loaddata(fname, comments='#',
                     dtype=None, encoding='ascii')

    # convert data to a dictionary
    rgbdict = {}
    for line in data:
        k = line[3].astype(str)
        # v = np.array([line[0], line[1], line[2]])
        v = np.array([int(x) for x in line[0:2]])
        rgbdict[k] = v

    return rgbdict


def colorname(name, opt=None):
    """
    Map between color names and RGB values

    :param name: name of a color or name of a 3-element color array
    :type name: string or (numpy, tuple, list)
    :param opt: name of colorspace (eg 'rgb' or 'xyz' or 'xy' or 'ab')
    :type opt: string
    :return out: output
    :rtype out: named tuple, name of color, numpy array in colorspace

    - ``name`` is a string/list/set of color names, then ``colorname`` returns
      a 3-tuple of rgb tristimulus values.

    Example:

    .. autorun:: pycon

    .. note::

        - Color name may contain a wildcard, eg. "?burnt"
        - Based on the standard X11 color database rgb.txt
        - Tristiumuls values are [0,1]

    :references:

        - Robotics, Vision & Control, Chapter 14.3, P. Corke, Springer 2011.
    """
    # I'd say str in, 3 tuple out, or 3-element array like (numpy, tuple, list)
    #  in and str out

    assert isinstance(name, (str, tuple, list, np.ndarray)
                      ), 'name must be a string, tuple, list or np.ndarray'

    # load rgbtable (rbg.txt as a dictionary)
    print('loading rgb.txt')
    rgbfilename = (Path.cwd() / 'data' / 'rgb.txt').as_posix()
    rgbdict = _loadrgbdict(rgbfilename)

    if isinstance(name, str) or (isinstance(name, (list, set, tuple)) and
                                 isinstance(name[0], str)):
        if isinstance(name, str):
            name = list(name)  # convert to a list

        # make a new dictionary for those keys in name
        rgbout = {k: v for k, v in rgbdict.items() if k in name}

        # return rgbout as RGB 3-tuple
        if isinstance(name, str):
            return tuple(rgbout[name])
        else:
            return [tuple(rgbout[k] for k in rgbout.keys())]

    elif isinstance(name, (np.ndarray, tuple, list)):
        # map RGB tuple to name
        n = np.array(name)  # convert tuple or list into np array

        assert (n.shape[1] == 3), 'color value must have 3 elements'

        if (opt is not None) and (opt == 'xyz'):
            # TODO: if xyz, then convert to rgb
            print('TODO')

        rgbvals = list(rgbdict.values())
        rgbkeys = list(rgbdict.keys())
        rgbout = {}
        for i in range(n.shape[0]):
            dist = np.linalg.norm(np.array(rgbvals) - n[i, :], axis=1)
            # not sure why np.where is returning a tuple
            idist = np.where(dist == dist.min())
            idist = np.array(idist).flatten()
            # this loop can pick up multiple minima,
            # often only when there are identical colour cases
            for j in range(len(idist)):
                rgbout[rgbkeys[idist[j]]] = rgbvals[idist[j]]

        # TODO just return single string?
        return str(list(rgbout.keys()))
    else:
        raise TypeError('name is of unknown type')


def rg_addticks(ax):
    """
    Label spectral locus

    :param ax: axes reference for plotting
    :type ax: Matplotlib.pyplot axes object

    ``rg_addticks(ax)`` adds wavelength ticks to the spectral locus.
    """

    # well-spaced points around the locus
    lam = np.arange(460, 550, 10,)
    lam = np.hstack((lam, np.arange(560, 620, 20)))

    rgb = cmfrgb(lam * 1e-9)
    r = rgb[0:, 0] / np.sum(rgb, axis=1)
    g = rgb[0:, 1] / np.sum(rgb, axis=1)

    ax.plot(r, g, 'ko')

    for i in range(len(lam)):
        ax.text(r[i], g[i], '  {0}'.format(lam[i]))


def cie_primaries():
    """
    Define CIE primary colors

    ``cie_primaries`` is a 3-vector with the wavelengths [m] of the
    IE 1976 red, green and blue primaries respectively.

    """
    return np.array([700, 546.1, 435.8]) * 1e-9


if __name__ == '__main__':  # pragma: no cover

    # import pathlib
    # import os.path

    # exec(open(os.path.join(pathlib.Path(__file__).parent.absolute(),
    # "test_color.py")).read())
    import machinevisiontoolbox.color as color

    rg = color.lambda2rg(555e-9)
    print(rg)

    wcc = color.tristim2cc(np.r_[1, 1, 1])
    print(wcc)

    import code
    code.interact(local=dict(globals(), **locals()))
