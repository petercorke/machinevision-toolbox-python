#!/usr/bin/env python
import io as io
import numpy as np
# import scipy as sp
import spatialmath.base.argcheck as argcheck
# import matplotlib.pyplot as plt  # TODO: remove later, as only used for debugging

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
    
    References:

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


def loaddata(filename):
    """
    Load data

    :param filename: filename
    :type filename: string
    :return: data
    :rtype: numpy array

    ``loaddata(filename)`` returns ``data`` from ``filename``

    Example::

        # TODO

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
            data = np.genfromtxt(clean_lines, comments='%')
    except IOError:
        print('An exception occurred: Spectral file {} not found'.format(filename))
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

    Example::

        #TODO

    :notes:

        - The file is assumed to have its first column as wavelength in metres,
          the remainding columns are linearly interpolated and returned as
          columns of S.
        - The files are kept in the private folder inside the MVTB folder.

    References:

        - Robotics, Vision & Control, Section 10.1, P. Corke, Springer 2011.
    """

    # check valid input
    lam = argcheck.getvector(lam)
    data = loaddata(filename)

    # interpolate data
    data_wavelength = data[0:, 0]
    data_s = data[0:, 1:]

    f = interpolate.interp1d(data_wavelength, data_s, axis=0,
                             bounds_error=False, **kwargs)

    # TODO: check lam is contained within data_wavelength for valid
    # take min/max of lam, compared to min/max of data_wavelength
    # interpolation range
    s = f(lam)

    return namedtuple('spectrum', 's lam')(s, lam)


def lambda2rg(lam, e=None):
    """
    lambda2rgb RGB chromaticity coordinates

    :param lam: wavelength 𝜆 [m]
    :type lam: float or array_like
    :param e: illlumination spectrum defined at the wavelengths 𝜆
    :type e: numpy array (N,1)
    :return rg: rg-chromaticity
    :rtype:

    ``lambda2rg(𝜆)`` is the rg-chromaticity coordinate (1,2) for
    illumination at the specific wavelength 𝜆 [m]. If 𝜆 is a
    numpy array (N,1), then P (N,2) is a vector whose elements are the
    chromaticity coordinates at the corresponding elements of 𝜆.

    ``lambda2rg(𝜆, e)`` is the rg-chromaticity coordinate (1,2) for an
    illumination spectrum e (N,1) defined at corresponding wavelengths
    𝜆 (N,1).

    Example::

        #TODO

    :notes:

    - Data from http://cvrl.ioo.ucl.ac.uk
    - From Table I(5.5.3) of Wyszecki & Stiles (1982). (Table 1(5.5.3)
      of Wyszecki & Stiles (1982) gives the Stiles & Burch functions in
      250 cm-1 steps, while Table I(5.5.3) of Wyszecki & Stiles (1982)
      gives them in interpolated 1 nm steps.)
    - The Stiles & Burch 2-deg CMFs are based on measurements made on
      10 observers. The data are referred to as pilot data, but probably
      represent the best estimate of the 2 deg CMFs, since, unlike the CIE
      2 deg functions (which were reconstructed from chromaticity data),
      they were measured directly.
    - These CMFs differ slightly from those of Stiles & Burch (1955). As
      noted in footnote a on p. 335 of Table 1(5.5.3) of Wyszecki &
      Stiles (1982), the CMFs have been "corrected in accordance with
      instructions given by Stiles & Burch (1959)" and renormalized to
      primaries at 15500 (645.16), 19000 (526.32), and 22500 (444.44) cm-1

    References:

        - Robotics, Vision & Control, Section 10.2, P. Corke, Springer 2011.
    """

    # check input
    lam = argcheck.getvector(lam)

    if e is None:
        rgb = cmfrgb(lam)
    else:
        e = argcheck.getvector(e)
        rgb = cmfrgb(lam, e)

    cc = tristim2cc(rgb)
    r = cc[0:, 0]
    g = cc[0:, 1]

    return namedtuple('rg', 'r g')(r, g)


def cmfrgb(lam, e=None):
    """
    cmfrgb RGB color matching function

    :param lam: wavelength 𝜆 [m]
    :type lam: float or array_like
    :param e: illlumination spectrum defined at the wavelengths 𝜆
    :type e: numpy array (N,1)
    :return rgb: rg-chromaticity
    :rtype: numpy array, shape = (N,3)

    ``rgb = cmfrgb(𝜆)`` is the CIE color matching function (N,3)
    for illumination at wavelength 𝜆 (N,1) [m]. If 𝜆 is a vector
    then each row of RGB is the color matching function of the
    corresponding element of 𝜆.

    ``rgb = cmfrgb(𝜆, e) is the CIE color matching (1,3) function for an
    illumination spectrum e (N,1) defined at corresponding wavelengths
    𝜆 (N,1).

    Example::

        #TODO

    References:

        - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
    """

    lam = argcheck.getvector(lam)  # lam is (N,1)

    cmfrgb_data = Path('data') / 'cmfrgb.dat'
    rgb = loadspectrum(lam, cmfrgb_data.as_posix())
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

    Example::

        #TODO

    References:

        - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
    """

    # check if tri is correct shape
    if not argcheck.ismatrix(tri, tri.shape):
        raise TypeError('input must be numpy ndarray matrix')

    if tri.ndim < 3:
        # each row is R G B, or X Y Z
        s = np.sum(tri, axis=1)
        s = argcheck.getvector(s)
        cc = tri[:0, 0:2]/s

        # TODO decide on output format cc.c1, cc.c2?
    else:
        # tri is given as an image
        s = np.sum(tri, axis=2)  # np.tile?
        ss = np.stack((s, s), axis=-1)
        cc = tri[0:, 0:, 0:2] / ss  # / [s s]
        # TODO decide on output format a, b as separate channels or overall cc?

    return cc


def lambda2xy(lam, *args):
    """
    xy-chromaticity coordinates for a given wavelength 𝜆 [meters]

    :param lam: wavelength 𝜆 [m]
    :type lam: float or array_like
    :return xy: xy-chromaticity coordinates
    :rtype: numpy array, shape = (N,2)

    ``lambda2xy(𝜆)`` is the xy-chromaticity coordinate (1,2) for
    illumination at the specific wavelength 𝜆 [metres]. If 𝜆 is a
    vector (N,1), then the return is a vector (N,2) whose elements

    Example::

        #TODO

    References:
        - Robotics, Vision & Control, Section 10.2, P. Corke, Springer 2011.
    """

    # argcheck
    lam = argcheck.getvector(lam)

    cmf = cmfxyz(lam, *args)
    xy = tristim2cc(cmf)

    return xy


def cmfxyz(lam, e=None):
    """
    color matching function for xyz tristimulus

    :param lam: wavelength 𝜆 [m]
    :type lam: float or array_like
    :param e: illlumination spectrum defined at the wavelengths 𝜆
    :type e: numpy array (N,1)
    :return xy: xy-chromaticity
    :rtype: numpy array, shape = (N,2)

    The color matching function is the XYZ tristimulus required to match a
    particular wavelength excitation.

    ``cmfxyz(𝜆)`` is the CIE XYZ color matching function (N,3) for illumination
    at wavelength 𝜆 (N,1) [m].  If 𝜆 is a vector then each row of XYZ
    is the color matching function of the corresponding element of 𝜆.

    ``cmfxzy(𝜆, e)`` is the CIE XYZ color matching (1,3) function for an
    illumination spectrum e (N,1) defined at corresponding wavelengths 𝜆 (N,1).

    Example::

        #TODO

    :notes:
    - CIE 1931 2-deg XYZ CMFs from cvrl.ioo.ucl.ac.uk

    References:
        - Robotics, Vision & Control, Chapter 14.3, P. Corke, Springer 2011.
    """

    lam = argcheck.getvector(lam)
    cmfxyz_data_name = Path('data') / 'cmfxyz.dat'
    xyz = loaddata(cmfxyz_data_name.as_posix())

    xyz = interpolate.pchip_interpolate(xyz[0:, 0], xyz[0:, 0:], lam,
                                        axis=0, der=0)

    if e is not None:
        # approximate rectangular integration
        dlam = lam[1] - lam[0]
        xyz = e * xyz * dlam

    return xyz


def luminos(lam):
    """
    photopic luminosity function

    :param lam: wavelength 𝜆 [m]
    :type lam: float or array_like
    :return p: luminosity
    :rtype: numpy array, shape = (N,1)

    ``luminos(𝜆)`` is the photopic luminosity function for the
    wavelengths in 𝜆 (N,1) [m]. If 𝜆 is a vector then ``p`` is a vector
    whose elements are the luminosity at the corresponding 𝜆.

    Example::

        #TODO

    :notes:
    - luminosity has units of lumens, which are the intensity with which
      wavelengths are perceived by the light-adapted human eye

    References:

        - Robotics, Vision & Control, Chapter 10.1, P. Corke, Springer 2011.
    """

    lam = argcheck.getvector(lam)
    data = loaddata((Path('data') / 'photopicluminosity.dat').as_posix())

    flum = interpolate.interp1d(data[0:, 0], data[0:, 1],
                                bounds_error=False, fill_value=0)
    lum = flum(lam)

    return lum  # photopic luminosity is the Y color matching function



def rluminos(lam):
    """
    relative photopic luminosity function

    :param lam: wavelength 𝜆 [m]
    :type lam: float or array_like
    :return p: relative luminosity
    :rtype: numpy array, shape = (N,1)

    ``rluminos(𝜆)`` is the relative photopic luminosity function for the
    wavelengths in 𝜆 (N,1) [m]. If 𝜆 is a vector then ``p`` is a vector
    whose elements are the luminosity at the corresponding 𝜆.

    Example::

        #TODO

    :notes:
    - Relative luminosity lies in t he interval 0 to 1, which indicate the
      intensity with which wavelengths are perceived by the light-adapted
      human eye.

    References:

        - Robotics, Vision & Control, Chapter 10.1, P. Corke, Springer 2011.
    """

    lam = argcheck.getvector(lam)
    xyz = cmfxyz(lam)
    return xyz[0:, 1]  # photopic luminosity is the Y color matching function



def showcolorspace(somestr='xy', *args):
    """
    showcolorspace display spectral locus

    :param xy: 'xy'
    :type xy: string
    :param lab: 'lab'
    :type lab: string
    :param which: 'which'
    :type which: string
    :param p:
    :type p: numpy array, shape = (N,1)
    :return IM: image
    :rtype: nd.array
    :return AX: corresponding x-axis coordinates for IM
    :rtype: vector or 1D array
    :return AY: corresponding y-axis coordinates for IM
    :rtype: vector of 1D array

    # TODO: for now, just return plotting

    Example::
        #TODO

    :notes:
    - The colors shown within the locus only approximate the true colors, due
    to the gamut of the display device.

    References:
        - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
    """

    # parse input options
    # xy, lab or which
    # optional input p (default None)
    # set default options N, L
    # case xy
    #




if __name__ == '__main__':  # pragma: no cover
    import pathlib
    import os.path

    exec(open(os.path.join(pathlib.Path(__file__).parent.absolute(), "test_color.py")).read())
