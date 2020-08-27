#!/usr/bin/env python
import io as io
import numpy as np
# import scipy as sp
import spatialmath.base.argcheck as argcheck
# import matplotlib.pyplot as plt  # TODO: remove later, as only used for debugging

from scipy import interpolate
from collections import namedtuple

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


def loadspectrum(lam, filename, **kwargs):
    """
    Load spectrum data

    :param lam: wavelength 𝜆 [m]
    :type lam: float or array_like
    :param filename: filename
    :type filename: string
    :param variable/options? #TODO - kwargs**
    :return: interpolated irradiance and corresponding wavelength
    :rtype: collections.namedtuple

    S = LOADSPECTRUM(LAMBDA, FILENAME) is spectral data (NxD) from file
    FILENAME interpolated to wavelengths [meters] specified in LAMBDA (Nx1).
    The spectral data can be scalar (D=1) or vector (D>1) valued.

    [S, LAMBDA] = LOADSPECTRUM(LAMBDA, FILENAME) as above but also returns the
    passed wavelength LAMBDA.

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

    
    # parse options
    # defaults/assertions if filename is empty
    # tab = load data from filename
    # s = interpolation of data in tab
    # return s, lambda (if nargout == 2)

    # check valid input
    lam = argcheck.getvector(lam)

    # check filename is a string
    check_filename_isstr = isinstance(filename, str)
    if check_filename_isstr is False:
        print('Warning: input variable "filename" is not a valid string')

    if not ("." in filename):
        filename = filename + '.dat'

    try:
        # import filename, which we expect to be a .dat file
        # columns for wavelength and irradiance
        # assume column delimiters are whitespace, so for .csv files,
        # replace , with ' '
        with open(filename) as file:
            clean_lines = (line.replace(',', ' ') for line in file)
            # default delimiter whitespace
            data = np.genfromtxt(clean_lines, comments='%')
    
        # print(data)
    except IOError:
        print('An exception occurred: Spectral file {} not found'.format(filename))

    # do interpolation, TODO using kwargs** to pass on options information
    # to interp1d
    data_wavelength = data[0:, 0]
    data_s = data[0:, 1:]
    f = interpolate.interp1d(data_wavelength, data_s, axis=0,
                             bounds_error=False, **kwargs)

    # TODO: check lam is contained within data_wavelength for valid
    # take min/max of lam, compared to min/max of data_wavelength
    # interpolation range
    s = f(lam)

    # debug: plot the data:
    # plt.plot(data_wavelength, data_irradiance, 'o',
    #        lam, irradiance_interp, '-')
    # plt.grid(True)
    # plt.xlabel('wavelength [m]')
    # plt.ylabel('irradiance [W/m^2/m]')
    # plt.show()

    return namedtuple('spectrum', 's lam')(s, lam)


if __name__ == '__main__':  # pragma: no cover
    import pathlib
    import os.path

    exec(open(os.path.join(pathlib.Path(__file__).parent.absolute(), "test_color.py")).read())
