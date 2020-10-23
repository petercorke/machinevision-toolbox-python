#!/usr/bin/env python
import io as io
import numpy as np
import spatialmath.base.argcheck as argcheck
import cv2 as cv
import matplotlib.path as mpath
import machinevisiontoolbox.image as image

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

    Example::

        # TODO

    .. note::

        - Comments are assumed to be ' as original data files were part
          of the MATLAB machine vision toolbox.
        - # TODO can change this with the use of **kwargs.
    """

    # check filename is a string
    check_filename_isstr = isinstance(filename, str)
    if check_filename_isstr is False:
        print('Warning: input variable "filename" is not a valid string')

    if not ("." in filename):
        filename = filename + '.dat'

    # setup kwargs for np.genfromtxt options:
    #

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

    Example::

        #TODO

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

    Example::

        #TODO

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

    Example::

        #TODO

    :references:

        - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
    """

    # TODO check if tri is correct shape? can be vectror or matrix
    #import code
    #code.interact(local=dict(globals(), **locals()))

    tri = np.array(tri)
    if tri.ndim < 2:
        # we to make tri at least a 2D vector
        tri = argcheck.getvector(tri)
        tri = np.expand_dims(tri, axis=0)
    else:
        # I believe this works for a matrix as well
        tri = image.getimage(tri)

    #import code
    #code.interact(local=dict(globals(), **locals()))
    # if not argcheck.ismatrix(tri, tri.shape):
    #    raise TypeError('input must be numpy ndarray matrix')

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

    Example::

        #TODO

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

    Example::

        #TODO

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

    Example::

        #TODO

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

    Example::

        #TODO

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


def showcolorspace(cs='xy', N=501, L=90, *args):
    """
    Display spectral locus

    :param cs: 'xy', 'lab', 'ab' or None defines which colorspace to show
    :type xy: string
    :param N: number of points to sample in the x- and y-directions
    :type N: integer, N > 0, default 501
    :param L: length of points to sample for Lab colorspace
    :type L: integer, L > 0, default 90
    :return IM: image
    :rtype: nd.array

    # TODO: for now, just return plotting

    Example::

        #TODO

    .. note::

        - The colors shown within the locus only approximate the true colors, due to
          the gamut of the display device.

    :references:

        - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
    """

    # TODO check valid inputs
    # TODO cslist = [None, 'xy', 'ab', 'Lab']
    # which should be defined by cases (showcolorspace.m)

    assert isinstance(cs, str), 'color space must be a string'

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
        XYZ = np.stack((X, Y, Z), axis=2)

        # TODO replace with color.colorspace(im,conv,**kwargs)
        # (replace starts here)
        # note that using cv.COLOR_XYZ2RGB does not seem to work properly?
        BGR_raw = cv.cvtColor(np.float32(XYZ), cv.COLOR_XYZ2BGR)

        # desaturate and rescale to constrain resulting RGB values to [0,1]
        B = BGR_raw[:, :, 0]
        G = BGR_raw[:, :, 1]
        R = BGR_raw[:, :, 2]
        add_white = -np.minimum(np.minimum(np.minimum(R, G), B), 0)
        B += add_white
        G += add_white
        R += add_white
        # cv.imshow('lala',mat,)
        # inverse gamma correction
        B = _invgammacorrection(B)
        G = _invgammacorrection(G)
        R = _invgammacorrection(R)

        # combine layers into image:
        RGB = np.stack((R, G, B), axis=2)
        # (replace ends here. note: ensure consistent with BGR vs RGB)

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

        # determine which points from xx, yy, are contained within the polygon defined by xi, yi
        p = np.stack((xi, yi), axis=-1)
        polypath = mpath.Path(p)

        xxc = xx.flatten('F')
        yyc = yy.flatten('F')
        pts_in = polypath.contains_points(np.stack((xxc, yyc), axis=-1))
        # same for both xx and yy
        colors_in = np.reshape(pts_in, xx.shape, 'F')
        # colors_in_yy = pts_in.reshape(yy.shape)
        # plt.imshow(colors_in)
        # plt.show()
        # set outside pixels to white
        RGB[np.where(
            np.stack((colors_in, colors_in, colors_in), axis=2) == False)] = 1.0
        # color[~np.stack((colorsin, colorsin, colorsin), axis=2)] = 1.0
        #import matplotlib.pyplot as plt
        # plt.imshow(RGB)
        # plt.show(block=False)
        # plt.show()

        # for renaming purposes
        color = RGB

    elif (cs == 'ab') or (cs == 'Lab'):
        ax = np.linspace(-100, 100, N)
        ay = np.linspace(-100, 100, N)
        aa, bb = np.meshgrid(ax, ay)

        # convert from Lab to RGB
        avec = argcheck.getvector(aa)
        bvec = argcheck.getvector(bb)
        color = cv.cvtColor(np.stack((L*np.ones(avec.shape), avec, bvec),
                                     axis=2), cv.COLOR_Lab2BGR)

        color = col2im(color, [N, N])

        color = image.pixelswitch(image.kcircle(np.floor(N / 2)),
                                  color, [1, 1, 1])
    else:
        raise ValueError('no or unknown color space provided')

    # im = color
    return color


def col2im(col, im):
    """
    Convert pixel vector to image

    :param col: set of pixel values
    :type col: numpy array, shape (N, P)
    :param im: image
    :type im: numpy array, shape (N, M, P), or a 2-vector (N, M) indicating image size
    :return: image of specified shape
    :rtype: numpy array

    ``col2im(col, imsize)`` is an image (H, W, P) comprising the pixel values
    in col (N,P) with one row per pixel where N=HxW.
    ``imsize`` is a 2-vector (N,M).

    ``col2im(col, im)`` as above but the dimensions of the return are the
    same as ``im``.

    .. note::

        - The number of rows in ``col`` must match the product of the elements of ``imsize``.

    :references:

        - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
    """
    # TODO check valid input

    #col = argcheck.getvector(col)
    col = np.array(col)
    if col.ndim == 1:
        nc = len(col)
    elif col.ndim == 2:
        nc = col.shape[0]
    else:
        raise ValueError(col, 'col does not have valid shape')

    # second input can be either a 2-tuple/2-array, or a full image
    im = np.array(im)  # ensure we can use ndim and shape
    if im.ndim == 1:
        # input is a tuple/1D array
        sz = im
    elif im.ndim == 2:
        im = image.getimage(im)
        sz = im.shape
    elif im.ndim == 3:
        im = image.getimage(im)
        sz = np.array([im.shape[0], im.shape[1]])  # ignore 3rd channel
    else:
        raise ValueError(im, 'im does not have valid shape')

    if nc > 1:
        sz = np.hstack((sz, nc))

    # reshape:
    # TODO need to test this
    return np.reshape(col, sz)


def _invgammacorrection(Rg):
    """
    inverse gamma correction

    :param Rg: 2D image
    :type Rg: numpy array, shape (N,M)
    :return: R
    :rtype: numpy array

    ``_invgammacorrection(Rg)`` returns ``R`` from ``Rg``

    Example::

        # TODO

    .. note::

        - Based on code from Pascal Getreuer 2005-2010
        - And code in colorspace.m from Peter Corke's Machine Vision Toolbox
    """

    R = np.zeros(Rg.shape)
    a = 0.0404482362771076
    i = np.where(Rg <= a)
    noti = np.where(Rg > a)
    R[i] = Rg[i] / 12.92
    R[noti] = np.real(((Rg[noti] + 0.055)/1.055)**2.4)
    return R


def _gammacorrection(R):
    """
    Gamma correction

    :param R: 2D image
    :type R: numpy array, shape (N,M)
    :return: Rg
    :rtype: numpy array

    ``_gammacorrection(R)`` returns ``Rg`` from ``R``

    Example::

        # TODO

    .. note::

        - Based on code from Pascal Getreuer 2005-2010
        - And code in colorspace.m from Peter Corke's Machine Vision Toolbox
    """

    Rg = np.zeros(R.shape)
    a = 0.0031306684425005883
    b = 0.416666666666666667
    i = np.where(R <= a)
    noti = np.where(R > a)
    Rg[i] = R[i] * 12.92
    Rg[noti] = np.real(1.055 * (R[noti]**b) - 0.055)
    return Rg


def colorspace(im, conv, **kwargs):
    """
    Transform a color image between color representations

    :param im: image
    :type im: numpy array, shape (N,M) or (N,3)
    :param conv: color code for color conversion, based on OpenCV's cvtColor
    :type conv: string (see below)
    :param kwargs: keywords/options for OpenCV's cvtColor
    :type kwargs: name/value pairs
    :return: out
    :rtype: numpy array, shape (N,M) or (N,3)

    ``colorspace(im, conv)`` transforms the color representation of image ``im``
    where ``conv`` is a string specifying the conversion. The input array ``im``
    should be a real full double array of size (M,3) or (M,N,3). The output is
    the same size as ``im``.

    ``conv`` tells the source and destination color spaces,
    ``conv`` = 'dest<-src', or alternatively, ``conv`` = 'src->dest'.
    Supported color spaces are
    'RGB'              sRGB IEC 61966-2-1
    'YCbCr'            Luma + Chroma ("digitized" version of Y'PbPr)
    'JPEG-YCbCr'       Luma + Chroma space used in JFIF JPEG
    'YDbDr'            SECAM Y'DbDr Luma + Chroma
    'YPbPr'            Luma (ITU-R BT.601) + Chroma
    'YUV'              NTSC PAL Y'UV Luma + Chroma
    'YIQ'              NTSC Y'IQ Luma + Chroma
    'HSV' or 'HSB'     Hue Saturation Value/Brightness
    'HSL' or 'HLS'     Hue Saturation Luminance
    'HSI'              Hue Saturation Intensity
    'XYZ'              CIE 1931 XYZ
    'Lab'              CIE 1976 L*a*b* (CIELAB)
    'Luv'              CIE L*u*v* (CIELUV)
    'LCH'              CIE L*C*H* (CIELCH)
    'CAT02 LMS'        CIE CAT02 LMS

    .. note::

        - All conversions assume 2 degree observer and D65 illuminant.
          Color space names are case insensitive and spaces are ignored.  When
          sRGB is the source or destination, it can be omitted. For example
          'yuv<-' is short for 'yuv<-rgb'.
          For sRGB, the values should be scaled between 0 and 1.  Beware that
          transformations generally do not constrain colors to be "in gamut."
          Particularly, transforming from another space to sRGB may obtain
          R'G'B' values outside of the [0,1] range.  So the result should be
          clamped to [0,1] before displaying:
          image(min(max(B,0),1));  lamp B to [0,1] and display
          sRGB (Red Green Blue) is the (ITU-R BT.709 gamma-corrected) standard
          red-green-blue representation of colors used in digital imaging.  The
          components should be scaled between 0 and 1.  The space can be
          visualized geometrically as a cube.
        - Y'PbPr, Y'CbCr, Y'DbDr, Y'UV, and Y'IQ are related to sRGB by linear
          transformations.  These spaces separate a color into a grayscale
          luminance component Y and two chroma components.  The valid ranges of
          the components depends on the space.
        - HSV (Hue Saturation Value) is related to sRGB by
          H = hexagonal hue angle   (0 <= H < 360),
          S = C/V                   (0 <= S <= 1),
          V = max(R',G',B')         (0 <= V <= 1),
          where C = max(R',G',B') - min(R',G',B').  The hue angle H is computed on
          a hexagon.  The space is geometrically a hexagonal cone.
          - HSL (Hue Saturation Lightness) is related to sRGB by
          H = hexagonal hue angle                (0 <= H < 360),
          S = C/(1 - abs(2L-1))                     (0 <= S <= 1),
          L = (max(R',G',B') + min(R',G',B'))/2  (0 <= L <= 1),
          where H and C are the same as in HSV.  Geometrically, the space is a
          double hexagonal cone.
        - HSI (Hue Saturation Intensity) is related to sRGB by
          H = polar hue angle        (0 <= H < 360),
          S = 1 - min(R',G',B')/I    (0 <= S <= 1),
          I = (R'+G'+B')/3           (0 <= I <= 1).
          Unlike HSV and HSL, the hue angle H is computed on a circle rather than
          a hexagon.
        - CIE XYZ is related to sRGB by inverse gamma correction followed by a
          linear transform.  Other CIE color spaces are defined relative to XYZ.
        - CIE L*a*b*, L*u*v*, and L*C*H* are nonlinear functions of XYZ.  The L*
          component is designed to match closely with human perception of
          lightness.  The other two components describe the chroma.
        - CIE CAT02 LMS is the linear transformation of XYZ using the MCAT02
          chromatic adaptation matrix.  The space is designed to model the
          response of the three types of cones in the human eye, where L, M, S,
          correspond respectively to red ("long"), green ("medium"), and blue
          ("short").
        - TODO how to reference Pascal Getreuer 2005-2010?

    :references:

        - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
    """
    # check valid image input (image.isimage)``
    # identify which case we're dealing with, based on conv
    # for xyz to rgb case:
    # call cvtColor
    # desaturate
    # apply invgammacorrections
    # return out
    # TODO other color cases

    if not image.isimage(im):
        raise ValueError(im, 'im must be an image according to image.isimage')

    # ensure floats? unsure if cv.cvtColor operates on ints
    im = image.idouble(im)

    if conv == cv.COLOR_XYZ2BGR:
        # note that using cv.COLOR_XYZ2RGB does not seem to work properly?
        BGR_raw = cv.cvtColor(im, cv.COLOR_XYZ2BGR, **kwargs)

        # desaturate and rescale to constrain resulting RGB values to [0,1]
        B = BGR_raw[:, :, 0]
        G = BGR_raw[:, :, 1]
        R = BGR_raw[:, :, 2]
        add_white = -np.minimum(np.minimum(np.minimum(R, G), B), 0)
        B += add_white
        G += add_white
        R += add_white

        # inverse gamma correction
        B = _invgammacorrection(B)
        G = _invgammacorrection(G)
        R = _invgammacorrection(R)
        return np.stack((B, G, R), axis=2)  # BGR
    else:
        # TODO other color conversion cases
        return cv.cvtColor(np.float32(im), **kwargs)


def igamm(im, gam):
    """
    Inverse gamma correction

    :param im: image
    :type im: numpy array (N,M,3) or (N,M,1)?
    :param gam: string identifying srgb, or scalar to raise the image power
    :type gam: string or float TODO: variable input seems awkward
    :return: gamma corrected version of image im
    :rtype: numpy array, same shape as im

    Example::
        #TODO

    .. note::

        - Gamma decoding should be applied to any color image prior to colometric operations.
        - The exception to this is colorspace conversion using COLORSPACE which expects RGB images to be gamma encoded.
        - Gamma encoding is typically performed in a camera with GAMMA=0.45.
        - Gamma decoding is typically performed in the display with GAMMA=2.2.
        - For images with multiple planes the gamma correction is applied to all planes.
        - For images sequences the gamma correction is applied to all elements.
        - For images of type double the pixels are assumed to be in the range 0 to 1.
        - For images of type int the pixels are assumed in the range 0 to the
          maximum value of their class.  Pixels are converted first to double,
          processed, then converted back to the integer class.

    :references:

        - Robotics, Vision & Control, Chapter 10, P. Corke, Springer 2011.
    """

    # if not isinstance(gam, str):
    #    print('Warning: input variable "gam" is not a valid string')
    if not image.isimage(im):
        raise TypeError(im, 'im is not a valid image')
    im = np.array(im)

    if gam == 'srgb':
        # TODO check im is an image/valid input
        if np.issubdtype(im.dtype, np.float):
            f = im
        else:
            # convert image from int to float:
            f = np.float32(im) / np.iinfo(im.dtype).max

        # convert gamma-encoded sRGB to linear tristimulus values
        #Rg = im[:, :, 0]
        #Gg = im[:, :, 1]
        #Bg = im[:, :, 2]
        #R = _invgammacorrection(Rg)
        #G = _invgammacorrection(Gg)
        #B = _invgammacorrection(Bg)
        #g = np.stack((R, G, B), axis=2)
        g = _invgammacorrection(f)

        if not np.issubdtype(im.dtype, np.float):
            g *= np.iinfo(im.dtype).max
            g = g.astype(im.dtype)
    else:
        # normal power law:
        if np.issubdtype(im.dtype, np.float):
            g = im ** gam
        else:
            # int image
            maxg = np.float32((np.iinfo(im.dtype).max))
            g = ((im.astype(np.float32) / maxg) ** gam) * maxg
    return g


def ccxyz(lam, e=None):
    """
    Chromaticity coordinates

    :param lam: wavelength 𝜆 [m]
    :type lam: float or array_like
    :param e: illlumination spectrum defined at the wavelengths 𝜆
    :type e: numpy array (N,1)
    :return xyz: xyz-chromaticity coordinates
    :rtype: numpy array, shape = (N,3)

    Example::

        #TODO

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

    :param name: name of a color or name of a 3-element array corresponding to colors
    :type name: string or (numpy, tuple, list)
    :param opt: name of colorspace (eg 'rgb' or 'xyz' or 'xy' or 'ab')
    :type opt: string
    :return out: output
    :rtype: named tuple, name of color, numpy array in corresponding colorspace (if given)

    ``name`` is a string/list/set of color names, then colorname returns a 3-tuple of
    rgb tristimulus values.

    Example::

        #TODO

    .. note::

        - Color name may contain a wildcard, eg. "?burnt"
        - Based on the standard X11 color database rgb.txt
        - Tristiumuls values are [0,1]

    :references:

        - Robotics, Vision & Control, Chapter 14.3, P. Corke, Springer 2011.
    """
    # I'd say str in, 3 tuple out, or 3-element array like (numpy, tuple, list) in and str out

    assert isinstance(name, (str, tuple, list, np.ndarray)
                      ), 'name must be a string, tuple, list or np.ndarray'

    # load rgbtable (rbg.txt as a dictionary)
    print('loading rgb.txt')
    rgbfilename = (Path.cwd() / 'data' / 'rgb.txt').as_posix()
    rgbdict = _loadrgbdict(rgbfilename)

    if isinstance(name, str) or (isinstance(name, (list, set, tuple)) and isinstance(name[0], str)):
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
            # this loop can pick up multiple minima, often only when there are identical colour cases
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
    import pathlib
    import os.path

    # exec(open(os.path.join(pathlib.Path(__file__).parent.absolute(),
    # "test_color.py")).read())

    wcc = tristim2cc(np.r_[1, 1, 1])
