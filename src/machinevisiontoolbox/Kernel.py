"""
Convolution kernel classes for image processing operations.
"""

import matplotlib.pyplot as plt
import numpy as np
import spatialmath.base.argcheck as argcheck
from matplotlib import cm

from machinevisiontoolbox.mvtb_types import Dtype


class Kernel:
    def __init__(self, K, name=None):
        """
        Convolution kernel object

        :param K: kernel weighting matrix
        :type K: ndarray(N,M)
        :param name: name of the kernel, defaults to None
        :type name: str, optional
        :raises ValueError: ``K`` is not a 2D ndarray

        Kernel objects are used to represent convolution kernels for image
        processing operations. They are created by a number of class
        methods that generate common kernels such as Gaussian, Laplacian, etc.

        :class:`ImageCore.Image` :class:`machinevisiontoolbox.ImageCore.Image`  :class:`machinevisiontoolbox.Image`

        :seealso: :meth:`Gauss` :meth:`Laplace` :meth:`Sobel` :meth:`DoG` :meth:`LoG` :meth:`DGauss` :meth:`Circle` :meth:`Box`
        """
        if not isinstance(K, np.ndarray) and K.ndim != 2:
            raise ValueError("kernel must be a 2D ndarray")
        self.K = K
        self.name = name

    def __str__(self) -> str:
        """Human readable kernel description

        :return: summary description of the kernel
        :rtype: str

        The summary includes the size of the kernel, and its minimum, maximum and mean
        values .  If the kernel is symmetric this is noted.

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.Gauss(sigma=2)
            >>> print(K)

        """
        s = f"Kernel: {self.K.shape[0]}x{self.K.shape[1]}"
        s += f", min={self.K.min():.2g}, max={self.K.max():.2g}, mean={self.K.mean():.2g}"
        if np.allclose(self.K, self.K.T, rtol=1e-05, atol=1e-08):
            s += ", SYMMETRIC"
        if self.name is not None:
            s += f" ({self.name})"
        return s

    def __repr__(self) -> str:
        """Compact representation of the kernel

        :return: compact representation of the kernel
        :rtype: str

        The representation includes the size of the kernel, and its minimum, maximum and mean
        values.

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.Gauss(sigma=2)
            >>> K

        """
        return str(self)

    def disp3d(self, block=False, **kwargs):
        """Show kernel as a 3D surface plot

        :param block: block until plot is dismissed, defaults to False
        :type block: bool, optional

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.Gauss(5, h=15)
            >>> K.disp3d()

        .. plot::

            from machinevisiontoolbox import Kernel
            K = Kernel.Gauss(5, h=15)
            K.disp3d()
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        h = self.K.shape[0] // 2
        x = np.arange(-h, h + 1)
        y = np.arange(-h, h + 1)
        X, Y = np.meshgrid(x, y)
        kwargs.setdefault("linewidth", 0)
        kwargs.setdefault("antialiased", False)
        kwargs.setdefault("cmap", cm.coolwarm)
        ax.plot_surface(X, Y, self.K, **kwargs)
        ax.set_xlabel("u")
        ax.set_ylabel("v")

    @property
    def T(self):
        return Kernel(self.K.T)

    @property
    def shape(self):
        return self.K.shape

    def print(self, fmt=None, separator: str = " ", precision: int = 2) -> None:
        """
        Print kernel weights in compact format

        :param fmt: format string, defaults to None
        :type fmt: str, optional
        :param separator: value separator, defaults to single space
        :type separator: str, optional
        :param precision: precision for floating point kernel values, defaults to 2
        :type precision: int, optional

        Very compact display of kernel numerical values in grid layout.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.Gauss(sigma=2)
            >>> K.print()

        """
        if fmt is None:
            ff = f"{{:.{precision}f}}"
            width = max(len(ff.format(self.K.max())), len(ff.format(self.K.min())))
            fmt = f"{separator}{{:{width}.{precision}f}}"

        for v in range(self.K.shape[0]):
            row = ""
            for u in range(self.K.shape[1]):
                row += fmt.format(self.K[v, u])
            print(row)

    @classmethod
    def Gauss(cls, sigma, h=None):
        r"""
        Gaussian kernel

        :param sigma: standard deviation of Gaussian kernel
        :type sigma: float
        :param h: half width of the kernel
        :type h: integer, optional
        :return: 2h+1 x 2h+1 Gaussian kernel
        :rtype: :class:`Kernel`

        Return the 2-dimensional Gaussian kernel of standard deviation ``sigma``

        .. math::

            \mathbf{K} = \frac{1}{2\pi \sigma^2} e^{-(u^2 + v^2) / 2 \sigma^2}

        The kernel is centred within a square array with side length given by:

        - :math:`2 \mbox{ceil}(3 \sigma) + 1`, or
        - :math:`2 \mathtt{h} + 1`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.Gauss(sigma=1, h=2)
            >>> K.shape
            >>> print(K)
            >>> K.print()
            >>> K = Kernel.Gauss(sigma=2)
            >>> K.shape

        Example::

            >>> Kernel.Gauss(5, 15).disp3d()

        .. plot::

            from machinevisiontoolbox import Kernel
            K = Kernel.Gauss(5, h=15)
            K.disp3d()

        .. note::
            - The volume under the Gaussian kernel is one.
            - If the kernel is strongly truncated, ie. it is non-zero at the
              edges of the window then the volume will be less than one.

        :references:
            - |RVC3|, Section 11.5.1.1.

        :seealso: :meth:`DGauss`
        """

        # make sure sigma, w are valid input
        if h is None:
            h = np.ceil(3 * sigma)

        wi = np.arange(-h, h + 1)
        x, y = np.meshgrid(wi, wi)

        m = 1.0 / (2.0 * np.pi * sigma**2) * np.exp(-(x**2 + y**2) / 2.0 / sigma**2)
        # area under the curve should be 1, but the discrete case is only
        # an approximation
        # return m / np.sum(m)
        return cls(m / np.sum(m), name=f"Gaussian σ={sigma}")

    @classmethod
    def Laplace(cls):
        r"""
        Laplacian kernel

        :return: 3 x 3 Laplacian kernel
        :rtype: Kernel

        Return the Laplacian kernel

        .. math::

            \mathbf{K} = \begin{bmatrix}
                0 & 1 & 0 \\
                1 & -4 & 1 \\
                0 & 1 & 0
                \end{bmatrix}

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.Laplace()
            >>> K
            >>> K.print()

        .. note::
            - This kernel has an isotropic response to image gradient.

        :references:
            - |RVC3|, Section 11.5.1.3.

        :seealso: :meth:`LoG` :meth:`zerocross`
        """
        # fmt: off
        K = np.array([[ 0,  1,  0],
                      [ 1, -4,  1],
                      [ 0,  1,  0]])
        # fmt: on
        return cls(K, name="Laplacian")

    @classmethod
    def Sobel(cls):
        r"""
        Sobel edge detector

        :return: 3 x 3 Sobel kernel
        :rtype: Kernel

        Return the Sobel kernel for horizontal gradient

        .. math::

            \mathbf{K} = \frac{1}{8} \begin{bmatrix}
                1 & 0 & -1 \\
                2 & 0 & -2 \\
                1 & 0 & -1
                \end{bmatrix}

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.Sobel()
            >>> K
            >>> K.print()

        .. note::
            - This kernel is an effective vertical-edge detector
            - The y-derivative (horizontal-edge) kernel is ``K.T``

        :references:
            - |RVC3|, Section 11.5.1.3.

        :seealso: :meth:`DGauss`
        """
        # fmt: off
        K = np.array([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]]) / 8.0
        # fmt: on
        return cls(K, name="Sobel")

    @classmethod
    def DoG(cls, sigma1, sigma2=None, h=None):
        r"""
        Difference of Gaussians kernel

        :param sigma1: standard deviation of first Gaussian kernel
        :type sigma1: float
        :param sigma2: standard deviation of second Gaussian kernel
        :type sigma2: float, optional
        :param h: half-width of Gaussian kernel
        :type h: int, optional
        :return: 2h+1 x 2h+1 kernel
        :rtype: Kernel

        Return the 2-dimensional difference of Gaussian kernel defined by two
        standard deviation values:

        .. math::

            \mathbf{K} = G(\sigma_1) - G(\sigma_2)

        where :math:`\sigma_1 > \sigma_2`.
        By default, :math:`\sigma_2 = 1.6 \sigma_1`.

        The kernel is centred within a square array with side length given by:

        - :math:`2 \mbox{ceil}(3 \sigma) + 1`, or
        - :math:`2\mathtt{h} + 1`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.DoG(1)
            >>> K
            >>> K.print()

        Example::

            >>> Kernel.DoG(5, 15).disp3d()

        .. plot::

            from machinevisiontoolbox import Kernel
            K = Kernel.DoG(5, h=15)
            K.disp3d()

        .. note::
            - This kernel is similar to the Laplacian of Gaussian and is often
              used as an efficient approximation.
            - This is a "Mexican hat" shaped kernel

        :references:
            - |RVC3|, Section 11.5.1.3.

        :seealso: :meth:`LoG` :meth:`Gauss`
        """

        # sigma1 > sigma2
        if sigma2 is None:
            sigma2 = 1.6 * sigma1
        else:
            if sigma2 > sigma1:
                t = sigma1
                sigma1 = sigma2
                sigma2 = t

        # thus, sigma2 > sigma1
        if h is None:
            h = np.ceil(3.0 * sigma1)

        m1 = Kernel.Gauss(sigma1, h)  # thin kernel
        m2 = Kernel.Gauss(sigma2, h)  # wide kernel

        return cls(m2.K - m1.K, name=f"DoG σ1={sigma1}, σ2={sigma2}")

    @classmethod
    def LoG(cls, sigma, h=None):
        r"""
        Laplacian of Gaussian kernel

        :param sigma: standard deviation of first Gaussian kernel
        :type sigma: float
        :param h: half-width of kernel
        :type h: int, optional
        :return: 2h+1 x 2h+1 kernel
        :rtype: Kernel

        Return a 2-dimensional Laplacian of Gaussian kernel with
        standard deviation ``sigma``

        .. math::

            \mathbf{K} = \frac{1}{\pi \sigma^4} \left(\frac{u^2 + v^2}{2 \sigma^2} -1\right) e^{-(u^2 + v^2) / 2 \sigma^2}

        The kernel is centred within a square array with side length given by:

        - :math:`2 \mbox{ceil}(3 \sigma) + 1`, or
        - :math:`2\mathtt{h} + 1`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.LoG(1)
            >>> K
            >>> K.print()

        Example::

            >>> Kernel.LoG(5, 15).disp3d()

        .. plot::

            from machinevisiontoolbox import Kernel
            K = Kernel.LoG(5, h=15)
            K.disp3d()

        .. note:: This is the classic "Mexican hat" shaped kernel

        :references:
            - |RVC3|, Section 11.5.1.3.

        :seealso: :meth:`Laplace` :meth:`DoG` :meth:`Gauss` :meth:`zerocross`
        """

        if h is None:
            h = np.ceil(3.0 * sigma)
        wi = np.arange(-h, h + 1)
        x, y = np.meshgrid(wi, wi)

        log = (
            1.0
            / (np.pi * sigma**4.0)
            * ((x**2 + y**2) / (2.0 * sigma**2) - 1)
            * np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
        )

        # ensure that the mean is zero, for a truncated kernel this may not
        # be the case
        log -= log.mean()

        return cls(log, name=f"LoG σ={sigma}")

    @classmethod
    def DGauss(cls, sigma, h=None):
        r"""
        Derivative of Gaussian kernel

        :param sigma: standard deviation of Gaussian kernel
        :type sigma: float
        :param h: half-width of kernel
        :type h: int, optional
        :return: 2h+1 x 2h+1 kernel
        :rtype: Kernel

        Returns a 2-dimensional derivative of Gaussian
        kernel with standard deviation ``sigma``

        .. math::

            \mathbf{K} = \frac{-x}{2\pi \sigma^2} e^{-(x^2 + y^2) / 2 \sigma^2}

        The kernel is centred within a square array with side length given by:

        - :math:`2 \mbox{ceil}(3 \sigma) + 1`, or
        - :math:`2\mathtt{h} + 1`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.DGauss(1)
            >>> K
            >>> K.print()

        Example::

            >>> Kernel.DGauss(5, 15).disp3d()

        .. plot::

            from machinevisiontoolbox import Kernel
            K = Kernel.DGauss(5, h=15)
            K.disp3d()

        .. note::
            - This kernel is the horizontal derivative of the Gaussian, :math:`dG/dx`.
            - The vertical derivative, :math:`dG/dy`, is the transpose of this kernel.
            - This kernel is an effective edge detector.

        :references:
            - |RVC3|, Section 11.5.1.3.

        :seealso: :meth:`HGauss` :meth:`Gauss` :meth:`Sobel`
        """
        if h is None:
            h = np.ceil(3.0 * sigma)

        wi = np.arange(-h, h + 1)
        x, y = np.meshgrid(wi, wi)

        K = -x / sigma**2 / (2.0 * np.pi) * np.exp(-(x**2 + y**2) / 2.0 / sigma**2)
        return cls(K, name=f"DGauss σ={sigma}")

    @classmethod
    def HGauss(cls, sigma, h=None):
        r"""
        Hessian of Gaussian kernel

        :param sigma: standard deviation of Gaussian kernel
        :type sigma: float
        :param h: half-width of kernel
        :type h: int, optional
        :return: 2h+1 x 2h+1 kernels: Hxx, Hyy, Hxy
        :rtype: (Kernel, Kernel, Kernel)

        Returns the Hessian of Gaussian with standard deviation ``sigma`` as three
        2-dimensional kernels

        .. math::

            \mathbf{K}_{xx} &= \frac{x^2 - \sigma^2}{2\pi \sigma^3} e^{-(x^2 + y^2) / 2 \sigma^2} \\
            \mathbf{K}_{yy} &= \frac{y^2 - \sigma^2}{2\pi \sigma^3} e^{-(x^2 + y^2) / 2 \sigma^2} \\
            \mathbf{K}_{xy} &= \frac{xy}{2\pi \sigma^6} e^{-(x^2 + y^2) / 2 \sigma^2}


        The second derivative of an image :math:`\bf{I}` at point :math:`(x,y)` is
        given by:

        .. math::

            \begin{bmatrix} (\bf{K}_{xx} * \bf{I})_{x,y} & (\bf{K}_{xy} * \bf{I})_{x,y} \\ (\bf{K}_{xy} * \bf{I})_{x,y} & (\bf{K}_{yy} * \bf{I})_{x,y} \end{bmatrix}

        This second derivative matrix is the Gaussian curvature of the image at :math:`(x,y)`.

        The kernels are centred within a square array with side length given by:

        - :math:`2 \mbox{ceil}(3 \sigma) + 1`, or
        - :math:`2\mathtt{h} + 1`

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> Hxx, Hyy, Hxy = Kernel.HGauss(1)
            >>> Hxx
            >>> Hxx.print()

        Example::

            >>> Hxx, Hyy, Hxy = Kernel.HGauss(5, 15)
            >>> Hxx.disp3d()
            >>> Hyy.disp3d()
            >>> Hxy.disp3d()

        .. plot::

            from machinevisiontoolbox import Kernel
            Hxx, Hyy, Hxy = Kernel.HGauss(5, 15)
            Hxx.disp3d()

        .. plot::

            from machinevisiontoolbox import Kernel
            Hxx, Hyy, Hxy = Kernel.HGauss(5, 15)
            Hyy.disp3d()

        .. plot::

            from machinevisiontoolbox import Kernel
            Hxx, Hyy, Hxy = Kernel.HGauss(5, 15)
            Hxy.disp3d()

        :seealso: :meth:`DGauss` :meth:`Gauss` :meth:`Sobel`
        """
        if h is None:
            h = np.ceil(3.0 * sigma)

        wi = np.arange(-h, h + 1)
        x, y = np.meshgrid(wi, wi)

        K0 = np.exp(-(x**2 + y**2) / 2.0 / sigma**2)
        Kxx = (x**2 - sigma**2) / (2.0 * np.pi * sigma**3) * K0
        Kyy = (y**2 - sigma**2) / (2.0 * np.pi * sigma**3) * K0
        Kxy = (x * y) / (2.0 * np.pi * sigma**6) * K0

        return (
            cls(Kxx, name=f"Hxx σ={sigma}"),
            cls(Kyy, name=f"Hyy σ={sigma}"),
            cls(Kxy, name=f"Hxy σ={sigma}"),
        )

    @classmethod
    def Circle(cls, radius, h=None, normalize=False, dtype: Dtype = "uint8"):
        r"""
        Circular structuring element

        :param radius: radius of circular structuring element
        :type radius: scalar, array_like(2)
        :param h: half-width of kernel
        :type h: int
        :param normalize: normalize volume of kernel to one, defaults to False
        :type normalize: bool, optional
        :param dtype: data type for image, defaults to ``uint8``
        :type dtype: str or NumPy dtype, optional
        :return: 2h+1 x 2h+1 circular kernel
        :rtype: Kernel

        Returns a circular kernel of radius ``radius`` pixels. Sometimes referred
        to as a tophat kernel. Values inside the circle are set to one,
        outside are set to zero.

        If ``radius`` is a 2-element vector the result is an annulus of ones,
        and the two numbers are interpreted as inner and outer radii
        respectively.

        The kernel is centred within a square array with side length given
        by :math:`2\mathtt{h} + 1`.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.Circle(2)
            >>> K
            >>> K.print()
            >>> Kernel.Circle([2, 3])

        :references:
            - |RVC3|, Section 11.5.1.1.

        :seealso: :meth:`Box`
        """

        # check valid input:
        if not argcheck.isscalar(radius):  # r.shape[1] > 1:
            radius = argcheck.getvector(radius)
            rmax = radius.max()
            rmin = radius.min()
        else:
            rmax = radius

        if h is not None:
            w = h * 2 + 1
        elif h is None:
            w = 2 * rmax + 1

        s = np.zeros((int(w), int(w)), dtype=dtype)
        c = np.floor(w / 2.0)

        if not argcheck.isscalar(radius):
            # circle case
            x = np.arange(w) - c
            X, Y = np.meshgrid(x, x)
            r2 = X**2 + Y**2
            ll = np.where((r2 >= rmin**2) & (r2 <= rmax**2))
            s[ll] = 1
        else:
            # annulus case
            x = np.arange(w) - c
            X, Y = np.meshgrid(x, x)
            ll = np.where(np.round((X**2 + Y**2 - radius**2) <= 0))
            s[ll] = 1

        if normalize:
            s /= np.sum(s)
        return cls(s, name=f"Circle r={radius}")

    @classmethod
    def Box(cls, h, normalize=True):
        r"""
        Square structuring element

        :param h: half-width of kernel
        :type h: int
        :param normalize: normalize volume of kernel to one, defaults to True
        :type normalize: bool, optional
        :return: 2h+1 x 2h+1 kernel
        :rtype: Kernel

        Returns a square kernel with unit volume.

        The kernel is centred within a square array with side length given
        by :math:`2\mathtt{h} + 1`.

        Example:

        .. runblock:: pycon

            >>> from machinevisiontoolbox import Kernel
            >>> K = Kernel.Box(2)
            >>> K
            >>> K.print()
            >>> Kernel.Box(2, normalize=False)

        :references:
            - |RVC3|, Section 11.5.1.1.

        :seealso: :meth:`Circle`
        """
        # check valid input:
        wi = 2 * h + 1
        k = np.ones((wi, wi))
        if normalize:
            k /= np.sum(k)

        return cls(k, name=f"Box h={h}")
