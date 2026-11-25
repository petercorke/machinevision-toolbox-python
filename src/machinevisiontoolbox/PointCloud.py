"""
Thin wrapper around Open3D point-cloud objects with MVTB look and feel.

Open3D is an open-source library that supports rapid development of software that deals
with 3D data. The Open3D frontend exposes a set of carefully selected data structures
and algorithms in both C++ and Python. The backend is highly optimized and is set up for
parallelization.

The ``PointCloud`` class provides a subset of commonly used methods but does not
attempt to wrap every Open3D method.  Instead, any method or property not known to the
``PointCloud`` class, such as ``pc.some_method()``, falls through to the native Open3D
method ``some_method()`` on the encapsulated Open3D object, and all parameters are
passed through.  If the return value is an Open3D point cloud, or a tuple that contains
one, it is wrapped and returned as a ``PointCloud`` instance.  A name that neither class
knows raises :class:`AttributeError`.

Open3D uses NumPy arrays in which points are rows.  This is different to the MVTB
convention of points as columns.  The MVTB wrappers handle this transposition, but
anything that falls through to Open3D, or that is reached through :attr:`pcd`, uses the
Open3D convention.
"""

import difflib
from pathlib import Path
from typing import Any, Callable
from warnings import warn

import numpy as np
import spatialmath.base as smb
from spatialmath import SE3

from machinevisiontoolbox.base.color import name2color

try:
    import open3d as o3d

    _open3d = True
except ImportError:
    _open3d = False


def _show_in_notebook(
    geometry: Any, title: str, file: str | None, kwargs: dict[str, Any]
) -> None:
    """Draw an Open3D geometry in the output of the notebook cell, using Plotly

    :param geometry: an Open3D point cloud or triangle mesh
    :param title: title of the figure
    :param file: ``disp``'s ``file`` argument, which cannot be used here
    :param kwargs: view options passed to
        :func:`open3d.visualization.get_plotly_fig`
    :raises ValueError: ``file`` is given
    :raises ImportError: Plotly is not installed, it is not a dependency of the
        Toolbox
    """
    if file is not None:
        raise ValueError("file cannot be used with notebook=True")
    try:
        import plotly  # noqa: F401
    except ImportError:
        raise ImportError(
            "disp(notebook=True) draws with Plotly, which is not installed. "
            "Install it with: pip install plotly"
        ) from None

    # get_plotly_fig rather than draw_plotly, which shows the figure and also returns
    # it, so that a cell ending in the call shows it twice
    fig = o3d.visualization.get_plotly_fig([geometry], **kwargs)
    # the three axes have the same scale, so shapes have their true proportions as
    # in the Open3D window.  Plotly's default, "auto", does that unless one axis is
    # more than four times longer than the other two, when it draws a cube instead
    # and squashes a long or flat scene
    fig.update_layout(
        title=dict(text=title), margin=dict(t=40), scene_aspectmode="data"
    )
    fig.show()


class PointCloud:
    """
    A set of 3D points, optionally with colors and surface normals

    A thin wrapper around an Open3D :obj:`open3d.geometry.PointCloud` that keeps the
    Toolbox conventions: points, colors and normals are 3xN arrays with one point per
    column.  Methods that Open3D provides and that are not wrapped here are forwarded
    to it.  See the constructor for the ways to create a point cloud.

    .. seealso:: :meth:`__init__`, :meth:`Read`, :meth:`DepthImage`, :meth:`disp`
    """

    def __init__(self, arg: Any, depth_scale: float = 1.0, **kwargs) -> None:
        """
        Create new point cloud object

        :param arg: point cloud data as an Open3D or MVTB PointCloud object, 3xN array of points.
        :type arg: :obj:`o3d.geometry.PointCloud`, :obj:`~machinevisiontoolbox.PointCloud`, ndarray(3,N)
        :param depth_scale: depth scale factor, defaults to 1.0
        :type depth_scale: float, optional
        :raises RuntimeError: PointCloud class requires Open3D to be installed: pip install open3d or pip install machinevision-toolbox-python[open3d]
        :raises ValueError: bad arguments

        This object wraps an Open3D :obj:`open3d.geometry.PointCloud` object.  It can be
        created from:

        - an Open3D point cloud object
        - a 3xN array of point coordinates
        - a depth image as a 1-plane :class:`~machinevisiontoolbox.Image` instance
        - an RGBD image as a 4-plane :class:`~machinevisiontoolbox.Image` instance
        - a tuple of two images (RGB, D) as 3-plane and 1-plane :class:`~machinevisiontoolbox.Image` instances

        .. warning:: Open3D must be installed.

        The wrapped Open3D :obj:`open3d.geometry.PointCloud` object has many methods and properties
        but rather than writing individual wrappers the :meth:`__getattr__` picks up these
        references and invokes them on the underlying Open3D object.  Any returned Open3D
        point cloud objects are converted to a :class:`PointCloud` instance.

        .. seealso:: :meth:`__getattr__` :obj:`open3d.geometry.PointCloud` :class:`~machinevisiontoolbox.Image` :class:`CentralCamera`
        """
        if not _open3d:
            raise RuntimeError(
                "PointCloud class requires Open3D to be installed: "
                "pip install open3d or "
                "pip install machinevision-toolbox-python[open3d]\n"
                " ** Note that Open3D typically does not run on the latest Python version(s)"
            )

        if not _open3d:
            raise RuntimeError(
                "PointCloud class requires Open3D to be installed: "
                "pip install open3d or "
                "pip install machinevision-toolbox-python[open3d]"
            )

        if isinstance(arg, o3d.geometry.PointCloud):
            # Open3D point cloud
            pcd = arg

        elif isinstance(arg, np.ndarray):
            arg = smb.getmatrix(arg, (3, None)).astype("float32")

            # simple point cloud:
            # passed a 3xN array of point coordinates
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(arg.T)

        else:
            raise ValueError(
                "arg must be PointCloud or ndarray, for a depth image use DepthImage() constructor"
            )

        self._pcd = pcd

    @staticmethod
    def _intrinsics(camera: Any) -> Any:
        """Return an Open3D style camera intrinsic object

        :param camera: camera model
        :type camera: :class:`~machinevisiontoolbox.Camera.CentralCamera`
        :return: Open3D camera intrinsic object
        :rtype: :class:`open3d.camera.PinholeCameraIntrinsic`
        """
        return o3d.camera.PinholeCameraIntrinsic(
            camera.width, camera.height, *camera.fpix, *camera.pp
        )

    @staticmethod
    def _image(image: Any) -> Any:
        """Return an Open3D style image object

        :param image: MVTB image
        :type image: :class:`~machinevisiontoolbox.Image` instance
        :return: Open3D image
        :rtype: :class:`open3d.geometry.Image` instance
        """
        img = image._A
        if not img.data.c_contiguous:
            img = img.copy()  # make contiguous
        return o3d.geometry.Image(img)

    def copy(self) -> "PointCloud":
        """
        Copy point cloud

        :return: copy of point cloud
        :rtype: :class:`PointCloud`

        Return a new point cloud instance that contains a copy of
        the point data.
        """
        return self.__class__(o3d.geometry.PointCloud(self._pcd))

    def __len__(self) -> int:
        """
        Number of points

        :return: number of points
        :rtype: int
        """
        return np.asarray(self._pcd.points).shape[0]

    # allow methods of o3d.geometry.PointCloud to be invoked indirectly
    def __getattr__(self, name: str) -> Any:
        """
        Access Open3D PointCloud methods and attributes

        :param name: method or attribute name
        :type name: str
        :raises AttributeError: neither this class nor the Open3D point cloud has
            a method or property of this name

        If ``P`` is a :class:`PointCloud` then ``P.name`` invokes
        method ``name`` of the underlying :obj:`open3d.geometry.PointCloud`
        object.  For example:

            .. code-block:: python

                pcfiltered, ind = pc.remove_radius_outlier(radius=0.2, nb_points=100)
                pc.is_empty()  # invoke is_empty predicate method
                pc.colors   # get colors attribute


        .. note::
            - This is a convenience method to avoid having to explicitly all
              the properties and methods of the Open3D point cloud object.
            - If the method returns a new point cloud object, or a tuple that
              contains one, it is converted to a Toolbox :class:`PointCloud`
              instance.
            - Arguments and return values other than point clouds are passed
              through unchanged, so arrays use the Open3D convention of one point
              per row, not the Toolbox convention of one point per column.

        .. versionchanged:: 2.5.0
            An unknown name raises :class:`AttributeError`, as Python expects, so
            ``hasattr`` and ``getattr`` with a default work.  It used to raise
            :class:`ValueError`.  The message names the two classes that were
            searched and suggests similar names.  A point cloud that is returned
            by itself is converted, not only one inside a tuple.

        .. seealso:: :attr:`pcd`
        """
        # copy and pickle create an instance without calling __init__, so there is
        # no wrapped object, and looking for it here would call this method again
        if name == "_pcd":
            raise AttributeError(name)

        def wrap(value: Any) -> Any:
            if isinstance(value, o3d.geometry.PointCloud):
                return PointCloud(value)
            if isinstance(value, tuple):
                return tuple(wrap(v) for v in value)
            return value

        if hasattr(self._pcd, name):
            attr = getattr(self._pcd, name)
            if callable(attr):

                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    return wrap(attr(*args, **kwargs))

                return wrapper
            return wrap(attr)

        # not a method or property of either class, so suggest similar names
        known = {
            n
            for n in dir(self._pcd) + dir(type(self))
            if not n.startswith("_") and n[0].islower()
        }
        close = difflib.get_close_matches(name, sorted(known), n=2, cutoff=0.8)
        msg = (
            f"'{type(self).__name__}' object has no attribute '{name}': it is not "
            "a method or property of PointCloud, or of the Open3D point cloud "
            "that it wraps"
        )
        if close:
            msg += ". Did you mean " + " or ".join(f"'{c}'" for c in close) + "?"
        raise AttributeError(msg)

    def __str__(self) -> str:
        """
        Concise string representation of point cloud parameters

        :return: concise string representation of the wrapped point cloud
        :rtype: str
        """
        return str(self._pcd)

    def __repr__(self) -> str:
        """
        Short string representation of the point cloud

        :return: number of points, and whether the points have colors
        :rtype: str
        """
        return f"PointCloud(npoints={len(self)}, colored={self.iscolor})"

    @property
    def pcd(self) -> Any:
        """
        The wrapped Open3D point cloud

        :return: the Open3D object this instance wraps
        :rtype: :obj:`open3d.geometry.PointCloud`

        This is the object itself, not a copy, so changing it changes this
        point cloud.  It is for calling Open3D functions that the Toolbox does not
        wrap.  Open3D uses Nx3 arrays, where the Toolbox uses 3xN.

        .. seealso:: :attr:`points`, :attr:`colors`
        """
        return self._pcd

    @property
    def points(self) -> np.ndarray:
        """
        Get points as array

        :return: points as array
        :rtype: ndarray(3,N)

        Points are returned as columns of the array. The columns correspond
        to the same columns in the :attr:`colors` and :attr:`normal` properties.

        .. seealso:: :attr:`colors` :attr:`normals`
        """
        return np.asarray(self._pcd.points).T

    @property
    def colors(self) -> np.ndarray:
        """
        Get point color data as array

        :return: point color
        :rtype: ndarray(3,N)

        Point colors are returned as columns of the array. The columns correspond
        to the same columns in the :attr:`points` and :attr:`normal` properties.

        .. seealso:: :attr:`points`, :attr:`normals`, :attr:`iscolor`, :meth:`paint`
        """
        return np.asarray(self._pcd.colors).T

    @property
    def iscolor(self) -> bool:
        """
        Test if the points have colors

        :return: ``True`` if every point has a color
        :rtype: bool

        An empty point cloud has no colors.

        .. seealso:: :attr:`colors`, :meth:`paint`, :meth:`unpaint`
        """
        return self._pcd.has_colors()

    @classmethod
    def Read(cls, filename: str, *args: Any, **kwargs: Any) -> "PointCloud":
        """
        Create point cloud from file

        :param filename: name of file
        :type filename: str
        :return: new point cloud
        :rtype: :class:`PointCloud` instance

        Read point cloud data from a file using :func:`open3d.io.read_point_cloud`.
        Supported filetypes include PLY.

        If the path is not absolute it is first searched for relative
        to the current directory, and if not found, it is searched for in
        the ``data`` folder of the
        `mvtb_data package <https://github.com/petercorke/machinevision-toolbox-python/tree/master/packages/mvtb-data>`_.

        .. note:: ``filename`` may be given as a bare name (``'bunny.ply'``)
            or with a redundant leading ``data/`` (``'data/bunny.ply'``, the
            older convention some book examples still use) -- both resolve
            to the same file.

        """

        from machinevisiontoolbox import mvtb_path_to_datafile

        # accept a caller-supplied redundant "data/" prefix (older
        # convention) without double-joining it against the "data" this
        # method already prepends
        parts = Path(filename).parts
        if parts and parts[0] == "data":
            filename = str(Path(*parts[1:])) if len(parts) > 1 else ""

        path = mvtb_path_to_datafile("data", filename, string=True)
        pcd = o3d.io.read_point_cloud(str(path), *args, **kwargs)  # type: ignore[arg-type]  # stubs say PathLike
        return cls(pcd)

    @classmethod
    def DepthImage(
        cls,
        depth: Any,
        camera: Any,
        rgb: Any = None,
        depth_scale: float = 1.0,
        **kwargs: Any,
    ) -> "PointCloud":
        """Create point cloud from depth image

        Creates a colored or uncolored point cloud.

        :param depth: depth image or an RGBD image.
        :type depth: :class:`~machinevisiontoolbox.Image`
        :param camera: perspective camera model
        :type camera: :class:`~machinevisiontoolbox.Camera.CentralCamera`
        :param rgb: RGB image for colorizing point cloud, defaults to None
        :type rgb: :class:`~machinevisiontoolbox.Image`, optional
        :param depth_scale: depth scale factor, defaults to 1.0
        :type depth_scale: float, optional
        :raises RuntimeError: PointCloud class requires Open3D to be installed: pip install open3d or pip install machinevision-toolbox-python[open3d]
        :raises ValueError: input images has wrong number of planes

        The depth image is converted to a point cloud using the Open3D, and the pixel
        values are scaled by ``depth_scale``.  If the depth image is a multi-plane image
        then the ``D`` plane is used.

        There are two sources of color for point colorization:

            - RGB planes of the depth image, if present.
            - ``rgb`` is an optional RGB image.

        Camera intrinsics are required and provided by a :class:`~machinevisiontoolbox.CentralCamera` instance.

        """
        from machinevisiontoolbox import Image

        if camera is None:
            raise ValueError("camera model must be provided")
        if not isinstance(depth, Image):
            raise ValueError("depth image must be an Image instance")

        if depth.nplanes == 4:
            rgb = depth["RGB"]
            depth = depth["D"]

        if rgb is None:
            # no RGB, uncolored point cloud
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                PointCloud._image(depth),
                PointCloud._intrinsics(camera),
                depth_scale=depth_scale,
                **kwargs,
            )
        else:
            # colored point cloud
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color=PointCloud._image(rgb),
                depth=PointCloud._image(depth),
                depth_scale=depth_scale,
                **kwargs,
            )

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, PointCloud._intrinsics(camera)
            )

        return cls(pcd)

    def write(self, filename: str) -> bool:
        """
        Write point cloud to file

        :param filename: name of file
        :type filename: str
        :return: return status
        :rtype: bool

        The file format is determined by the file extension.
        """
        return o3d.io.write_point_cloud(str(filename), self._pcd)  # type: ignore[arg-type]

    def disp(
        self,
        block: bool = False,
        file: str | None = None,
        title: str = "Point Cloud",
        notebook: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Display point cloud using Open3D

        :param block: block until window dismissed, defaults to False
        :type block: bool, optional
        :param title: window title, defaults to "Point Cloud"
        :type title: str, optional
        :param file: save display as an image, defaults to None
        :type file: str, optional
        :param notebook: draw in the output of the notebook cell, rather than in an
            Open3D window, defaults to False
        :type notebook: bool, optional
        :raises ValueError: ``file`` is given with ``notebook=True``
        :raises ImportError: ``notebook=True`` and Plotly is not installed
        :param front: set front vector
        :type front: array_like(3)
        :param lookat: set lookat vector
        :type lookat: array_like(3)
        :param up: set up vector
        :type up: array_like(3)
        :param zoom: set zoom value
        :type zoom: float

        Various viewing options are passed to the Open3D :class:`open3d.visualization.ViewControl`.

        Typing "h" in the display window will print help information to the console.

        With ``notebook=True`` the point cloud is drawn in the output of the notebook
        cell, as a Plotly figure that can be rotated and zoomed.  This needs
        `Plotly <https://plotly.com/python/>`_, which is not installed with the
        Toolbox: ``pip install plotly``.  ``block`` and ``file`` do not apply, and
        besides the view options above, ``width``, ``height`` and
        ``point_sample_factor`` (the fraction of the points drawn, 0 to 1, which
        keeps a large cloud responsive) are passed to
        :func:`open3d.visualization.draw_plotly`.  The three axes have the same
        scale, so shapes have their true proportions, as in the Open3D window.

        .. versionchanged:: 2.5.0
            Added the ``notebook`` option.

        .. seealso:: :class:`open3d.visualization.ViewControl`
        """
        if notebook:
            _show_in_notebook(self._pcd, title, file, kwargs)
            return

        if block:
            o3d.visualization.draw_geometries([self._pcd], window_name=title, **kwargs)
        else:
            # nonblocking display
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=title)
            vis.add_geometry(self._pcd)

            # get object to control viewpoint
            view_control = vis.get_view_control()

            # handle the possible options
            if "front" in kwargs:
                view_control.set_front(kwargs["front"])
            if "lookat" in kwargs:
                view_control.set_lookat(kwargs["lookat"])
            if "up" in kwargs:
                view_control.set_up(kwargs["up"])
            if "zoom" in kwargs:
                view_control.set_zoom(kwargs["zoom"])

            # update the display
            vis.poll_events()
            vis.update_renderer()

            # save to file if requested
            if file is not None:
                vis.capture_screen_image(str(file), do_render=False)  # type: ignore[arg-type]

    def __rmul__(self, T: SE3) -> "PointCloud":
        """
        Overloaded * operator to transform points

        :param T: rigid-body transform to apply to the point cloud
        :type T: :class:`~spatialmath.pose3d.SE3`
        :return: point cloud
        :rtype: :class:`PointCloud`

        Transform a :class:`PointCloud` by premultiplying by an
        :class:`~spatialmath.pose3d.SE3` instance.

        .. seealso:: :meth:`transform` :meth:`__imul__`
        """
        if isinstance(T, SE3):
            return self.copy().transform(T)
        else:
            return NotImplemented

    def __imul__(self, T: SE3) -> "PointCloud":
        """
        Overloaded ``*=`` operator to transform points

        :param T: rigid-body transform to apply to the point cloud
        :type T: :class:`~spatialmath.pose3d.SE3`
        :return: point cloud
        :rtype: :class:`PointCloud`

        Transform a :class:`PointCloud` by inplace multiplication by an
        :class:`~spatialmath.pose3d.SE3` instance.

        .. seealso:: :meth:`__rmul__` :meth:`transform`
        """
        if isinstance(T, SE3):
            self.transform(T)
            return self
        else:
            return NotImplemented

    def transform(self, T: SE3, inplace: bool = True) -> "PointCloud":
        """
        Transform point cloud

        :param T: rigid-body transform to apply to the point cloud
        :type T: :class:`~spatialmath.pose3d.SE3`
        :param inplace: transform points in place, defaults to True
        :type inplace: bool, optional
        :return: point cloud
        :rtype: :class:`PointCloud`

        If ``inplace`` is False then the point cloud data is copied
        before transformation.

        .. seealso:: :meth:`__rmul__` :meth:`__imul__`
        """
        if inplace:
            self._pcd.transform(T.A)
            return self
        else:
            new = self.copy()
            new._pcd.transform(T.A)
            return new

    def concatenate(
        self,
        *others: "PointCloud",
        color: Any = (0.7, 0.7, 0.7),
    ) -> "PointCloud":
        """
        Join point clouds

        :param others: one or more point clouds to join to this one
        :type others: :class:`PointCloud`
        :param color: color for the points of clouds that have no color: an RGB
            value in the range [0, 1] or a color name, or ``None`` to remove the
            colors from the result, defaults to light gray
        :type color: array_like(3), str or None, optional
        :return: a new point cloud containing the points of all the clouds
        :rtype: :class:`PointCloud`
        :raises ValueError: ``color`` is a string that is not a known color name

        The points are joined in order, this cloud first.  The clouds that are
        joined are not changed.

        If none of the clouds has color, the result has none.  If some of them
        have color and others do not, every point must have a color, so the
        points that have none are given ``color``.  With ``color=None`` the
        colors of all points are dropped instead.  An empty point cloud has no
        color, and does not affect the colors of the result.

        The ``+`` operator does the same with the default ``color``.

        .. versionadded:: 2.5.0

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import numpy as np
            >>> from machinevisiontoolbox import PointCloud
            >>> red = PointCloud(np.random.rand(3, 4)).paint("red")
            >>> plain = PointCloud(np.random.rand(3, 3))
            >>> both = red.concatenate(plain, color="blue")
            >>> both.colors
            >>> red.concatenate(plain, color=None).iscolor

        .. seealso:: :meth:`paint`, :meth:`unpaint`, :meth:`__add__`, :attr:`iscolor`
        """
        if color is not None and not isinstance(color, str):
            color = np.asarray(color, dtype=float)
        elif isinstance(color, str):
            rgb = name2color(color)
            if rgb is None:
                raise ValueError(f"unknown color name: {color!r}")
            color = rgb

        clouds = (self,) + others
        pcd = o3d.geometry.PointCloud()
        keep_color = color is not None and any(c.iscolor for c in clouds)
        for c in clouds:
            pcd.points.extend(c.points.T)
            if keep_color:
                if c.iscolor:
                    pcd.colors.extend(c.colors.T)
                else:
                    pcd.colors.extend(np.tile(color, (len(c), 1)))
        return self.__class__(pcd)

    def __add__(self, other: "PointCloud") -> "PointCloud":
        """
        Overloaded the + operator to concatenate point clouds

        :param other: second point cloud
        :type other: :class:`PointCloud`
        :return: concatenated point clouds
        :rtype: :class:`PointCloud`

        Points that have no color are given light gray if the other cloud has
        color.  Use :meth:`concatenate` to choose the color, or to drop the colors.

        .. versionchanged:: 2.5.0
            Joining a colored and an uncolored cloud gives every point a color.  The
            result used to have fewer colors than points, so it displayed with none.

        .. seealso:: :meth:`concatenate`, :meth:`__radd__`
        """
        return self.concatenate(other)

    def __radd__(self, other: Any) -> "PointCloud":
        """
        Overloaded the + operator so that :func:`sum` works on point clouds

        :param other: the integer 0, which :func:`sum` starts from
        :return: this point cloud, unchanged
        :rtype: :class:`PointCloud`

        ``sum(clouds)`` begins by computing ``0 + clouds[0]``.  This method
        accepts that zero, so a list of point clouds can be joined with
        ``sum(clouds)``.

        .. versionadded:: 2.5.0

        .. seealso:: :meth:`__add__`, :meth:`concatenate`
        """
        if isinstance(other, int) and other == 0:
            return self
        return NotImplemented

    def downsample_voxel(self, voxel_size: float) -> "PointCloud":
        """
        Downsample point cloud by voxelization

        :param voxel_size: voxel dimension
        :type voxel_size: float
        :return: downsampled point cloud
        :rtype: :class:`PointCloud`

        Point cloud resolution is reduced by keeping only one
        point per voxel.

        .. seealso:: :meth:`downsample_random`
        """
        return self.__class__(self._pcd.voxel_down_sample(voxel_size))

    # random downsample
    def downsample_random(
        self, fraction: float, seed: int | None = None
    ) -> "PointCloud":
        """
        Downsample point cloud by random selection

        :param fraction: fraction of points to retain
        :type fraction: float
        :param seed: random number seed, defaults to None
        :type seed: int, optional
        :return: downsampled point cloud
        :rtype: :class:`PointCloud`

        Point cloud resolution is reduced by randomly selecting
        a subset of points.

        .. seealso:: :meth:`downsample_random`
        """
        if seed is None:
            return self.__class__(self._pcd.random_down_sample(fraction))

        if seed >= 0:
            np.random.seed(seed)
        n = len(self)
        ind = np.random.choice(n, int(fraction * n), replace=False)
        return self.__class__(self._pcd.select_by_index(ind.tolist()))

    def estimate_normals(self, **kwargs: Any) -> None:
        """
        Estimate point normals

        Normals are computed and stored within the Open3D point cloud
        object.  They can be:

        - accessed using the ``normals`` property
        - toggled on for display when the point cloud is displayed.

        .. seealso:: :meth:`disp` :meth:`open3d.geometry.PointCloud.estimate_normals`
        """
        # radius=0.1, max_nn=30
        self._pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(**kwargs)
        )

    @property
    def normals(self) -> np.ndarray:
        """
        Get point normal data as array

        :return: point normal
        :rtype: ndarray(3,N)

        Point normals are returned as columns of the array.  The columns correspond
        to the same columns in the :attr:`points` and :attr:`colors` properties.

        .. seealso:: :meth:`estimate_normals` :attr:`points` :attr:`colors`
        """
        return np.asarray(self._pcd.normals).T

    def remove_outlier(self, **kwargs: Any) -> "PointCloud":
        """
        Remove point cloud outliers

        :return: cleaned up point cloud
        :rtype: :class:`PointCloud`

        Remove outlying points. Any point with fewer than ``nb_points`` neighbors
        within a radius of ``radius`` is removed.

        .. deprecated:: 1.0.3
            Use :meth:`open3d.geometry.PointCloud.remove_radius_outlier` directly.

        .. seealso:: :meth:`open3d.geometry.PointCloud.remove_radius_outlier`
        """
        warn(
            "Deprecated in 1.0.3: use pc.remove_radius_outlier(radius=, nb_points=)[0] instead of remove_outlier().",
            DeprecationWarning,
            stacklevel=2,
        )
        # nb_points=16, radius=0.2
        pcd, ind = self._pcd.remove_radius_outlier(**kwargs)
        return self.__class__(pcd)

    def segment_plane(self, **kwargs: Any) -> tuple[Any, "PointCloud", "PointCloud"]:
        """
        Find the dominant plane using RANSAC

        :param distance_threshold: largest distance of a point from the plane for
            it to count as part of the plane, in the units of the points
        :type distance_threshold: float
        :param ransac_n: number of points sampled to fit each candidate plane,
            defaults to 3
        :type ransac_n: int, optional
        :param num_iterations: number of candidate planes tried, defaults to 1000
        :type num_iterations: int, optional
        :param probability: required probability of finding the best plane,
            Open3D's default is 0.99999999
        :type probability: float, optional
        :return: plane parameters, points on the plane, all the other points
        :rtype: ndarray(4), :class:`PointCloud`, :class:`PointCloud`
        :raises TypeError: ``distance_threshold`` is not given

        Uses RANSAC to fit a plane :math:`aX + bY + cZ + d = 0` to the points.  Each
        iteration fits a plane to ``ransac_n`` randomly chosen points, and keeps the
        plane that has the most points within ``distance_threshold`` of it.  The
        arguments are passed to
        :meth:`open3d.geometry.PointCloud.segment_plane`.

        The first return value is the array :math:`(a, b, c, d)`.  The second is
        a new point cloud of the points within ``distance_threshold`` of the plane,
        the inliers, and the third is a new point cloud of the remaining points.  The
        original point cloud is not changed.  Points of an object that lie close
        to the plane, such as the foot of an object standing on a ground plane, are
        inliers.

        .. versionchanged:: 2.5.0
            The default ``num_iterations`` is 1000.  It was 100, which sometimes
            returned a poor plane.

        RANSAC chooses its points at random, so the plane and the number of inliers
        differ slightly from run to run.  With few iterations a poor plane is
        sometimes returned.  Increase ``num_iterations``, or seed Open3D's random
        number generator with ``open3d.utility.random.seed(n)`` to get the same result
        every time.

        .. rubric:: Examples

        .. runblock:: pycon

            >>> import numpy as np
            >>> from machinevisiontoolbox import PointCloud
            >>> rng = np.random.default_rng(0)
            >>> ground = np.vstack((rng.random((2, 300)), np.zeros(300)))  # z = 0
            >>> objects = rng.random((3, 100)) + np.array([[0], [0], [0.5]])  # above it
            >>> pc = PointCloud(np.hstack((ground, objects)))
            >>> plane, on_plane, rest = pc.segment_plane(distance_threshold=0.01)
            >>> len(on_plane), len(rest)

        .. seealso:: :meth:`select`, :meth:`remove_outlier`
        """
        default_dict = dict(ransac_n=3, num_iterations=1000)
        kwargs = {**default_dict, **kwargs}

        plane_model, inliers = self._pcd.segment_plane(**kwargs)
        inlier_cloud = self._pcd.select_by_index(inliers)
        outlier_cloud = self._pcd.select_by_index(inliers, invert=True)
        return plane_model, self.__class__(inlier_cloud), self.__class__(outlier_cloud)

    def select(
        self, ind: np.ndarray | Callable[..., bool], invert: bool = False
    ) -> "PointCloud":
        """
        Select points by index

        :param ind: indices or lambda function
        :type ind: ndarray() or callable
        :param invert: exclude points, defaults to False
        :type invert: bool, optional
        :return: subset of point cloud
        :rtype: :class:`PointCloud`

        Create point cloud from points with indices given by integer
        array ``ind``.  If ``invert`` is True then select those points
        not given by ``ind``.

        If ``ind`` is a callable function then it is applied to each column of the
        3xN array of points and the point is included if the function returns True.
        For example:

            .. code-block:: python

                pcd.select(lambda p: p[2] < 2)


        will select all points with a z-coordinate less than 2.

        """
        if callable(ind):
            idx = np.where(np.apply_along_axis(ind, axis=0, arr=self.points))[0]
        else:
            idx = np.array(ind)
        return self.__class__(self._pcd.select_by_index(idx.tolist(), invert=invert))

    def paint(self, color: Any) -> "PointCloud":
        """
        Colorize point cloud

        :param color: color for points, either an RGB value in the range [0, 1]
            or the name of a color, such as ``"red"`` or ``"skyblue"``
        :type color: array_like(3), str
        :return: colorized point cloud
        :rtype: :class:`PointCloud`
        :raises ValueError: ``color`` is a string that is not a known color name

        Paint all points in the point cloud this color.

        .. versionchanged:: 2.5.0
            ``color`` can be a color name, which is converted by
            :func:`~machinevisiontoolbox.base.color.name2color`.

        .. seealso:: :meth:`unpaint`, :meth:`concatenate`, :attr:`colors`, :attr:`iscolor`
        """
        if isinstance(color, str):
            rgb = name2color(color)
            if rgb is None:
                raise ValueError(f"unknown color name: {color!r}")
            color = rgb
        self._pcd.paint_uniform_color(color)
        return self

    def unpaint(self) -> "PointCloud":
        """
        Remove the colors from a point cloud

        :return: the point cloud, now without color
        :rtype: :class:`PointCloud`

        The opposite of :meth:`paint`.  The points are kept.  The point cloud is
        changed, and returned so calls can be chained.

        .. versionadded:: 2.5.0

        .. seealso:: :meth:`paint`, :attr:`iscolor`, :meth:`concatenate`
        """
        self._pcd.colors = o3d.utility.Vector3dVector()
        return self

    def ICP(
        self,
        data: "PointCloud",
        T0: SE3 | None = None,
        max_correspondence_distance: float = 1,
        **kwargs: Any,
    ) -> tuple[SE3, Any]:
        """
        Register point cloud using ICP

        :param data: point cloud to register
        :type data: :class:`PointCloud`
        :param T0: initial transform, defaults to None
        :type T0: :class:`~spatialmath.pose3d.SE3`, optional
        :param max_correspondence_distance: distance beyond which correspondence is broken, defaults to 1
        :type max_correspondence_distance: float, optional
        :return: pose of ``data`` with respect to instance points
        :rtype: :class:`~spatialmath.pose3d.SE3`

        .. seealso:: :obj:`open3d.pipelines.registration.TransformationEstimationPointToPoint`
        """
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

        # Convergence-Criteria for Vanilla ICP
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(**kwargs)

        if T0 is None:
            T0 = np.eye(4)
        else:
            T0 = T0.A

        status = o3d.pipelines.registration.registration_icp(
            self._pcd, data._pcd, max_correspondence_distance, T0, estimation, criteria
        )
        # voxel_size, save_loss_log)

        T = SE3(smb.trnorm(status.transformation))

        return T, status

    def voxel_grid(self, voxel_size: float) -> "VoxelGrid":
        """
        Voxelize point cloud

        :param voxel_size: voxel dimension
        :type voxel_size: float
        :return: voxel grid
        :rtype: :class:`VoxelGrid`
        """
        return VoxelGrid(self, voxel_size)


class VoxelGrid:
    """
    A point cloud divided into cubic cells

    Each cell, or voxel, that contains at least one point of the point cloud is
    occupied.  It is created by :meth:`PointCloud.voxel_grid`.

    .. seealso:: :meth:`PointCloud.voxel_grid`, :meth:`disp`, :meth:`write`
    """

    def __init__(self, pcd: PointCloud, voxel_size: float) -> None:
        """
        Create a voxel grid from a point cloud

        :param pcd: the point cloud
        :type pcd: :class:`PointCloud`
        :param voxel_size: edge length of a voxel, in the units of the points
        :type voxel_size: float

        A voxel takes the color of the points it contains, if they have color.
        Use :meth:`PointCloud.voxel_grid` rather than calling this directly.

        .. seealso:: :meth:`PointCloud.voxel_grid`
        """
        self._voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd._pcd, voxel_size=voxel_size
        )

    def _cubes(self) -> Any:
        """The voxels as a triangle mesh, one cube of 12 triangles per voxel

        Open3D's Plotly drawing does not handle a voxel grid, it draws nothing.  The
        voxels keep their colors, or are gray if none has a color, so that the
        lighting shows their shape.
        """
        voxels = self._voxels.get_voxels()
        index = np.array([v.grid_index for v in voxels], dtype=float).reshape(-1, 3)
        colors = np.array([v.color for v in voxels], dtype=float).reshape(-1, 3)
        if not np.any(colors):
            colors = np.full_like(colors, 0.7)
        n = len(voxels)

        # the unit cube: 8 corners, and 12 triangles which are 2 on each face
        corners = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
            + [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
            dtype=float,
        )
        faces = np.array(
            [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7]]
            + [[0, 1, 5], [0, 5, 4], [3, 6, 2], [3, 7, 6]]
            + [[0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]]
        )

        # the grid index is of the voxel's minimum corner, from the grid's origin
        minimum = np.asarray(self._voxels.origin) + index * self._voxels.voxel_size
        vertices = minimum[:, None, :] + corners[None, :, :] * self._voxels.voxel_size
        triangles = faces[None, :, :] + 8 * np.arange(n)[:, None, None]

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.reshape(-1, 3))
        mesh.triangles = o3d.utility.Vector3iVector(triangles.reshape(-1, 3))
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.repeat(colors, 8, axis=0))
        return mesh

    def write(self, filename: str) -> None:
        """
        Write voxel grid to file

        :param filename: filename
        :type filename: str

        .. seealso:: :obj:`open3d.io.write_voxel_grid`
        """
        o3d.io.write_voxel_grid(str(filename), self._voxels)  # type: ignore[arg-type]

    def disp(
        self,
        block: bool = True,
        file: str | None = None,
        title: str = "Voxel Grid",
        notebook: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Display voxel grid

        :param block: block until window is dismissed, defaults to True
        :type block: bool, optional
        :param title: window title, defaults to "Voxel Grid"
        :type title: str, optional
        :param file: save display to this filename, defaults to None
        :type file: str, optional
        :param notebook: draw in the output of the notebook cell, rather than in an
            Open3D window, defaults to False
        :type notebook: bool, optional
        :raises ValueError: ``file`` is given with ``notebook=True``
        :raises ImportError: ``notebook=True`` and Plotly is not installed

        With ``notebook=True`` the voxels are drawn in the output of the notebook
        cell, as a Plotly figure that can be rotated and zoomed, in the same way as
        :meth:`PointCloud.disp`, which describes the options, and with the same
        scale on the three axes.  Each voxel is drawn as a cube of 12 triangles, so a
        grid of many thousands of voxels is slow.

        .. versionchanged:: 2.5.0
            Added the ``notebook`` option.

        .. seealso:: :meth:`PointCloud.disp`
        """
        if notebook:
            _show_in_notebook(self._cubes(), title, file, kwargs)
            return

        if block:
            o3d.visualization.draw_geometries(
                [self._voxels], window_name=title, **kwargs
            )
        else:
            # nonblocking display
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=title)
            vis.add_geometry(self._voxels)

            # get object to control viewpoint
            view_control = vis.get_view_control()

            # handle the possible options
            if "front" in kwargs:
                view_control.set_front(kwargs["front"])
            if "lookat" in kwargs:
                view_control.set_lookat(kwargs["lookat"])
            if "up" in kwargs:
                view_control.set_up(kwargs["up"])
            if "zoom" in kwargs:
                view_control.set_zoom(kwargs["zoom"])

            render = vis.get_render_option()
            render.mesh_show_wireframe = True

            # update the display
            vis.poll_events()
            vis.update_renderer()

            # save to file if requested
            if file is not None:
                vis.capture_screen_image(str(file), do_render=False)  # type: ignore[arg-type]


if __name__ == "__main__":
    from pathlib import Path

    import pytest

    pytest.main(
        [
            str(Path(__file__).parent.parent.parent / "tests" / "test_pointcloud.py"),
            "-v",
        ]
    )
