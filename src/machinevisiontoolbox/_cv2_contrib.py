"""A clear error when OpenCV's contrib modules are not installed.

OpenCV's extra modules (``cv2.ximgproc``, ``cv2.xfeatures2d`` and so on) come only with
``opencv-contrib-python``.  Pyodide, and so JupyterLite, ships plain OpenCV without them,
as does an environment with ``opencv-python`` alone.  Calling one of these then fails with
``AttributeError: module 'cv2' has no attribute 'ximgproc'``, which says nothing about why.

:func:`install` puts a stand-in for each missing contrib module in ``cv2``.  Using it
raises :class:`ContribNotAvailableError`, which says what is missing and why.  That is an
:class:`AttributeError`, so ``hasattr(cv2.ximgproc, "x")`` is still False, and code that
tests for a feature this way keeps working.  A module that is present is left alone.
"""

from __future__ import annotations

import types
from typing import Any

# The modules of opencv_contrib that are not in plain OpenCV.  This is OpenCV's list, not
# MVTB's.  ``aruco`` is not here: it moved into the main modules in OpenCV 4.7.
CONTRIB_MODULES = (
    "bgsegm",
    "bioinspired",
    "ccalib",
    "face",
    "fuzzy",
    "hfs",
    "img_hash",
    "intensity_transform",
    "line_descriptor",
    "mcc",
    "optflow",
    "phase_unwrapping",
    "plot",
    "quality",
    "rapid",
    "reg",
    "rgbd",
    "saliency",
    "stereo",
    "structured_light",
    "surface_matching",
    "text",
    "tracking",
    "wechat_qrcode",
    "xfeatures2d",
    "ximgproc",
    "xobjdetect",
    "xphoto",
)


class ContribNotAvailableError(AttributeError):
    """A part of OpenCV's contrib modules was used, but they are not installed."""


class _MissingContrib(types.ModuleType):
    """Stands in for a contrib module that is not installed."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)  # not a use of the module: copy, pickle, ...
        raise ContribNotAvailableError(
            f"cv2.{self.__name__}.{name} is not available: OpenCV's contrib modules "
            f"are not installed, and {self.__name__} is one of them.  On a desktop, "
            "install them with:  pip install opencv-contrib-python.  They cannot be "
            "installed in JupyterLite, which has only plain OpenCV, so this feature "
            "is not available there."
        )


def install(cv2_module: Any) -> list[str]:
    """Add a stand-in for each contrib module that ``cv2_module`` lacks.

    :param cv2_module: the ``cv2`` module
    :returns: the names of the stand-ins added
    """
    added = []
    for name in CONTRIB_MODULES:
        if not hasattr(cv2_module, name):
            setattr(cv2_module, name, _MissingContrib(name))
            added.append(name)
    return added
