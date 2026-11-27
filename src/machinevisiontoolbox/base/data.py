"""
Locates Machine Vision Toolbox data files within the separately installed ``mvtb-data``
package, sidestepping PyPI size limits by keeping large data independent of code.
"""

from __future__ import annotations

import difflib
import importlib
import importlib.resources
from pathlib import Path
from typing import Any, Callable


def mvtb_load_matfile(filename: str) -> dict[str, Any]:
    """
    Load toolbox mat format data file

    :param filename: relative pathname of datafile
    :type filename: str
    :raises ValueError: File does not exist
    :return: contents of mat data file
    :rtype: dict

    Reads a MATLAB format *mat* file which can contain multiple variables, in
    a binary or ASCII format.  Returns a dict where the keys are the variable
    names and the values are NumPy arrays.

    .. note::
        - Uses SciPy ``io.loadmat`` to do the work.
        - If the filename has no path component it will be
          first be looked for in the folder ``machinevisiontoolbox/data``, then
          the current working directory.

    .. seealso:: :func:`mvtb_path_to_datafile` :func:`scipy.io.loadmat`
    """
    from collections import namedtuple

    from scipy.io import loadmat
    from scipy.io.matlab import mat_struct

    # get results as a dict
    data = mvtb_load_data(filename, loadmat, squeeze_me=True, struct_as_record=False)

    # if elements are a scipy.io.matlab.mat_struct, that is, they
    # were a MATLAB struct, convert them to a namedtuple
    for key, value in data.items():
        if isinstance(value, mat_struct):
            # print("fixing")
            nt = namedtuple("matstruct", value._fieldnames)  # type: ignore[misc]  # field names are only known at run time
            data[key] = nt(*[getattr(value, n) for n in value._fieldnames])

    return data


def mvtb_load_jsonfile(filename: str) -> dict[str, Any]:
    """
    Load toolbox JSON format data file

    :param filename: relative pathname of datafile
    :type filename: str
    :raises ValueError: File does not exist
    :return: contents of JSON data file
    :rtype: dict

    Reads a JSON format file which can contain multiple variables and return a
    dict where the keys are the variable names and the values are Python data
    types.

    .. note::
        - If the filename has no path component it will be
          first be looked for in the folder ``machinevisiontoolbox/data``, then
          the current working directory.

    .. seealso:: :func:`mvtb_path_to_datafile`
    """
    import json

    return mvtb_load_data(filename, lambda f: json.load(open(f, "r")))


def mvtb_load_data(filename: str, handler: Callable[..., Any], **kwargs: Any) -> Any:
    """
    Load toolbox data file

    :param filename: relative pathname of datafile
    :type filename: str
    :param handler: function to read data
    :type handler: callable
    :param kwargs: keyword arguments passed to ``handler``
    :type kwargs: dict
    :raises ValueError: File does not exist
    :return: data object
    :rtype: Any

    Resolves the relative pathname to an absolute name and then invokes the
    data reading function::

        handler(abs_file_name, **kwargs)

    For example::

        data = mvtb_load_data('data/foo.dat', lambda f: data_load(open(f, 'r')))

    .. note:: If the filename has no path component it will
        first be looked for in the folder ``machinevisiontoolbox/data``, then
        the current working directory.

    .. seealso:: :func:`mvtb_path_to_datafile`
    """
    path = mvtb_path_to_datafile(filename)
    return handler(path, **kwargs)


def _not_found_message(filename: tuple[str | Path, ...], root: Path) -> str:
    """Readable message for a file that is neither local nor in mvtbdata

    The message names the file as the caller wrote it, without the folder
    inside ``mvtbdata`` that was added to the search, and suggests files with
    similar names if there are any.
    """
    folder = root.joinpath(*filename[:-1])
    where = "mvtbdata"
    if len(filename) > 1:
        where += f"/{Path(*filename[:1])}"
        name = Path(*filename[1:])
    else:
        name = Path(filename[0])
    msg = f"file {str(name)!r} not found in the current folder or in {where}"
    if folder.is_dir():
        # best matches first, ties in alphabetical order
        names = sorted(p.name for p in folder.iterdir() if not p.name.startswith("."))
        score = {n: difflib.SequenceMatcher(None, name.name, n).ratio() for n in names}
        close = sorted((n for n in names if score[n] >= 0.6), key=lambda n: -score[n])[
            :3
        ]
        if close:
            msg += "; did you mean " + ", ".join(close) + "?"
    return msg


def mvtb_path_to_datafile(
    *filename: str | Path, local: bool = True, string: bool = False
) -> Path | str:
    """
    Get absolute path to file in MVTB data package

    :param filename: pathname components of image file
    :type filename: str or Path
    :param local: search for file locally first, default True
    :type local: bool
    :param string: return as string, default False
    :type string: bool
    :raises FileNotFoundError: File is neither local nor in the mvtbdata package
    :return: Absolute path
    :rtype: Path or str

    The data associated with the Machine Vision Toolbox for Python is shipped
    as a separate package.

    The positional arguments are joined, like ``os.path.join``, for example::

        mvtb_path_to_datafile('data', 'solar.dat')  # data/solar.dat

    If ``local`` is True then ``~`` is expanded and if the file exists, the
    path is made absolute, and symlinks resolved::

        mvtb_path_to_datafile("foo.dat")         # find ./foo.dat
        mvtb_path_to_datafile("~/foo.dat")       # find $HOME/foo.dat

    Otherwise, the file is sought within the ``mvtbdata`` package and if found,
    return that absolute path.

    .. rubric:: Examples

    .. runblock:: pycon

        >>> from machinevisiontoolbox import mvtb_path_to_datafile
        >>> mvtb_path_to_datafile('data', 'solar.dat')   # read mvtbdata/data/solar.dat

    """

    path = Path(filename[-1]).expanduser()

    if path.is_absolute():
        # path is absolute
        path = path.resolve()  # make it absolute
        if path.exists():
            if string:
                return str(path)
            else:
                return path
        else:
            raise FileNotFoundError(f"file {str(path)!r} not found")

    # file is either local or in mvtbdata
    path = Path(filename[-1]).expanduser()

    if local:
        #  ignore the first element of filename if given, it's only for the mvtbdata case

        path = path.resolve()  # make it absolute
        if path.exists():
            if string:
                return str(path)
            else:
                return path

    # otherwise, look for it in mvtbdata

    root = Path(str(importlib.resources.files("mvtbdata")))
    # if folder:
    #     root = root / folder
    # root = Path(__file__).parent.parent / "images"

    path = root.joinpath(*filename).resolve()
    if path.exists():
        if string:
            return str(path)
        else:
            return path
    else:
        raise FileNotFoundError(_not_found_message(filename, root))
