from pathlib import Path
import importlib

"""
The data associated with the Machine Vision Toolbox for Python is shipped
as a separate package::

$ pip install mvtb-data

which is a dependency.  This is to sidestep PyPI package size limits, and
reflect the reality that the data changes much less frequently than the code.

The functions in this module locate the specified files within the separately
installed package in the user's filesystem.
"""

def mvtb_load_matfile(filename):
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
    
    :seealso: :func:`mvtb_path_to_datafile` :func:`scipy.io.loadmat`
    """
    from scipy.io import loadmat
    from scipy.io.matlab.mio5_params import mat_struct
    from collections import namedtuple

    # get results as a dict
    data = mvtb_load_data(filename, loadmat, squeeze_me=True, struct_as_record=False)

    # if elements are a scipy.io.matlab.mio5_params.mat_struct, that is, they
    # were a MATLAB struct, convert them to a namedtuple
    for key, value in data.items():
        if isinstance(value, mat_struct):
            print('fixing')
            nt = namedtuple("matstruct", value._fieldnames)
            data[key] = nt(*[getattr(value, n) for n in value._fieldnames])
        
    return data

def mvtb_load_jsonfile(filename):
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
    
    :seealso: :func:`mvtb_path_to_datafile` 
    """
    import json

    return mvtb_load_data(filename, lambda f: json.load(open(f, 'r')))

def mvtb_load_data(filename, handler, **kwargs):
    """
    Load toolbox data file

    :param filename: relative pathname of datafile
    :type filename: str
    :param handler: function to read data
    :type handler: callable
    :raises ValueError: File does not exist
    :return: data object

    Resolves the relative pathname to an absolute name and then invokes the
    data reading function::

        handler(abs_file_name, **kwargs)

    For example::

        data = mvtb_load_data('data/foo.dat', lambda f: data_load(open(f, 'r')))

    .. note:: If the filename has no path component it will 
        first be looked for in the folder ``machinevisiontoolbox/data``, then
        the current working directory.

    :seealso: :func:`mvtb_path_to_datafile`
    """
    path = mvtb_path_to_datafile(filename)
    return handler(path, **kwargs)

def mvtb_path_to_datafile(*filename, local=True, string=False):
    """
    Get absolute path to file in MVTB data package

    :param filename: pathname of image file
    :type filename: str
    :param local: search for file locally first, default True
    :type local: bool
    :raises FileNotFoundError: File does not exist
    :return: Absolute path
    :rtype: Path

    The data associated with the Machine Vision Toolbox for Python is shipped
    as a separate package.
    
    The positional arguments are joined, like ``os.path.join``, for example::

        mvtb_path_to_datafile('data', 'solar.dat')  # data/solar.dat

    If ``local`` is True then ``~`` is expanded and if the file exists, the
    path is made absolute, and symlinks resolved::
        
        mvtb_path_to_datafile('foo.dat')         # find ./foo.dat
        mvtb_path_to_datafile('~/foo.dat')       # find $HOME/foo.dat

    Otherwise, the file is sought within the ``mvtbdata`` package and if found,
    return that absolute path.

    Example:

    .. runblock:: pycon

        >>> from machinevisiontoolbox import mvtb_path_to_datafile
        >>> mvtb_path_to_datafile('data', 'solar.dat')   # read mvtbdata/data/solar.dat

    """

    filename = Path(*filename)

    if local:
        # check if file is in user's local filesystem

        p = filename.expanduser()
        p = p.resolve()
        if p.exists():
            if string:
                p = str(p)
            return p

    # otherwise, look for it in mvtbdata

    mvtbdata = importlib.import_module("mvtbdata")
    root = Path(mvtbdata.__path__[0])
    # if folder:
    #     root = root / folder
    # root = Path(__file__).parent.parent / "images"
    
    path = root / filename
    if path.exists():
        p = path.resolve()
        if string:
            p = str(p)
        return p
    else:
        raise ValueError(f"file {filename} not found locally or in mvtbdata")
