from pathlib import Path
import importlib


def mvtb_path_to_datafile(*filename, folder=None, local=True):
    """
    Get absolute path to image file

    :param filename: pathname of image file
    :type filename: str
    :param local: search for file locally first, default True
    :type local: bool
    :raises FileNotFoundError: File does not exist
    :return: Absolute path
    :rtype: Path

    The positional arguments are joined, like ``os.path.join``.

    If ``local`` is True then ``~`` is expanded and if the file exists, the
    path is made absolute, and symlinks resolved.

    Otherwise, the file is sought within the ``rtbdata`` package and if found,
    return that absolute path.

    Example::

        loadmat('data/map1.mat')   # read rtbdata/data/map1.mat
        loadmat('foo.dat')         # read ./foo.dat
        loadmat('~/foo.dat')       # read $HOME/foo.dat
    """

    filename = Path(*filename)

    if local:
        # check if file is in user's local filesystem

        p = filename.expanduser()
        p = p.resolve()
        if p.exists():
            return p

    # otherwise, look for it in mvtbdata

    mvtbdata = importlib.import_module("mvtbdata")
    root = Path(mvtbdata.__path__[0])
    if folder:
        root = root / folder
    # root = Path(__file__).parent.parent / "images"
    
    path = root / filename
    if path.exists():
        return path.resolve()
    else:
        raise ValueError(f"file {filename} not found locally or in mvtbdata")
