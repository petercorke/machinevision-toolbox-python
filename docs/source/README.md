
# How to references methods and classes

Note that intersphinx support has been dropped by OpenCV.

## Within a class definition

You can reference the class itself using ``:class:`~classname` ``

You can reference its own methods by ``:meth:`~methodname` `` where the `~` suppresses the name prefix.

A method in another class is referenced by ``:meth:~.~methodname` ``.