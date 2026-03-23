* docstrings are ReST format for use with Sphinx.  Check syntax with ``rstcheck`` and use the Sphinx directives for parameters, return values, etc.  See https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html for details.
* when an API argument might be a scalar or an iterable I prefer to accept either and decide at run time, promoting the scalar to a list if needed using spatialmath.getvector()
* when a return value might be a scalar or single-element iterable (list, numpy array) I prefer to return the scalar value if possible.
* methods,arguments and docstrings use Australian English.  That's English English but with color and program, rather than colour and programme.
* all new code should use type hinting.
* mvtb_types.py includes some generic 1D and 2D array definitions, use when appropriate.
* don't add type annotations for local variables, only for function arguments and return values.
* use the new style of type hinting for class attributes, i.e. `field: type` rather than `field: type = None` and then assigning in the constructor.  This is more concise and less error prone.
