
1.0.0 January 2025

* `Kernel` methods now return `Kernel` instances rather than NumPy arrays. Methods that
accept a kernel can accept a `Kernel` instance or a NumPy array.  Methods exist to 
sringify or print a kernel.

* The indexing order of an `Image` object (using square bracket `__getitem__` access) has
  changed and is now `img[u,v]` where `u` is the column coordinate and `v` is the row
  coordinate.  This is consistent with the column-first convention used across the
  Toolbox and is consistent with the $(u,v)$ coordinate system for images. But, this
  is the __opposite__ order to that used for NumPy index on the underlying array, and
  to earlier versions of the Toolbox. 

    - a 2-tuple of integers, select this pixel.  If the image has multiple planes, the
      result is a vector over planes.
    - a 3-tuple of integers, for a multiplane image select this pixel from the specified
      plane.
    - a 2-tuple of slice objects, select this region. If the image has multiple planes,
      the result is a 3D array.
    - a 3-tuple of slice objects, select this region of uv and planes
    - an int, select this plane
    - a string, select this named plane or planes

* added `pixel(u,v)` method for faster access to a single pixel value, scalar or vector.

* the children of a `Blob` is now given as a list of `Blob` objects, not their indices
within the overall list of blobs.  This simplies traversing the blob hierarchy tree.
Similarly, the parent is a reference to the parent `Blob` object rather than an index,
and is `None` if the blob has no parent (its parent is the background).

* Documentation overhaul, both in-code docstrings, and the organization of the overall Sphinx document.

* Additional unit tests