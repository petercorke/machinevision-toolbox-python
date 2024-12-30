import numpy as np
from machinevisiontoolbox import Image

# from numba import jit
import cProfile

# ilabel.c from MVTB for MATLAB converted to Python by CoPilot
#
# Copyright (C) 1995-2009, by Peter I. Corke

from ilabel import ilabel

UNKNOWN = 0
import time


def pprint(array, before=None, width=1):
    """Pretty print a small image array

    :param array: _description_
    :type array: _type_
    :param before: _description_, defaults to None
    :type before: _type_, optional
    :param width: _description_, defaults to 1
    :type width: int, optional
    :return: multiline string containing formatted array
    :rtype: str

    .. runblock:: pycon

        >>> import numpy as np
        >>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> print(showarray(A))
        >>> s2 = showarray(2*A, width=2)
        >>> print(showarray(A, before=s2))

    :seealso: :func:`Image.showpixels` :func:`Image`
    """

    # make the pixel value format string based on the width of the value
    fmt = f" {{:{width}d}}"

    # add the header rows, which indicate the column number.  2-digit column
    # numbers are shown with the digits one above the other.
    s10 = " " * 5
    s1 = " " * 5
    sep = " " * 5
    s = ""
    for col in range(array.shape[1]):
        if col // 10 == 0:
            s10 += " " * (width + 1)
        else:
            s10 += fmt.format(col // 10)
        s1 += fmt.format(col % 10)
        sep += " " * width + "-"
    s = s10 + "\n" + s1 + "\n" + sep + "\n"

    # add the pixel values, row by row
    for row in range(array.shape[0]):
        s += f"{row:3d}: "
        for col in range(array.shape[1]):
            s += fmt.format(array[row, col])
        s += "\n"

    # if there is a before image, then join the two images side by side
    # this assumes that the images have the same number of rows
    if before is not None:
        # horizontal join
        return "\n".join(
            [x + " |" + y[5:] for x, y in zip(before.splitlines(), s.splitlines())]
        )
    else:
        return s


# im = Image(
#     r"""
#             ..........
#             ..........
#             ..........
#             ....##....
#             ....##....
#             ..........
#             ..........
#             ..........
#             ..........
#             ..........
#             """,
#     binary=True,
# )
# im = Image(
#     r"""
#             ..#.......
#             ..........
#             ..........
#             ...####...
#             ...#..#...
#             ...####...
#             ..........
#             ..........
#             ..........
#             ..........
#             """,
#     binary=True,
# )
# im = Image(
#     r"""
#             ..........
#             ..........
#             .#.#.#.#..
#             .#.#.#.#..
#             .#.#.#.#..
#             .########.
#             ..#.#.#.#.
#             ..#.#.#.#.
#             ..........
#             ..........
#             """,
#     binary=True,
# )
# im = Image(
#     r"""
#             ............
#             ............
#             .########...
#             ..#.........
#             ..#.........
#             ............
#             ............
#             """,
#     binary=True,
# )
# z = False
# print(f"{'T' if z else 'F'}")
# fmt = "{'T' if z else 'F'}"
# fmt.format(True)
# aa = pprint(im.A)
im = Image.Read("multiblobs.png")
# time execution

nlabels, limage, parent, blobsize, edge, color = ilabel(im.A, 4)


print(f"{nlabels} labels")
# print(pprint(limage, before=aa))
# Image(limage).disp(block=True)
print(f"parents: {parent}")
print(f"blobsize: {blobsize}")
print(im.size, sum(parent.values()))
print(f"edge: {edge}")
print(f"color: {color}")

# cProfile.run("ilabel(im.A, 4, 0)")
