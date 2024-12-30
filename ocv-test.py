import numpy as np
from machinevisiontoolbox import Image


def fmtarrays(*arrays, widths=1, arraysep=" |", labels=None):
    """Pretty print small arrays

    :param array: one or more arrays to be formatted horizontally concantenated
    :type array: Numpy array
    :param width: number of digits for the formatted array elements, defaults to 1
    :type width: int, optional
    :param arraysep: separator between arrays, defaults to " |"
    :type arraysep: str, optional
    :param labels: list of labels for each array, defaults to None
    :type labels: list of str, optional
    :return: multiline string containing formatted arrays
    :rtype: str
    :raises ValueError: if the arrays have different numbers of rows

    For image processing this is useful for displaying small test images.

    The arrays are formatted and concatenated horizontally with a vertical separator.
    Each array has a header row that indicates the column number.  Each row has a
    header column that indicates the row number.

    .. runblock:: pycon

        >>> import numpy as np
        >>> rng = np.random.default_rng()
        >>> A = rng.integers(low=0, high=9, size=(5,5)
        >>> print(fmtarrays(A))
         >>> print(fmtarrays(A, width=2))
        >>> B = rng.integers(low=0, high=9, size=(5,5)
        >>> print(fmtarrays(A, B))
        >>> print(fmtarrays(A, B, labels=("A:", "B:")))

    The number of rows in each array must be the same, but the number of columns can
    vary.

    :seealso: :func:`Image.showpixels` :func:`Image`
    """

    # check that all arrays have the same number of rows
    if len(set([array.shape[0] for array in arrays])) != 1:
        raise ValueError("All arrays must have the same number of rows")

    if isinstance(widths, int):
        widths = [widths] * len(arrays)

    # add the header rows, which indicate the column number.  2-digit column
    # numbers are shown with the digits one above the other.
    stitle = ""  # array title row
    s10 = " " * 5  # array column number, 10s digit
    s1 = " " * 5  # array column number, 1s digit
    divider = " " * 5  # divider between column number header and array values
    s = ""
    tens = False  # has a tens row
    for i, array in enumerate(arrays):  # iterate over the input arrays
        width = widths[i]
        # make the pixel value format string based on the width of the value
        fmt = f" {{:{width}d}}"

        # build the title row
        if labels is not None:
            stitle += " " * (len(s10) - len(stitle)) + labels[i]

        # build the column number header rows
        for col in range(array.shape[1]):
            # the 10s digits
            if col // 10 == 0:
                s10 += " " * (width + 1)
            else:
                s10 += fmt.format(col // 10)
                tens = True
            # the 1s digits
            s1 += fmt.format(col % 10)
            divider += " " * width + "-"

        s10 += " " * len(arraysep)
        s1 += " " * len(arraysep)
        divider += " " * len(arraysep)

    # concatenate the header rows
    s = stitle + "\n"
    if tens:
        s += s10 + "\n"  # only include if there are 10s digits
    s += s1 + "\n" + divider + "\n"

    # add the element values, row by row
    for row in range(array.shape[0]):
        # add the row number
        s += f"{row:3d}: "

        # for each array, add the elements for this row
        for array, width in zip(arrays, widths):
            # make the pixel value format string based on the width of the value
            fmt = f" {{:{width}d}}"
            for col in range(array.shape[1]):
                s += fmt.format(array[row, col])
            if array is not arrays[-1]:
                s += arraysep
        s += "\n"

    return s


def fmtarray(array, width=1, label=None):
    """Format a 2D array as a string with a left-hand label

    :param array: array to format
    :type array: numpy array
    :param width: number of digits for the formatted array elements, defaults to 1
    :type width: int, optional
    :param label: labels for array, defaults to None
    :type label: str, optional
    :return: formatted array as a string
    :rtype: str

    .. runblock:: pycon

        >>> import numpy as np
        >>> rng = np.random.default_rng()
        >>> A = rng.integers(low=0, high=9, size=(5,5)
        >>> print(fmtarray(A, "A:"))
    """
    # make the pixel value format string based on the width of the value
    fmt = f" {{:{width}d}}"
    s = ""

    # add the element values, row by row
    labels = label.split("\n")
    longest = max([len(label) for label in labels])
    for row in range(array.shape[0]):
        if row < len(labels):
            s += labels[row] + " " * (longest - len(labels[row])) + "| "
        else:
            s += " " * longest + "| "
        # for each array, add the elements for this row
        for col in range(array.shape[1]):
            s += fmt.format(array[row, col])

        if row < array.shape[0] - 1:
            s += "\n"

    return s


# im = Image(
#     r"""
#             ..........
#             ..........
#             .#####....
#             ...###....
#             ...######.
#             ..........
#             ..........
#             """,
#     binary=True,
# )
# im = Image(
#     r"""
#             ..........
#             ....#.....
#             ....#.....
#             ....#.....
#             .#######..
#             ....#.....
#             ....#.....
#             ....#.....
#             ..........
#             """,
#     binary=True,
# )
# im = Image(
#     r"""
#         ##########
#         ########.#
#         #######.##
#         ######.###
#         ##.##.####
#         ###.######
#         ####.#####
#         #####.####
#         ##########

#             """,
#     binary=True,
# )
# im = Image(
#     r"""
#         #.#.#.
#         .#.#.#
#         #.#.#.
#         .#.#.#
#             """,
#     binary=True,
# )
im = Image(
    r"""
        ..........
        ....##....
        ....##....
        ..........
        ..........
        ..######..
        ..........

            """,
    binary=True,
)

# im = Image(
#     r"""
#             ..........
#             ..........
#             ...###....
#             ...###....
#             ...###....
#             ..........
#             ..........
#             """,
#     binary=True,
# )
# im = Image(
#     r"""
#             ..#.......
#             ....###...
#             .#........
#             ...####...
#             ...#..#...
#             ...####...
#             ..........
#             .#......#.
#             ..#....#..
#             ...#..#...
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


import cv2 as cv

contours, hierarchy = cv.findContours(
    im.to_int(), mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE
)
retval, labels = cv.connectedComponentsWithAlgorithm(
    im.to_int(), connectivity=8, ltype=cv.CV_32S, ccltype=cv.CCL_BBDT
)
perim = np.zeros(im.shape, dtype="uint8")
for i in range(len(contours)):
    cv.drawContours(perim, contours, i, i + 1, 1)
# print(fmtarrays(im.A, labels, perim, labels=("image:", "labels:", "perim:")))
print(
    fmtarrays(
        im.A,
        labels,
        perim,
        widths=(1, 1, 1),
        labels=("image:", "labels:", "contour idx+1:"),
    )
)

for i, contour in enumerate(contours):
    m = cv.moments(contour)
    print(
        fmtarray(
            np.squeeze(contour, axis=1).T, label=f"contour {i}\nm00 = {m['m00']:.0f}"
        )
    )
    print()
