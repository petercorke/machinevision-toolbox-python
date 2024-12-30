import numpy
cimport numpy

# ilabel.c from MVTB for MATLAB converted to Python by CoPilot
#
# Copyright (C) 1995-2009, by Peter I. Corke


cdef unsigned int UNKNOWN = 0
import time


# def ilabel(numpy.ndarray[numpy.uint8_t, ndim=2] im, connectivity=4):
def ilabel(im, int connectivity=4):


    print("*****ilabel.pyx")

    def merge(int label1, int label2) -> int:

        # merge label1 and label2 (maybe indirected)
        # print(f"merge: {label1}, {label2}")
        label2 = LMAP[label2]
        # choose which label to keep
        if label1 > label2:
            label1, label2 = label2, label1
        # label1 dominates
        # print(
        #     f"merge ({row}, {col}): {label2}({blobsize[label2]}) -> {label1}({blobsize[label1]})"
        # )
        # print(limage)
        # print(lmap)
        # print()
        LMAP[label2] = label1
        blobsize[label1] += blobsize[
            label2
        ]  # dominant blob absorbs pixels from the other

        return label1

    tstart = time.time()

    cdef int height, width, row, col
    cdef unsigned int newlabel, curlab
    #cdef int curpix

    height, width = im.shape

    # LMAP = {}  # map old labels to new labels
    LMAP = numpy.zeros((20_000,), dtype=int)
    blobsize = {}  # size of each blob
    parent = {}  # parent of each blob
    color = {}  # color of each blob
    epoint = {}  # the enclosure point of each blob, guaranteed on the boundary

    # create the label image, initially all zeros (ie. no labels)
    # note that performance is much worse for uint16 or int32 types
    cdef numpy.ndarray[numpy.int_t, ndim=2] limage = numpy.zeros((height, width), dtype=int)

    newlabel = 0  # the next label to be assigned, incremented first
    for row in range(height):
        for col in range(width):

            ## assign a label based on already labelled neighbours to
            ## west or row above that have the same color
            curpix = im[row, col]
            curlab = UNKNOWN
            if col == 0:
                # if pix is the first pixel in the row, then the label is the same as the pixel above
                if curpix == im[row - 1, col]:
                    curlab = limage[row - 1, col]  # inherit label from the north
            else:
                # if pix is the same as the W pixel, then the label is the same
                if curpix == im[row, col - 1]:
                    curlab = limage[row, col - 1]  # inherit label from the west
                elif row > 0 and curpix == im[row - 1, col]:
                    curlab = LMAP[limage[row - 1, col]]
                    # )  # inherit label from the north
                # add 8-way stuff here

            if curlab == UNKNOWN:
                # current label is not inherited from a neighbour, assign a new label
                newlabel += 1  # assign new label
                curlab = newlabel
                color[curlab] = curpix  # set blob color to current pixel
                epoint[curlab] = None  # no enclosure point yet
                blobsize[curlab] = 0  # no pixels in blob yet
                LMAP[curlab] = curlab  # map new label to itself

            # check if a blob merge is required or an enclosure has occurred
            # these events can only occur on the second row or later
            if row > 0:
                if im[row - 1, col] == curpix:
                    # the current pixel is the same as the N pixel
                    if LMAP[limage[row - 1, col]] != curlab:
                        # but the label is different, we have a merge
                        curlab = merge(curlab, limage[row - 1, col])

                    elif im[row, col - 1] == curpix and im[row - 1, col - 1] != curpix:
                        # the current pixel is the same as the N and W pixel, but
                        # different to the NW pixel, so the NW pixel represents a blob
                        # that has been enclosed.

                        # print(f"enclosure at ({row}, {col})")
                        parent[limage[row - 1, col - 1]] = curlab
                        epoint[limage[row - 1, col - 1]] = (
                            row - 1,
                            col - 1,
                        )
                elif connectivity == 8:
                    # for 8-way connectivity, if the pixel above and to the left is the same as the current pixel, but the label is different

                    if (
                        col > 0
                        and im[row - 1, col - 1] == curpix
                        and LMAP[limage[row - 1, col - 1]] != curlab
                    ):
                        # we have a merge to the NW
                        curlab = merge(curlab, limage[row - 1, col - 1])
                    # for 8-way connectivity, if the pixel above and to the right is the same as the current pixel, but the label is different
                    elif (
                        col < (width - 1)
                        and im[row - 1, col + 1] == curpix
                        and LMAP[limage[row - 1, col + 1]] != curlab
                    ):
                        # we have a merge to the NE
                        curlab = merge(curlab, limage[row - 1, col + 1])

            blobsize[curlab] += 1  # bump the blob size by 1
            limage[row, col] = curlab  # stash label in label image
            # prevlab = curlab

    tc = time.time()
    print(f"connectivity: {tc - tstart:.2f}")
    # print(tc - tstart)

    # create a mapping from unique (after redirection) labels to sequential numbers
    # starting from zero
    lmap2 = {}
    #ulabels = set(LMAP.values())  # all the label redirection targets
    ulabels = list(numpy.unique(LMAP))
    ulabels.remove(0)
    for i, u in enumerate(ulabels):
        lmap2[u] = i
    # print(lmap2)

    # create a mapping from labels to sequential numbers starting from zero
    #lmap3 = {old: lmap2[new] for (old, new) in LMAP.items()}
    lmap3 = {old: lmap2[LMAP[old]] for old in range(1, newlabel + 1)}

    # print(lmap3)

    # apply the mapping to the label image
    limage2 = numpy.vectorize(lambda oldlabel: lmap3[oldlabel])(limage)
    # print(limage2)

    # apply the mapping to the keys of the other dicts
    parent = {lmap3[child]: lmap3[parent] for (child, parent) in parent.items()}
    color = {
        lmap3[oldlabel]: color
        for (oldlabel, color) in color.items()
        if oldlabel in lmap2
    }
    epoint = {
        lmap3[oldlabel]: edge
        for (oldlabel, edge) in epoint.items()
        if oldlabel in lmap2
    }
    blobsize = {
        lmap3[oldlabel]: size
        for (oldlabel, size) in blobsize.items()
        if oldlabel in lmap2
    }

    tf = time.time()
    print(f"total: {tf - tstart:.2f}, tc: {tc - tstart:.2f}, tf: {tf - tc:.2f}")
    print(f"newlabel: {newlabel}")
    # print(tf - tstart)
    return len(lmap2), limage2, parent, blobsize, epoint, color
