from machinevisiontoolbox import Image
import cv2 as cv
import numpy as np

im = Image(
    r"""
            ..........
            ..........
            ..........
            ....##....
            ....##....
            ..........
            ..........
            ..........
            ..........
            ..........
            """,
    binary=True,
)
im.showpixels()


class Blob:

    def __init__(self, label, moments):
        self.label = label
        self.contour = None
        self.moments = moments
        self.children = []
        self.parent = None
        self.level = None

    def __str__(self):
        return f"Blob {self.label}, level {self.level}, parent {self.parent.label if self.parent is not None else '-'} with children {[child.label for child in self.children]}"

    def __repr__(self):
        return self.__str__()


# im = Image.Read("sharks.png")
im = Image.Read("multiblobs.png")

contours, hierarchy = cv.findContours(
    im.to_int(), mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE
)
# change contours to list of 2xN arraay
contours = [c[:, 0, :].T for c in contours]

retval, labels = cv.connectedComponentsWithAlgorithm(
    image=im.to_int(), connectivity=4, ltype=cv.CV_32S, ccltype=cv.CCL_BBDT
)
print(retval)

blobs = []
for label in range(retval):
    m = cv.moments((labels == label).astype("uint8"), True)
    print(f"Label {label} has {m['m00']} pixels")
    blobs.append(Blob(label, m))

blobdict = {}
for i, contour in enumerate(contours):
    u, v = contour[:, 0]
    label = labels[v, u]
    print(
        f" contour #{i} with point ({u}, {v}), belongs to label {label}, length {contour.shape[1]}"
    )
    blobs[label].contour = contour
    blobdict[i] = label  # map contour number to label

print(retval)
print(blobdict)

Image(labels).disp(block=True)

hierarchy = hierarchy.squeeze()
# print(hierarchy.shape)
# print(hierarchy)


def levels(c, hierarchy, blobs, level):
    blob = blobs[blobdict[c]]
    thislevel = []
    while c != -1:
        thislevel.append(c)

        if hierarchy[c, 2] != -1:
            # has a child
            blob.children = levels(hierarchy[c, 2], hierarchy, blobs, level + 1)

        # add reference to parent
        if hierarchy[c, 3] == -1:
            blob.parent = None
        else:
            blob.parent = blobdict[hierarchy[c, 3]]
        blob.level = level
        # get next at this level
        c = hierarchy[c, 0]

    return [blobdict[c] for c in thislevel]


print(hierarchy)
topblobs = levels(0, hierarchy, blobs, 0)

for blob in blobs:
    print(blob)
print()
for blob in topblobs:
    print(blob)

# v, u = np.nonzero(labels == 1)
# m00 = len(v)
# m10 = np.sum(u)
# m01 = np.sum(v)
# m11 = np.sum(u * v)
# m20 = np.sum(u**2)
# m02 = np.sum(v**2)
# m30 = np.sum(u**3)
# u0 = u - m10 / m00

m = cv.moments((labels == 1).astype("uint8"), True)
print(m)

Image(labels).disp(block=True)

# time execution using timeit
# import timeit

# print(
#     timeit.timeit(
#         "cv.findContours(im.to_int(), mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)",
#         globals=globals(),
#         number=1000,
#     )
# )  # 0.0001

# print(
#     timeit.timeit(
#         "cv.connectedComponents(image=im.to_int(), connectivity=4, ltype=cv.CV_32S)",
#         globals=globals(),
#         number=1000,
#     )
# )

# contours 0.458 ms, connectedComponents 0.680 ms
