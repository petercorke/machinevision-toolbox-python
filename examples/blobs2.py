from machinevisiontoolbox import Image
import matplotlib.pyplot as plt


mb = Image.Read("multiblobs.png")
mb.disp()

blobs = mb.blobs()
print(f"there are {len(blobs)} in this image")

print(blobs)

print("we can slice out a single blob")
b5 = blobs[5]
print(b5)
print(f"it has {len(b5.children)} child blobs")

blobs.dotfile(show=True)

labels, n = mb.labels_binary()
labels.disp(title="label image", colorbar=True, block=True)
