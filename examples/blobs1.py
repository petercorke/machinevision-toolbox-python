from machinevisiontoolbox import Image
import matplotlib.pyplot as plt

mb = Image.Read("shark2.png")
print(mb)
mb.disp()
blobs = mb.blobs()
print(len(blobs))

blobs.plot_labelbox(color="yellow")


print(blobs)

plt.show(block=True)
