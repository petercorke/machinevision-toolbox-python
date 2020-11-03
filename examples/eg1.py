import machinevisiontoolbox as mvtb 
from machinevisiontoolbox import Image, Blob 

# # im = Image("machinevisiontoolbox/images/flowers?.png")
im = Image("flowers1.png")
# # im.disp()
print(im)

mb = Image("multiblobs.png")
# mb.disp()

blobs = mb.blobs()
print(len(blobs))

print(blobs)

a = blobs[0]
print(len(a))
print(a.u)
print(a)

sift = mb.SIFT(nfeatures=10)
print(sift[:10])



print(sift[0])