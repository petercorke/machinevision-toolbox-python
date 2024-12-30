import code

# import machinevisiontoolbox as mvtb
# from machinevisiontoolbox import Image, Blob
from machinevisiontoolbox import Image

# # im = Image("flowers1.png")

im = Image.Read("flowers1.png")
# im.disp()
print(im)

red = im.red()
blue = im.blue()
print(red)

# im.disp(block=False)
# red.disp()

grey = im.mono()
print(grey)
# grey.disp()

print(im.isint)
im.stats()
z = im.to("float") ** 2
print(z)
z.stats()
z.disp()

z = im * 0.5
z.stats()

# read from web

im = Image.Read("http://petercorke.com/files/images/monalisa.png")
print("monalisa:", im)


# blobs

mb = Image.Read("multiblobs.png")
# mb.disp()

blobs = mb.blobs()
print(len(blobs))

print(blobs)

a = blobs[0]
print(len(a))
print(a.u)
print(a)

# features

sift = mb.SIFT(nfeatures=10)
print(sift[:10])

# sift.drawKeypoints(mb)  # TODO import errors with machinevisiontoolbox.Image

print(sift[0])

orb = mb.ORB(nfeatures=10)

# mser = mb.MSER()
# mser doesn't have detectAndCompute - I think it only has detectRegions()
