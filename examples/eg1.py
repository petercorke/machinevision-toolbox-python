import machinevisiontoolbox as mvtb 
from machinevisiontoolbox import Image, Blob 

# # im = Image("machinevisiontoolbox/images/flowers?.png")

if 0:
    im = Image("flowers1.png")
    im.disp()
    print(im)

    red = im.red()
    blue = im.blue()
    print(red)

    im.disp(block=False)
    red.disp()

    grey = im.mono()
    print(grey)
    grey.disp()

## read from web

im = Image("http://petercorke.com/files/images/monalisa.png")
print("monalisa:", im)
im.disp()
im = Image("http://petercorke.com/files/images/flowers7.png")
print("flowers7:", im)
im.disp()

    # the images all load with 4 planes
    # so they are not tagged as color, but they display as color
    # some issue with imdecode()


## blobs

mb = Image("multiblobs.png")
# mb.disp()

blobs = mb.blobs()
print(len(blobs))

print(blobs)

a = blobs[0]
print(len(a))
print(a.u)
print(a)

## features

sift = mb.SIFT(nfeatures=10)
print(sift[:10])



print(sift[0])