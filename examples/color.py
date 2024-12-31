from machinevisiontoolbox import Image
import matplotlib.pyplot as plt

im = Image.Read("flowers1.png")
im.disp()
print(im)

red = im.red()
blue = im.blue()
print(red)

red.disp(title="red channel of flowers1.png")

grey = im.mono()
print(grey)
grey.disp(title="grey scale version of flowers1.png")

f = im.to("float")
f.disp(title="float version of flowers1.png")

im.stats()

hist = im.hist()
plt.figure()
hist.plot(title="histogram of flowers1.png")

plt.show(block=True)
