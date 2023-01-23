from machinevisiontoolbox import Image
img = Image.Random(10)
window = img.showpixels(windowsize=1)
window.move(2,3)