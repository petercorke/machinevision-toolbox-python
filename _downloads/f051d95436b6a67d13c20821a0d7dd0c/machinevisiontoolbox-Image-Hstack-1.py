from machinevisiontoolbox import Image
img = Image.Read('street.png')
Image.Hstack((img, img, img)).disp()