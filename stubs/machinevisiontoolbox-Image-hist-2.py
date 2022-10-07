from machinevisiontoolbox import Image
img = Image.Read('flowers1.png')
img.hist().plot(style='overlay')