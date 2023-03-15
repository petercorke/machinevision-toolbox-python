from machinevisiontoolbox import Image
img = Image.Read('street.png')
img.hist().plot()