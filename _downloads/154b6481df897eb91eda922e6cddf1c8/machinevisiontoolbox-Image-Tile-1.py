from machinevisiontoolbox import Image, ImageCollection
images = ImageCollection('campus/*.png')  # image iterator
Image.Tile(images).disp()