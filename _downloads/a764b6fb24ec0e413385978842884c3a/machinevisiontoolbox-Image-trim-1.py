from machinevisiontoolbox import Image
img = Image.Read('flowers1.png').trim(left=100, bottom=100).disp()