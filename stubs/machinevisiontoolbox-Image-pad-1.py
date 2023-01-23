from machinevisiontoolbox import Image
img = Image.Read('flowers1.png', dtype='float').pad(left=10, bottom=10, top=10, right=10, value='r').disp()