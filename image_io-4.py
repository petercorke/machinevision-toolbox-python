from machinevisiontoolbox import Image
left = Image.Read("rocks2-l.png", reduce=2)
right = Image.Read("rocks2-r.png", reduce=2)
left.anaglyph(right).disp()