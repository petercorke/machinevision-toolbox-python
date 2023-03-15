from machinevisiontoolbox import draw_box, idisp
import numpy as np
img = np.zeros((1000, 1000), dtype='uint8')
draw_box(img, ltrb=[100, 300, 700, 500], thickness=2, color=200)
draw_box(img, ltrb=[100, 300, 700, 500], thickness=-1, color=50)
idisp(img)