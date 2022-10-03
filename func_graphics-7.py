from machinevisiontoolbox import draw_circle, idisp
import numpy as np
img = np.zeros((1000, 1000), dtype='uint8')
draw_circle(img, (400,600), 150, thickness=2, color=200)
draw_circle(img, (400,600), 150, thickness=-1, color=50)
idisp(img)