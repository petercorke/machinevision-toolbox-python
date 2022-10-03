from machinevisiontoolbox import draw_point, idisp
import numpy as np
img = np.zeros((1000, 1000), dtype='uint8')
draw_point(img, (100, 300), '*', fontsize=1, color=200)
draw_point(img, (500, 300), '*', 'labelled point', fontsize=1, color=200)
draw_point(img, np.random.randint(1000, size=(2,10)), '+', 'point {0}', 100, fontsize=0.8)
idisp(img)