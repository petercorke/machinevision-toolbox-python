from machinevisiontoolbox import draw_line, idisp
import numpy as np
img = np.zeros((1000, 1000), dtype='uint8')
draw_line(img, (100, 300), (700, 900), color=200, thickness=10)
idisp(img)