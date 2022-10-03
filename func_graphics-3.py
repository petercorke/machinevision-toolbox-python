from machinevisiontoolbox import draw_labelbox, idisp
import numpy as np
img = np.zeros((1000, 1000), dtype='uint8')
draw_labelbox(img, 'labelled box', bbox=[100, 500, 300, 600], textcolor=0, labelcolor=100, color=200, thickness=2, fontsize=1)
idisp(img)