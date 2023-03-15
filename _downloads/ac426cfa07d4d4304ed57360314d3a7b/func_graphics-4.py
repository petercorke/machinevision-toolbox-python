from machinevisiontoolbox import draw_text, idisp
import numpy as np
img = np.zeros((1000, 1000), dtype='uint8')
draw_text(img, (100, 150), 'hello world!', color=200, fontsize=2)
idisp(img)