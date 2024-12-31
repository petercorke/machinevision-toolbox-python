# create random greyscale pixel pattern and allow interactive threshold
# to change the dot density
#
# could be the basis of texture for stereo calibration

from machinevisiontoolbox import Image
import matplotlib.pyplot as plt

image = Image.Random(200)
image.threshold_interactive()
