# create random greyscale pixel pattern and allow interactive threshold
# to change the dot density
#
# could be the basis of texture for stereo calibration

from machinevisiontoolbox import Image

image = Image.Random(200)
image.threshold_interactive(block=True)
