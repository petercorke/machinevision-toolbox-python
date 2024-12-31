# animate the display of frames stored within a zip file

from machinevisiontoolbox import ZipArchive
from machinevisiontoolbox.base.data import mvtb_path_to_datafile
import matplotlib.pyplot as plt

path = mvtb_path_to_datafile("images", "bridge-l.zip")
for image in ZipArchive(path, filter="*.png"):
    image.disp(reuse=True, fps=10)
