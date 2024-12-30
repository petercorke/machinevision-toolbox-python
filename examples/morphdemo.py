from machinevisiontoolbox import Image, set_window_title
import matplotlib.pyplot as plt
import numpy as np

morph_op = np.min
# morph_op = np.max

# pause = None
pause = 0.5

a = Image.Random(10)  # random input image
b = Image.Zeros(10)  # "empty" output image

# create two adjacent subplots, and display the input image on the left
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
set_window_title("Morphological operation demonstration")

wa = a.showpixels(textcolors="blue", ax=ax[0], windowsize=1)
b.showpixels(ax=ax[1])

# wa is an object that can highlight windows in the corresponding image

# loop over all pixels in the input image, excluding the border
for v in range(1, a.shape[0] - 1):
    for u in range(1, a.shape[1] - 1):
        window = wa.move(u, v, "red")  # highlight the window

        # window is the 3x3 window centered on (u,v)
        b.image[v, u] = morph_op(window)  # perform the morphological operation

        # update the display of the output image
        ax[1].cla()
        b.showpixels(ax=ax[1])

        if pause is None:
            plt.waitforbuttonpress()
        else:
            plt.pause(pause)
