import machinevisiontoolbox as mvtb
import matplotlib.pyplot as plt
import numpy as np

a = np.random.rand(10, 10)
b = np.random.rand(10, 10)





fig, ax = plt.subplots(1, 2)
showpixels(a, fmt="{:.1f}", ax=ax[0])
showpixels(a, fmt="{:.1f}", ax=ax[1])

w = Window(h=1)

for v in range(1, a.shape[0] - 1):
    for u in range(1, a.shape[1] - 1):
        if u == 5:
            w.move(u, v, 'blue')
        else:
            w.move(u, v)
        plt.pause(0.5)