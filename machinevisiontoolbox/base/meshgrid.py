import numpy as np

def meshgrid(width, height):
    u = np.arange(width)
    v = np.arange(height)

    return np.meshgrid(u, v)#, indexing='ij')