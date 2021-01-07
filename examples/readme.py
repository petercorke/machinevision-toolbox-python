import machinevisiontoolbox as mvtb
import matplotlib.pyplot as plt
from spatialmath import SE3
import numpy as np

nm = 1e-9

# im = mvtb.Image('shark2.png')   # read a binary image of two sharks
# fig = im.disp();   # display it with interactive viewing tool
# f = im.blobs()  # find all the white blobs
# print(f)
# print(fig)

# f.plot_box(fig, color='g')  # put a green bounding box on each blob
# f.plot_centroid(fig, 'o', color='y')  # put a circle+cross on the centroid of each blob
# f.plot_centroid(fig, 'x', color='y')
# plt.show(block=True)

# out = f.drawBlobs(im)
# out.disp(block=True)

# --------------------------------------------------------------------- #

# im = mvtb.Image('multiblobs.png')
# im.disp()

# f  = im.blobs()
# print(f)
# out = f.labelImage(im)
# out.disp(block=True, colormap='jet', cbar=True, vrange=[0,len(f)-1])

# --------------------------------------------------------------------- #


# cam = mvtb.CentralCamera(f=0.015, rho=10e-6,
#     imagesize=[1280, 1024], pp=[640, 512], name='mycamera')

# print(cam)

# print(cam.K)
# P = [0.3, 0.4, 3.0]
# p = cam.project(P)
# print(p)

# p = cam.project(P, pose=SE3(0.1, 0, 0) )
# print(p)
# [X,Y,Z] = mvtb.mkcube(0.2, pose=SE3(0, 0, 1), edge=True);
# print(X)
# print(Y)
# print(Z)
# cam.mesh(X, Y, Z, color='k')

# plt.show(block=True)

# out = mvtb.showcolorspace('xy')
# plt.show(block=True)

λ = np.arange(400, 701, 5) * nm # visible light
sun_at_ground = mvtb.loadspectrum(λ, 'solar.dat')
xy = mvtb.lambda2xy(λ, sun_at_ground)
print(xy)
print(mvtb.colorname(xy, 'xy'))


# im = iread('church.png', 'grey', 'double');
# edges = icanny(im);
# h = Hough(edges, 'suppress', 10);
# lines = h.lines();

# idisp(im, 'dark');
# lines(1:10).plot('g');

# lines = lines.seglength(edges);

# lines(1)

# k = find( lines.length > 80);

# lines(k).plot('b--')