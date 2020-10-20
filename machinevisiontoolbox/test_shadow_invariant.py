import cv2
from skimage import exposure
from skimage import img_as_float
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import interpolate
from scipy import optimize
import time
import multiprocessing

'''
References:
 https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
 https://stackoverflow.com/questions/19070943/numpy-scipy-analog-of-matlabs-fminsearch
'''

# class ShadowInvariant:
# 	def __init__(self):
# 		# initialization (options from the invariant script)

# 	def invariant(self, image):
# 		# invariant class

# 	def esttheta(self, image):
# 		# Estimate theta

# 	def sharpen(self, image):
# 		# Sharpen image

# 	def plot_img(self, image, axes, bins):
# 		# Plot image

# if __name__ == '__main__':


def plot_img(image, axes, bins=256):
	image = img_as_float(image)
	ax_img = axes
	ax_img.imshow(image, cmap=plt.cm.gray)
	ax_img.set_axis_off()
	return ax_img

'''
Spectral data (NxD) of bb2 interpolated to wavelengths [metres] specified in LAMBDA (Nx1).
'''
def loadspectrum(lam,filename='bb2.dat'):
	tab = np.loadtxt('bb2.dat',comments='%')#,delimiter=' ',dtype=float
	tab = np.asarray(tab.reshape(-1,4),dtype=float) # This is hacky if you change the filename
	Q = []
	for i in range(1,4):
		f = interpolate.interp1d(tab[:,0], tab[:,i],kind='cubic') # This will give us the interpolation function
		tmp = f(lam)
		Q.append(tmp)

	Q = np.asarray(Q,dtype=float)
	# print(Q.shape)
	return Q.transpose()


def sharpen_matrix(x):
	theta = x[0]
	phi = x[1]
	c1 = np.array([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)])
	c1 = c1.transpose()
	
	theta = x[2]
	phi = x[3]
	c2 = np.array([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)])
	c2 = c2.transpose()

	theta = x[4] 
	phi = x[5]
	c3 = np.array([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)])
	c3 = c3.transpose()

	T = np.array([c1,c2,c3]) #3x3 matrix
	return T

def sharpen_spectrum(T,spectrum):
	sharp = spectrum.dot(T)
	return sharp

def costfunc(x,spectrum):
	
	# Transformation matrix
	T = sharpen_matrix(x)
	# compute sharpened spectrum
	sharp = sharpen_spectrum(T,spectrum) #Nx3

	# compute spectral overlap
	overlap = np.sum((sharp[:,0]*sharp[:,1])) + np.sum((sharp[:,0]*sharp[:,2])) + np.sum((sharp[:,1]*sharp[:,2]))

	# compute sum of non-positive values, we don't really want this 
	c = sharp[:]
	
	negativeness = np.sum(c[c<0])
	
	# final cost function
	e = overlap - 2*negativeness
	return e 

def sharpen_rgb(image,M):
	# applying the sharpening matrix M (3x3) to every pixel in the input image of (HxWx3)
	im = np.asarray(image).reshape(-1,3)
	im.astype(float)
	img = im.dot(M)
	# clamp negative values to zero
	img = img.clip(min=0)
	img = img.reshape(image.shape)
	return img


def gamma(image,gamma='sRGB'):

	# Convert image to double, asssuming it is uint8 image
	f = skimage.img_as_float(image)
	if gamma=='sRGB':
		# convert gamma encoded sRGB to linear tristimulus values
		k = f <= 0.04045
		f[k] = np.divide(f[k],12.92)
		k = f > 0.04045
		f[k] = ( (f[k]+0.055)/1.055)**2.4
		g = f
	else:
		print('This encoding is not implemented')
	return g


'''
input: gamma corrected image
output: log-chrom coordinates to compute the invariant direction
'''
def log_chrom_coord(image):
	# Planckian illumination constant (we only this constant if theta is not given)
	# c2 = 1.4388e-2;

	# Convert the gamma corrected image into vector (h*w,channel)
	im = np.asarray(image).reshape(-1,3)
	im.astype(float)
	# print(im.shape)

	# compute chromaticity
	# denom = prod(im, 2).^(1/3);
	A = np.prod(im,axis=1)
	denom = A**(1/3)

	r_r = np.divide(im[:,0],denom, out=np.zeros(im[:,0].shape, dtype=float), where=denom!=0)
	r_b = np.divide(im[:,2],denom, out=np.zeros(im[:,0].shape, dtype=float), where=denom!=0)

	# Exact log
	r_rp = np.log(r_r)
	r_bp = np.log(r_b)

	return r_rp, r_bp


'''
input: gamma corrected image
output: gray-scale shadow invariant image
'''
def invariant(image, theta):

	# compute the log-chrom coordinates
	r_rp, r_bp = log_chrom_coord(image)

	# figure the illumination invariant direction
	c = np.array([[math.cos(theta)], [math.sin(theta)]])
	# print(c.shape)

	# create an array of [r_rp and r_bp]
	d = np.array([r_rp,r_bp])
	# print(d.transpose().shape)

	# gs = [r_rp r_bp] * c';
	gs = d.transpose().dot(c)
	# print(gs.shape)

	# reshape the gs vector to the original image size

	gs_im = np.asarray(gs).reshape(height,width)
	# print(gs_im.shape)

	return gs_im


'''
This function is similar to nvariant but with different input arguments for theta estimate
input: log chrom coordinates
output: gray scale image
'''
def esttheta_invariant(r_rp, r_bp, theta):
	# figure the illumination invariant direction
	c = np.array([[math.cos(theta)], [math.sin(theta)]])
	# print(c.shape)

	# create an array of [r_rp and r_bp]
	d = np.array([r_rp,r_bp])
	# print(d.transpose().shape)

	# gs = [r_rp r_bp] * c';
	gs = d.transpose().dot(c)
	# print(gs.shape)

	# reshape the gs vector to the original image size

	gs_im = np.asarray(gs).reshape(height,width)
	# print(gs_im.shape)

	return gs_im

'''
input: gamma corrected image
output: estimated invariant line angle (radians) 
'''
def esttheta(image):
	theta_v = np.asarray(np.arange(0.0,math.pi,0.02),dtype=float)
	# print(theta_v.shape)

	# compute the log chrom coordinates for the image
	r_rp,r_bp = log_chrom_coord(image)

	sim = []
	for i in theta_v:
		start_time = time.time()
		gs = esttheta_invariant(r_rp,r_bp, i)
		elapsed_time = time.time() - start_time
		gs = gs[~np.isnan(gs)]  
		gs = gs[~np.isinf(gs)]
  		# slower process - get the indices of the elements which are finite and then find the elements
		# ii = np.isfinite(gs)
		# gs = gs[ii]
		# print(np.std(gs))
		# print('Elapsed time in sec:  ', elapsed_time)
		sim.append(np.std(gs))

	sim = np.asarray(sim)
	# print(sim)
	k = np.where(sim==np.min(sim))
	return theta_v[k]



#====================================================================
# filename = '/media/bilal/DATA1/MachineVisionToolbox/shadow_images/0671_FR_1162_01_10.jpg'
# filename = '0671_FR_1162_01_10.jpg'
filename = '/media/bilal/DATA1/ZED_QUT_Data/shadow_invariant_test_imgs/lizard.jpg'

img = skimage.io.imread(filename)
(height,width,channel) = img.shape
# print(img.shape)

# ToDo: gamma correction using igamm.m using sRGB encoding.
# gamma_corrected_img = exposure.adjust_gamma(img, 2.2)
# print('gamma corrected image shape',gamma_corrected_img.shape)


# Testing gamma correction
gamma_corrected_img = gamma(img)
# print('gamma_corrected image size',gamma_corrected_img.shape)


# Sharpen the gamma corrected image

# lam = [400:10:700]*1e-9;
lam = np.asarray(range(400,700,10),dtype=float)
lam = np.multiply(lam,1e-9)
# print(lam.shape)

# Spectrum
Q = loadspectrum(lam)
# print(Q.shape)

x0 = [math.pi/2, 0.5, math.pi/2, math.pi/2, 0, 0.5]

# e=costfunc(x0,Q)
xopt = optimize.fmin(func=costfunc, x0=x0, args=(Q,))
M = sharpen_matrix(xopt)
Qsharp = sharpen_spectrum(M,Q)

# Display Q and Qsharp results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((3, 1), dtype=np.object)
axes[0,0] = plt.subplot(3, 1, 1)
axes[0,0].plot(lam*1e9, Q[:,0])
axes[0,0].plot(lam*1e9, Qsharp[:,0],'--')
axes[0,0].set_title('Q1')

axes[1,0] = plt.subplot(3, 1, 2, sharex=axes[0,0], sharey=axes[0,0])
axes[1,0].plot(lam*1e9, Q[:,1])
axes[1,0].plot(lam*1e9, Qsharp[:,1],'--')
axes[1,0].set_title('Q2')

axes[2,0] = plt.subplot(3, 1, 3, sharex=axes[0,0], sharey=axes[0,0])
axes[2,0].plot(lam*1e9, Q[:,2])
axes[2,0].plot(lam*1e9, Qsharp[:,2],'--')
axes[2,0].set_title('Q2')


# Sharpen gamma corrected image
sharp_gamma_corr = sharpen_rgb(gamma_corrected_img, M)

# Show gamma corrected sharpen image
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((1, 1), dtype=np.object)
axes[0, 0] = plt.subplot(1, 1, 1)
ax_img = plot_img(sharp_gamma_corr, axes[0, 0])
ax_img.set_title('Gamma corrected sharp image')


# Display gamma corrected results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((1, 2), dtype=np.object)

axes[0,0] = plt.subplot(1, 2, 1)
ax_img = plot_img(img, axes[0, 0])
ax_img.set_title('Raw image')

axes[0, 1] = plt.subplot(1, 2, 2, sharex=axes[0, 0], sharey=axes[0, 0])
ax_img = plot_img(gamma_corrected_img, axes[0, 1])
ax_img.set_title('Gamma correction')


# Estimating theta
theta = esttheta(sharp_gamma_corr)
print('Estimated theta in radians: ', theta)

# Shadow invariant image
# dummy_theta = 0.8
gs_im = invariant(gamma_corrected_img, theta) # sharp_gamma_corr, gamma_corrected_img

# show shadow invariant image
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((1, 1), dtype=np.object)
axes[0, 0] = plt.subplot(1, 1, 1)
ax_img = plot_img(gs_im, axes[0, 0])
ax_img.set_title('shadow invariant')


fig.tight_layout()
plt.show()

