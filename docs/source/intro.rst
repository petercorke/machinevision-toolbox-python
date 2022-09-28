************
Introduction
************

The goal of this package is to simplify the expression of computer vision
algorithms in Python.  Images can be represented as 2D or 3D arrays which are
the domain of `NumPy <https://numpy.org>`_` but many power image specific operations are provided by 
`OpenCV  <https://opencv.org>`_, `SciPy <https://scipy.org>`_` and `Open3D <open3d.org>`_.  `matplotlib <https://matplotlib.org>`_ is a portable and powerful
way to display graphical data, including images, whereas OpenCV does a great job of displaying
images but other graphics, not so much.

In practice, using these various
packages is complex, each have their own way of working, similar options are
accessed differently and some function require image pixels to have particular types.
None of them consider the image as an object with a set of useful image and vision
processing methods and operators.  

For example, to read an image using OpenCV, smooth it, and display it::

    import cv2
    import numpy

    # read image
    src = cv2.imread('.../flowers1.png', cv2.IMREAD_UNCHANGED)
    
    # apply Gaussian blur on src image
    dst = cv2.GaussianBlur(src, (5,5), cv2.BORDER_DEFAULT)
    
    # display input and output image
    cv2.imshow("Gaussian Smoothing",numpy.hstack((src, dst)))
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image

Using this toolbox we would write instead::

    from machinevisiontoolbox import Image

    img = Image.Read('flowers1.png')
    smooth = img.smooth(hw=2)
    smooth.disp()

or even

    from machinevisiontoolbox import Image

    img = Image.Read('flowers1.png').smooth(hw=2).disp()


While all this is very subjective, you get the idea that the Toolbox allows 
succinct coding without the need for lots of OpenCV flags like ``cv2.IMREAD_UNCHANGED`` in the example above.

There are lots of ways to create an image.  We can read an image from a file::

    img = Image.Read('street.png')

or create it from code

	img = Image.Zeros(100, dtype='uint8')


Under the hood the :class:`Image` object contains some image parameters, a lot
of methods, and a reference to a 2D or 3D NumPy ndarray.

Color images are handled a bit more sensibly than raw OpenCV.  A multi-channel
or multi-plane image is a NumPy ndarray with an arbitrary number of planes and a
dictionary that maps channel names to an integer index.  We can see that

    img = Image.Read('flowers.png')

Rather than have the meaning of the plane implicit, it is explicity, for example

    img = Image.Read('flowers1.png')
    img.red().disp()
    img.colorspace('hsv').plane('h').disp()

	img = Image.Zeros(100, colororder="RGB")

An image object has a lot of useful predicates

	img.nplanes
	img.iscolor  # image is multichannel
	img.ismono   # image is single channel
	img.isfloat  # image has floating point pixels
	img.to_float().isfloat

image[i]
for plane in image:


## Synopsis

The Machine Vision Toolbox for Python (MVTB-P) provides many functions that are useful in machine vision and vision-based control.  The main components are:

- An object-oriented wrapper of OpenCV functions that supports operator overloading and handles the gnarly details of OpenCV like conversion to/from float32 and the BGR color order.

It is a somewhat eclectic collection reflecting my personal interest in areas of photometry, photogrammetry, colorimetry.  It includes over 100 functions spanning operations such as image file reading and writing, acquisition, display, filtering, blob, point and line feature extraction,  mathematical morphology, homographies, visual Jacobians, camera calibration and color space conversion. With input from a web camera and output to a robot (not provided) it would be possible to implement a visual servo system entirely in Python.

An image is usually treated as a rectangular array of scalar values representing intensity or perhaps range, or 3-vector values representing a color image.  The matrix is the natural datatype of [NumPy](https://numpy.org) and thus makes the manipulation of images easily expressible in terms of arithmetic statements in Python.  
Advantages of this Python Toolbox are that:

  * it uses, as much as possibe, [OpenCV](https://opencv.org), which is a portable, efficient, comprehensive and mature collection of functions for image processing and feature extraction;
  * it wraps the OpenCV functions in a consistent way, hiding some of the complexity of OpenCV;
  * it is has similarity to the Machine Vision Toolbox for MATLAB.

# Getting going

## Using pip

Install a snapshot from PyPI

```
% pip install machinevision-toolbox-python
```

## From GitHub

Install the current code base from GitHub and pip install a link to that cloned copy

```
% git clone https://github.com/petercorke/machinevision-toolbox-python.git
% cd machinevision-toolbox-python
% pip install -e .
```

# Examples


### Binary blobs

```python
import machinevisiontoolbox as mvtb
import matplotlib.pyplot as plt
im = mvtb.Image('shark2.png')   # read a binary image of two sharks
fig = im.disp();   # display it with interactive viewing tool
f = im.blobs()  # find all the white blobs
print(f)

	┌───┬────────┬──────────────┬──────────┬───────┬───────┬─────────────┬────────┬────────┐
	│id │ parent │     centroid │     area │ touch │ perim │ circularity │ orient │ aspect │
	├───┼────────┼──────────────┼──────────┼───────┼───────┼─────────────┼────────┼────────┤
	│ 0 │     -1 │ 371.2, 355.2 │ 7.59e+03 │ False │ 557.6 │       0.341 │  82.9° │  0.976 │
	│ 1 │     -1 │ 171.2, 155.2 │ 7.59e+03 │ False │ 557.6 │       0.341 │  82.9° │  0.976 │
	└───┴────────┴──────────────┴──────────┴───────┴───────┴─────────────┴────────┴────────┘

f.plot_box(fig, color='g')  # put a green bounding box on each blob
f.plot_centroid(fig, 'o', color='y')  # put a circle+cross on the centroid of each blob
f.plot_centroid(fig, 'x', color='y')
plt.show(block=True)  # display the result
```
![Binary image showing bounding boxes and centroids](https://github.com/petercorke/machinevision-toolbox-python/raw/master/figs/shark2+boxes.png)


### Binary blob hierarchy

We can load a binary image with nested objects

```python
im = mvtb.Image('multiblobs.png')
im.disp()
```

![Binary image showing bounding boxes and centroids](https://github.com/petercorke/machinevision-toolbox-python/raw/master/figs/multi.png)

```python
f  = im.blobs()
print(f)
	┌───┬────────┬───────────────┬──────────┬───────┬────────┬─────────────┬────────┬────────┐
	│id │ parent │      centroid │     area │ touch │  perim │ circularity │ orient │ aspect │
	├───┼────────┼───────────────┼──────────┼───────┼────────┼─────────────┼────────┼────────┤
	│ 0 │      1 │  898.8, 725.3 │ 1.65e+05 │ False │ 2220.0 │       0.467 │  86.7° │  0.754 │
	│ 1 │      2 │ 1025.0, 813.7 │ 1.06e+05 │ False │ 1387.9 │       0.769 │ -88.9° │  0.739 │
	│ 2 │     -1 │  938.1, 855.2 │ 1.72e+04 │ False │  490.7 │       1.001 │  88.7° │  0.862 │
	│ 3 │     -1 │  988.1, 697.2 │ 1.21e+04 │ False │  412.5 │       0.994 │ -87.8° │  0.809 │
	│ 4 │     -1 │  846.0, 511.7 │ 1.75e+04 │ False │  496.9 │       0.992 │ -90.0° │  0.778 │
	│ 5 │      6 │  291.7, 377.8 │  1.7e+05 │ False │ 1712.6 │       0.810 │ -85.3° │  0.767 │
	│ 6 │     -1 │  312.7, 472.1 │ 1.75e+04 │ False │  495.5 │       0.997 │ -89.9° │  0.777 │
	│ 7 │     -1 │  241.9, 245.0 │ 1.75e+04 │ False │  496.9 │       0.992 │ -90.0° │  0.777 │
	│ 8 │      9 │ 1228.0, 254.3 │ 8.14e+04 │ False │ 1215.2 │       0.771 │ -77.2° │  0.713 │
	│ 9 │     -1 │ 1225.2, 220.0 │ 1.75e+04 │ False │  496.9 │       0.992 │ -90.0° │  0.777 │
	└───┴────────┴───────────────┴──────────┴───────┴────────┴─────────────┴────────┴────────┘
```

We can display a label image, where the value of each pixel is the label of the blob that the pixel
belongs to

```python
out = f.labelImage(im)
out.stats()
out.disp(block=True, colormap='jet', cbar=True, vrange=[0,len(f)-1])
```


and request the blob label image which we then display

```matlab
>> [label, m] = ilabel(im);
>> idisp(label, 'colormap', jet, 'bar')
```
![Binary image showing bounding boxes and centroids](https://github.com/petercorke/machinevision-toolbox-python/raw/master/figs/multi_labelled.png)

### Camera modelling

```python
cam = mvtb.CentralCamera(f=0.015, rho=10e-6, imagesize=[1280, 1024], pp=[640, 512], name='mycamera')
print(cam)
	           Name: mycamera [CentralCamera]
	   focal length: (array([0.015]), array([0.015]))
	     pixel size: 1e-05 x 1e-05
	   principal pt: (640.0, 512.0)
	     image size: 1280.0 x 1024.0
	   focal length: (array([0.015]), array([0.015]))
	           pose: t = 0, 0, 0; rpy/zyx = 0°, 0°, 0°
```

and its intrinsic parameters are

```matlab 
print(cam.K)
	[[1.50e+03 0.00e+00 6.40e+02]
	 [0.00e+00 1.50e+03 5.12e+02]
	 [0.00e+00 0.00e+00 1.00e+00]]
```
We can define an arbitrary point in the world

```python 
P = [0.3, 0.4, 3.0]
```
and then project it into the camera

```python
p = cam.project(P)
print(p)
	[790. 712.]
```
which is the corresponding coordinate in pixels.  If we shift the camera slightly the image plane coordinate will also change

```python 
p = cam.project(P, T=SE3(0.1, 0, 0) )
print(p)
[740. 712.]
```

We can define an edge-based cube model and project it into the camera's image plane

```python 
X, Y, Z = mkcube(0.2, pose=SE3(0, 0, 1), edge=True)
cam.mesh(X, Y, Z)
```
![Perspective camera view](figs/cube.png)

<!---or with a fisheye camera

```matlab
>> cam = FishEyeCamera('name', 'fisheye', ...
'projection', 'equiangular', ...
'pixel', 10e-6, ...
'resolution', [1280 1024]);
>> [X,Y,Z] = mkcube(0.2, 'centre', [0.2, 0, 0.3], 'edge');
>> cam.mesh(X, Y, Z);
```
![Fisheye lens camera view](figs/cube_fisheye.png)


### Bundle adjustment
--->
### Color space
Plot the CIE chromaticity space

```python
showcolorspace('xy')
```
![CIE chromaticity space](figs/colorspace.png)

Load the spectrum of sunlight at the Earth's surface and compute the CIE xy chromaticity coordinates

```python
nm = 1e-9
lam = np.linspace(400, 701, 5) * nm # visible light
sun_at_ground = loadspectrum(lam, 'solar')
xy = lambda2xy(lambda, sun_at_ground)
print(xy)
	[[0.33272798 0.3454013 ]]
print(colorname(xy, 'xy'))
	khaki
```

### Hough transform

```matlab
im = iread('church.png', 'grey', 'double');
edges = icanny(im);
h = Hough(edges, 'suppress', 10);
lines = h.lines();

idisp(im, 'dark');
lines(1:10).plot('g');

lines = lines.seglength(edges);

lines(1)

k = find( lines.length > 80);

lines(k).plot('b--')
```
![Hough transform](figs/hough.png)

### SURF features

We load two images and compute a set of SURF features for each

```matlab
>> im1 = iread('eiffel2-1.jpg', 'mono', 'double');
>> im2 = iread('eiffel2-2.jpg', 'mono', 'double');
>> sf1 = isurf(im1);
>> sf2 = isurf(im2);
```
We can match features between images based purely on the similarity of the features, and display the correspondences found

```matlab
>> m = sf1.match(sf2)
m = 
644 corresponding points (listing suppressed)
>> m(1:5)
ans = 
 
(819.56, 358.557) <-> (708.008, 563.342), dist=0.002137
(1028.3, 231.748) <-> (880.14, 461.094), dist=0.004057 
(1027.6, 571.118) <-> (885.147, 742.088), dist=0.004297
(927.724, 509.93) <-> (800.833, 692.564), dist=0.004371
(854.35, 401.633) <-> (737.504, 602.187), dist=0.004417
>> idisp({im1, im2})
>> m.subset(100).plot('w')
```
![Feature matching](figs/matching.png)

Clearly there are some bad matches here, but we we can use RANSAC and the epipolar constraint implied by the fundamental matrix to estimate the fundamental matrix and classify correspondences as inliers or outliers

```matlab
>> F = m.ransac(@fmatrix, 1e-4, 'verbose')
617 trials
295 outliers
0.000145171 final residual
F =
    0.0000   -0.0000    0.0087
    0.0000    0.0000   -0.0135
   -0.0106    0.0116    3.3601
>> m.inlier.subset(100).plot('g')
>> hold on
>> m.outlier.subset(100).plot('r')
>> hold off
```
where green lines show correct correspondences (inliers) and red lines show bad correspondences (outliers) 
![Feature matching after RANSAC](figs/matching_ransac.png)

### Fundamental matrix


