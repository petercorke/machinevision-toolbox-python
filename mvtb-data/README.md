# Machine Vision Toolbox for Python - data package

[![A Python Robotics Package](https://raw.githubusercontent.com/petercorke/robotics-toolbox-python/master/.github/svg/py_collection.min.svg)](https://github.com/petercorke/machine-vision-toolbox-python)
[![PyPI version](https://badge.fury.io/py/mvtb-data.svg)](https://badge.fury.io/py/mvtb-data)
[![Anaconda version](https://anaconda.org/conda-forge/mvtb-data/badges/version.svg)](https://anaconda.org/conda-forge/mvtb-data)

<table style="border:0px">
<tr style="border:0px">
<td style="border:0px">
<img src="https://github.com/petercorke/machinevision-toolbox-python/raw/master/figs/MVTBDataLogo.png" width="250"></td>
<td style="border:0px">
This package includes large data files associated with the <a href="https://pypi.org/project/machinevision-toolbox-python">Machine Vision Toolbox for Python (MVTB-P)</a>.
</td>
</tr>
</table>


## Rationale

The data files are provided as a separate package to work around disk space limitations on PyPI.  Including these data with the MVTB code adds nearly 200MB to every release, which will blow the PyPI limit quite quickly.  
Since the data doesn't change very much, it's mostly images models and a few data files, it makes sense for it to be a standalone package.

## Package contents

| Folder | Purpose                        |
| ------ | ------------------------------ |
| data   | miscellaneous spectral data           |
| images | example images, videos                       |

## Accessing data within the package

The Toolbox function `mvtb_path_to_datafile(file)` will return an absolute
`Path` object that contains the path to `file` which is given relative to the
root of the data package:

```
mvtb_path_to_datafile("images/monalisa.png")
```

which can also be used like `os.path.join` as

```
mvtb_path_to_datafile("images", "monalisa.png")
```

Image files are assumed to be in the `images` folder of the data package, and this will be searched
by the image loading function

```
iread("myimage.png")       # read ./myimage.png
iread("monalisa.png")      # read from data package
```

or class method

```
Image.Read("myimage.png")       # read ./myimage.png
Image.Read("monalisa.png")      # read from data package
```


A matching local file takes precendence over a file in the data package.

## Installing the package

You don't need to explicitly install this package, it happens automatically when you when you install MVTB-P

```
pip install machinevisiontoolbox-python
```
since it is a dependency.

## Install big image files

There are two very large zip files containing image sequences which are used in
Sec. 14.8.3 Visual Odometry, each is 116M and exceeds the total PyPI quota. They
are not included in the `mvtbdata` package, but you can download them into your
*local* `mvtbdata` package by running

```
import mvtbdata.mvtb_load_image_data
```

from inside a Python session.  You only need to do this once.
