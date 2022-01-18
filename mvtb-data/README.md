# Machine Vision Toolbox for Python - data files

[![PyPI version](https://badge.fury.io/py/rtb-data.svg)](https://badge.fury.io/py/mvtb-data)
[![Anaconda version](https://anaconda.org/conda-forge/mvtb-data/badges/version.svg)](https://anaconda.org/conda-forge/mvtb-data)

<table style="border:0px">
<tr style="border:0px">
<td style="border:0px">
<img src="https://github.com/petercorke/machinevision-toolbox-python/raw/master/docs/figs/MVTBDataLogo.png" width="200"></td>
<td style="border:0px">
This package includes large data files associated with the Machine Vision Toolbox for Python (MVTB-P).
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
| images | example images                        |

## Accessing data within the package

The Toolbox function `path_to_datafile(file)` will return an absolute
`Path` to `file` which is relative to the root of the data package.  For example

```
iread('myimage.png')       # read ./myimage.png
iread('monalisa.png')      # read from data package
```

A matching local file takes precendence over a file in the data package.

## Installing the package

You don't need to explicitly install this package, it happens automatically when you when you install MVTB-P

```
pip install machinevisiontoolbox-python
```
since it is a dependency.
