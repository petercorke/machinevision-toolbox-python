.. currentmodule:: machinevisiontoolbox.Sources
.. _sources_label:

:class:`Image` and :class:`PointCloud` sources
======================================================

The toolbox provides a variety of sources for these objects.  They can be obtained from
files, from cameras, from ROS topics and bag files, from the web, and from other
sources.  The toolbox provides a convenient interface to these sources, and the objects
they produce are instances of the :class:`~machinevisiontoolbox.ImageCore.Image` and
:class:`~machinevisiontoolbox.PointCloudCore.PointCloud` classes, which have a large
number of methods for processing and displaying them.   

All provide an iterator interface, so they can be used in a for loop to process a
sequence of images or point clouds.  For example, to read a video file and display each
frame::

    from machinevisiontoolbox import VideoFile

    for im in VideoFile("myvideo.mp4"):
        im.disp()

All sources also have a ``read`` method that returns the next image or point cloud.  For
example, to read a video file and display each frame::

    from machinevisiontoolbox import VideoFile

    video = VideoFile("myvideo.mp4")
    while True:
        im = video.read()
        if im is None:
            break
        im.disp()

Some sources also have a ``__getitem__`` method that allows you to index into the source to
get a specific image or point cloud.  For example, to read the 10th frame of a video
file::

    from machinevisiontoolbox import VideoFile

    video = VideoFile("myvideo.mp4")
    im = video[9]  # index starts at 0
    im.disp()

All sources serve as a context manager, so they can be used in a ``with`` statement to
ensure that resources are properly released.  For example, to read a video file and
display each frame::

    from machinevisiontoolbox import VideoFile

    with VideoFile("myvideo.mp4") as video:
        for im in video:
            im.disp()

All sources have a ``disp`` method that displays the image or point cloud using Matplotlib.
For example, to read a video file and display each frame::

    from machinevisiontoolbox import VideoFile

    with VideoFile("myvideo.mp4") as video:
        for im in video:
            im.disp()

All sources have a ``close`` method that releases any resources associated with the source.
For example, to read a video file and display each frame::

    from machinevisiontoolbox import VideoFile

    video = VideoFile("myvideo.mp4")
    for im in video:
        im.disp()
    video.close()

All sources have a ``tensor`` method that returns the image or point cloud as a PyTorch
tensor.  For example, to read a video file and display each frame::

    from machinevisiontoolbox import VideoFile

    with VideoFile("myvideo.mp4") as video:
        for im in video:
            tensor = im.tensor()
            # do something with the tensor

Image Sources
-------------

:class:`~machinevisiontoolbox.ImageCore.Image` objects can be conveniently obtained from a variety of sources:

.. autosummary::
   :toctree: stubs
   :nosignatures:
   :template: mvtbtemplate.rst

    ~ImageSequence
    ~FileCollection
    ~FileArchive
    ~VideoFile
    ~VideoCamera
    ~WebCam
    ~EarthView
    ~ROSBag
    ~ROSTopic
    ~TensorStack
    ~LabelMeReader

Deprecated aliases
------------------

These names are kept for backward compatibility and will emit a deprecation warning.

.. autosummary::
    :toctree: stubs
    :nosignatures:
    :template: mvtbtemplate.rst

     ~ImageCollection
     ~ZipArchive


PointCloud Sources
------------------

:class:`~machinevisiontoolbox.PointCloud` objects can be conveniently obtained from a variety of sources:

.. autosummary::
   :toctree: stubs
   :nosignatures:
   :template: mvtbtemplate.rst

    ~PointCloudSequence

- :class:`~machinevisiontoolbox.Sources.ROSBag`
- :class:`~machinevisiontoolbox.Sources.ROSTopic`

