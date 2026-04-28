PyTorch tensor interface
========================

.. code-block:: python

   import machinevisiontoolbox
   print(machinevisiontoolbox.__version__)

The toolbox provides interfaces to PyTorch -- an important machine learning framework.
The fundamental datatype in PyTorch is the tensor which is a multidimensional array,
with shape (N, H, W, C) where
N is the batch size, H and W are the image dimensions and C is the number of channels.
For a single image the batch size is 1, so the shape is (1, H, W, C) or sometimes it is
*squeezed* into the shape (H, W, C).


Image → tensor
--------------

Use the :meth:`tensor` method to convert an :class:`Image` to a tensor.  For example:

.. runblock:: pycon

    >>> from machinevisiontoolbox import Image
    >>> im = Image.Read("eiffel-1.png")
    >>> tensor = im.tensor()
    >>> print(tensor.shape)

By default, the resulting tensor is normalized to have the Imagenet mean and standard
deviation, which is a common preprocessing step for feeding images into a neural
network.  If you want to get the raw pixel values as a tensor, you can set the
``normalized`` argument to False.

tensor → Image
--------------

Use the constructor :meth:`Tensor` method to create an :class:`Image` from a tensor.  For example:

.. runblock:: pycon

    >>> from machinevisiontoolbox import Image
    >>> from torch import rand
    >>> tensor = rand(3, 480, 640)  # random 3-channel image
    >>> img = Image.Tensor(tensor)
    >>> print(img)

An exception is thrown if the tensor has a batch dimension greater than 1.

If the tensor is from a segmentation model and contains logits, then the ``logits``
argument should be set to True.  This will apply a softmax to the tensor and convert it
to a color image where each pixel is colored according to the class with the highest
probability.  For example:

.. code-block:: pycon

    >>> outputs = model(img.tensor())
    >>> classes = Image.Tensor(outputs, logits=True)
    >>> classes.disp() # display the class labels as colors

Image source → batch tensor
---------------------------

An image source (a concrete instance of the abstract class :class:`ImageSource` that yields images) represents a set of images
and can be converted to a batch tensor where N>1.  All sources have a :meth:`tensor`
method that creates a batch tensor containing all the images in the source.

For example, a video file is a set of images and a tensor can be created that contains all of its frames:

.. code-block:: python

     from machinevisiontoolbox import VideoFile
     with VideoFile("traffic_sequence.mp4") as video:
         tensor = video.tensor()
     print(tensor.shape)

Note the use of the context manager to ensure that the video file is properly closed
after reading.  The resulting tensor has shape (N, H, W, C) where N is the number of
frames in the video.

We can similarly create a batch tensorfor a local file folder, a ROS bag, or a Zip archive.


batch tensor → Image iterator
-----------------------------

A batch tensor can be converted to a set of images.  This is done using an :class:`Image` iterator:

.. code-block:: python

    from machinevisiontoolbox import TensorStack
    from torch import rand
    tensor = rand(16, 480, 640, 3)  # batch of 16 random RGB images
    for img in TensorStack(tensor):
        img.disp(fps=4)

This particular example could be achieved a little more concisely by using
the :meth:`ImageSource.disp` method inherited by all image sources, including :class:`TensorStack`.
A single line of code iterates over a tensor and displays the frames as a video:

.. code-block:: python

    from machinevisiontoolbox import TensorStack
    from torch import rand
    tensor = rand(16, 480, 640, 3)  # batch of 16 random RGB images
    TensorStack(tensor).disp(fps=4)


