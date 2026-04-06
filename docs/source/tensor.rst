PyTorch tensor interface
========================

.. code-block:: python

   import machinevisiontoolbox
   print(machinevisiontoolbox.__version__)

The toolbox provides interfaces to PyTorch -- an important machine learning framework.
The fundamental datatype in PyTorch is the tensor which is a multidimensional array,
with shape (N, H, W, C) where
N is the batch size, H and W are the image dimensions and C is the number of channels.
For a single image the batch size is 1, so the shape is (1, H, W, C) or sometimes in
its _squeezed_ form (H, W, C).


Image → tensor
--------------

Use the :class:`Image` :meth:`tensor` method to convert an :class:`Image` to a tensor.  For example:

.. runblock:: pycon

    >>> from machinevisiontoolbox import Image
    >>> im = Image.Read("eiffel-1.png")
    >>> tensor = im.tensor()
    >>> print(tensor.shape)

tensor → Image
--------------

Use the :class:`Image` constructor :meth:`Tensor` method to create an :class:`Image` from a tensor.  For example:

.. runblock:: pycon

    >>> from machinevisiontoolbox import Image
    >>> from torch import rand
    >>> tensor = rand(3, 480, 640)  # random 3-channel image
    >>> img = Image.Tensor(tensor)
    >>> print(img)

An exception is thrown if the tensor has a batch dimension greater than 1.

Image source → batch tensor
---------------------------

An image source (an instance of a class that yields images) represents a set of images
and can be converted to a batch tensor where N>1.  All sources have a :meth:`tensor`
method that creates a batch tensor containing all the images in the source.

For example, a video file is a set of images and a tensor can be created that contains all its frames:

.. code-block:: python

     from machinevisiontoolbox import VideoFile
     with VideoFile("traffic_sequence.mp4") as video:
         tensor = video.tensor()
     print(tensor.shape)



Note the use of the context manager to ensure that the video file is properly closed
after reading.  The resulting tensor has shape (N, H, W, C) where N is the number of
frames in the video.

batch tensor → Image iterator
-----------------------------

A batch tensor can be converted to a set of images.  This is done using an :class:`Image` iterator:

.. code-block:: python

    from machinevisiontoolbox import TensorStack
    from torch import rand
    tensor = rand(16, 480, 640, 3)  # random batch of 16 RGB images
    for img in TensorStack(tensor):
        img.disp(fps=4)

This particular example could be achieved a little more concisely by using
the :meth:`ImageSource.disp` method inherited by all image sources, including :class:`TensorStack`.
A single line of code iterates over a tensor and displays the frames as a video:

.. code-block:: python

    from machinevisiontoolbox import TensorStack
    from torch import rand
    tensor = rand(16, 480, 640, 3)  # random batch of 16 RGB images
    TensorStack(tensor).disp(fps=4)


