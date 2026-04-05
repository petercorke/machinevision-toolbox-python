

ROS and PyTorch interfaces
==========================

The toolbox provides interfaces to ROS and PyTorch -- two very important connections
that connect the toolbox to physical sensors and machine learning frameworks.  

ROS interface
-------------

The ROS interface allows you to
read images, point clouds and other message from ROS bag files or published topics from a running
ROS system.  

Reading a ROS bag file
^^^^^^^^^^^^^^^^^^^^^^

We can 

.. runblock:: pycon

    >>> from machinevisiontoolbox import RosBag
    >>> bag = RosBag('test_ros1.bag')
    >>> bag.print()
    >>> bag = RosBag('test_ros2.bag', topicfilter="camera")
    >>> for image in bag:
    ...     print(image)


Subscribing to a ROS message stream
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Publishing a ROS message
^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch interface
-----------------

The ROS interface allows you to
read images and point clouds from ROS bag files, and display them.  The PyTorch
interface allows you to use the toolbox's image processing functions in PyTorch
pipelines, and to convert between the toolbox's Image class and PyTorch tensors.




