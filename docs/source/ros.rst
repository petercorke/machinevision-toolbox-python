ROS interfaces
==============

The toolbox provides interfaces to ROS -- an important framework for
robotics and connecting to the physical world through sensors and actuators.  

The ROS interface allows you to
read images, point clouds and other message from ROS bag files or published topics from a running
ROS system.  

Reading a ROS bag file
----------------------
We can 

.. runblock:: pycon

    >>> from machinevisiontoolbox import ROSBag
    >>> bag = ROSBag('test_ros1.bag')
    >>> bag.print()
    >>> bag = ROSBag('test_ros2.bag', topicfilter="camera")
    >>> for image in bag:
    ...     print(image)


Subscribing/publishing to a ROS message stream
----------------------------------------------

This is similar to reading a ROS bag file, but instead of reading from a file we read from a running ROS system.  For example, to subscribe to a topic called ``/camera/image`` that publishes images::

.. important:: 
    You need to have ROS installed and running with a rosbridge2 node in order to use this feature.  The toolbox
    supports both ROS1 and ROS2, but you need to have the appropriate version of ROS
    installed and sourced in your environment. rosbridge

    from machinevisiontoolbox import ROSTopic

    with ROSTopic("/camera/image", "/msg/sensor_msgs/Image") as camera:
        for image in camera:
            print(image)

Synchronizing messages from multiple topics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some ROS publishers publish related information on multiple topics, for example a RealSense
camera might publish RGB images on one topic and Depth images on another topic.  The toolbox
provides a convenient interface to read these related messages together using the
``ROSTopic`` source.  For example, to read images from a topic called
``/camera/rgb`` and the corresponding camera info from a topic called
``/camera/depth``:

.. code:: python

    from machinevisiontoolbox import ROSTopic, SyncROSStreams, Image

    rgb = ROSTopic("/camera/rgb", "/msg/sensor_msgs/Image", blocking=True)
    depth = ROSTopic("/camera/depth", "/msg/sensor_msgs/Image", blocking=True)
    with SyncROSStreams("/camera/rgb", "/camera/depth", tol=20e-3) as rgbd:
         for rgb_msg, depth_msg in rgbd:
            # rgb_msg and depth_msg are both Image instances and emitted
            # by the publisher at the same time step
            print(rgb_msg)


Publishing a ROS message
^^^^^^^^^^^^^^^^^^^^^^^^

A :class:`ROSTopic` instance can also be used to *publish* messages on a ROS topic.  For
example, to periodically publish an image to a topic called ``/camera/rgb``::

    from machinevisiontoolbox import ROSTopic
    from time import sleep
    img = Image.Read("monalisa.png")
    camera = ROSTopic("/camera/rgb", "/msg/sensor_msgs/Image") # connect to topic
    for _ in range(10):
        camera.publish(img)
        sleep(0.1)

For compressed image message types the image will be automatically compressed before
publishing, and decompressed when reading.

The argument to :meth:`ROSTopic.publish` can be an :class:`Image`, a :class:`PointCloud` or
a dict containing elements of a general message.
