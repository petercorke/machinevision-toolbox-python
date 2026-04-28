ROS interfaces
==============

The toolbox provides interfaces to ROS -- an important framework for
robotics and connecting to the physical world through sensors and actuators.  

The ROS interface allows you to
read images, point clouds and other message from ROS bag files or published topics from a running
ROS system.  

Reading a ROS bag file
----------------------
We can read images from a ROS bag file using the :class:`ROSBag` source.

.. code-block:: pycon

    >>> from machinevisiontoolbox import ROSBag
    >>> bag = ROSBag('test_ros1.bag')
    >>> bag.print()
    >>> bag = ROSBag('test_ros2.bag')
    >>> for image in bag:
    ...     print(image)

Messages returned by the ``ROSBag`` source are filtered based on the topic and message
filters. The default message filter is ``"Image"``, which means only image messages will
be returned. 

For an application like VSLAM we might want to read additional, non-image, messages from the bag file. We do this by specifying a message filter using the
``msgfilter`` argument. For example, to read all images and IMU measurements

.. code-block:: pycon

    >>> bag = ROSBag('test_ros1.bag', msgfilter=['Image', 'Imu'])

We can check the filter is working by looking at the last column of the table printed by
``bag.print()`` which indicates whether each topic is allowed (according to the message
and topic filters applied) to be read from the bag file.

.. code-block:: pycon

    >>> bag.print()

    ┌────────────────────────────┬───────────────────────┬───────┬─────────┐
    │           topic            │        msgtype        │ count │ allowed │
    ├────────────────────────────┼───────────────────────┼───────┼─────────┤
    │ /camera/fisheye2/image_raw │ sensor_msgs/msg/Image │   855 │    ✓    │
    │ /camera/odom/sample        │ nav_msgs/msg/Odometry │  5679 │    ✗    │
    │ /camera/imu                │ sensor_msgs/msg/Imu   │  5679 │    ✓    │
    └────────────────────────────┴───────────────────────┴───────┴─────────┘

Now we can read the allowed messages from the bag file using a loop:

    >>> for msg in bag:
    ...     print(msg)


In this case ``msg`` will be an :class:`Image`, or a class ``rosbags.typesys.stores.ros1_noetic.nav_msgs__msg__Odometry`` created by the `rosbags <https://pypi.org/project/rosbags/>`_
package.  For a message of the latter type we can access the fields of the message using dot notation.  For example:

.. code-block:: pycon

    >>> m.pose.pose.orientation
    geometry_msgs__msg__Quaternion(x=0.004015379585325718, y=-0.2130533903837204, z=-0.000446687190560624, w=0.9770321846008301, __msgtype__='geometry_msgs/msg/Quaternion')
    >>> m.pose.pose.orientation.x
    0.004015379585325718

*All* objects returned by the ``ROSBag`` source have additional attributes which provide information about the topic the message was received on and timestamp (ROS standard, nanoseconds since the epoch) from the message header.  For example:

.. code-block:: pycon

    >>> m.topic
    '/camera/imu'
    >>> m.timestamp
    1679876543210.1234

.. note:: The message and topic filters can be a single string, or a list of strings.  
        If a single string is provided, it is treated as a list with one element.  
        The filter passes if any of the strings in the list are a substring of the message type (for the message filter) or topic name (for the topic filter).
        It is an OR condition.

Subscribing/publishing to a ROS message stream
----------------------------------------------

This is similar to reading a ROS bag file, but instead of reading from a file we read from a running ROS system.

.. important:: 
    You need to have ROS installed and running with a ``rosbridge2`` node in order to use this feature.  The toolbox
    supports both ROS1 and ROS2, but you need to have the appropriate version of ROS
    installed and sourced in your environment.

For example, to subscribe to a topic called ``/camera/image`` that publishes images:

.. code-block:: python

    from machinevisiontoolbox import ROSTopic

    with ROSTopic("/camera/image", "/msg/sensor_msgs/Image") as camera:
        for image in camera:
            print(image.ts)  # print the timestamp of each received image message
            print(image) # print the image metadata (size, dtype, colororder, etc.)

Synchronizing messages from multiple topics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some ROS publishers publish related information on multiple topics, for example a RealSense
camera might publish RGB images on one topic and Depth images on another topic.  Naively reading from
two ``ROSTopic`` instances may result in non-corresponding images, eg. an RGB image from one sample time, and a Depth image from the previous time step.  

The Toolbox provides a convenient interface to read these related messages together using the
``ROSTopic`` source.  For example, to read images from a topic called
``/camera/rgb`` and the corresponding depth image from a topic called
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

The tolerance should be set based on the expected time difference between messages on
the two topics.  Setting it too low may result in no messages being returned, while
setting it too high may result in non-corresponding messages being returned.  In
practice, a tolerance of around 20 milliseconds is often a good starting point for
synchronizing RGB and Depth images from a RealSense camera.

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

The argument to :meth:`ROSTopic.publish` can be an :class:`Image`, a :class:`PointCloud` or
a dict containing elements of a general message.

.. note:: For compressed image message types the image will be automatically compressed before publishing, and decompressed when reading.