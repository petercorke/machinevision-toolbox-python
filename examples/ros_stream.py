"""
Simple test for RosTopic: display frames from /camera/image/compressed via rosbridge.

Usage::

    $ python examples/ros_stream.py [host] [port]

The rosbridge server must be running, e.g.::

    $ roslaunch rosbridge_server rosbridge_websocket.launch
"""

import sys
from machinevisiontoolbox import RosTopic

host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 9090

print(f"Connecting to rosbridge at {host}:{port} ...")

with RosTopic("/camera/image/compressed", host=host, port=port) as stream:
    print(stream)
    print("Press [space] to step, [q] to quit.")
    stream.disp()
