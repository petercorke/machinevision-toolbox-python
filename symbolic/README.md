# Symbolic

This folder contains a SymPy script that computes the Jacobians and projection function
for bundle adjustment. Pose is expressed as a translation vector plus the vector
part of a unit quaternion, the `vector` method of the `UnitQuaternion` class.

This code is generated as `camera_derivatives.py` which is wrapped by the `derivatives` 
method of the `CentralCamera` class.