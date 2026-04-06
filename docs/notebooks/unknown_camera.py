import numpy as np
from spatialmath import SE3, SO3
from machinevisiontoolbox import CentralCamera


# Create a MVTB central camera model
def CreateSimulatedCamera(x=1, y=-1, z=0.01, roll=-92, pitch=2, yaw=50, image_size=(1280,1024), f=0.015):
    """Create a Machine Vision Toolbox central camera model given 6 DoF pose, image size and f.
        
        Args In: 
            x - position of camera in x-axis world frame (in metres)
            y - position of camera in y-axis world frame (in metres)
            z - position of camera in z-axis world frame (in metres)
            roll - rotation of the camera about the x-axis world frame (in degrees)
            pitch - rotation of the camera about the y-axis world frame (in degrees)
            yaw - rotation of the camera about the z-axis world frame (in degrees)
            image_size - two element tuple specifying the width and height of the image (in pixels)
            f - focal length
            
       Returns:
           a camera model
        
    """
    
    # Establish a camera position with respect to the world frame
    # position
    t_cam = np.r_[x, y, z] 
    
    # orientation
    R = SO3.Rz(yaw, 'deg') * SO3.Ry(pitch, 'deg') * SO3.Rx(roll, 'deg')
    
    # Create full transformation matrix
    T = SE3(t_cam) * SE3.SO3(R)
    
    # print(T)
    
    # Create camera model
    cam_model = CentralCamera(imagesize=image_size, f=f, pose=T)
    
    return cam_model