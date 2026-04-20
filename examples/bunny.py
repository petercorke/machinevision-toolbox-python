# display images within a zip file as an animation

from machinevisiontoolbox import PointCloud

# load bunny point cloud from the included data files
pcd = PointCloud.Read("bunny.ply")

print(pcd)
print("\nUse your mouse to interact with the 'bunny' point cloud. 'q' to quit")
pcd.disp(block=True)
