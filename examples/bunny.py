# display images within a zip file as an animation

from machinevisiontoolbox import PointCloud

# load bunny point cloud from the included data files
pcd = PointCloud.Read("bunny.ply")

print(pcd)
pcd.disp(block=True)
