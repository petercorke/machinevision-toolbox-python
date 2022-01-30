# simple wrapper for Open3D that has look and feel like MVTB

import numpy as np
import spatialmath.base as smbase
from spatialmath import SE3

try:
    import open3d as o3d
    _open3d = True
except ModuleNotFoundError:
    _open3d = False

class PointCloud:

    def __init__(self, arg, image=None, colors=None, camera=None, depth_scale=1.0, **kwargs):

        if not _open3d:
            raise RuntimeError("PointCloud class requires Open3D to be installed: pip install open3d")

        if isinstance(arg, o3d.geometry.PointCloud):
            pcd = arg

        elif isinstance(arg, np.ndarray) and arg.ndim == 2 and arg.shape[0] == 3:
            # simple point cloud:
            # passed a 3xN array of point coordinates
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(arg.T)
            if colors is not None and colors.shape == arg.shape:
                if np.issubdtype(colors.dtype, np.integer):
                    colors = colors / np.iinfo(colors.dtype).max
                pcd.colors = o3d.utility.Vector3dVector(colors.T)

        elif isinstance(arg, np.ndarray) and image is not None and camera is not None:
            # colored point cloud:
            # passed a WxH array of depth plus a WxH image
            if arg.shape != image.shape[:2]:
                print(arg.shape, image.image.shape)
                raise ValueError('depth array and image must be same shape')

            if image.iscolor and "convert_rgb_to_intensity" not in kwargs:
                kwargs["convert_rgb_to_intensity"] = False
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(image.image), 
                o3d.geometry.Image(arg), 
                depth_scale=depth_scale,
                **kwargs)

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(
                    image.width, image.height, 
                    *camera.fpix, *camera.pp))
        else:
            raise ValueError('bad arguments')

        self._pcd = pcd

    def copy(self):
        return self.__class__(o3d.geometry.PointCloud(self._pcd))

    def __len__(self):
        return np.asarray(self._pcd.points).shape[0]

    # allow methods of o3d.geometry.PointCloud to be invoked indirectly
    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            meth = getattr(self._pcd, name)
            return meth(*args, **kwargs)

        if hasattr(self._pcd, name):
            return wrapper


    def __str__(self):
        return str(self._pcd)
    
    def __repr__(self):
        return str(self)

    @property
    def pcd(self):
        return self._pcd

    @property
    def points(self):
        return np.asarray(self._pcd.points).T

    @property
    def colors(self):
        return np.asarray(self._pcd.colors).T
    
    @classmethod
    def Read(cls, filename, *args, **kwargs):

        from machinevisiontoolbox import mvtb_path_to_datafile

        filename = mvtb_path_to_datafile(filename, string=True)
        pcd = o3d.io.read_point_cloud(filename, *args, **kwargs)
        return cls(pcd)

    def disp(self, block=True, file=None, **kwargs):
        if block:
            o3d.visualization.draw_geometries([self._pcd], **kwargs)
        else:
            # nonblocking display
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(self._pcd)

            # get object to control viewpoint
            view_control = vis.get_view_control()

            # handle the possible options
            if "front" in kwargs:
                view_control.set_front(kwargs["front"])
            if "lookat" in kwargs:
                view_control.set_lookat(kwargs["lookat"])
            if "up" in kwargs:
                view_control.set_up(kwargs["up"])
            if "zoom" in kwargs:
                view_control.set_zoom(kwargs["zoom"])

            # update the display
            vis.poll_events()
            vis.update_renderer()

            # save to file if requested
            if file is not None:
                vis.capture_screen_image(str(file), do_render=False)

    def __rmul__(self, T):
        if isinstance(T, SE3):
            return self.copy().transform(T)
        else:
            return NotImplemented

    def __imul__(self, T):
        if isinstance(T, SE3):
            self.transform(T)
            return self
        else:
            return NotImplemented

    def transform(self, T, inplace=True):
        if inplace:
            self._pcd.transform(T.A)
            return self
        else:
            new = self.copy()
            new._pcd.transform(T.A)
            return new

    def __add__(self, other):
        pcd = o3d.geometry.PointCloud()
        pcd.points.extend(self.points.T)
        pcd.points.extend(other.points.T)
        pcd.colors.extend(self.colors.T)
        pcd.colors.extend(other.colors.T)
        return self.__class__(pcd)

    def downsample_voxel(self, voxel_size):
        return self.__class__(self._pcd.voxel_down_sample(voxel_size))

    # random downsample
    def downsample_random(self, fraction, seed=None):
        if seed is None:
            return self.__class__(self._pcd.random_down_sample(fraction))
        
        if seed >= 0:
            np.random.seed(seed)
        n = len(self)
        ind = np.random.choice(n, int(fraction * n), replace=False)
        return self.__class__(self._pcd.select_by_index(ind))

    def normals(self, **kwargs):
        # radius=0.1, max_nn=30
        self._pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(**kwargs))

    def remove_outlier(self, **kwargs):
        # nb_points=16, radius=0.2
        pcd, ind = self._pcd.remove_radius_outlier(**kwargs)
        return self.__class__(pcd)

    def segment_plane(self, **kwargs):
        # distance_threshold=0.05,
        # ransac_n=3,
        # num_iterations=1000

        default_dict = dict(ransac_n=3, num_iterations=100)
        kwargs = {**default_dict, **kwargs}

        plane_model, inliers = self._pcd.segment_plane(**kwargs)
        inlier_cloud = self._pcd.select_by_index(inliers)
        outlier_cloud = self._pcd.select_by_index(inliers, invert=True)
        return plane_model, self.__class__(inlier_cloud), self.__class__(outlier_cloud)

    def select(self, ind, invert=False):
        return self.__class__(self._pcd.select_by_index(ind, invert=invert))

    def paint(self, color):
        self._pcd.paint_uniform_color(color)
        return self

    def ICP(self, data, T0=None, max_correspondence_distance=1, **kwargs):
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

        # Convergence-Criteria for Vanilla ICP
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(**kwargs)

        if T0 is None:
            T0 = np.eye(4)
        else:
            T0 = T0.A

        status = o3d.pipelines.registration.registration_icp(
            self._pcd, 
            data._pcd, 
            max_correspondence_distance,
            T0,
            estimation, 
            criteria)
            #voxel_size, save_loss_log)

        T = SE3(smbase.trnorm(status.transformation))

        return T, status

    def voxel_grid(self, voxel_size):
        return VoxelGrid(self, voxel_size)
class VoxelGrid:

    def __init__(self, pcd, voxel_size):
        self._voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd._pcd, voxel_size=voxel_size)

    def disp(self, block=True, file=None, **kwargs):
        if block:
            o3d.visualization.draw_geometries([self._voxels], **kwargs)
        else:
            # nonblocking display
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(self._voxels)

            # get object to control viewpoint
            view_control = vis.get_view_control()

            # handle the possible options
            if "front" in kwargs:
                view_control.set_front(kwargs["front"])
            if "lookat" in kwargs:
                view_control.set_lookat(kwargs["lookat"])
            if "up" in kwargs:
                view_control.set_up(kwargs["up"])
            if "zoom" in kwargs:
                view_control.set_zoom(kwargs["zoom"])

            render = vis.get_render_option()
            render.mesh_show_wireframe = True

            # update the display
            vis.poll_events()
            vis.update_renderer()

            # save to file if requested
            if file is not None:
                vis.capture_screen_image(str(file), do_render=False)

if __name__ == "__main__":
    from machinevisiontoolbox import mvtb_path_to_datafile

    pcd = PointCloud.Read(mvtb_path_to_datafile('data/bunny.ply'))
    print(pcd)
    pcd.disp(block=False, file='bun.png')

    # import time
    # time.sleep(4)