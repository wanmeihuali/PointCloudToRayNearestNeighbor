# %%
import torch
import taichi as ti
from dataclasses import dataclass

# %%
@ti.func
def extract_point_information(
    pointcloud: ti.types.ndarray(ti.f32, ndim=2), # (N, 3)
    point_id: ti.i32,
    T_camera_pointcloud_mat,
    camera_intrinsics_mat
):
    homogenous_point_in_global = ti.Vector(
        [pointcloud[point_id, 0], pointcloud[point_id, 1], pointcloud[point_id, 2], 1.0])
    homogenous_point_in_camera = T_camera_pointcloud_mat @ homogenous_point_in_global
    point_in_camera = ti.Vector(
        [homogenous_point_in_camera[0], homogenous_point_in_camera[1], homogenous_point_in_camera[2]])
    pixel_in_camera = camera_intrinsics_mat @ (point_in_camera / point_in_camera[2])
    pixel_u = pixel_in_camera[0]
    pixel_v = pixel_in_camera[1]
    return point_in_camera, pixel_u, pixel_v

@ti.dataclass
class NNPoint:
    point_id: ti.i64
    depth: ti.f32

@ti.func
def bubble_sort_by_depth(
    ray_nn_points: ti.types.ndarray(ti.i64, ndim=3), # (W, H, K)
    ray_nn_depth: ti.types.ndarray(ti.f32, ndim=3), # (W, H, K)
    v: ti.i32,
    u: ti.i32
):
    num_points = ray_nn_points.shape[2]
    for i in range(num_points):
        for j in range(num_points - i - 1):
            if ray_nn_depth[v, u, j] > ray_nn_depth[v, u, j + 1]:
                temp_0 = ray_nn_depth[v, u, j]
                ray_nn_depth[v, u, j] = ray_nn_depth[v, u, j + 1]
                ray_nn_depth[v, u, j + 1] = temp_0
                temp = ray_nn_points[v, u, j]
                ray_nn_points[v, u, j] = ray_nn_points[v, u, j + 1]
                ray_nn_points[v, u, j + 1] = temp

@ti.kernel
def filter_point_in_camera(
    pointcloud: ti.types.ndarray(ti.f32, ndim=2), # (N, 3)
    camera_intrinsics: ti.types.ndarray(ti.f32, ndim=2), # (3, 3)
    T_camera_pointcloud: ti.types.ndarray(ti.f32, ndim=2), # (4, 4)
    point_in_camera_tensor: ti.types.ndarray(ti.i8, ndim=1), # (N)
    far_plane: ti.f32,
    camera_width: ti.i32,
    camera_height: ti.i32,
):
    T_camera_pointcloud_mat = ti.Matrix([[T_camera_pointcloud[row, col] for col in range(4)] for row in range(4)])
    camera_intrinsics_mat = ti.Matrix([[camera_intrinsics[row, col] for col in range(3)] for row in range(3)])
                                        
    # filter points in camera
    for point_id in range(pointcloud.shape[0]):
        point_in_camera, pixel_u, pixel_v = extract_point_information(
            pointcloud=pointcloud,
            point_id=point_id,
            T_camera_pointcloud_mat=T_camera_pointcloud_mat,
            camera_intrinsics_mat=camera_intrinsics_mat)
        depth_in_camera = point_in_camera[2]
        if depth_in_camera > 0 and pixel_u >= 0 and pixel_u < camera_width and pixel_v >= 0 and pixel_v < camera_height and depth_in_camera < far_plane:
            point_in_camera_tensor[point_id] = ti.cast(1, ti.i8)
        else:
            point_in_camera_tensor[point_id] = ti.cast(0, ti.i8)
 
@ti.kernel
def point_cloud_to_ray_nn_taichi(
    pointcloud: ti.types.ndarray(ti.f32, ndim=2), # (N, 3)
    camera_intrinsics: ti.types.ndarray(ti.f32, ndim=2), # (3, 3)
    T_camera_pointcloud: ti.types.ndarray(ti.f32, ndim=2), # (4, 4)
    ray_nn_points_tensor: ti.types.ndarray(ti.i64, ndim=3), # (W, H, K)
    ray_nn_depth_tensor: ti.types.ndarray(ti.f32, ndim=3), # (W, H, K)
    ray_nn_offset_vector_tensor: ti.types.ndarray(ti.f32, ndim=4), # (W, H, K, 3)
    nn_radius: ti.f32,
    max_camera_plane_radius: ti.template(), # float
    far_plane: ti.f32,
    camera_width: ti.i32,
    camera_height: ti.i32,
    in_camera_point_id_list: ti.types.ndarray(ti.i64), 
    ray_nn_points_count_tensor: ti.types.ndarray(ti.i32, ndim=2), # (W, H)
):
    T_camera_pointcloud_mat = ti.Matrix([[T_camera_pointcloud[row, col] for col in range(4)] for row in range(4)])
    T_pointcloud_camera_mat = T_camera_pointcloud_mat.inverse()
    R_pointcloud_camera_mat = T_pointcloud_camera_mat[0:3, 0:3]
    camera_intrinsics_mat = ti.Matrix([[camera_intrinsics[row, col] for col in range(3)] for row in range(3)])
    inv_camera_intrinsics_mat = camera_intrinsics_mat.inverse()
    background_id = pointcloud.shape[0]
    focal_length = camera_intrinsics_mat[0, 0]
    # for each point in camera, the rays which have the point as the nearest neighbor will form a circle
    # in the image plane. The radius of the circle is determined by the depth of the point.
    for idx in ti.ndrange(in_camera_point_id_list.shape[0]):
        in_camera_point_id = in_camera_point_id_list[idx]
        point_in_camera, pixel_u, pixel_v = extract_point_information(
            pointcloud=pointcloud,
            point_id=in_camera_point_id,
            T_camera_pointcloud_mat=T_camera_pointcloud_mat,
            camera_intrinsics_mat=camera_intrinsics_mat)
        depth_in_camera = point_in_camera[2]
        camera_plane_radius = focal_length * nn_radius / depth_in_camera
        camera_plane_radius = ti.min(camera_plane_radius, max_camera_plane_radius)
        camera_plane_radius_int32 = ti.cast(camera_plane_radius, ti.i32)
        camera_plane_radius_int32 = ti.max(camera_plane_radius_int32, 1)
        for row_offset in range(-camera_plane_radius_int32, camera_plane_radius_int32 + 1):
            for col_offset in range(-camera_plane_radius_int32, camera_plane_radius_int32 + 1):
                if (row_offset * row_offset + col_offset * col_offset) > (camera_plane_radius_int32 * camera_plane_radius_int32):        
                    continue
                affected_pixel_u = ti.cast(pixel_u + col_offset, ti.i32)
                affected_pixel_v = ti.cast(pixel_v + row_offset, ti.i32)
                if affected_pixel_u < 0 or affected_pixel_u >= camera_width or affected_pixel_v < 0 or affected_pixel_v >= camera_height:
                    continue
                # fixed array is 30x faster than dynamic array in taichi in this case... give up using dynamic array
                # although dynamic array is more correct, it is not worth the performance penalty.
                offset = ti.atomic_add(ray_nn_points_count_tensor[affected_pixel_v, affected_pixel_u], 1) 
                if offset < ray_nn_points_tensor.shape[2]:
                    ray_nn_points_tensor[affected_pixel_v, affected_pixel_u, offset] = in_camera_point_id
                    ray_nn_depth_tensor[affected_pixel_v, affected_pixel_u, offset] = depth_in_camera
    
    # for each ray, the nearest neighbor points are not sorted by depth. We need to sort them.
    # Because pytorch cannot handle dynamic length fields, we only keep the first K nearest neighbor points.
    for v, u in ti.ndrange(camera_height, camera_width):
        bubble_sort_by_depth(ray_nn_points_tensor, ray_nn_depth_tensor, v, u)

    # fill the output tensors
    for v, u, k in ti.ndrange(camera_height, camera_width, ray_nn_points_tensor.shape[2]):
        if k < ray_nn_points_count_tensor[v, u]:
            nn_point_id = ray_nn_points_tensor[v, u, k]
            nn_point_in_camera, point_u, point_v = extract_point_information(
                pointcloud=pointcloud,
                point_id=nn_point_id,
                T_camera_pointcloud_mat=T_camera_pointcloud_mat,
                camera_intrinsics_mat=camera_intrinsics_mat)
            ray_vector = (inv_camera_intrinsics_mat @ ti.Vector([u + 0.5, v + 0.5, 1.0])).normalized()
            projection_scalar = nn_point_in_camera.dot(ray_vector)
            projected_point = projection_scalar * ray_vector
            offset_vector_in_camera = projected_point - nn_point_in_camera
            offset_vector_in_pointcloud = R_pointcloud_camera_mat @ offset_vector_in_camera
            ray_nn_offset_vector_tensor[v, u, k, 0] = offset_vector_in_pointcloud[0]
            ray_nn_offset_vector_tensor[v, u, k, 1] = offset_vector_in_pointcloud[1]
            ray_nn_offset_vector_tensor[v, u, k, 2] = offset_vector_in_pointcloud[2]
        else:
            ray_nn_points_tensor[v, u, k] = background_id
            ray_nn_depth_tensor[v, u, k] = far_plane
            ray_nn_offset_vector_tensor[v, u, k, 0] = 0.0
            ray_nn_offset_vector_tensor[v, u, k, 1] = 0.0
            ray_nn_offset_vector_tensor[v, u, k, 2] = 0.0
        
class PointCloudToRayNearestNeighbor:
    """ Given a point cloud and a camera, for rays corresponding to each pixel in the camera image, find the 
    points in the point cloud that are near enough to the ray. The points are sorted by depth. 
    Returns:
        RenderResult: See the definition of RenderResult for details.
    """
    
    @dataclass
    class Config:
        camera_intrinsics: torch.Tensor # 3x3 tensor, the camera intrinsics matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        camera_width: int = 640 # the width of the camera image
        camera_height: int = 480 # the height of the camera image
        nn_radius: float = 0.2 # the radius of the nearest neighbor search
        max_points_per_pixel: int = 50 # the maximum number of points per pixel to record
        far_plane: float = 1000.0 # the far plane of the camera
        max_camera_plane_radius: float = 60.0 # the maximum radius of the camera plane in pixel space

    @dataclass
    class RenderResult:
        # 3D tensor with shape (camera_height, camera_width, max_points_per_pixel) of int64, the point indices in the point cloud, if the point is not valid, the index is set to the number of points in the point cloud
        ray_nn_points: torch.Tensor 
        # 3D tensor with shape (camera_height, camera_width, max_points_per_pixel) of float32, the depth of the points in the camera coordinate system, if the point is not valid, the depth is set to the far plane
        ray_nn_depth: torch.Tensor
        # 4D tensor with shape (camera_height, camera_width, max_points_per_pixel, 3) of float32, the offset vector from the nearest neighbor point to the projected point on the ray, if the point is not valid, the offset vector is set to (0, 0, 0)
        ray_nn_offset_vector: torch.Tensor 

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(self, pointcloud: torch.Tensor, T_camera_pointcloud: torch.Tensor) -> RenderResult:
        assert pointcloud.dim() == 2 and pointcloud.shape[1] == 3, "pointcloud must be a 2D tensor with shape (N, 3)"
        background_id = pointcloud.shape[0]
        num_of_points = pointcloud.shape[0]

        # create the torch tensor for the result
        ray_nn_points_tensor = torch.full(
            size=(self.config.camera_height, self.config.camera_width, self.config.max_points_per_pixel),
            fill_value=background_id,
            dtype=torch.int64,
            device=pointcloud.device)
        ray_nn_points_count_tensor = torch.zeros(
            size=(self.config.camera_height, self.config.camera_width),
            dtype=torch.int32,
            device=pointcloud.device)
        ray_nn_depth_tensor = torch.full(
            size=(self.config.camera_height, self.config.camera_width, self.config.max_points_per_pixel),
            fill_value=self.config.far_plane,
            dtype=torch.float32,
            device=pointcloud.device)
        ray_nn_offset_vector_tensor = torch.zeros(
            size=(self.config.camera_height, self.config.camera_width, self.config.max_points_per_pixel, 3),
            dtype=torch.float32,
            device=pointcloud.device)
        point_in_camera = torch.zeros(
            size=(num_of_points,),
            dtype=torch.int8,
            device=pointcloud.device)

        filter_point_in_camera(
            pointcloud=pointcloud,
            camera_height=self.config.camera_height,
            camera_width=self.config.camera_width,
            far_plane=self.config.far_plane,
            camera_intrinsics=self.config.camera_intrinsics,
            T_camera_pointcloud=T_camera_pointcloud,
            point_in_camera_tensor=point_in_camera)
        
        in_camera_point_id_list = torch.arange(num_of_points, device=pointcloud.device, dtype=torch.int64)[
            point_in_camera.to(torch.bool)].contiguous()

        point_cloud_to_ray_nn_taichi(
            pointcloud=pointcloud,
            camera_intrinsics=self.config.camera_intrinsics,
            T_camera_pointcloud=T_camera_pointcloud,
            ray_nn_points_tensor=ray_nn_points_tensor,
            ray_nn_depth_tensor=ray_nn_depth_tensor,
            ray_nn_offset_vector_tensor=ray_nn_offset_vector_tensor,
            nn_radius=self.config.nn_radius,
            max_camera_plane_radius=self.config.max_camera_plane_radius,
            far_plane=self.config.far_plane,
            camera_height=self.config.camera_height,
            camera_width=self.config.camera_width,
            in_camera_point_id_list=in_camera_point_id_list,
            ray_nn_points_count_tensor=ray_nn_points_count_tensor)
        result = PointCloudToRayNearestNeighbor.RenderResult(
            ray_nn_points=ray_nn_points_tensor,
            ray_nn_depth=ray_nn_depth_tensor,
            ray_nn_offset_vector=ray_nn_offset_vector_tensor)
        return result
