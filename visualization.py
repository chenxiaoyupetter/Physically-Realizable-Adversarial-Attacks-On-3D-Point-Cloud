import open3d as o3d
from open3d import geometry
import numpy as np
import torch
print(torch.cuda.device_count())
torch.cuda.is_available()

vis = o3d.visualization.Visualizer()
vis.create_window()
mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
    size=1, origin=[0, 0, 0])
vis.add_geometry(mesh_frame)
# mesh,voxel_mesh= load_ply_point_cloud(file,size)
# open3d.visualization.draw_geometries([mesh])
ply1 = o3d.io.read_point_cloud('/home/car/桌面/pythonProject/MSF-ADV-master/object/cube.ply')

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(0.0001)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh
import torch

ply2 = o3d.io.read_point_cloud('/home/car/桌面/pythonProject/MSF-ADV-master/object/obj_savelidar_v2.ply')
pcd_down, pcd_fpfh =preprocess_point_cloud(ply1, 0.2)
a= np.array(pcd_fpfh.data).T
o3d.visualization.draw_geometries([ply1])

o3d.visualization.draw_geometries([ply2])

# def load_ply_point_cloud(filename,voxel_size=0.5):
#     print("Load a ply point cloud, print it, and render it")
#     mesh= o3d.io.read_triangle_mesh(filename)
#     # voxel_mesh = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
#     return mesh,voxel_mesh

file0 = '/home/car/桌面/pythonProject/MSF-ADV-master/object/1.ply'
file1 = '/home/car/桌面/pythonProject/MSF-ADV-master/object/obj_savelidar_v2.ply'



size = 0.55
mesh = o3d.io.read_triangle_mesh(file0)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
mesh = o3d.io.read_triangle_mesh(file1)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])

# 点云

# points_ar = []
# for x in voxel_mesh.voxels:
#     points_ar.append(x.grid_index)
# points = np.array(points_ar)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
#
# # 可视化
# o3d.visualization.draw_geometries([pcd])
