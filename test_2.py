

import open3d as o3d
import numpy as np
print("Testing kdtree in Open3D...")
print("Load a point cloud and paint it gray.")

sample_pcd_data = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud(sample_pcd_data.path)
pcd.paint_uniform_color([0.5, 0.5, 0.5])
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

print("Paint the 1501st point red.")
pcd.colors[1500] = [1, 0, 0]

print("Find its 200 nearest neighbors, and paint them blue.")
[k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], 200)
np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
