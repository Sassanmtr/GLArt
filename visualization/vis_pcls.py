import mesh_to_sdf
import os
from pathlib import Path
import numpy as np
from visualize import RerunViewer, Grid3D, object_to_point_cloud, object_to_trimesh
import torch

# Description: Visualize the pointclouds (For debugging purposes)
points = Grid3D(density=64, grid_dim=1.0, device="cpu", precision=torch.float32).points
grid_points = points.detach().numpy()
current_directory = Path.cwd()
base_dir = current_directory
pcl_dir = str(base_dir / "datasets" / "pointclouds")
files = os.listdir(pcl_dir)
files.sort()
vis = RerunViewer()
for file in files:
    if file.endswith(".npz"):
        # get the object name
        object_name = file.split("_")[0]
        scale = float(file.split("_")[1])
        joint_state = file.split("_")[2][:-4]
        print(
            "Object name: ", object_name, "Scale: ", scale, "Joint state: ", joint_state
        )
        pcl_path = os.path.join(pcl_dir, file)
        # load pointclouds and generate pointcloud
        loaded_data = np.load(pcl_path, allow_pickle=True)
        pc_points = loaded_data["balanced_points"]    # partial pointcloud
        urdf_path = str(base_dir / "DataGenObj" / object_name / "mobility.urdf")
        object_trimesh = object_to_trimesh(urdf_path, scale, joint_state)
        pc_full, _ = object_to_point_cloud(object_trimesh)   # full pointcloud
        # print(
        #         f"Number of points {object_name}_{scale}_{joint_state}: ", len(pc_points)
        #     )
        
        # vis.add_pointcloud("pcd", grid_points, radii=0.002)
        vis.clear()
        vis.add_sphere()
        vis.add_grid_box()
        # vis.add_pointcloud("vis/pcd_full", pc_full, radii=0.002)
        # vis.add_pointcloud("vis/pcd_partial", pc_points, radii=0.002)
        input("Press enter to continue")
        