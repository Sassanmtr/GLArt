import os
from pathlib import Path
import numpy as np
from visualize import RerunViewer

current_directory = Path.cwd()
base_dir = current_directory
pcl_dir = str(base_dir / "datasets" / "pointclouds")
files = os.listdir(pcl_dir)
files.sort()
for file in files:
    # if ends with .npy
    if file.endswith(".npz"):
        # get the object name
        object_name = file.split("_")[0]
        scale = float(file.split("_")[1])
        joint_state = file.split("_")[2][:-4]
        print(
            "Object name: ", object_name, "Scale: ", scale, "Jointstate: ", joint_state
        )
        pcl_path = os.path.join(pcl_dir, file)
        # load pointclouds and generate pointcloud
        loaded_data = np.load(pcl_path, allow_pickle=True)
        pc_points = loaded_data["balanced_points"]
        print(
                f"Number of grasps {object_name}_{scale}_{joint_state}: ", len(pc_points)
            )
        vis = RerunViewer()
        vis.add_pointcloud("pcd", pc_points, radii=0.002)
        input("Press enter to continue")