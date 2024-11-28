import mesh_to_sdf
import os
from pathlib import Path
import numpy as np
import trimesh
from visualize import RerunViewer
import matplotlib.pyplot as plt
from visualize import create_markers_multiple, object_to_point_cloud, object_to_trimesh

def visualize_pointcloud_and_grasps(pointcloud, grasp_poses):
    # Plot the point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pointcloud[:,0], pointcloud[:,1], pointcloud[:,2], c='b', marker='.', label='Point Cloud')

    # Plot the grasp poses
    for grasp in grasp_poses:
        fk = grasp.visual_trimesh_fk()
        # trimesh_scene = trimesh.Scene()
        for mesh, pose in fk.items():
            try:
                trimesh.Scene.add_geometry(trimesh.Scene(), mesh, transform=pose)
            except:
                trimesh.Scene.add_geometry(trimesh.Scene(), mesh, transform=pose["pose"])



if __name__ == "__main__":
    # Description: Visualize the grasps on the object point cloud (For debugging purposes)
    current_directory = Path.cwd()
    base_dir = current_directory
    grasp_dir = str(base_dir / "datasets" / "grasps")
    files = os.listdir(grasp_dir)
    files.sort()
    sum_grasps = 0
    for file in files:
        if file.endswith(".npy"): #and tar in file:
            # get the object name
            object_name = file.split("_")[0]
            scale = float(file.split("_")[1])
            joint_state = file.split("_")[2][:-4]
            if object_name == "101943": #and scale == 0.5 and float(joint_state) >1.3:
                urdf_path = str(base_dir / "urdfs" / object_name / "mobility.urdf")
                grasp_path = os.path.join(grasp_dir, file)
                # load grasps and generate pointcloud
                grasps = np.load(grasp_path)
                grasp_trimesh = create_markers_multiple(
                    grasps, color=[0.0, 1.0, 0.0], axis_frame=True, highlight_first=True
                )
                sum_grasps += len(grasps)
                if len(grasps) < 500:
                    print(
                        f"Number of grasps {object_name}_{scale}_{joint_state}: ", len(grasps)
                    )
                object_trimesh = object_to_trimesh(urdf_path, scale, joint_state)
                pc_points, _ = object_to_point_cloud(object_trimesh)
                # visualize_pointcloud_and_grasps(pc_points, grasp_trimesh)
                vis = RerunViewer()
                vis.clear()
                vis.add_sphere()
                vis.add_pointcloud("vis/pcd_full", pc_points, radii=0.002)
                # vis.add_trimeshes("object", [object_trimesh])
                vis.add_grasps("vis/grasps", grasps)
                input("Press enter to continue")
    print("Total number of grasps: ", sum_grasps)