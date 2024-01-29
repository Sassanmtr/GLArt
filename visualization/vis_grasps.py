import mesh_to_sdf
import os
from pathlib import Path
import numpy as np
import trimesh
import urdfpy
from typing import Tuple
from visualize import RerunViewer

def object_to_trimesh(urdf_file_path, scale, joint_state):
    # Load the URDF robot model
    arti_obj = urdfpy.URDF.load(urdf_file_path)
    joint_name = arti_obj.actuated_joints[0].name
    fk = arti_obj.visual_trimesh_fk({joint_name: joint_state})
    trimesh_scene = trimesh.Scene()
    for mesh, pose in fk.items():
        scale_mat = np.eye(4) * scale
        scale_mat[3, 3] = 1.0
        try:
            trimesh_scene.add_geometry(mesh, transform=scale_mat @ pose)
        except:
            trimesh_scene.add_geometry(mesh, transform=scale_mat @ pose["pose"])
        # trimesh_scene.add_geometry(mesh, transform=scale_mat @ internal_transformed @ pose)
    ## Uncomment for visualization
    # trimesh_scene.show()

    return trimesh_scene


def object_to_point_cloud(
    object_trimesh, number_samples: int = 200000
) -> Tuple[np.ndarray, np.ndarray]:
    mesh, transform = mesh_to_sdf.utils.scale_to_unit_sphere(
        object_trimesh, get_transform=True
    )
    surface_point_cloud = mesh_to_sdf.get_surface_point_cloud(
        mesh,
        sample_point_count=number_samples,
        surface_point_method="sample",  # [scan, sample] To allow inside of the mesh?
        calculate_normals=True,
    )
    points = surface_point_cloud.points * transform["scale"] + transform["translation"]
    normals = surface_point_cloud.normals
    return points, normals


current_directory = Path.cwd()
base_dir = current_directory
grasp_dir = str(base_dir / "datasets" / "grasps")
files = os.listdir(grasp_dir)
files.sort()
for file in files:
    # if ends with .npy
    #TODO: Hardcode for testing
    # object_name = "7130"
    # scale = 0.43
    # joint_state = "0.6981317007977318"
    # tar = f"{object_name}_{scale}_{joint_state}.npy"
    if file.endswith(".npy"): #and tar in file:
        # get the object name
        object_name = file.split("_")[0]
        scale = float(file.split("_")[1])
        joint_state = file.split("_")[2][:-4]
        
        print(
            "Object name: ", object_name, "Scale: ", scale, "Jointstate: ", joint_state
        )
        # if object_name == "7310" and scale == 0.4 and joint_state == "0.6981317007977318":
        if object_name == "7310":
        # if True:
            urdf_path = str(base_dir / "DataGenObj" / object_name / "mobility.urdf")
            grasp_path = os.path.join(grasp_dir, file)
            # load grasps and generate pointcloud
            grasps = np.load(grasp_path)
            if len(grasps) < 500:
                print(
                    f"Number of grasps {object_name}_{scale}_{joint_state}: ", len(grasps)
                )
            object_trimesh = object_to_trimesh(urdf_path, scale, joint_state)
            pc_points, _ = object_to_point_cloud(object_trimesh)
            vis = RerunViewer()
            vis.clear()
            vis.add_pointcloud("pcd", pc_points, radii=0.002)
            vis.add_grasps(f"grasps/_", grasps)
            input("Press enter to continue")