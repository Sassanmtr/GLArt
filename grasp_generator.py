import os
from pathlib import Path
import time
import json
import numpy as np
import gc
from tqdm import tqdm
from glob import glob
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation
from simulation import run_simulation
from loader import load_assets
from sim_env import setup_env, setup_physics_client
from helper import quaternion_from_vectors, calc_forward_vec, change_quaternion_format_to_xyz_w, give_quaternion_roll, create_transformation_matrix


def run(data_dir, assets_dir, gripper_start_position, gripper_start_orientation, object_number, center_of_object, balanced_cloud, scale, joint_state):
    setup_env(start_position_camera = gripper_start_position)

    gripper, object_id = load_assets(data_dir, assets_dir, gripper_start_position, gripper_start_orientation, object_number, center_of_object, balanced_cloud, scale, joint_state)
     
    result = run_simulation(gripper, object_id, steps = 6000000, sleep = 1/1000)

    return result

def limit_rotation_quat_link(initial_orientation_euler):
    quaternion = []
    rotation_matrix = Rotation.from_euler('xyz', initial_orientation_euler).as_matrix()
    additional_angles0 = np.array([0, 0, 0])
    rotation_matrix0 = np.dot(rotation_matrix, Rotation.from_euler('xyz', additional_angles0).as_matrix())
    quaternion.append(Rotation.from_matrix(rotation_matrix0).as_quat())
    additional_angles1 = np.array([0, -np.pi/2, 0])
    rotation_matrix1 = np.dot(rotation_matrix, Rotation.from_euler('xyz', additional_angles1).as_matrix())
    quaternion.append(Rotation.from_matrix(rotation_matrix1).as_quat())
    additional_angles2 = np.array([0, np.pi/2, 0])
    rotation_matrix2 = np.dot(rotation_matrix, Rotation.from_euler('xyz', additional_angles2).as_matrix())
    quaternion.append(Rotation.from_matrix(rotation_matrix2).as_quat())
    return quaternion

def rotation_quat_handle(initial_orientation_euler, handle_mode):
    rotation_matrix = Rotation.from_euler('xyz', initial_orientation_euler).as_matrix()
    if handle_mode == "vertical":
        additional_angles = np.array([0, 0, 0])
    elif handle_mode == "horizontal":
        additional_angles = np.array([0, 0, np.pi/2])
    else:
        raise Exception("Invalid handle mode")
    rotation_matrix = np.dot(rotation_matrix, Rotation.from_euler('xyz', additional_angles).as_matrix())
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
    return quaternion

def valid_grasp_orientations(link_data, handle_data, joint_state):
    """
    Returns a list of valid grasp orientations for the object
    """
    valid_orientations = []
    if link_data == "right":
        initial_orientation_euler = [0, np.pi/2, np.pi/2-float(joint_state)]
        valid_orientations = limit_rotation_quat_link(initial_orientation_euler)
        if handle_data != "none":
            initial_orientation_euler_handle = [float(joint_state), np.pi/2, 0]
            orientation_handle_quat = rotation_quat_handle(initial_orientation_euler_handle, handle_data)
            if joint_state == 0: # If joint_state is closed, then the only valid grasp is the handle
                valid_orientations = []
                valid_orientations.append(orientation_handle_quat)
            else:
                valid_orientations.append(orientation_handle_quat)

    elif link_data == "left":
        if joint_state == 0:
            valid_orientations = valid_orientations.append([-float(joint_state), np.pi/2, 0])
        else:
            initial_orientation_euler = [0, -np.pi/2, np.pi/2+float(joint_state)]
            valid_orientations = limit_rotation_quat_link(initial_orientation_euler)
            if handle_data != "none":
                initial_orientation_euler_handle = [-float(joint_state), np.pi/2, 0]
                orientation_handle_quat = rotation_quat_handle(initial_orientation_euler_handle, handle_data)
                if joint_state == 0: # If joint_state is closed, then the only valid grasp is the handle
                    valid_orientations = []
                    valid_orientations.append(orientation_handle_quat)
                else:
                    valid_orientations.append(orientation_handle_quat)
    elif link_data == "top":
        if joint_state == 0:
            valid_orientations = valid_orientations.append([float(joint_state), np.pi, np.pi/2])
        else:
            initial_orientation_euler_link = [float(joint_state), np.pi, np.pi/2]
            valid_orientations = limit_rotation_quat_link(initial_orientation_euler_link)
            if handle_data != "none":
                initial_orientation_euler_handle = [0, np.pi/2 - float(joint_state), 0]
                orientation_handle_quat = rotation_quat_handle(initial_orientation_euler_handle, handle_data)
                if joint_state == 0: # If joint_state is closed, then the only valid grasp is the handle
                    valid_orientations = []
                    valid_orientations.append(orientation_handle_quat)
                else:
                    valid_orientations.append(orientation_handle_quat)
    
    elif link_data == "bottom":
        initial_orientation_euler_link = [float(joint_state), 0, np.pi]
        valid_orientations = limit_rotation_quat_link(initial_orientation_euler_link)
        if handle_data != "none":
            initial_orientation_euler_handle = [0, np.pi/2 + float(joint_state), 0]
            orientation_handle_quat = rotation_quat_handle(initial_orientation_euler_handle, handle_data)
            if joint_state == 0: # If joint_state is closed, then the only valid grasp is the handle
                valid_orientations = []
                valid_orientations.append(orientation_handle_quat)
            else:
                valid_orientations.append(orientation_handle_quat)
    else:
        raise Exception("Invalid link data")
    
    return valid_orientations

def grasp_orientation_sampler(valid_orientations):
    """
    Returns a random grasp orientation from the list of valid orientations. 
    If len of valid_orientations is 3, then unoformly return an element. 
    If the len is 4, return the last element with 0.4 probability, and each of the other elements with 0.2 probability.
    """
    if len(valid_orientations) == 1:
        return valid_orientations[0]
    elif len(valid_orientations) == 3:
        probabilities = [1/3, 1/3, 1/3]
        index = np.random.choice([0, 1, 2], p=probabilities)
        return valid_orientations[index]
    elif len(valid_orientations) == 4:
        probabilities = [0.2, 0.2, 0.2, 0.4]
        index = np.random.choice([0, 1, 2, 3], p=probabilities)
        return valid_orientations[index]
    else:
        raise Exception("Invalid length of valid orientations")

def precomputed_main(assets_dir, dataset_dir, data_dir, link_info, handle_info, object_name, 
                     scale, joint_state, grasp_limit, rot_sample_mode="nope"):
    setup_physics_client()
    # joint_state = 1.5707963267948966 hardcoded for debugging
    pointcloud_path = os.path.join(dataset_dir, "pointclouds", f"{object_name}_{scale}_{joint_state}.npz")
    success_poses = []
    result_dict = {}
    # Load the data from the .npz file
    loaded_data = np.load(pointcloud_path, allow_pickle=True)
    # Access individual arrays by their keys
    balanced_points = loaded_data['balanced_points']
    balanced_normals = loaded_data['balanced_normals']
    points = list(loaded_data['points'])
    center_of_object = loaded_data['center_of_object']
    success_count = 0
    for episode in range(len(balanced_points)):
        angle = joint_state
        print("episode: ", episode)
        #choose a random point
        point_index = np.random.choice(points)
        points.remove(point_index)
        #retrieve gripper trajectory and point
        point = balanced_points[point_index]
        if rot_sample_mode == "rand":
            normal = balanced_normals[point_index]
            # normal = (-1.0, 0.0, 0.0)
            gripper_start_orientation = give_quaternion_roll(change_quaternion_format_to_xyz_w(quaternion_from_vectors([0,0,-1], normal)), angle)
        else:
            valid_orientations = valid_grasp_orientations(link_info, handle_info, joint_state)
            gripper_start_orientation = grasp_orientation_sampler(valid_orientations)
        gripper_start_position = calc_forward_vec(point, gripper_start_orientation, -0.12)

        result = run(data_dir, assets_dir, gripper_start_position, gripper_start_orientation, object_name, center_of_object, point, scale, joint_state)
        p.resetSimulation()
        print("result: ", result)
        if result == "Success" :#or result == "Partial success"
            result_dict[int(point_index)] = [gripper_start_position, gripper_start_orientation]
            # points_success.append(point_index)
            selected_pose = create_transformation_matrix(point, gripper_start_orientation)
            success_poses.append(selected_pose)
            success_count += 1
            print("Number of successful grasps: ", success_count)
        gc.collect()
        if success_count == grasp_limit:
            break   
    success_grasps = np.array(success_poses)
    grasp_path = os.path.join(dataset_dir, "grasps", f"{object_name}_{scale}_{joint_state}.npy")
    np.save(grasp_path, success_grasps)





if __name__ == "__main__":
    # Initialize the PyBullet GUI mode and configure the visualizer
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    current_directory = Path.cwd()
    dataset_dir = current_directory / 'datasets'
    pcl_dir = dataset_dir / 'pointclouds'
    save_directory = str(dataset_dir / 'grasps')
    data_dir = str(current_directory / 'DataGenObj/')
    # Load the link handle info
    link_handle_path = os.path.join(str(current_directory), "configs", "current_link_handle_info.json")
    with open(link_handle_path, "r") as f:
        link_handle_info = json.load(f)
    for object_name, lh_info in link_handle_info.items():
        link_info = lh_info["link"]
        handle_info = lh_info["handle"]
        # Create a pattern to match .npz files for the current object_name
        pattern = str(pcl_dir / f"{object_name}_*.npz")
        matching_files = glob(pattern)
        for file in tqdm(matching_files):
            # Extract the scale and joint state from the file name
            scale = float(file.split("_")[1])
            joint_state = float(file.split("_")[2][:-4])
            # Run the precomputed_main function
            precomputed_main(str(current_directory), str(dataset_dir), data_dir, link_info, handle_info, object_name, 
                             scale, joint_state, 100, rot_sample_mode="nope")
        print()

    