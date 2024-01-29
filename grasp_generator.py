import os
from pathlib import Path
import time
import random
import json
import numpy as np
import gc
from tqdm import tqdm
from glob import glob
import pybullet as p
import pybullet_data
import multiprocessing
from functools import partial
from scipy.spatial.transform import Rotation
from simulation import run_simulation
from loader import load_assets
from sim_env import setup_env, setup_physics_client
from helper import quaternion_from_vectors, calc_forward_vec, change_quaternion_format_to_xyz_w, give_quaternion_roll, create_transformation_matrix

# fix the seed for HPO
random.seed(42)
np.random.seed(42)

def run(data_dir, assets_dir, point, gripper_start_position, gripper_start_orientation, object_number, center_of_object, balanced_cloud, scale, joint_state):
    setup_env(start_position_camera = gripper_start_position)

    gripper, object_id = load_assets(data_dir, assets_dir, gripper_start_position, gripper_start_orientation, object_number, center_of_object, balanced_cloud, scale, joint_state)
    try:
        result = run_simulation(point, gripper_start_position, gripper_start_orientation, gripper, object_id, steps = 6000000, sleep = 1/1000)
        return result
    except:
        return gripper_start_position, "Fail"

def trans_mat_creator(point, quaternion):
    rotation = Rotation.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = point

    return transform_matrix



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
    pointcloud_path = os.path.join(dataset_dir, "pointclouds", f"{object_name}_{scale}_{joint_state}.npz")
    success_poses = []
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

        result = run(data_dir, assets_dir, point, gripper_start_position, gripper_start_orientation, object_name, center_of_object, point, scale, joint_state)
        p.resetSimulation()
        print("RESULTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT: ", result)
        if result == "Success" :#or result == "Partial success"
            selected_pose = trans_mat_creator(point, gripper_start_orientation)
            success_poses.append(selected_pose)
            success_count += 1
            print("Successful grasps: ", success_count, "out of: ", episode+1)
            print()
        gc.collect()
        if success_count == grasp_limit:
            break
    
    success_grasps = np.array(success_poses)
    grasp_path = os.path.join(dataset_dir, "grasps", f"{object_name}_{scale}_{joint_state}.npy")
    np.save(grasp_path, success_grasps)
    print()


def list_all_paths(directory):
    all_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_paths.append(file_path)
    return all_paths


if __name__ == "__main__":
    ## Initialize the PyBullet GUI mode and configure the visualizer
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    multiproc = False
    starting_time = time.time()
    current_directory = Path.cwd()
    dataset_dir = current_directory / 'datasets'
    franka_dir = dataset_dir / 'franka'
    pcl_dir = dataset_dir / 'pointclouds'
    save_directory = str(dataset_dir / 'grasps')
    data_dir = str(current_directory / 'DataGenObj/')
    # Load the link handle info
    link_handle_path = os.path.join(str(current_directory), "configs", "current_link_handle_info.json")
    with open(link_handle_path, "r") as f:
        link_handle_info = json.load(f)
    
    # Run the precomputed_main function

    if multiproc:
        # Create a pool of worker processes
        def process_file(args):
            file, current_directory, dataset_dir, data_dir, link_info, handle_info, object_name = args
            scale = float(file.split("_")[1])
            joint_state = float(file.split("_")[2][:-4])
            precomputed_main(str(franka_dir), str(dataset_dir), data_dir, link_info, handle_info, object_name, 
                            scale, joint_state, 500, rot_sample_mode="nope")
            
        num_processes = multiprocessing.cpu_count()  
        pool = multiprocessing.Pool(processes=num_processes)

        # Create a list of arguments for each file and object
        file_args_list = []
        paths_list = list_all_paths(save_directory)
        for object_name, lh_info in link_handle_info.items():
            link_info = lh_info["link"]
            handle_info = lh_info["handle"]
            # Create a pattern to match .npz files for the current object_name
            pattern = str(pcl_dir / f"{object_name}_*.npz")
            matching_files = glob(pattern)
            for file in matching_files:
                if file not in paths_list: # If the file has not been processed yet
                    file_args_list.append((file, current_directory, dataset_dir, data_dir, link_info, handle_info, object_name))

        # Use partial to create a function with fixed arguments
        partial_process_file = partial(process_file)

        # Map the function to the list of file arguments using multiprocessing
        pool.map(partial_process_file, file_args_list)

        # Close the pool and wait for the worker processes to finish
        pool.close()
        pool.join()
        elapsed_time = time.time() - starting_time
        print("Elapsed time: ", elapsed_time)

    else:
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
                # TODO: Remove the hardcoded scale, joint_state, object_name
                # if scale != 0.43 or joint_state != 0.6981317007977318 or object_name != "7130":
                # if scale != 0.4 or joint_state != 0.6981317007977318 or object_name != "7310":
                if scale != 0.4 or joint_state != 0.0 or object_name != "7310":
                # if object_name != "7119":
                    # 0.6981317007977318
                    continue
                # # If the file does not exist, Run the precomputed_main function
                # if os.path.isfile(os.path.join(save_directory, f'{object_name}_{scale}_{joint_state}.npy')):
                #         continue
                precomputed_main(str(franka_dir), str(dataset_dir), data_dir, link_info, handle_info, object_name, 
                                scale, joint_state, 500, rot_sample_mode="nope")
