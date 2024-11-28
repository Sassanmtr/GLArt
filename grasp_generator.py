import argparse
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
from multiprocessing import Pool
import multiprocessing
from utils.simulation import run_simulation
from utils.loader import load_assets
from utils.sim_env import setup_env, setup_physics_client
from utils.helper import quaternion_from_vectors, calc_forward_vec, change_quaternion_format_to_xyz_w, give_quaternion_roll, trans_mat_creator, valid_grasp_orientations, grasp_orientation_sampler


def run(data_dir, assets_dir, gripper_start_position, gripper_start_orientation, object_number, center_of_object, scale, joint_state, mode="partial"):
    setup_env(start_position_camera = gripper_start_position)
    gripper, object_id = load_assets(data_dir, assets_dir, gripper_start_position, gripper_start_orientation, object_number, center_of_object, scale, joint_state)
    try:
        result = run_simulation(gripper, object_id, steps = 6000000, sleep = 1/1000, joint_state = joint_state, mode=mode)
        return result
    except:
        return "Fail"


def generate_grasps(assets_dir, dataset_dir, data_dir, link_info, handle_info, object_name, 
                     scale, joint_state, grasp_limit, rot_sample_mode="nope", success_mode="partial"):
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
        result= run(data_dir, assets_dir, gripper_start_position, gripper_start_orientation, object_name, center_of_object, scale, joint_state, success_mode)
        p.resetSimulation()

        if result == "Success" or result == "Partial success":
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


def generate_grasps_task(args):
    assets_dir, dataset_dir, data_dir, link_info, handle_info, object_name, scale, joint_state, grasp_limit, rot_sample_mode = args
    generate_grasps(assets_dir, dataset_dir, data_dir, link_info, handle_info, object_name, scale, joint_state, grasp_limit, rot_sample_mode)


def main(args):
    # fix the seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    starting_time = time.time()
    current_directory = Path.cwd()
    dataset_dir = current_directory / 'datasets'
    franka_dir = dataset_dir / 'franka'
    pcl_dir = dataset_dir / 'pointclouds'
    data_dir = str(current_directory / 'urdfs')
    save_directory = str(dataset_dir / 'grasps')
    link_handle_path = os.path.join(str(current_directory), "configs", "objects_info.json")
    with open(link_handle_path, "r") as f:
        link_handle_info = json.load(f)
    if args.multiproc:
        pool = Pool(processes=multiprocessing.cpu_count())  
        tasks = []
        for object_name, lh_info in link_handle_info.items():
            link_info = lh_info["link"]
            handle_info = lh_info["handle"]
            pattern = str(pcl_dir / f"{object_name}_*.npz")
            matching_files = glob(pattern)
            for file in matching_files:
                scale = float(file.split("_")[1])
                joint_state = float(file.split("_")[2][:-4])
                result_file_path = os.path.join(save_directory, f'{object_name}_{scale}_{joint_state}.npy')
                if not os.path.isfile(result_file_path):
                    tasks.append((str(franka_dir), str(dataset_dir), data_dir, link_info, handle_info, object_name, scale, joint_state, 500, "nope", "partial"))   
        pool.map(generate_grasps_task, tasks)
        pool.close()
        pool.join()
       
    else:
        for object_name, lh_info in link_handle_info.items():
            link_info = lh_info["link"]
            handle_info = lh_info["handle"]
            pattern = str(pcl_dir / f"{object_name}_*.npz")
            matching_files = glob(pattern)
            for file in tqdm(matching_files):
                scale = float(file.split("_")[1])
                joint_state = float(file.split("_")[2][:-4])
                if os.path.isfile(os.path.join(save_directory, f'{object_name}_{scale}_{joint_state}.npy')):
                        continue
                generate_grasps(str(franka_dir), str(dataset_dir), data_dir, link_info, handle_info, object_name, 
                                scale, joint_state, 500, rot_sample_mode="nope", success_mode="partial")
    
    elapsed_time = time.time() - starting_time
    print("Elapsed time: ", elapsed_time)


def parse_args():
    parser = argparse.ArgumentParser(description="Run grasping simulation")
    parser.add_argument('--multiproc', dest='multiproc', default=False, action=argparse.BooleanOptionalAction, help='Use multiprocessing. Default is False.')
    # Add more arguments as needed
    return parser.parse_args()


if __name__ == "__main__":
    ## (For debugging) Initialize the PyBullet GUI mode and configure the visualizer. Only in single-threaded mode
    # p.connect(p.GUI)
    # p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    args = parse_args()
    main(args)