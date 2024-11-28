import os
import numpy as np
import json
import urdfpy
import time
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import argparse
from utils.file_parser import FileParser
from utils.sampling import get_balanced_pointcloud
from utils.helper import transform_open3d_to_pybullet_pointcloud, transform_open3d_to_pybullet



def joint_states_generator(data_dir, object_name, object_handle_info, nr_states):
    """
    Generates joint states for a given object
    """
    # Load the object URDF
    object_urdf_path = os.path.join(data_dir, object_name, 'mobility.urdf')
    object_urdf = urdfpy.URDF.load(object_urdf_path)

    actuated_joint = object_urdf.actuated_joints
    if len(actuated_joint) > 1:
        raise Exception(print(f"More than one actuated joint for {object_name}"))
    else:
        actuated_joint = actuated_joint[0]

    joint_type = actuated_joint.joint_type
    if joint_type != 'revolute':
        raise Exception(print(f"Joint type {joint_type} not supported"))
    joint_upper_limit = actuated_joint.limit.upper
    joint_lower_limit = actuated_joint.limit.lower
    
    if object_handle_info[object_name]["handle"] == "none":
        joint_states = np.linspace(joint_lower_limit + 0.25, joint_upper_limit, nr_states)
    else:
        joint_states = np.linspace(joint_lower_limit, joint_upper_limit, nr_states)
    return joint_states


def precompute_pointcloud(data_dir, save_dir, object_name, scale, joint_state, number_of_points):
    balanced_cloud = get_balanced_pointcloud(data_dir, object_name, scale, joint_state, number_of_points)
    obj = FileParser(data_dir, object_name)
    center_of_object = obj.get_center_of_object()
    center_of_object = transform_open3d_to_pybullet(center_of_object)
    balanced_points = transform_open3d_to_pybullet_pointcloud(balanced_cloud.points)
    balanced_normals = transform_open3d_to_pybullet_pointcloud(balanced_cloud.normals)
    points = list(range(len(balanced_points)))
    data_dict = {
        'balanced_points': balanced_points,
        'balanced_normals': balanced_normals,
        'points': points,
        'center_of_object': center_of_object
    }
    np.savez(os.path.join(save_dir, f'{object_name}_{scale}_{joint_state}.npz'), **data_dict)




def main(args):
    start_time = time.time()
    current_directory = Path.cwd()
    data_directory = str(current_directory / 'datasets' / 'urdfs/')
    save_directory = str(current_directory / 'datasets' / 'pointclouds')
    
    # Load the link handle info
    objects_info_path = os.path.join(str(current_directory), "configs", "objects_info.json")
    with open(objects_info_path, "r") as f:
        objects_info = json.load(f)
    
    number_of_points = args.number_of_points

    if args.multi_proc:
        def precompute_pointcloud_worker(arguments):
            data_directory, save_directory, object_name, scale, joint_state, number_of_points = arguments
            print(f"Processing: Object - {object_name}, Scale - {scale}")
            # if file exists, skip
            if os.path.isfile(os.path.join(save_directory, f'{object_name}_{scale}_{joint_state}.npz')):
                return
            else:
                precompute_pointcloud(data_directory, save_directory, object_name, scale, joint_state, number_of_points)

        args_list = []
        for object_name in objects_info.keys():
            joint_states = joint_states_generator(data_directory, object_name, objects_info, args.nr_states)
            scales = objects_info[object_name]['scales']
            for scale in scales:
                for joint_state in joint_states:
                    args_list.append((data_directory, save_directory, object_name, scale, joint_state, number_of_points))

        with Pool(processes=args.nr_processes) as pool:
            list(pool.imap(precompute_pointcloud_worker, args_list))
    else:
        for object_name in objects_info.keys():
           joint_states = joint_states_generator(data_directory, object_name, objects_info, args.nr_states)
           scales = objects_info[object_name]['scales']
           for scale in scales:
               for joint_state in tqdm(joint_states, desc=f"Object: {object_name}, Scale: {scale}"):
                   if os.path.isfile(os.path.join(save_directory, f'{object_name}_{scale}_{joint_state}.npz')):
                       continue
                   else:
                       precompute_pointcloud(data_directory, save_directory, object_name, scale, joint_state, number_of_points)
                       elapsed_time = time.time() - start_time
                       print("Elapsed time: ", elapsed_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute point clouds for objects.")
    parser.add_argument("--multi_proc", action="store_true", help="Use multiprocessing")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of processes for multiprocessing")
    parser.add_argument("--nr_states", type=int, default=10, help="Number of joint states to generate")
    parser.add_argument("--number_of_points", type=int, default=1_000_000, help="Number of points to sample")
    args = parser.parse_args()

    main(args)