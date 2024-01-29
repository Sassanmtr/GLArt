import os
import numpy as np
import json
import urdfpy
import time
from multiprocessing import Pool
import multiprocessing
from functools import partial
from pathlib import Path
from tqdm import tqdm
from file_parser import FileParser
from sampling import get_balanced_pointcloud
from helper import transform_open3d_to_pybullet_pointcloud, transform_open3d_to_pybullet



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
        if object_name == "7236":
            joint_states = np.linspace(joint_lower_limit+0.4, joint_upper_limit, nr_states)
        else:
            joint_states = np.linspace(joint_lower_limit+0.25, joint_upper_limit, nr_states)
    else:
        joint_states = np.linspace(joint_lower_limit, joint_upper_limit, nr_states)
    return joint_states


def precompute_pointcloud(data_dir, save_dir, object_name, scale, joint_state, number_of_points, ratio):
    print("Started")
    balanced_cloud = get_balanced_pointcloud(data_dir, object_name, scale, joint_state, number_of_points, ratio)
    # o3d.visualization.draw_geometries([balanced_cloud], point_show_normal=True)
    print("Finished")
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


if __name__ == "__main__":
    multi_proc = False
    starting_time = time.time()
    current_directory = Path.cwd()
    data_directory = str(current_directory / 'DataGenObj/')
    save_directory = str(current_directory / 'datasets' / 'pointclouds')

    # Load the link handle info
    link_handle_path = os.path.join(str(current_directory), "configs", "current_link_handle_info.json")
    with open(link_handle_path, "r") as f:
        link_handle_info = json.load(f)

    # Number of points to sample from the pointcloud and ratio of non articulated points
    number_of_points=1_000_000
    ratio=[0.0, 0.0]

    # Hard code scales for microwaves (TODO: generalize with generating and parsing a file for scales)
    gt_scales = {"7119": [0.3, 0.4], "101943": [0.8, 0.9], "7130": [0.43, 0.51], "7138": [0.52, 0.6], 
                 "7290": [0.52, 0.56], "10144": [0.75, 0.8], "10373": [0.82, 0.85], 
                 "11211": [0.78, 0.82], "11260": [0.83, 0.86], "12054": [0.83, 0.86], 
                 "12055": [0.83, 0.86], "12249": [0.81, 0.84], "12252": [0.8, 0.83],
                 "11887": [0.45, 0.49], "12071": [0.46, 0.51], "12092": [0.46, 0.5],
                 "12259": [0.48, 0.52], "12480": [0.75, 0.82], "12542": [0.44, 0.48],
                 "12560": [0.45, 0.49], "12565": [0.43, 0.46], "12587": [0.46, 0.49],
                 "12614": [0.45, 0.48], "12617": [0.44, 0.48], "12621": [0.45, 0.48], "12654": [0.65, 0.7],
                 "38516": [0.62, 0.67], "41452": [0.64, 0.68], "45087": [0.62, 0.66], "45162": [0.52, 0.56],
                 "45267": [0.53, 0.57], "45297": [0.63, 0.66], "45448": [0.42, 0.45], "45524": [0.45, 0.48],
                 "45575": [0.43, 0.47], "45600": [0.51, 0.55], "45606": [0.4, 0.43], "45691": [0.45, 0.48],
                 }

    if multi_proc:
        def precompute_pointcloud_worker(args):
            data_directory, save_directory, object_name, scale, joint_state, number_of_points, ratio = args
            print(f"Processing: Object - {object_name}, Scale - {scale}")
            # if file exists, skip
            if os.path.isfile(os.path.join(save_directory, f'{object_name}_{scale}_{joint_state}.npz')):
                return
            else:
                precompute_pointcloud(data_directory, save_directory, object_name, scale, joint_state, number_of_points, ratio)

        args_list = []    
        for object_name in link_handle_info.keys():
            joint_states = joint_states_generator(data_directory, object_name, link_handle_info, 10)
            scales = gt_scales[object_name]
            for scale in scales:
                for joint_state in joint_states:
                    args_list.append((data_directory, save_directory, object_name, scale, joint_state, number_of_points, ratio))

        # num_processes = multiprocessing.cpu_count()  
        # pool = multiprocessing.Pool(processes=int(num_processes/2))
        # partial_process_file = partial(precompute_pointcloud_worker)
        # pool.imap(precompute_pointcloud_worker, args_list)
        # pool.close()
        # pool.join()
        with Pool(processes=4) as pool:
            # list(tqdm(pool.imap(precompute_pointcloud_worker, args_list), total=len(args_list), desc="Processing"))
            list(pool.imap(precompute_pointcloud_worker, args_list))
    else:
        for object_name in link_handle_info.keys():
            joint_states = joint_states_generator(data_directory, object_name, link_handle_info, 10)
            scales = gt_scales[object_name]
            for scale in scales:
                for joint_state in tqdm(joint_states, desc=f"Object: {object_name}, Scale: {scale}"):
                    if os.path.isfile(os.path.join(save_directory, f'{object_name}_{scale}_{joint_state}.npz')):
                        continue
                    else:
                        # if object_name == "45691":
                        precompute_pointcloud(data_directory, save_directory, object_name, scale, joint_state, number_of_points, ratio)
                        elapsed_time = time.time() - starting_time
                        print("Elapsed time: ", elapsed_time)
                        print()