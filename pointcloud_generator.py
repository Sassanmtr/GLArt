import os
import numpy as np
import json
import urdfpy
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
        joint_states = np.linspace(joint_lower_limit+0.25, joint_upper_limit, nr_states)
    else:
        joint_states = np.linspace(joint_lower_limit, joint_upper_limit, nr_states)
    return joint_states


def precompute_pointcloud(data_dir, save_dir, object_name, scale, joint_state, number_of_points, ratio):
    # setup_physics_client()
    # scale = object_config['scale']
    # joint_state = object_config['joint_state']
    # object_name = object_config['object_name']
    # save_dir = os.path.join(save_dir, "pointclouds")
    balanced_cloud = get_balanced_pointcloud(data_dir, object_name, scale, joint_state, number_of_points, ratio)
    # o3d.visualization.draw_geometries([balanced_cloud], point_show_normal=True)
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


def process_object(object_args):
    data_directory, save_directory, object_name, scale, joint_state, number_of_points, ratio = object_args
    precompute_pointcloud(data_directory, save_directory, object_name, scale, joint_state, number_of_points, ratio)



if __name__ == "__main__":
    current_directory = Path.cwd()
    # parent_directory = current_directory.parent
    data_directory = str(current_directory / 'DataGenObj/')
    save_directory = str(current_directory / 'datasets' / 'pointclouds')

    # Load the link handle info
    link_handle_path = os.path.join(str(current_directory), "configs", "current_link_handle_info.json")
    with open(link_handle_path, "r") as f:
        link_handle_info = json.load(f)

    # Number of points to sample from the pointcloud and ratio of non articulated points
    number_of_points=1_000_000
    ratio=[0.0,0.0]

    # Hard code scales for microwaves (TODO: generalize with generating and parsing a file for scales)
    scales = [0.3, 0.4]

    # num_processes = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(processes=num_processes)

    # jobs = []

    # for object_name in link_handle_info.keys():
    #     joint_states = joint_states_generator(data_directory, object_name, link_handle_info, 10)
    #     for scale in scales:
    #         for joint_state in joint_states:
    #             args = (data_directory, save_directory, object_name, scale, joint_state, number_of_points, ratio)
    #             jobs.append(args)

    # # Use partial to create a function with fixed arguments
    # partial_process_file = partial(process_object)
    # # Map the function to the list of file arguments using multiprocessing
    # pool.map(partial_process_file, jobs)
    # pool.close()
    # pool.join()





    for object_name in link_handle_info.keys():
        joint_states = joint_states_generator(data_directory, object_name, link_handle_info, 10)
        for scale in scales:
            for joint_state in tqdm(joint_states, desc=f"Object: {object_name}, Scale: {scale}"):
                precompute_pointcloud(data_directory, save_directory, object_name, scale, joint_state, number_of_points, ratio)

