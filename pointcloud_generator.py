import os
import numpy as np
import json
import urdfpy
from episode_handler import precompute_pointcloud
from pathlib import Path
from tqdm import tqdm

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

    for object_name in link_handle_info.keys():
        joint_states = joint_states_generator(data_directory, object_name, link_handle_info, 10)
        for scale in scales:
            for joint_state in tqdm(joint_states, desc=f"Object: {object_name}, Scale: {scale}"):
                precompute_pointcloud(data_directory, save_directory, object_name, scale, joint_state, number_of_points, ratio)

