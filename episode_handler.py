from simulation import run_simulation
from loader import load_assets
from sim_env import setup_env, setup_physics_client
import time
import os
from sampling import Sampling, get_balanced_pointcloud
import random
import numpy as np
import yaml
import itertools
import pickle
from helper import quaternion_from_vectors, calc_forward_vec, transform_open3d_to_pybullet_pointcloud, transform_open3d_to_pybullet, change_quaternion_format_to_xyz_w, give_quaternion_roll, create_transformation_matrix
from file_parser import FileParser
from scipy.spatial.transform import Rotation
import open3d as o3d
import json
import gc
import cProfile
import pybullet as p
import pybullet_data

# fix the seed for HPO
random.seed(42)
np.random.seed(42)


def run(param_config, data_dir, assets_dir, gripper_start_position, gripper_start_orientation, object_number, center_of_object, balanced_cloud, scale, joint_state):
    setup_env(start_position_camera = gripper_start_position)

    gripper, object_id = load_assets(param_config, data_dir, assets_dir, gripper_start_position, gripper_start_orientation, object_number, center_of_object, balanced_cloud, scale, joint_state)
     
    result = run_simulation(gripper, object_id, steps = 6000000, sleep = 1/1000)

    return result

def main(config, data_dir, assets_dir, object_number, number_of_episodes, number_of_points, scale, joint_state):
    setup_physics_client()
    points_success = []
    """point_cloud = Sampling(data_dir, object_number, number_of_points=60000)"""
    
    balanced_cloud = get_balanced_pointcloud(data_dir, object_number, scale, joint_state, number_of_points, [0, 0])
    # o3d.visualization.draw_geometries([balanced_cloud], point_show_normal=True)
    obj = FileParser(data_dir, object_number)
    center_of_object = obj.get_center_of_object()
    center_of_object = transform_open3d_to_pybullet(center_of_object)

    print("COM", center_of_object)
    balanced_points = transform_open3d_to_pybullet_pointcloud(balanced_cloud.points)
    balanced_normals = transform_open3d_to_pybullet_pointcloud(balanced_cloud.normals)
    result_dict = {}
    points = list(range(len(balanced_points)))
    if len(points) < number_of_episodes: #check that there are enough graspable points for the amount of episodes chosen

        raise ValueError("Only " + str(len(points)) + " points available, but " + str(number_of_episodes) + " are needed. Increase the point cloud size or decrease the number of episodes.")


    
    # Extract parameter ranges from the config
    param_ranges = {
        'GripperSpinningFriction': config['GripperSpinningFriction'],
        'ObjectSpinningFriction': config['ObjectSpinningFriction'],
        'GripperLateralFriction': config['GripperLateralFriction'],
        'ObjectLateralFriction': config['ObjectLateralFriction'],
        'ObjectMass': config['ObjectMass'],
        'GripperMass': config['GripperMass'],
    }
    
    # Define the predefined values for ObjectCollisionMargin and GripperCollisionMargin
    num_samples = 50  # Number of random samples you want to test

    random_configurations = []

    for _ in range(num_samples):
        config_dict = {}
        for param_name, param_range in param_ranges.items():
            if param_name in ['ObjectMass']:
                # Randomly sample from the predefined object mass values
                sampled_value = random.uniform(0.5, 0.5)
            else:
                # Assuming a range [min, max] for integer values
                sampled_value = random.randint(param_range[0], param_range[1])
            config_dict[param_name] = sampled_value

        random_configurations.append(config_dict)

    result_dict = {}
    final_dict = {}
    for config_dict in random_configurations:
        print("Testing configuration dictionary:", config_dict)
        #run the actual episode
        new_points = points.copy()
        for episode in range(number_of_episodes):
            #random roll for the gripper
            # angle = random.uniform(-np.pi, np.pi)
            angle = 0.0
            print("episode: ", episode)
            #choose a random point
            point_index = np.random.choice(new_points)
            new_points.remove(point_index)
            #retrieve gripper trajectory and point
            point = balanced_points[point_index]
            normal = balanced_normals[point_index]
            normal = (-1.0, 0.0, 0.0)
            gripper_start_orientation = give_quaternion_roll(change_quaternion_format_to_xyz_w(quaternion_from_vectors([0,0,-1], normal)), angle)#TODO dont use random angle
            #the actual gripper starting position is by determined by taking the point in the object and then calculating the backwords trajectory
            gripper_start_position = calc_forward_vec(point, gripper_start_orientation, -0.12)

            """if episode > 0:
                cProfile("run(gripper_start_position, gripper_start_orientation, object_number, center_of_object, point)", sort="cumtime").print_stats"""
            result = run(config_dict, data_dir, assets_dir, gripper_start_position, gripper_start_orientation, object_number, center_of_object, point, scale)
            p.resetSimulation()
            if result == "Success" :#or result == "Partial success"
                
                result_dict[int(point_index)] = [gripper_start_position, gripper_start_orientation]
                points_success.append(point_index)
            #o3d.visualization.draw_geometries([balanced_cloud, point_cloud.mesh], point_show_normal=True)
            gc.collect()
        success_count = len(points_success)
        final_dict[config_dict] = success_count
    with open("my_dict.pkl", "wb") as pickle_file:
        pickle.dump(final_dict, pickle_file)
    # # Convert NumPy arrays to lists recursively
    # result_dict_serializable = convert_to_serializable(result_dict)

    # # Specify the file path where you want to save the dictionary
    # file_path = "result_dict.json"

    # # Write the dictionary to the file
    # with open(file_path, "w") as json_file:
    #     json.dump(result_dict_serializable, json_file)

    # print("Dictionary saved to", file_path)
    # print(result_dict)
    # pc_new = o3d.geometry.PointCloud()
    # for index in points_success:
    #     point = balanced_cloud.points[index]
    #     normal = balanced_cloud.normals[index]
    #     pc_new.points.append(point)
    #     pc_new.normals.append(normal)
    # #o3d.visualization.draw_geometries([pc_new], point_show_normal=True)
    # return pc_new
    
def precompute_pointcloud(data_dir, save_dir, object_uuid, object_config, number_of_points, ratio):
    setup_physics_client()
    scale = object_config['scale']
    joint_state = object_config['joint_state']
    object_name = object_config['object_name']
    save_dir = os.path.join(save_dir, "test_pointclouds")
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
    np.savez(os.path.join(save_dir, f'{object_uuid}_plc.npz'), **data_dict)



def precomputed_main(config_dict, assets_dir, object_uuid, object_config):
    setup_physics_client()
    scale = object_config['scale']
    joint_state = object_config['joint_state']
    object_name = object_config['object_name']
    pointcloud_path = os.path.join(assets_dir, "test_pointclouds") #TODO change to pointclouds
    pointcloud_path = os.path.join(pointcloud_path, f'{object_uuid}_plc.npz')
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
        #random roll for the gripper
        # angle = random.uniform(-np.pi, np.pi)
        angle = joint_state
        print("episode: ", episode)
        #choose a random point
        point_index = np.random.choice(points)
        points.remove(point_index)
        #retrieve gripper trajectory and point
        point = balanced_points[point_index]
        normal = balanced_normals[point_index]
        normal = (-1.0, 0.0, 0.0)
        gripper_start_orientation = give_quaternion_roll(change_quaternion_format_to_xyz_w(quaternion_from_vectors([0,0,-1], normal)), angle)
        # gripper_start_orientation = change_quaternion_format_to_xyz_w(quaternion_from_vectors([0,0,-1], normal))
        #the actual gripper starting position is by determined by taking the point in the object and then calculating the backwords trajectory
        # gripper_start_orientation = give_quaternion_roll([0.707,-0.707,0,0], angle)
        gripper_start_position = calc_forward_vec(point, gripper_start_orientation, -0.12)

        """if episode > 0:
            cProfile("run(gripper_start_position, gripper_start_orientation, object_number, center_of_object, point)", sort="cumtime").print_stats"""
        result = run(config_dict, data_dir, assets_dir, gripper_start_position, gripper_start_orientation, object_name, center_of_object, point, scale, joint_state)
        p.resetSimulation()
        print("result: ", result)
        if result == "Success" :#or result == "Partial success"
            result_dict[int(point_index)] = [gripper_start_position, gripper_start_orientation]
            # points_success.append(point_index)
            selected_pose = create_transformation_matrix(point, gripper_start_orientation)
            success_poses.append(selected_pose)
            success_count += 1
            print("Number of successful grasps: ", success_count)
        #o3d.visualization.draw_geometries([balanced_cloud, point_cloud.mesh], point_show_normal=True)
        gc.collect()
        if success_count == 100:
            break   
    success_grasps = np.array(success_poses)
    grasp_dir = os.path.join(assets_dir, "grasps")
    file_path = os.path.join(grasp_dir, f'{object_uuid}_grasps.npy')
    np.save(file_path, success_grasps)
    


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj

if __name__ == "__main__":
    # # Initialize the PyBullet GUI mode and configure the visualizer
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    # # End of initialization
    numbers  = [7221]
    # data_dir = "/home/mokhtars/Documents/Thesis/datasets/others/CatData/Microwave/"
    data_dir = "/home/mokhtars/Documents/articulatedobjectsgraspsampling/"
    assets_dir = "/home/mokhtars/Documents/articulatedobjectsgraspsampling/"
    out = []
    config_path = "/home/mokhtars/Documents/articulatedobjectsgraspsampling/test_config.yaml"
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # pointcloud_path = precompute_pointcloud(data_dir=data_dir, save_dir=assets_dir, joint_state=0.5, scale=0.3, object_name=7310, number_of_points=100000, ratio=[0.0,0.0])
    # pointcloud_path = '/home/mokhtars/Documents/articulatedobjectsgraspsampling/7310_0.3_0.5.npz'
    # precomputed_main(config_data, pointcloud_path, object_name=7310, scale=0.3, joint_state=0.5, number_of_episodes=1000)
    # for number in numbers:
    #     cloud = main(config=config_data, data_dir=data_dir, assets_dir=assets_dir, object_number = number, number_of_episodes=1000, number_of_points=100000, scale=0.3, joint_state=0.4)    
    #     out.append(f"{number} has {len(cloud.points)} points")
    # Load the JSON file
    json_file_path = "/home/mokhtars/Documents/Thesis/object_configurations_debug1.json"
    # json_file_path = "/home/mokhtars/Documents/Thesis/test1.json"
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    # # Loop over the objects and save the point clouds
    # for object_id, object_data in data.items():
    #     precompute_pointcloud(data_dir=data_dir, save_dir=assets_dir, object_uuid=object_id, object_config=object_data, number_of_points=1000000, ratio=[0.0,0.0])
    
    # Loop over the objects and generate the grasps

    for object_id, object_data in data.items():
        precomputed_main(config_data, assets_dir, object_uuid=object_id, object_config=object_data)
    p.disconnect()