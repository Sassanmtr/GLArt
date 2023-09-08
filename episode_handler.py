from simulation import run_simulation
from loader import load_assets
from sim_env import setup_env, setup_physics_client
import time
from sampling import Sampling, get_balenced_pointcloud
import random
import numpy as np
from helper import quaternion_from_vectors, calc_forward_vec, transform_open3d_to_pybullet_pointcloud, transform_open3d_to_pybullet, vector_to_euler, change_quaternion_format_to_xyz_w, give_quaternion_roll
from file_parser import FileParser
from scipy.spatial.transform import Rotation
import open3d as o3d
import json
import gc
import cProfile
import pybullet as p

def run(gripper_start_position, gripper_start_orientation, object_number, center_of_object, balenced_cloud):
    setup_env(start_position_camera = gripper_start_position)

    gripper, object_id = load_assets(gripper_start_position, gripper_start_orientation, object_number, center_of_object, balenced_cloud )
     
    result = run_simulation(gripper, object_id, steps = 6000000, sleep = 1/1000)

    return result

def main(object_number, number_of_episodes, number_of_points):
    setup_physics_client()
    points_success = []
    """point_cloud = Sampling(object_number, number_of_points=60000)"""
    
    balenced_cloud = get_balenced_pointcloud(object_number, number_of_points, [0.0,0.0])
    #o3d.visualization.draw_geometries([balenced_cloud], point_show_normal=True)
    obj = FileParser(object_number)
    center_of_object = obj.get_center_of_object()
    center_of_object = transform_open3d_to_pybullet(center_of_object)

    print("COM", center_of_object)
    balenced_points = transform_open3d_to_pybullet_pointcloud(balenced_cloud.points)
    balenced_normals = transform_open3d_to_pybullet_pointcloud(balenced_cloud.normals)

    result_dict = {}
    points = list(range(len(balenced_points)))
    if len(points) < number_of_episodes: #check that there are enough graspable points for the amount of episodes chosen
        dummy = o3d.geometry.PointCloud()
        for _ in range(666):
            dummy.points.append([0,0,0])
        return dummy
        raise ValueError("Only " + str(len(points)) + " points available, but " + str(number_of_episodes) + " are needed. Increase the point cloud size or decrease the number of episodes.")
    #run the actual episode
    for episode in range(number_of_episodes):
        start_time = time.time()
        #random roll for the gripper
        angle = random.uniform(-np.pi, np.pi)
        print("episode: ", episode)
        #choose a random point
        point_index = np.random.choice(points)
        points.remove(point_index)
        #retrieve gripper trajectory and point
        point = balenced_points[point_index]
        print("normal: ", balenced_normals[point_index])
        normal = balenced_points[point_index]
        
        gripper_start_orientation = give_quaternion_roll(change_quaternion_format_to_xyz_w(quaternion_from_vectors([0,0,-1], normal)), angle)#TODO dont use random angle
        #the actual gripper starting position is by determined by taking the point in the object and then calculating the backwords trajectory
        gripper_start_position = calc_forward_vec(point, gripper_start_orientation, -0.12)

        """if episode > 0:
            cProfile("run(gripper_start_position, gripper_start_orientation, object_number, center_of_object, point)", sort="cumtime").print_stats"""
        result = run(gripper_start_position, gripper_start_orientation, object_number, center_of_object, point)
        p.resetSimulation()
        print("result: ", result)
        print("gripper_start_position: ", gripper_start_position)
        print("gripper_start_orientation: ", gripper_start_orientation)
        if result == "Success" :#or result == "Partial success"
            
            result_dict[int(point_index)] = [gripper_start_position, gripper_start_orientation]
            points_success.append(point_index)
        #o3d.visualization.draw_geometries([balenced_cloud, point_cloud.mesh], point_show_normal=True)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"The code took {execution_time} seconds to complete.")
        gc.collect()
    # Convert NumPy arrays to lists recursively
    result_dict_serializable = convert_to_serializable(result_dict)

    # Specify the file path where you want to save the dictionary
    file_path = "result_dict.json"

    # Write the dictionary to the file
    with open(file_path, "w") as json_file:
        json.dump(result_dict_serializable, json_file)

    print("Dictionary saved to", file_path)
    print(result_dict)
    pc_new = o3d.geometry.PointCloud()
    for index in points_success:
        point = balenced_cloud.points[index]
        normal = balenced_cloud.normals[index]
        pc_new.points.append(point)
        pc_new.normals.append(normal)
    #o3d.visualization.draw_geometries([pc_new], point_show_normal=True)
    return pc_new
    


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
    numbers  = [10068, 10144, 10373, 10489, 10586, 10612, 10751, 10655, 10627, 10638, 10849, 10867, 10900, 10944, 11178, 12055,12059,10905, 11211, 11231, 11299, 11304, 11550, 11709, 12043, 12042, 11846, 12036, 12038, 12066, 12055, 12059, 12054, 12248, 12249, 12250, 12252]
    out = []
    start_time = time.time()
    for number in numbers:
        cloud = main(object_number = number, number_of_episodes=100, number_of_points=100000)
        out.append(f"{number} has {len(cloud.points)} points")


        end_time = time.time()

        print(out)
    execution_time = end_time - start_time
    print(f"The code took {execution_time} seconds to complete.")