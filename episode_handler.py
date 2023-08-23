from simulation import run_simulation
from loader import load_assets
from sim_env import setup_env
import time
from sampling import Sampling
import random
import numpy as np
from helper import quaternion_from_vectors, calc_forward_vec, transform_open3d_to_pybullet_pointcloud, transform_open3d_to_pybullet, vector_to_euler, change_quaternion_format_to_xyz_w
from file_parser import FileParser
from scipy.spatial.transform import Rotation
import open3d as o3d

def run(gripper_start_position, gripper_start_orientation, object_number, center_of_object, balenced_cloud):
    setup_env(start_position_camera = gripper_start_position)

    load_assets(gripper_start_position, gripper_start_orientation, object_number, center_of_object, balenced_cloud )
     
    result = run_simulation(0, 1, steps = 600000, sleep = 1/1000000)

    return result

def main(object_number, number_of_episodes):

    result_dict = {}
    point_cloud = Sampling(object_number, number_of_points=2000)
    balenced_cloud = point_cloud.create_balenced_cloud([0, 0])
    
    obj = FileParser(object_number)
    center_of_object = obj.get_center_of_object()
    center_of_object = transform_open3d_to_pybullet(center_of_object)

    print("COM", center_of_object)
    balenced_points = transform_open3d_to_pybullet_pointcloud(balenced_cloud.points)
    balenced_normals = transform_open3d_to_pybullet_pointcloud(balenced_cloud.normals)


    length_points = len(balenced_points)
    points_used = []
    for episode in range(number_of_episodes):
        
        point_index = np.random.choice(length_points)
        while point_index in points_used:
            point_index = np.random.choice(length_points)
        points_used.append(point_index)
        point = balenced_points[point_index]
        print("normal: ", balenced_normals[point_index])
        normal = balenced_points[point_index]
        
        gripper_start_orientation = [change_quaternion_format_to_xyz_w(quaternion_from_vectors([0,0,-1], normal)) for normal in balenced_normals]#TODO dont use random angle
        point = [[balenced_points[idx], gripper_start_orientation[idx]]for idx, _ in enumerate(balenced_points)] 
        gripper_start_position = point[0]#calc_forward_vec(point[0], gripper_start_orientation, -0.12)
        result = run(gripper_start_position, gripper_start_orientation, object_number, center_of_object, point)
        result_dict[episode] = [object_number, result, gripper_start_position, gripper_start_orientation]
        o3d.visualization.draw_geometries([balenced_cloud, point_cloud.mesh], point_show_normal=True)
        print("normal: ", balenced_normals[point_index])
        print("euler: ", vector_to_euler(balenced_normals[point_index]))
        print("Hi")
    print(result_dict)

if __name__ == "__main__":
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    start_time = time.time()
    main(object_number = 7263, number_of_episodes=10)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"The code took {execution_time} seconds to complete.")