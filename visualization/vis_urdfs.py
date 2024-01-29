import sys
sys.path.append("/home/mokhtars/Documents/GraspSampler")
import urdfpy
import numpy as np
from file_parser import FileParser
import pyrender

def load_urdf_meshes(obj, urdf_file_path, scale, joint_state, only_articulated=False):
    if only_articulated == True:
        mesh_dict, origin = obj.get_articulated_meshes()
    if only_articulated == False:
        mesh_dict, origin = obj.get_all_meshes()
    # Load the URDF robot model
    arti_obj = urdfpy.URDF.load(urdf_file_path)
    # Open the articulated joints to the desired joint state
    articulated_link = [joint.child for joint in arti_obj.actuated_joints]
    joint_name = arti_obj.actuated_joints[0].name
    # fk = arti_obj.link_fk({joint_name: joint_state})
    transformed_meshes = []
    actuated_meshes = arti_obj.visual_trimesh_fk(links=articulated_link).keys()
    # new added code
    corres_meshes = []
    for key, value in mesh_dict.items():
        for filename in value:
            corres_meshes.append(filename)

    # Visualization
    scene = pyrender.Scene()
    # Define the length of the axes
    axis_length = 1

    # Define the colors for the axes (R, G, B)
    x_color = [1.0, 0.0, 0.0]  # Red for X-axis
    y_color = [0.0, 1.0, 0.0]  # Green for Y-axis
    z_color = [0.0, 0.0, 1.0]  # Blue for Z-axis

    num_points = 100
    # Create points for lines along each axis
    x_points = np.array([[i * axis_length / num_points, 0, 0] for i in range(num_points)])
    y_points = np.array([[0, i * axis_length / num_points, 0] for i in range(num_points)])
    z_points = np.array([[0, 0, i * axis_length / num_points] for i in range(num_points)])

    # Create mesh objects for the lines
    for i in range(num_points - 1):
        x_segment = pyrender.Mesh.from_points(x_points[i:i+2], colors=np.tile(x_color, (2, 1)))
        y_segment = pyrender.Mesh.from_points(y_points[i:i+2], colors=np.tile(y_color, (2, 1)))
        z_segment = pyrender.Mesh.from_points(z_points[i:i+2], colors=np.tile(z_color, (2, 1)))
        
        # Add the line segments to the scene
        scene.add(x_segment)
        scene.add(y_segment)
        scene.add(z_segment)

    rx = np.deg2rad(-90)  # Rotate 90 degrees around X-axis
    ry = np.deg2rad(90)  # Rotate 90 degrees around Y-axis

    # Create the corrective rotation matrices
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    rotation_matrix_y = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # Combine the corrective rotations
    combined_rotation_matrix = rotation_matrix_y @ rotation_matrix_x
    internal_transformed = np.eye(4)
    internal_transformed[:3, :3] = combined_rotation_matrix
    for mesh, pose in arti_obj.visual_trimesh_fk({joint_name: joint_state}).items():
        try: 
            mesh_name = 'textured_objs/'+mesh.metadata['file_name']
        except:
            mesh_name = 'textured_objs/'+mesh.metadata['name']
        if mesh_name in corres_meshes:
            # # mesh.apply_scale(scale)
            scale_mat = np.eye(4) * scale
            scale_mat[3, 3] = 1.0
            mesh.apply_transform(scale_mat @ internal_transformed @ pose['pose'])  
            transformed_meshes.append(mesh)
        mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene.add(mesh)#, pose=pose['pose'])
 
    pyrender.Viewer(scene, use_raymond_lighting=True)

    return transformed_meshes


obj = FileParser("/home/mokhtars/Documents/GraspSampler/DataGenObj/Categories/StorageFurniture/", "45915")
urdf_file_path = '/home/mokhtars/Documents/GraspSampler/DataGenObj/Categories/StorageFurniture/45915/mobility.urdf'
load_urdf_meshes(obj, urdf_file_path=urdf_file_path, scale=1.0, joint_state=0.8, only_articulated=False)
