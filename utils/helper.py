import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

def local_to_world(axis, orientation):
    """
    Transforms a local axis to world coordinates.

    Parameters:
    - axis (list): Local axis.
    - orientation (list): Quaternion orientation.

    Returns:
    - np.array: World coordinates of the axis.
    """
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    axis_world = rotation_matrix @ np.array(axis)
    return axis_world


def calc_forward_vec(position, orientation, distance):
    """
    Calculate the forward vector based on a position, orientation, and distance.

    Parameters:
    - position (list): Current position.
    - orientation (list): Current orientation.
    - distance (float): Distance to calculate the forward vector for.

    Returns:
    - np.array: The forward vector.
    """
    forward_direction = local_to_world([0, 0, 1], orientation)
    # Scale the forward direction by the distance and add it to the current position
    forward_vec = position + forward_direction * distance
    return forward_vec



def apply_rotation(vector, quaternion):
    """
    Apply rotations to a vector based on given Euler angles.

    Parameters:
    - vector (list): The vector to rotate.
    - theta, phi, psi (float): Euler angles for yaw, pitch, and roll.

    Returns:
    - np.array: The rotated vector.
    """
    # Convert the quaternion to a rotation matrix
    rot_matrix = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
    # Rotate the vector
    rotated_vector = np.dot(rot_matrix, vector)
    return rotated_vector


def closest_point_on_line(line_point, line_direction, point):
    """
    Finds the closest point on a line to a given point.

    Parameters:
    - line_point (list): A point on the line.
    - line_direction (list): The direction vector of the line.
    - point (list): The point to find the closest point on the line to.

    Returns:
    - np.array: The closest point on the line to 'point'.
    """
    line_direction = np.array(line_direction)
    line_direction /= np.linalg.norm(line_direction)
    v = np.array(point) - np.array(line_point)
    d = np.dot(v, line_direction)
    return np.array(line_point) + d * line_direction


def calc_ray_to_world(gripper_id, distance=1):
    """
    Calculates the start and end points of a ray extending from the gripper.

    Parameters:
    - gripper_id (int): The ID of the gripper in the simulation.
    - distance (float): The distance the ray extends from the gripper.

    Returns:
    - tuple: The start and end points of the ray.
    """
    # get the gripper's position and orientation
    base_pos, base_orientation = p.getBasePositionAndOrientation(gripper_id)
    start_point = calc_forward_vec(base_pos, base_orientation, 0.0)
    # calculate the end point of the ray
    end_point = calc_forward_vec(start_point, base_orientation, distance)
    return start_point, end_point


def calculate_normal(v1, v2, v3):
    """
    Calculates the normal vector of a plane defined by three points.

    Parameters:
    - v1, v2, v3 (list): The coordinates of the three points.

    Returns:
    - np.array: The normal vector of the plane.
    """
    # These are vectors defined by subtracting coordinates
    a = v2 - v1
    b = v3 - v1
    # The cross product gives a vector perpendicular to the plane defined by a and b
    return np.cross(a, b)


def transform_open3d_to_pybullet_pointcloud(points):
    """
    Transforms a point cloud from Open3D format to PyBullet partnet format.
    we apply a extra rotation to adjust for the rotation done bythe first joint in ever partnet mobility object

    Parameters:
    - points (list): The point cloud in Open3D format.

    Returns:
    - list: The point cloud in PyBullet format.
    """
    transformed_points = []
    # print("before: ", points[0])
    for point in points:
        transformed_points.append(transform_open3d_to_pybullet(point))
    # print("before: ", transformed_points[0])
    return transformed_points


def transform_open3d_to_pybullet(point):
    x, y, z = point
    return np.array([-z, -x, y]) #[-z, -x, y]


def quaternion_from_vectors(v1, v2):
    """
    Creates a quaternion that represents the rotation from vector v1 to v2.

    Parameters:
    - v1, v2 (list): The vectors.

    Returns:
    - np.array: The quaternion representing the rotation.
    """
    # Calculate the rotation axis
    rotation_axis = np.cross(v1, v2)
    rotation_axis = rotation_axis.astype(float)  # Convert to floating-point
    norm_axis = np.linalg.norm(rotation_axis)
    if norm_axis < 1e-8:
        # Handle nearly parallel vectors
        if np.dot(v1, v2) < 0:  # Check if vectors are pointing in opposite directions
            return np.array([0, 0.0, 1, 0.0])  # Quaternion for 180-degree rotation around y-axis
        else:
            return np.array([1.0, 0.0, 0.0, 0.0])
    rotation_axis /= norm_axis  # Normalize the axis
    # Calculate the cosine of the angle between the vectors
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # Calculate the angle of rotation
    rotation_angle = np.arccos(cos_theta)
    # Calculate the quaternion components
    scalar = np.cos(rotation_angle / 2)
    vector_part = rotation_axis * np.sin(rotation_angle / 2)
    quaternion = np.array([scalar] + list(vector_part))
    return quaternion


def change_quaternion_format_to_xyz_w(quaternion):
    quaternion = list(quaternion)
    return quaternion[1:] + quaternion[:1]


def give_quaternion_roll(quaternion, roll):
    euler = p.getEulerFromQuaternion(quaternion)
    euler = list(euler)
    euler[0] = roll
    # Given Euler angles (in radians)
    euler_angles = np.array([-np.pi/2, np.pi/2, 0])
    additional_angles = np.array([0, 0, np.pi/2])
    # # Convert Euler angles to rotation matrix
    rotation_matrix = Rotation.from_euler('xyz', euler_angles).as_matrix()
    rotation_matrix = np.dot(rotation_matrix, Rotation.from_euler('xyz', additional_angles).as_matrix())
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()  
    return quaternion


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
        initial_orientation_euler_link = [float(joint_state), 0, np.pi/2]
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