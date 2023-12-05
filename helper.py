import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from pyquaternion import Quaternion


def multiply_quaternions_np(q1, q2):
    """
    Multiplies two quaternions.

    Parameters:
    - q1, q2 (list): Quaternions represented as [x, y, z, w].

    Returns:
    - np.array: The product of the input quaternions.
    """
    v1, w1 =  np.array(q1[:3]), q1[3]
    v2, w2 =  np.array(q2[:3]), q2[3]
    
    w = w1 * w2 - np.dot(v1, v2)
    v = w1 * v2 + w2 * v1 - np.cross(v1, v2)

    return np.insert(v, 0, w)


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

def get_difference(before, after):
    """
    This is an extreemly helpfull documentation for a extreemly complex function, if you ask me, the documentation for this function doesnt make a difference, UNLIKE THE FUNCTION HAHA...
    """
    return np.array(after) - np.array(before)

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

def normalize_quaternion(q):
    """
    Normalizes a quaternion.

    Parameters:
    - q (list): The quaternion to normalize.

    Returns:
    - list: The normalized quaternion.
    """
    norm = np.sqrt(sum([x**2 for x in q]))
    return [x/norm for x in q]



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

def get_angle_between(v1, v2):
    """
    Calculates the angle in degrees between two vectors.

    Parameters:
    - v1, v2 (list): The vectors to find the angle between.

    Returns:
    - float: The angle in degrees.
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def vector_to_quaternion(vector, angle):
    """
    Converts a vector and an angle to a quaternion.

    Parameters:
    - vector (list): The vector to convert.
    - angle (float): The angle to use in the conversion.

    Returns:
    - np.array: The resulting quaternion.
    """
    euler = vector_to_euler(vector)
    quaternion = p.getQuaternionFromEuler(euler)
    np.array(quaternion)
    return np.array(quaternion)

import math

def vector_to_euler(A):
    """
    Converts a vector to Euler angles (yaw, pitch, roll).

    Parameters:
    - A (list): The vector to convert.

    Returns:
    - tuple: The Euler angles (theta, phi, psi).
    """
    x, y, z = A
    
    # Yaw (theta)
    theta = math.atan2(y, -x)
    
    # Pitch (phi)
    phi = math.atan2(z, math.sqrt(x**2 + y**2))
    
    # Roll (psi) - still 0 for this transformation
    psi = 0
    
    return (theta, phi, psi)



def apply_rotations(vector, theta, phi, psi):
    """
    Apply rotations to a vector based on Euler angles.

    Parameters:
    - vector (list): The vector to rotate.
    - theta, phi, psi (float): Euler angles for yaw, pitch, and roll.

    Returns:
    - np.array: The rotated vector.
    """

    R_z = np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]
    ])

    R_y = np.array([
        [math.cos(phi), 0, math.sin(phi)],
        [0, 1, 0],
        [-math.sin(phi), 0, math.cos(phi)]
    ])

    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(psi), -math.sin(psi)],
        [0, math.sin(psi), math.cos(psi)]
    ])

    rotated_vector = np.dot(R_z, np.dot(R_y, np.dot(R_x, vector)))
    return rotated_vector



def transform_open3d_to_pybullet_pointcloud(points):
    """
    Transforms a point cloud from Open3D format to PyBullet partnet format.
    we apply a extra rotation to adjust for the rotation done bythe first joint in ever partnet mobility obj

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
    """
    because different libaries use different quaternion representatoins :/
    """

    quaternion = list(quaternion)

    return quaternion[1:] + quaternion[:1]

def apply_quaternion_to_vector(quaternion, vector):
    """
    does what it says
    """
    q = Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
    rotated_vector = q.rotate(vector)
    return rotated_vector

def give_quaternion_roll(quaternion, roll):
    """
    used to give a quaternion a roll angle, this is needed because the gripper trajectory is initially only determined by the point normal which doesnt include roll
    """
    euler = p.getEulerFromQuaternion(quaternion)
    euler = list(euler)
    euler[0] = roll
    #
    # euler = [0.7, np.pi/2, np.pi/2]
    # euler = [0.4, (0.5*np.pi), (np.pi/2)]
    #
    from scipy.spatial.transform import Rotation

    # Given Euler angles (in radians)
    euler_angles = np.array([-np.pi/2, np.pi/2, 0])
    additional_angles = np.array([0, 0, np.pi/2])
    # # Convert Euler angles to rotation matrix
    rotation_matrix = Rotation.from_euler('xyz', euler_angles).as_matrix()
    rotation_matrix = np.dot(rotation_matrix, Rotation.from_euler('xyz', additional_angles).as_matrix())
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()

    # quaternion = p.getQuaternionFromEuler(euler_angles)
    euler2 = p.getEulerFromQuaternion(quaternion)    
    return quaternion
    # return randomized_orientation_quaternion
def quaternion_to_rotation_matrix(quaternion):
    q0, q1, q2, q3 = quaternion
    return np.array([
        [1 - 2 * q2**2 - 2 * q3**2, 2 * (q1*q2 - q0*q3), 2 * (q1*q3 + q0*q2), 0],
        [2 * (q1*q2 + q0*q3), 1 - 2 * q1**2 - 2 * q3**2, 2 * (q2*q3 - q0*q1), 0],
        [2 * (q1*q3 - q0*q2), 2 * (q2*q3 + q0*q1), 1 - 2 * q1**2 - 2 * q2**2, 0],
        [0, 0, 0, 1]
    ])

def create_transformation_matrix(position, quaternion):
    # transformation_matrix = np.identity(4)
    # transformation_matrix[:3, 3] = position
    transformation_matrix = quaternion_to_rotation_matrix(quaternion)
    transformation_matrix[:3, 3] = position
    return transformation_matrix





"""
vectors = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
    [1, 1, 1],
    [-1, -1, -1]
]

for vector in vectors:
    print("vector: ", vector)
    q = quaternion_from_vectors([-1, 0, 0], vector)
    print("quaternion: ", q)
    print("applying quaternion back to vector to verify function:", apply_quaternion_to_vector(q, [-1, 0, 0]))
    print("------------")
""""""

def change_quaternion_format_to_w_xyz(quaternion):

    x, y, z, w = quaternion

    return Quaternion(w, x, y, z)

def change_quaternion_format_to_xyz_w(quaternion):

    quaternion = list(quaternion)

    return quaternion[1:] + quaternion[:1]

def invert_quaternion(quaternion):
    p = change_quaternion_format_to_w_xyz(quaternion)  # your original quaternion

    q = Quaternion(axis=[0,0,1], degrees=180)  # rotation quaternion for 180 degrees around z-axis
    print(q)
    p_rotated = q * p * q.inverse  # apply the rotation

    rotated = change_quaternion_format_to_xyz_w(p_rotated)
    return rotated


def urdf_orientation_to_world(orientation):
    q_base = Quaternion(0.5, 0.5, -0.5, -0.5) #base orientation of the robot is w_x_y_z 0.5, 0.5, -0.5, -0.5
    orientation = change_quaternion_format_to_w_xyz(orientation)
    world_orientation = q_base * orientation * q_base.inverse
    return world_orientation

"""