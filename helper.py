import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
import open3d as o3d

def multiply_quaternions_np(q1, q2):
    v1, w1 =  np.array(q1[:3]), q1[3]
    v2, w2 =  np.array(q2[:3]), q2[3]
    
    w = w1 * w2 - np.dot(v1, v2)
    v = w1 * v2 + w2 * v1 - np.cross(v1, v2)

    return np.insert(v, 0, w)


def local_to_world(axis, orientation):
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    axis_world = rotation_matrix @ np.array(axis)
    return axis_world

def calc_forward_vec(position, orientation, distance):
    forward_direction = local_to_world([0, 0, 1], orientation)

    # Scale the forward direction by the distance and add it to the current position
    forward_vec = position + forward_direction * distance

    return forward_vec


def apply_rotation(vector, quaternion):

    # Convert the quaternion to a rotation matrix
    rot_matrix = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)

    # Rotate the vector
    rotated_vector = np.dot(rot_matrix, vector)

    return rotated_vector

def get_difference(before, after):
    return np.array(after) - np.array(before)

def closest_point_on_line(line_point, line_direction, point):
    line_direction = np.array(line_direction)
    line_direction /= np.linalg.norm(line_direction)
    v = np.array(point) - np.array(line_point)
    d = np.dot(v, line_direction)
    return np.array(line_point) + d * line_direction

def normalize_quaternion(q):
    norm = np.sqrt(sum([x**2 for x in q]))
    return [x/norm for x in q]



def calc_ray_to_world(gripper_id, distance=0.5):
    # get the gripper's position and orientation
    base_pos, base_orientation = p.getBasePositionAndOrientation(gripper_id)
    start_point = calc_forward_vec(base_pos, base_orientation, 0.12)
    # calculate the end point of the ray
    end_point = calc_forward_vec(start_point, base_orientation, distance)

    return start_point, end_point

def calculate_normal(v1, v2, v3):
    # These are vectors defined by subtracting coordinates
    a = v2 - v1
    b = v3 - v1
    
    # The cross product gives a vector perpendicular to the plane defined by a and b
    return np.cross(a, b)

def get_angle_between(v1, v2):
    """Return the angle in degrees between vectors 'v1' and 'v2'."""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def vector_to_quaternion(vector, angle):
    euler = vector_to_euler(vector)
    quaternion = p.getQuaternionFromEuler(euler)
    np.array(quaternion)
    return np.array(quaternion)

import math

def vector_to_euler(A):
    x, y, z = A
    
    # Yaw (theta)
    theta = math.atan2(y, -x)
    
    # Pitch (phi)
    phi = math.atan2(z, math.sqrt(x**2 + y**2))
    
    # Roll (psi) - still 0 for this transformation
    psi = 0
    
    return (theta, phi, psi)



def apply_rotations(vector, theta, phi, psi):
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

    transformed_points = []
    print("before: ", points[0])
    for point in points:
        transformed_points.append(transform_open3d_to_pybullet(point))
    print("before: ", transformed_points[0])
    return transformed_points

def transform_open3d_to_pybullet(point):
    x, y, z = point
    return np.array([-z, -x, y]) #[-z, -x, y]


from pyquaternion import Quaternion
import numpy as np


def quaternion_from_vectors(v1, v2):
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

def apply_quaternion_to_vector(quaternion, vector):
    q = Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
    rotated_vector = q.rotate(vector)
    return rotated_vector




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