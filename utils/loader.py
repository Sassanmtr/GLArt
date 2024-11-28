import os
import pybullet as p
from utils.gripper import Gripper

def load_gripper(assets_dir, startPosition, start_orientation_quaternion) -> None:
    """
    Loads the gripper into the simulation.

    Parameters:
    - startPosition (list): The starting position of the gripper.
    - start_orientation_quaternion (list): The starting orientation of the gripper in quaternion form.

    scale the gripper to fit the object size!

    Returns:
    - Gripper: An instance of the Gripper class.
    """
    urdf_path = os.path.join(assets_dir, "hand.urdf")
    gripperID = p.loadURDF(urdf_path, startPosition, start_orientation_quaternion,useFixedBase=False, globalScaling = 1.0)
    p.resetJointState(0, 0, 0.04)
    p.resetJointState(0, 1, 0.04)
    joints = range(p.getNumJoints(gripperID))
    p.changeDynamics(
        gripperID, 
        -1
        )
    finger_joints = [0, 1]
    for joints in finger_joints:
        p.changeDynamics(
            gripperID, 
            joints, 
            lateralFriction=7,
            mass=4,
            )
    gripper = Gripper(gripperID)
    return gripper


def load_obj(data_dir, object_number, center_of_object, scale, joint_state):
    """
    Loads the object into the simulation.

    Parameters:
    - object_number (int): The object number to load.
    - center_of_object (list): The center of mass of the object.

    Returns:
    - int: The object ID in the simulation.
    """
    location = os.path.join(data_dir, str(object_number), "mobility.urdf")
    p.multiplyTransforms([0,0,0], [0, 0, 0, 1], center_of_object, [0, 0, 0, 1])
    objID = p.loadURDF(location, globalScaling=scale, useFixedBase=True, 
                       basePosition=
                       [0, 0,  0], 
                       baseOrientation=[0,0,0,1]) 
    joints = range(p.getNumJoints(objID))
    for joint in joints:
        '''
        Mapping of joint types:
        0: JOINT_REVOLUTE
        1: JOINT_PRISMATIC
        2: JOINT_SPHERICAL
        3: JOINT_PLANAR
        4: JOINT_FIXED
        '''
        joint_index = p.getJointInfo(objID, joint)[0]
        joint_type = p.getJointInfo(objID, joint)[2]
        if joint_type == 0:
            revolute_joint_index = joint_index
        elif joint_type == 1:
            prismatic_joint_index = joint_index
    
    try:
        p.resetJointState(objID, revolute_joint_index, joint_state)
    except NameError:
        try:
            p.resetJointState(objID, prismatic_joint_index, joint_state)
        except NameError:
            raise ValueError("No revolute or prismatic joints found for object.")

    for joint in joints:
        p.getDynamicsInfo(objID, joint)
        p.changeDynamics(objID, joint,
                        mass = 0.3,
                        )
    return objID

def load_assets(data_dir, assets_dir, gripper_start_position, gripper_start_orientation, object_number, center_of_object, scale, joint_state):
    """
    Loads all assets into the simulation.

    Parameters:
    - gripper_start_position (list): Starting position of the gripper.
    - gripper_start_orientation (list): Starting orientation of the gripper.
    - object_number (int): The object number to load.
    - center_of_object (list): The center of mass of the object.
    - balanced_cloud (list): The balanced point cloud.

    Returns:
    - tuple: A tuple containing the gripper and object IDs.
    """
    gripper = load_gripper(assets_dir, gripper_start_position, gripper_start_orientation)
    object_id = load_obj(data_dir, object_number, center_of_object, scale, joint_state)
    return gripper, object_id