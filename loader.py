import os
import pybullet as p
from gripper import Gripper
import urdfpy


def load_gripper(param_config, assets_dir, startPosition, start_orientation_quaternion) -> None:
    """
    Loads the gripper into the simulation.

    Parameters:
    - startPosition (list): The starting position of the gripper.
    - start_orientation_quaternion (list): The starting orientation of the gripper in quaternion form.

    scale the gripper to fit the object size!

    Returns:
    - Gripper: An instance of the Gripper class.
    """
    gripperID = p.loadURDF(assets_dir + "panda_gripper.urdf", startPosition, start_orientation_quaternion,useFixedBase=False, globalScaling = 1.0) #scale important parameter

    p.resetJointState(0, 0, 0.04)
    p.resetJointState(0, 1, 0.04)
    # friction = param_config['GripperLateralFriction']
    # gripper_mass = param_config["GripperMass"]
    # spinning_friction = param_config["GripperSpinningFriction"]
    # resti = 0.99
    joints = range(p.getNumJoints(gripperID))

    p.changeDynamics(
        gripperID, 
        -1, 
        # lateralFriction=friction, 
        # mass=gripper_mass,
        # restitution=resti,
        )
    dyn_info = p.getDynamicsInfo(gripperID, -1)
    # print("dyn_info before", dyn_info)
    finger_joints = [0, 1]
    for joints in finger_joints:
        p.changeDynamics(
            gripperID, 
            joints, 
            # restitution=resti,
            lateralFriction=7,
            # spinningFriction=spinning_friction,
            mass=2,
            )

    gripper = Gripper(gripperID)


    return gripper


def modify_urdf_joint_orientation(data_dir, object_number, joint_state):
    # raw_path = os.path.dirname(data_dir)
    # Load the URDF file
    arti_obj = urdfpy.URDF.load(data_dir+str(object_number)+"/mobility.urdf")
    joint_name =  arti_obj.actuated_joints[0].name
    # Find the joint by name
    joint = arti_obj.joint_map[joint_name]
    orientation_matrix = urdfpy.rpy_to_matrix((0, -joint_state, 0))
    # Update the joint's origin with the specified orientation
    joint.origin[0:3, 0:3] = orientation_matrix
    # Save the modified URDF to a temporary file
    modified_urdf_file = data_dir+str(object_number)+"/modified_urdf.urdf"
    arti_obj.save(modified_urdf_file)

    return modified_urdf_file

def load_obj(param_config, data_dir, object_number, center_of_object, scale, joint_state):
    """
    Loads the object into the simulation.

    Parameters:
    - object_number (int): The object number to load.
    - center_of_object (list): The center of mass of the object.

    Returns:
    - int: The object ID in the simulation.
    """

    # modified_urdf_file = modify_urdf_joint_orientation(data_dir, object_number, joint_state)

    location = data_dir+str(object_number)+"/mobility.urdf"
    world_com, _ = p.multiplyTransforms([0,0,0], [0, 0, 0, 1], center_of_object, [0, 0, 0, 1])

    objID = p.loadURDF(location, globalScaling=scale, useFixedBase=True, 
                       basePosition=
                       [0, 0,  0], 
                       baseOrientation=[0,0,0,1]) #[-center_of_object[2],-center_of_object[0],center_of_object[1]] [-0.01255009, 0.14543785,  -0.23111956], 
    # print(p.getLinkState(objID, 0))
    # lateral_friction = param_config['ObjectLateralFriction']
    # object_mass = param_config["ObjectMass"]
    # spinning_friction = param_config["ObjectSpinningFriction"]

    
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
        joint_name = p.getJointInfo(objID, joint)[1].decode("utf-8")
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
        dyn_info = p.getDynamicsInfo(objID, joint)
        # print("dyn_info before", dyn_info)
        p.changeDynamics(objID, joint,
                        # lateralFriction=lateral_friction, 
                        mass = scale,
                        # spinningFriction=spinning_friction,
                        )

        dyn_info = p.getDynamicsInfo(objID, joint)
        # print("dyn_info", dyn_info)
    return objID

def visualize_points_loader(points, orientation):
    """
    Visualizes points in the simulation.

    Parameters:
    - points (list): List of points to visualize.
    - orientation (list): Orientation of the points.
    """
    

    for point in points:
        p.loadURDF("/block.urdf", globalScaling=1, useFixedBase=True, basePosition=point, baseOrientation=[0,0,0,1])
        #p.loadURDF("/arrow.urdf", globalScaling=0.07, useFixedBase=True, basePosition=point, baseOrientation=orientation)

"""def load_point_cloud_debugging(points):
    box_half_extents = [0.01, 0.01, 0.01]
    gripperID = p.loadURDF("robots/robots/panda_gripper.urdf", [1,1,1], [0,0,0,1], useFixedBase=False)
    p.loadURDF("/arrow.urdf", globalScaling=0.07, useFixedBase=True, basePosition=[1,1,1.4], baseOrientation=[0,0,0,1])
    gripperID = p.loadURDF("robots/robots/panda_gripper.urdf", [1,2,1], [1,1,1,1], useFixedBase=False)
    p.loadURDF("/arrow.urdf", globalScaling=0.07, useFixedBase=True, basePosition=[1,2,1.4], baseOrientation=[1,1,1,1])

    for point in points:
        p.loadURDF("/block.urdf", globalScaling=0.1, useFixedBase=True, basePosition=point[0].tolist(), baseOrientation=[0,0,0,1])
        p.loadURDF("/arrow.urdf", globalScaling=0.07, useFixedBase=True, basePosition=point[0].tolist(), baseOrientation=point[1])
    print(point)"""

def load_point_cloud(point):

    """
    there for debugging
    """

    #p.loadURDF("/block.urdf", globalScaling=0.1, useFixedBase=True, basePosition=point, baseOrientation=[0,0,0,1])
    print(point)

def load_assets(param_config, data_dir, assets_dir, gripper_start_position, gripper_start_orientation, object_number, center_of_object, balanced_cloud, scale, joint_state):
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
    gripper = load_gripper(param_config, assets_dir, gripper_start_position, gripper_start_orientation)
    points = load_point_cloud(balanced_cloud)
    object_id = load_obj(param_config, data_dir, object_number, center_of_object, scale, joint_state)


    return gripper, object_id