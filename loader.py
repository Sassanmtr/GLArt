import pybullet as p
from gripper import Gripper


def load_gripper(startPosition, start_orientation_quaternion) -> None:
    """
    Loads the gripper into the simulation.

    Parameters:
    - startPosition (list): The starting position of the gripper.
    - start_orientation_quaternion (list): The starting orientation of the gripper in quaternion form.

    scale the gripper to fit the object size!

    Returns:
    - Gripper: An instance of the Gripper class.
    """

    gripperID = p.loadURDF("robots/robots/panda_gripper.urdf", startPosition, start_orientation_quaternion,useFixedBase=False, globalScaling = 1.0) #scale important parameter

    p.resetJointState(0, 0, 0.04)
    p.resetJointState(0, 1, 0.04)
    friction = 50
    obj_mass = 20

    resti = 1
    joints = range(p.getNumJoints(gripperID))

    p.changeDynamics(
        gripperID, 
        -1, 
        lateralFriction=friction, 
        mass=obj_mass,
        restitution=resti
        )
    finger_joints = [0, 1]
    for joints in finger_joints:
        p.changeDynamics(
            gripperID, 
            joints, 

            restitution=resti,
            lateralFriction=friction
            )

    gripper = Gripper(gripperID)


    return gripper

def load_obj(object_number, center_of_object):
    """
    Loads the object into the simulation.

    Parameters:
    - object_number (int): The object number to load.
    - center_of_object (list): The center of mass of the object.

    Returns:
    - int: The object ID in the simulation.
    """
    object_number_str = str(object_number)

    location = object_number_str+"/mobility.urdf"
    world_com, _ = p.multiplyTransforms([0,0,0], [0, 0, 0, 1], center_of_object, [0, 0, 0, 1])

    objID = p.loadURDF(location, globalScaling=1, useFixedBase=True, 
                       basePosition=
                       [0, 0,  0], 
                       baseOrientation=[0,0,0,1]) #[-center_of_object[2],-center_of_object[0],center_of_object[1]] [-0.01255009, 0.14543785,  -0.23111956], 
    print(p.getLinkState(objID, 0))
    lateral_friction = 50


    joints = range(p.getNumJoints(objID))
    for joint in joints:
        p.changeDynamics(objID, joint,
                        lateralFriction=lateral_friction, 
                        mass = 1,
                        localInertiaDiagonal = [1,1,1],
                        )
        

    return objID

def visualize_points_loader(points, orientation):
    """
    Visualizes points in the simulation.

    Parameters:
    - points (list): List of points to visualize.
    - orientation (list): Orientation of the points.
    """
    

    for point in points:
        p.loadURDF("/block.urdf", globalScaling=0.1, useFixedBase=True, basePosition=point, baseOrientation=[0,0,0,1])
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

def load_assets(gripper_start_position, gripper_start_orientation, object_number, center_of_object, balenced_cloud):
    """
    Loads all assets into the simulation.

    Parameters:
    - gripper_start_position (list): Starting position of the gripper.
    - gripper_start_orientation (list): Starting orientation of the gripper.
    - object_number (int): The object number to load.
    - center_of_object (list): The center of mass of the object.
    - balenced_cloud (list): The balanced point cloud.

    Returns:
    - tuple: A tuple containing the gripper and object IDs.
    """

    gripper = load_gripper(gripper_start_position, gripper_start_orientation)
    points = load_point_cloud(balenced_cloud)
    object_id = load_obj(object_number, center_of_object)


    return gripper, object_id