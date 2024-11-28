import pybullet as p
import numpy as np
from utils.object_info import calc_forward_vec
from scipy.spatial.transform import Rotation as R

class Gripper:
    """
    A class for controlling a robotic gripper in a simulation environment.
    """
    def __init__(self, gripper_id, max_force=50, mass = 50, lateral_friction = 10, gravity = (0,0,0)):
        """
        Initialize the Gripper class.

        Parameters:
        - gripper_id (int): The ID of the gripper in the simulation environment.
        - max_force (int): The maximum force the gripper can exert.
        - mass (int): The mass of the gripper.
        - lateral_friction (int): The lateral friction of the gripper.
        - gravity (tuple): The gravity vector affecting the gripper.
        """
        self.gripper_id = gripper_id
        self.joint_indices = [0, 1]
        self.max_force = max_force
        self.mass = mass
        self.gravity = gravity
        self.lateral_friction = lateral_friction 
        self.target_orientation = p.getBasePositionAndOrientation(self.gripper_id)[1]
        self.open = True    #gripper starts of opening an object and then closes it


    def stabilize_gripper(self):
        """
        Stabilizes the gripper's position by adjusting its base and orientation.
        """
        grip_pos = self.base_to_grip()
        base_position = p.getBasePositionAndOrientation(self.gripper_id)[0]
        next_gripper_pos = calc_forward_vec(base_position, self.target_orientation, 0.1)
        difference = np.array(next_gripper_pos) - np.array(grip_pos)
        new_base_pos = calc_forward_vec(next_gripper_pos - difference, self.target_orientation, -0.1)  
        p.resetBasePositionAndOrientation(self.gripper_id, new_base_pos, self.target_orientation)


    def base_to_grip(self):
        """
        Calculate the position of the gripper's end-effector based on its base position and orientation.

        Returns:
        - The position of the gripper's grasping point.
        """
        base_position, base_orientation = p.getBasePositionAndOrientation(self.gripper_id)
        base_to_grip = calc_forward_vec(base_position, base_orientation, 0.10)
        return base_to_grip


    def close_gripper_pos_control(self, close_length=0.1, grip_force=70) -> None:
        """
        Close the gripper using position control.

        Parameters:
        - close_length (float): The length to close the gripper to.
        - grip_force (int): The force to apply while closing.
        """
        for joint_index in self.joint_indices:
            p.setJointMotorControl2(
                self.gripper_id, 
                joint_index, 
                p.POSITION_CONTROL, 
                targetPosition=close_length, 
                force=grip_force
                )


    def move_gripper(self, direction: list, velocity: float) -> None:
        """
        Move the gripper in a given direction with a given velocity.

        Parameters:
        - direction (list): The direction vector.
        - velocity (float): The velocity to move at.
        """
        linear_velocity = [d * velocity - g*(1/240)  for d,g in zip(direction, self.gravity)]  #add gravity to the velocity so it floats       
        p.resetBaseVelocity(self.gripper_id, linearVelocity=linear_velocity, angularVelocity=[0, 0, 0])


    def revolute_gripper(self, joint_point, axis):
        """
        Rotate the gripper around a given joint and axis.

        Parameters:
        - joint_point (list): The point around which to rotate.
        - axis (list): The axis to rotate around.
        """
        position = self.base_to_grip()
        vector = np.array(joint_point) - np.array(position)
        orthogonal_vector = np.cross(vector, axis)
        orthogonal_vector /= np.linalg.norm(orthogonal_vector)

 
        # if self.open:
        #     self.move_gripper(orthogonal_vector, 0.15)
        # else:
        #     self.move_gripper(orthogonal_vector, -0.15)
        if self.open:
            self.move_gripper(np.array(position), 0.15)
        else:
            self.move_gripper(np.array(position), -0.15)


    def adaptive_orientation(self, relative_orientation, object_id, joint_index):
        """
        Adapt the gripper's orientation based on the orientation of an object's link.

        Parameters:
        - relative_orientation (list): The relative orientation between the gripper and the object.
        - object_id (int): The ID of the object in the simulation.
        - joint_index (int): The index of the joint to consider for orientation.
        """
        link_orientation = p.getLinkState(object_id, joint_index)[1]
        q_link = R.from_quat(link_orientation)              #converting to class quaternions
        q_rel = R.from_quat(relative_orientation)
        q_resulting =  q_link * q_rel                       #multiplying quaternions to get the resulting orientation
        q_resulting = q_resulting.as_quat()
        self.target_orientation = q_resulting
        

    def get_relative_orientation(self, objID, link_index):
        """
        Get the relative orientation between the gripper and an object's link.

        Parameters:
        - objID (int): The ID of the object in the simulation.
        - link_index (int): The index of the link to consider for orientation.

        Returns:
        - The relative orientation as a quaternion.
        """
        link_orientation = p.getLinkState(objID, link_index)[1]
        q_link = R.from_quat(link_orientation)
        q_link_inv = q_link.inv()
        q_gripper = p.getBasePositionAndOrientation(self.gripper_id)[1]
        q_gripper = R.from_quat(q_gripper)
        q_rel_robot = q_link_inv * q_gripper
        return q_rel_robot.as_quat()
    

    def interact(self, object):
        """
        Interact with an object based on its joint type.

        Parameters:
        - object (Object): The object to interact with.

        Raises:
        - Exception: If the object's joint type is unsupported.
        """
        if object.interaction_joint_type == "Rotational":
            self.revolute_gripper(object.interaction_joint_position, object.interaction_joint_axis)
        elif object.interaction_joint_type == "Prismatic":
            if self.open:
                self.move_gripper(object.interaction_joint_axis, 0.3)
            else:
                self.move_gripper(object.interaction_joint_axis, -0.3)
        else:
            raise Exception("No supported movable joint type")
        

    def open_and_close(self, object):
        """
        Open and close the gripper to interact with an object.

        Parameters:
        - object (Object): The object to interact with.

        Raises:
        - Exception: If the open state is neither True nor False.
        """
        if self.open:
            self.interact(object)
            if object.reached_max_state(open=True):
                self.open = False
        elif not self.open:
            self.interact(object)
            if object.reached_max_state(open=False):
                self.open = None
        else:
            raise Exception("Open variable not set to True or False")