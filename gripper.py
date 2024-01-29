import pybullet as p
import time
import numpy as np
from object_info import calc_forward_vec
from helper import get_difference
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
        difference = get_difference(grip_pos, next_gripper_pos)
        new_base_pos = calc_forward_vec(next_gripper_pos - difference, self.target_orientation, -0.1)  

        p.resetBasePositionAndOrientation(self.gripper_id, new_base_pos, self.target_orientation)

    def set_joint_angles(self, joint_angles):
        """
        Set the joint angles of the gripper.

        Parameters:
        - joint_angles (list): List of joint angles to set.
        """
        for i, angle in enumerate(joint_angles):
            p.setJointMotorControl2(self.gripper_id, self.joint_indices[0], p.POSITION_CONTROL, targetPosition=angle)

    def base_to_grip(self):
        """
        Calculate the position of the gripper's end-effector based on its base position and orientation.

        Returns:
        - The position of the gripper's grasping point.
        """
        base_position, base_orientation = p.getBasePositionAndOrientation(self.gripper_id)
        base_to_grip = calc_forward_vec(base_position, base_orientation, 0.10)
        return base_to_grip
    

    #using position control had best results
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


    #gives the velocity impulse directly to the gripper
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
        this is for opening doors
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

        
        #link_orientation = urdf_orientation_to_world(link_orientation)
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
        

        #q_rel_world =  q_handle * q_gripper.inverse
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

        
                
    """def get_relative_orientation(self, objID, link_index):
        link_orientation = p.getLinkState(objID, link_index)[1]
        link_orientation = local_to_world([0,0,1], link_orientation)
        quaternion = p.getDifferenceQuaternion(self.target_orientation, link_orientation)
            
        print("relative orientationSDUHUDFHWAUDHN", quaternion)
        return quaternion


    def get_relative_orientation(self, objID, link_index):
        link_orientation = p.getLinkState(objID, link_index)[1]
        quaternion = p.getDifferenceQuaternion(link_orientation, self.target_orientation)
        
        print("relative orientationSDUHUDFHWAUDHN", quaternion)
        return normalize_quaternion(quaternion)

    def adaptive_orientation(self, relative_orientation, object_id, joint_index):
        last_orientation = self.last_link_orientation
        self.last_link_orientation = p.getLinkState(object_id, joint_index)[1]
        change_in_link_orientation = p.getDifferenceQuaternion(last_orientation, self.last_link_orientation)

        self.target_orientation = multiply_quaternions_np(self.target_orientation, change_in_link_orientation)
    


    def stabilize_gripper(self):
        current_position, current_orientation = p.getBasePositionAndOrientation(self.gripper_id)
        current_velocity, current_angular_velocity = p.getBaseVelocity(self.gripper_id)

        # Compute the position error
        position_error = [target - current for target, current in zip(self.target_position, current_position)]

        # Compute the velocity error
        velocity_error = [-vel for vel in current_velocity]

        # Compute the control inputs (force and torque)
        force = [self.kp * pos_error + self.kd * vel_error for pos_error, vel_error in zip(position_error, velocity_error)]
        torque = [0, 0, 0]  # You might want to control the orientation as well

        p.applyExternalForce(
            objectUniqueId=self.gripper_id, 
            linkIndex=-1, 
            forceObj=force, 
            posObj=current_position, 
            flags=p.WORLD_FRAME
            )
        p.applyExternalTorque(
            objectUniqueId=self.gripper_id, 
            linkIndex=-1, 
            torqueObj=torque, 
            flags=p.WORLD_FRAME)




    


    #Stuff that didnt work
   

    #maybe unnecessary, doesnt work right now
    def move_gripper_test(self, direction: list, velocity: float) -> None:
        linear_velocity = [i * velocity for i in direction]
        basePosition = p.getBasePositionAndOrientation(self.gripper_id)[0]
        print("base position: ", basePosition)
        p.applyExternalForce(
                             self.gripper_id, 
                             linkIndex=-1, 
                             forceObj=linear_velocity, 
                             posObj=basePosition, 
                             flags=p.WORLD_FRAME
                             )

    def close_gripper_simple(self, closing_steps=100):
        joint_angles_list = [[0.5 - 0.5 * i / closing_steps, -0.5 + 0.5 * i / closing_steps] for i in range(closing_steps)]
        print("joint list: ", joint_angles_list)
        for i in range(closing_steps):
            joint_angles = joint_angles_list[closing_steps - i - 1]
            self.set_joint_angles(joint_angles)
            p.stepSimulation()
            time.sleep(1. / 1000.)

    def close_gripper_experiment(self):
        joint1_index = p.getJointInfo(self.gripper_id, 0)[0]
        joint2_index = p.getJointInfo(self.gripper_id, 1)[0]
        force_magnitude = -20
        print("close gripper")
        p.setJointMotorControl2(
            bodyUniqueId=self.gripper_id, 
            jointIndex=joint1_index, 
            controlMode=p.TORQUE_CONTROL, 
            force=force_magnitude
            )
        
    def open_gripper_experiment(self):
        joint1_index = p.getJointInfo(self.gripper_id, 0)[0]
        joint2_index = p.getJointInfo(self.gripper_id, 1)[0]
        force_magnitude = -20
        print("opening gripper")
        p.setJointMotorControl2(
            bodyUniqueId=self.gripper_id, 
            jointIndex=joint1_index, 
            controlMode=p.TORQUE_CONTROL, 
            force=-force_magnitude
            )
        


    def close_gripper_with_force(self, objId, closing_steps=1000, max_force=2):
        joint_angles_list = [[0.5 - 0.5 * i / closing_steps, -0.5 + 0.5 * i / closing_steps] for i in range(closing_steps)]

        for i in range(closing_steps):
            joint_angles = joint_angles_list[closing_steps - i - 1]
            self.set_joint_angles(joint_angles)

            # Apply force control
            for joint_index in self.joint_indices:
                p.setJointMotorControl2(
                    bodyUniqueId=self.gripper_id,
                    jointIndex=joint_index,
                    controlMode=p.TORQUE_CONTROL,
                    force=max_force
                )

            # Check for collisions and stop applying force when contact is detected
            contact_points = p.getContactPoints(self.gripper_id, objId)
            if len(contact_points) > 0:
                break

            p.stepSimulation()
            time.sleep(1. / 1000.)

        # Switch to POSITION_CONTROL to hold the gripper's position
        self.set_joint_angles(joint_angles)


    def close_gripper_with_force_only(self, objId, force=2000):
        # Apply force control
        contact_points = p.getContactPoints(self.gripper_id, objId)
        print(len(contact_points) == 0)
        print("Joint indices:", self.joint_indices)  # add this line
        while len(contact_points) == 0:
            contact_points = p.getContactPoints(self.gripper_id, objId)
            
            joint_index = self.joint_indices[0]  # Only use the first joint index
            for force_direction in [force, -force]:
                p.setJointMotorControl2(
                    bodyUniqueId=self.gripper_id,
                    jointIndex=joint_index,
                    controlMode=p.TORQUE_CONTROL,
                    force=force_direction
                )

            p.stepSimulation()
            time.sleep(0.01)




    def holding_force(self, force=20):
            for joint_index, force_direction in zip(self.joint_indices, [-force, -force]):
                p.setJointMotorControl2(
                    bodyUniqueId=self.gripper_id,
                    jointIndex=joint_index,
                    controlMode=p.TORQUE_CONTROL,
                    force=force_direction
                )

    def close_gripper_with_external_force_only(self, objId, force_magnitude=2000):
        while True:
            # Apply external force
            for link_index, force_direction in zip(self.joint_indices, [force_magnitude, -force_magnitude]):
                p.applyExternalForce(
                    objectUniqueId=self.gripper_id,
                    linkIndex=link_index,
                    forceObj=[force_direction, 0, 0],  # Change this to the desired force direction
                    posObj=p.getLinkState(self.gripper_id, link_index)[0],
                    flags=p.WORLD_FRAME
                )

        # Check for collisions and stop applying force when contact is detected
        contact_points = p.getContactPoints(self.gripper_id, objId)
        #broken dont use (no contact break condition)
        """