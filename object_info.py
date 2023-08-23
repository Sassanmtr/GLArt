import pybullet as p
import numpy as np
from helper import local_to_world, apply_rotation, calc_forward_vec, apply_rotation, closest_point_on_line, calc_ray_to_world

class Object:
    def __init__(self, object_id, gripper_id):
        self.object_id = object_id
        self.gripper_id = gripper_id
        self.interaction_joint_link_index = self.get_nearest_link()
        self.interaction_joint_type, self.interaction_joint_axis = self.get_joint_info()
        self.interaction_joint_position = None
        self.min_max_joint_position = None
        self.interaction_joint_angle = None

        self.no_movement_counter = 0

    def setup(self):

        self.interaction_joint_position = self.get_joint_location()
        self.min_max_joint_position = p.getJointInfo(self.object_id, self.interaction_joint_link_index)[8:10]
        self.interaction_joint_angle = p.getJointState(self.object_id, self.interaction_joint_link_index)[0]

    def joint_to_world(self):
        link_world_pos = p.getLinkState(self.object_id, self.interaction_joint_link_index)[4]
        relative_joint_pos = p.getJointInfo(self.object_id, self.interaction_joint_link_index)[14]
        joint_world_positon = [x + y for x, y in zip(link_world_pos, relative_joint_pos)]
        return joint_world_positon


    def get_nearest_link(self):
        # cast a ray from the position to a distant point

        ray_to_world = calc_ray_to_world(self.gripper_id)
        result = p.rayTest(ray_to_world[0], ray_to_world[1])
        if result[0][1] == -1:
            return None
        
        hit_link_index = result[0][1]

        return hit_link_index

    def get_joint_info(self):
        if self.interaction_joint_link_index is None:
            return None, None
        joint_info = p.getJointInfo(self.object_id, self.interaction_joint_link_index)
        joint_type = joint_info[2]
        if joint_type == p.JOINT_REVOLUTE:
            joint_type_str = "Rotational"
        elif joint_type == p.JOINT_PRISMATIC:
            joint_type_str = "Prismatic"
        else:
            return None, None
    

        joint_axis = joint_info[13]  # this is a vector in the joint's local reference frame

        # convert the joint axis to the world frame
        joint_orientation = p.getLinkState(self.object_id, self.interaction_joint_link_index)[1]

        joint_axis_world = local_to_world(joint_axis, joint_orientation)

        return joint_type_str, joint_axis_world.tolist()

    def get_joint_location(self):
        relative_joint_position = p.getJointInfo(self.object_id, self.interaction_joint_link_index)[14]
        axis = apply_rotation(self.interaction_joint_axis, [0.5,-0.5,-0.5,0.5])
        joint_point = apply_rotation(relative_joint_position, [0.5,-0.5,-0.5,0.5])
        location_of_gripper = p.getBasePositionAndOrientation(self.gripper_id)[0]
        joint_location = closest_point_on_line(joint_point, axis, location_of_gripper)
        return joint_location.tolist()
    
    def reached_max_state(self, open, error_margin=0.01):
        if open:
            return p.getJointState(self.object_id, self.interaction_joint_link_index)[0] >= self.min_max_joint_position[1] - error_margin
        if open == False:
            return p.getJointState(self.object_id, self.interaction_joint_link_index)[0] <= self.min_max_joint_position[0] + error_margin
        
    def desired_state(self, error_margin=0.01): #making sure other joints have not moved from their stating position
        joints  = list(filter(lambda i: i != self.interaction_joint_link_index, range(p.getNumJoints(self.object_id))))  #I take out the joint which I am opening and closing since I just want to check that the others stay the same
        for joint in joints:
            base_state = p.getJointState(self.object_id, joint)[0]
            if 0-error_margin >= base_state  or base_state >= 0+error_margin:
                return False
        return True

    def check_movement_of_interaction_joint(self, error_margin=0.01):
        current_joint_angle = p.getJointState(self.object_id, self.interaction_joint_link_index)[0]
        if self.interaction_joint_angle - error_margin <= current_joint_angle <= self.interaction_joint_angle + error_margin:
            self.no_movement_counter += 1
        else:
            self.no_movement_counter = 0
            self.interaction_joint_angle = current_joint_angle

    def success(self, gripper):
        desired_state = self.desired_state()
        if gripper.open == None and desired_state:
            p.disconnect()
            return "Success"
        elif gripper.open == False and desired_state:
            p.disconnect()
            return "Partial success"
        else:
            p.disconnect()
            return "Fail"

