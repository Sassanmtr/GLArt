import pybullet as p
from utils.helper import local_to_world, apply_rotation, apply_rotation, closest_point_on_line, calc_ray_to_world

class Object:
    """
    Represents a manipulatable object in a simulation.
    """

    def __init__(self, object_id, gripper_id):
        """
        Initializes the Object with an object ID and associated gripper ID.

        Parameters:
        - object_id: ID of the object in the simulation.
        - gripper_id: ID of the gripper used to manipulate the object.
        """
        self.object_id = object_id
        self.gripper_id = gripper_id
        self.interaction_joint_link_index = self.get_nearest_link()
        self.interaction_joint_type, self.interaction_joint_axis = self.get_joint_info()
        self.interaction_joint_position = None
        self.min_max_joint_position = None
        self.interaction_joint_angle = None
        self.no_movement_counter = 0


    def setup(self):
        """
        Sets up the object by determining the position and orientation of its joints.
        """
        self.interaction_joint_position = self.get_joint_location()
        self.min_max_joint_position = p.getJointInfo(self.object_id, self.interaction_joint_link_index)[8:10]
        self.interaction_joint_angle = p.getJointState(self.object_id, self.interaction_joint_link_index)[0]


    def get_nearest_link(self):
        """
        Finds the nearest link on the object to the gripper.

        Returns:
        - int: Index of the nearest link.
        """
        # cast a ray from the position to a distant point

        ray_to_world = calc_ray_to_world(self.gripper_id)
        result = p.rayTest(ray_to_world[0], ray_to_world[1])
        if result[0][1] == -1:
            return None
        
        hit_link_index = result[0][1]

        return hit_link_index


    def get_joint_info(self):
        """
        Retrieves information about the joint used for interaction.

        Returns:
        - tuple: Joint type and its axis in world coordinates.
        """
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
        joint_axis = joint_info[13]  # vector in the joint's local reference frame
        # convert the joint axis to the world frame
        joint_orientation = p.getLinkState(self.object_id, self.interaction_joint_link_index)[1]
        joint_axis_world = local_to_world(joint_axis, joint_orientation)
        return joint_type_str, joint_axis_world.tolist()


    def get_joint_location(self):
        """
        Determines the location of the joint relative to the gripper.

        Returns:
        - list: Coordinates of the joint location.
        """
        relative_joint_position = p.getJointInfo(self.object_id, self.interaction_joint_link_index)[14]
        axis = apply_rotation(self.interaction_joint_axis, [0.5,-0.5,-0.5,0.5])
        joint_point = apply_rotation(relative_joint_position, [0.5,-0.5,-0.5,0.5])
        location_of_gripper = p.getBasePositionAndOrientation(self.gripper_id)[0]
        joint_location = closest_point_on_line(joint_point, axis, location_of_gripper)
        return joint_location.tolist()
    

    def reached_max_state(self, open, error_margin=0.01):
        """
        Checks if the object has reached its maximum state of interaction.

        Parameters:
        - open (bool): Indicates the open/close state.
        - error_margin (float, optional): Margin of error for comparison. Default is 0.01.

        Returns:
        - bool: True if the object reached its maximum state, False otherwise.
        """
        close_error_margin = error_margin + 0.6
        if open:
            return p.getJointState(self.object_id, self.interaction_joint_link_index)[0] >= self.min_max_joint_position[1] - error_margin
        if open == False:
            return p.getJointState(self.object_id, self.interaction_joint_link_index)[0] <= self.min_max_joint_position[0] + close_error_margin
        
    
    def get_joint_state(self):
        return p.getJointState(self.object_id, self.interaction_joint_link_index)[0]
    

    def desired_state(self, error_margin=0.01): #making sure other joints have not moved from their stating position
        """
        Ensures that only the interaction joint is moving, and all other joints remain stationary.

        Parameters:
        - error_margin (float, optional): Margin of error for comparison. Default is 0.01.

        Returns:
        - bool: True if the object is in the desired state, False otherwise.
        """
        joints  = list(filter(lambda i: i != self.interaction_joint_link_index, range(p.getNumJoints(self.object_id))))  #I take out the joint which I am opening and closing since I just want to check that the others stay the same
        for joint in joints:
            base_state = p.getJointState(self.object_id, joint)[0]
            if 0-error_margin >= base_state  or base_state >= 0+error_margin:
                return False
        return True


    def check_movement_of_interaction_joint(self, error_margin=0.01):
        """
        Checks the movement of the interaction joint and updates the no movement counter

        Parameters:
        - error_margin (float, optional): Margin of error for movement comparison
        """
        current_joint_angle = p.getJointState(self.object_id, self.interaction_joint_link_index)[0]
        if self.interaction_joint_angle - error_margin <= current_joint_angle <= self.interaction_joint_angle + error_margin:
            self.no_movement_counter += 1
        else:
            self.no_movement_counter = 0
            self.interaction_joint_angle = current_joint_angle


    def success(self, gripper):
        """
        Determines the success of the manipulation based on the gripper and object state.

        Parameters:
        - gripper (Gripper): The gripper used for manipulation.
        """
        desired_state = self.desired_state()
        if gripper.open == None and desired_state:
            return "Success"
        elif gripper.open == False and desired_state:
            return "Partial success"
        else:
            return "Fail"
        

    def collision(self):
        """
        Checks if there's a collision between the gripper and the object.

        Returns:
        - bool: True if there's a collision, False otherwise.
        """
        collision = p.getContactPoints(self.gripper_id, self.object_id)
        if len(collision) > 0:
            return True
        else:
            return False

