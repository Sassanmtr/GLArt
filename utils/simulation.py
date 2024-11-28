import time
import pybullet as p
import utils.sim_env as sim_env
from utils.object_info import Object

def run_simulation(gripper, object_id, steps=10000, sleep=1/1000, joint_state=0.0, mode="partial"):
    i = 0
    object = Object(object_id, gripper.gripper_id)
    sim_env.setGravity_global(gripper, 0, 0, 0)
    p.performCollisionDetection() 
    if object.collision() is True:
        return "fail due to gripper_object loading collision"

    if object.interaction_joint_type is None:
        return "Fail due to no object infront"
    
    object.setup()
    relative_orientation = gripper.get_relative_orientation(object.object_id, object.interaction_joint_link_index)

    #camera setup
    start_time = time.time()
    max_simulation_time = 100
    for i in range(steps):
        elapsed_time = time.time() - start_time
        if elapsed_time > max_simulation_time:
            return "Simulation timeout exceeded"  
        if gripper.open == None:
            break
        
        if i % 20 == 0:
            gripper.get_relative_orientation(object.object_id, object.interaction_joint_link_index)
            gripper.adaptive_orientation(relative_orientation, object.object_id, object.interaction_joint_link_index)
            gripper.stabilize_gripper()
  
        if i == 50:
            # Activate gravity after 50 steps
            sim_env.setGravity_global(gripper, 0, 0, -10)

        if  i <= 130:
            # Move the gripper to the object
            gripper.move_gripper([0.5,0,0], 0.1)
            gripper.close_gripper_pos_control(close_length=0.5)        
        
        if i > 130 and i <= 150:
            gripper.move_gripper([0,0,0],0)
            gripper.close_gripper_pos_control(close_length=0.0)
        
        if i > 150 and gripper.open != None:
            # Move the gripper to the initial position
            object.check_movement_of_interaction_joint()

            gripper.open_and_close(object)
            if mode == "partial":
                current_joint_state = object.get_joint_state()
                diff_joint_state = abs(joint_state - current_joint_state)
                if diff_joint_state > 0.2:
                    return "Partial success"

            if object.no_movement_counter > 200:
                return "Fail due to no movement"

        p.stepSimulation()
        time.sleep(sleep)

    return object.success(gripper)
