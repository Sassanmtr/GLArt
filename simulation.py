import time
import pybullet as p
import sim_env
from object_info import Object





def run_simulation(gripper, object_id, steps=1000000, sleep=1/1000, open = True):
    i = 0
    while i < 2000:
        p.stepSimulation()
        time.sleep(sleep)
        i += 1
    return """object.success(gripper)"""
    object = Object(object_id, gripper.gripper_id)
    sim_env.setGravity_global(gripper, 0, 0, 0)
    if object.interaction_joint_type is None:
        p.disconnect()
        return "Fail due to no object infront"
    object.setup()
    relative_orientation = gripper.get_relative_orientation(object.object_id, object.interaction_joint_link_index)

    print("joint lower and upper limit: ", p.getJointInfo(object.object_id, object.interaction_joint_link_index)[8:10])

    #camera setup
    for i in range(steps):  
        if gripper.open == None:
            break
        
        if i % 20 == 0:
            gripper.get_relative_orientation(object.object_id, object.interaction_joint_link_index)
            gripper.adaptive_orientation(relative_orientation, object.object_id, object.interaction_joint_link_index)
            gripper.stabilize_gripper()

  
        if i == 50:
            sim_env.setGravity_global(gripper, 0, 0, -10)

        if  i <= 130:
            gripper.move_gripper([0.5,0,0], 0.1)
            gripper.close_gripper_pos_control(close_length=0.5)        
        
        if i > 130 and i <= 150:
            gripper.move_gripper([0,0,0],0)
            gripper.close_gripper_pos_control(close_length=0.0)
        
        if i > 150 and gripper.open != None:

            object.check_movement_of_interaction_joint()

            gripper.open_and_close(object)

            if object.no_movement_counter > 100:
                p.disconnect()
                return "Fail due to no movement"

        p.stepSimulation()
        time.sleep(sleep)

    
    return object.success(gripper)

    
