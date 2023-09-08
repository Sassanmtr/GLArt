import time
import pybullet as p
import sim_env
from object_info import Object





def run_simulation(gripper, object_id, steps=10000, sleep=1/1000, open = True):
    i = 0

    object = Object(object_id, gripper.gripper_id)
    sim_env.setGravity_global(gripper, 0, 0, 0)
    p.performCollisionDetection() # because of this checking for the collision is so time efficient 0.04s, we dont actually have to run a sim check, 
    # we setup invirometn and instantly do the collision check
    if object.collision() is True:
        return "fail due to gripper_object loading collision"

    if object.interaction_joint_type is None:

        return "Fail due to no object infront"
    object.setup()
    relative_orientation = gripper.get_relative_orientation(object.object_id, object.interaction_joint_link_index)


    #camera setup
    for i in range(steps):  
        #now we run the actual simulation
        if gripper.open == None:
            break
        
        if i % 20 == 0:
            #every 20 steps we reorient the gripper to the object, this is the unrealistic part of the simulation because we are simply
            #hard reseting the position and orientation of the gripper, due to this we cant do it too often because it doesnt run the friction update 
            #in this step. 20 has turned out to be a good number that we still get a good simulation
            gripper.get_relative_orientation(object.object_id, object.interaction_joint_link_index)
            gripper.adaptive_orientation(relative_orientation, object.object_id, object.interaction_joint_link_index)
            gripper.stabilize_gripper()

  
        if i == 50:
            #once we have grasped we activate gravity
            sim_env.setGravity_global(gripper, 0, 0, -10)

        if  i <= 130:
            #we move the gripper to the object 
            gripper.move_gripper([0.5,0,0], 0.1)
            gripper.close_gripper_pos_control(close_length=0.5)        
        
        if i > 130 and i <= 150:
            #try to grasp the object
            gripper.move_gripper([0,0,0],0)
            gripper.close_gripper_pos_control(close_length=0.0)
        
        if i > 150 and gripper.open != None:
            #if we have grasped the object we move it around
            object.check_movement_of_interaction_joint()

            gripper.open_and_close(object)
            #we have a counter which counts the steps after we have gripped in which the object does not move,
            #if it reaches 200 we fail the simulation
            if object.no_movement_counter > 200:


                return "Fail due to no movement"

        p.stepSimulation()
        time.sleep(sleep)

    
    return object.success(gripper)

    
