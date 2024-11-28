from __future__ import division

class Robot(object):
    def __init__(self, env, urdf, material, open_gripper=False):
        self.env = env
        self.timestep = env.scene.get_timestep()
        # load robot
        loader = env.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(urdf, {"material": material})
        #self.robot = loader.load(urdf, material)
        self.robot.name = "robot"
        # hand (EE), two grippers, the rest arm joints (if any)
        self.end_effector_index, self.end_effector = \
            [(i, l) for i, l in enumerate(self.robot.get_links()) if l.name == 'panda_hand'][0]
        self.hand_actor_id = self.end_effector.get_id()
        self.gripper_joints = [joint for joint in self.robot.get_joints() if 
                joint.get_name().startswith("panda_finger_joint")]
        self.gripper_actor_ids = [joint.get_child_link().get_id() for joint in self.gripper_joints]
        self.arm_joints = [joint for joint in self.robot.get_joints() if
                joint.get_dof() > 0 and not joint.get_name().startswith("panda_finger")]
        # set drive joint property
        for joint in self.arm_joints:
            joint.set_drive_property(1000, 400)
        for joint in self.gripper_joints:
            joint.set_drive_property(200, 60)
        # open/close the gripper at start
        if open_gripper:
            joint_angles = []
            for j in self.robot.get_joints():
                if j.get_dof() == 1:
                    if j.get_name().startswith("panda_finger_joint"):
                        joint_angles.append(0.04)
                    else:
                        joint_angles.append(0)
            self.robot.set_qpos(joint_angles)


    def close_gripper(self):
        for joint in self.gripper_joints:
            joint.set_drive_target(0.0)


    def open_gripper(self):
        for joint in self.gripper_joints:
            joint.set_drive_target(0.04)


